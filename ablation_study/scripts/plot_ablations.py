#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FILENAME_RE = re.compile(r"k(?P<kernel>\d+)_c(?P<c>\d+)_gru(?P<gru>\d+)")


@dataclass
class RunSummary:
    filename: str
    kernel: int
    c: int
    gru: int
    epoch: int
    train_loss: float
    val_loss: float
    gap: float  # val - train
    ratio: float  # val / train (safe)
    mode: str  # 'final' or 'minval'


def parse_config_from_path(path: str) -> Optional[Dict[str, Any]]:
    base = os.path.basename(os.path.dirname(path))
    # Try base directory like k3_c16_gru32_YYYY...
    m = FILENAME_RE.search(base)
    if not m:
        # Fallback: try full path
        m = FILENAME_RE.search(path)
    if not m:
        return None
    return {
        "kernel": int(m.group("kernel")),
        "c": int(m.group("c")),
        "gru": int(m.group("gru")),
    }


def safe_ratio(val: float, train: float, eps: float = 1e-8) -> float:
    return val / max(train, eps)


def pick_epoch(data: Dict[str, List[float]], mode: str) -> int:
    vl = data.get("val_loss") or []
    if not vl:
        return len(data.get("train_loss", [])) - 1
    if mode == "minval":
        return int(pd.Series(vl).idxmin())
    # default: final
    return len(vl) - 1


def summarize_runs(all_data: List[Dict[str, Any]], mode: str) -> List[RunSummary]:
    out: List[RunSummary] = []
    for item in all_data:
        filename = item.get("filename", "")
        cfg = parse_config_from_path(filename) or {"kernel": None, "c": None, "gru": None}
        data = item.get("data", {})
        if not data:
            continue
        epoch = pick_epoch(data, mode)
        tl = data.get("train_loss", [])
        vl = data.get("val_loss", [])
        if not tl or not vl:
            # Attempt to fallback to policy+value if total not provided
            tl = data.get("train_policy_loss", [])
            tvl = data.get("train_value_loss", [])
            vl = data.get("val_policy_loss", [])
            vvl = data.get("val_value_loss", [])
            if tl and tvl:
                tl = list(pd.Series(tl).add(pd.Series(tvl), fill_value=0.0))
            if vl and vvl:
                vl = list(pd.Series(vl).add(pd.Series(vvl), fill_value=0.0))
        if not tl or not vl:
            continue
        # Clamp epoch within available range
        epoch = max(0, min(epoch, min(len(tl), len(vl)) - 1))
        t = float(tl[epoch])
        v = float(vl[epoch])
        out.append(
            RunSummary(
                filename=filename,
                kernel=int(cfg["kernel"]) if cfg["kernel"] is not None else -1,
                c=int(cfg["c"]) if cfg["c"] is not None else -1,
                gru=int(cfg["gru"]) if cfg["gru"] is not None else -1,
                epoch=epoch,
                train_loss=t,
                val_loss=v,
                gap=v - t,
                ratio=safe_ratio(v, t),
                mode=mode,
            )
        )
    return out


def to_dataframe(summaries: List[RunSummary]) -> pd.DataFrame:
    rows = [s.__dict__ for s in summaries]
    df = pd.DataFrame(rows)
    # Order columns
    col_order = [
        "filename",
        "kernel",
        "c",
        "gru",
        "epoch",
        "train_loss",
        "val_loss",
        "gap",
        "ratio",
        "mode",
    ]
    return df[col_order]


def plot_scatter(df: pd.DataFrame, threshold: float, outdir: str, title_suffix: str):
    plt.figure(figsize=(7.5, 6))
    sns.set(style="whitegrid")
    kernel_values = sorted(df["kernel"].unique())
    palette = dict(zip(kernel_values, sns.color_palette("Set2", len(kernel_values))))
    ax = sns.scatterplot(
        data=df,
        x="train_loss",
        y="val_loss",
        hue="kernel",
        style="kernel",
        s=80,
        palette=palette,
    )
    lim_max = float(max(df[["train_loss", "val_loss"]].max()))
    ax.plot([0, lim_max], [0, lim_max], "--", color="gray", linewidth=1)

    # Overfit highlighting
    overfit = df[df["ratio"] >= threshold]
    if not overfit.empty:
        ax.scatter(overfit["train_loss"], overfit["val_loss"], s=160, facecolors='none', edgecolors='red', linewidths=1.5, label=f"overfit (ratio ≥ {threshold:g})")
        # Annotate briefly
        for _, r in overfit.iterrows():
            label = f"k{int(r.kernel)}-c{int(r.c)}-g{int(r.gru)}"
            ax.annotate(label, (r.train_loss, r.val_loss), textcoords="offset points", xytext=(6, 6), fontsize=8, color="red")

    ax.set_title(f"Train vs Val Loss {title_suffix}")
    ax.set_xlabel("Train loss")
    ax.set_ylabel("Validation loss")
    ax.legend(title="kernel")
    plt.tight_layout()
    outfile = os.path.join(outdir, f"scatter_train_vs_val{title_suffix.replace(' ', '_')}.png")
    plt.savefig(outfile, dpi=160)
    plt.close()


def plot_heatmaps(df: pd.DataFrame, metric: str, outdir: str, title_suffix: str):
    # One heatmap per kernel setting
    for kernel_val, sub in df.groupby("kernel"):
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="c", columns="gru", values=metric, aggfunc="mean")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot.sort_index().sort_index(axis=1), annot=True, fmt=".3g", cmap="coolwarm", center=0 if metric == "gap" else None)
        plt.title(f"{metric} by c x gru | kernel={kernel_val} {title_suffix}")
        plt.xlabel("gru")
        plt.ylabel("c")
        plt.tight_layout()
        outfile = os.path.join(outdir, f"heatmap_{metric}_kernel_{kernel_val}{title_suffix.replace(' ', '_')}.png")
        plt.savefig(outfile, dpi=160)
        plt.close()


def plot_slopegraph(df: pd.DataFrame, outdir: str, title_suffix: str):
    # Each run: line from train->val; color by overfit
    d = df.copy()
    d["overfit"] = d["ratio"] >= 1.15
    # Build a compact label
    d["label"] = d.apply(lambda r: f"k{int(r.kernel)}-c{int(r.c)}-g{int(r.gru)}", axis=1)
    # Sort by gap to reduce overlap
    d = d.sort_values("gap", ascending=False)

    plt.figure(figsize=(8, 6))
    for _, r in d.iterrows():
        color = "#d7301f" if r["overfit"] else "#1a9850"
        alpha = 0.9 if r["overfit"] else 0.6
        lw = 1.8 if r["overfit"] else 1.2
        plt.plot([0, 1], [r.train_loss, r.val_loss], color=color, alpha=alpha, linewidth=lw)
        plt.scatter([0, 1], [r.train_loss, r.val_loss], color=color, s=30)

    # Axis cosmetics
    plt.xticks([0, 1], ["train", "val"])
    plt.ylabel("Loss")
    plt.title(f"Train→Val slope by run {title_suffix}\nRed=overfit (ratio≥1.15), Green=balanced")
    plt.tight_layout()
    outfile = os.path.join(outdir, f"slope_train_to_val{title_suffix.replace(' ', '_')}.png")
    plt.savefig(outfile, dpi=160)
    plt.close()


def plot_point_and_strip(df: pd.DataFrame, outdir: str, title_suffix: str):
    # Point plot: y=val_loss vs gru, hue=c, facet by kernel
    d = df.copy()
    d["kernel_label"] = d["kernel"].apply(lambda x: f"k{x}" if x >= 0 else "unknown")
    g = sns.catplot(
        data=d,
        x="gru",
        y="val_loss",
        hue="c",
        col="kernel_label",
        kind="point",
        dodge=True,
        height=4,
        aspect=1.1,
        ci=None,
        markers="o",
    )
    g.set_titles("kernel={col_name} " + title_suffix)
    g.set_axis_labels("gru", "Val loss")
    g.fig.tight_layout()
    g.savefig(os.path.join(outdir, f"point_val_vs_gru_by_c{title_suffix.replace(' ', '_')}.png"), dpi=160)
    plt.close(g.fig)

    # Strip plot: distribution of val_loss by c, hue=gru, facet by kernel
    g2 = sns.catplot(
        data=d,
        x="c",
        y="val_loss",
        hue="gru",
        col="kernel_label",
        kind="strip",
        dodge=True,
        height=4,
        aspect=1.1,
        jitter=0.08,
    )
    g2.set_titles("kernel={col_name} " + title_suffix)
    g2.set_axis_labels("c", "Val loss")
    g2.fig.tight_layout()
    g2.savefig(os.path.join(outdir, f"strip_val_by_c_hue_gru{title_suffix.replace(' ', '_')}.png"), dpi=160)
    plt.close(g2.fig)


def plot_gap_lollipop(df: pd.DataFrame, outdir: str, title_suffix: str):
    d = df.copy()
    d["overfit"] = d["ratio"] >= 1.15
    d["cfg"] = d.apply(lambda r: f"k{int(r.kernel)}-c{int(r.c)}-g{int(r.gru)}", axis=1)
    d = d.sort_values("gap", ascending=False)

    plt.figure(figsize=(8, max(4, 0.35 * len(d))))
    y = range(len(d))
    colors = d["overfit"].map({True: "#d7301f", False: "#1a9850"}).tolist()
    # stems
    for i, (_, r) in enumerate(d.iterrows()):
        plt.plot([0, r.gap], [i, i], color=colors[i], linewidth=2)
        plt.scatter([r.gap], [i], color=colors[i], s=25)
    plt.yticks(y, d["cfg"].tolist())
    plt.xlabel("Gap (val - train)")
    plt.title(f"Overfit gap by run {title_suffix} (red=overfit)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"lollipop_gap_by_run{title_suffix.replace(' ', '_')}.png"), dpi=160)
    plt.close()


def plot_learning_curves(
    all_data: List[Dict[str, Any]],
    df: pd.DataFrame,
    outdir: str,
    title_suffix: str,
    top_k: Optional[int] = None,
    per_page: int = 9,
):
    # Map filename->data
    idx = {item.get("filename"): item.get("data", {}) for item in all_data}

    def cfg_label(r):
        return f"k{int(r.kernel)}-c{int(r.c)}-g{int(r.gru)}"

    if top_k is not None and top_k > 0:
        sub = df.nsmallest(top_k, "val_loss")
    else:
        sub = df.sort_values("val_loss")

    runs = list(sub.itertuples(index=False))
    if not runs:
        return

    cols = 3
    page_count = 0
    for start in range(0, len(runs), per_page):
        chunk = runs[start:start + per_page]
        n = len(chunk)
        rows = max(1, (n + cols - 1) // cols)
        plt.figure(figsize=(cols * 4.2, rows * 3.2))
        for i, r in enumerate(chunk, 1):
            data = idx.get(r.filename, {})
            if not data:
                continue
            tl = data.get("train_loss")
            vl = data.get("val_loss")
            if not tl or not vl:
                # try recompute from components
                tl = data.get("train_policy_loss", [])
                tvl = data.get("train_value_loss", [])
                vl = data.get("val_policy_loss", [])
                vvl = data.get("val_value_loss", [])
                if tl and tvl:
                    tl = list(pd.Series(tl).add(pd.Series(tvl), fill_value=0.0))
                if vl and vvl:
                    vl = list(pd.Series(vl).add(pd.Series(vvl), fill_value=0.0))
            if not tl or not vl:
                continue
            ax = plt.subplot(rows, cols, i)
            ax.plot(tl, label="train", color="#3182bd")
            ax.plot(vl, label="val", color="#e6550d")
            mv = int(pd.Series(vl).idxmin())
            ax.scatter([mv], [vl[mv]], color="#e6550d", s=25)
            ax.set_title(cfg_label(r))
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            if start == 0 and i == 1:
                ax.legend()
        page_count += 1
        if top_k is not None and top_k > 0:
            page_label = f"top{top_k}_p{page_count:02d}"
        else:
            page_label = f"all_p{page_count:02d}"
        plt.suptitle(f"Learning curves {page_label} {title_suffix}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        outfile = os.path.join(outdir, f"learning_curves_{page_label}{title_suffix.replace(' ', '_')}.png")
        plt.savefig(outfile, dpi=160)
        plt.close()

def main():
    p = argparse.ArgumentParser(description="Summarize ablations from all.json and plot.")
    p.add_argument("--input", default="all.json", help="Path to all.json")
    p.add_argument("--outdir", default="outputs/plots", help="Directory to save plots")
    p.add_argument("--mode", choices=["final", "minval"], default="final", help="Which epoch to summarize: final or epoch with min val loss")
    p.add_argument("--overfit-threshold", type=float, default=1.15, help="Threshold on (val/train) ratio to flag overfitting")
    p.add_argument("--csv", action="store_true", help="Also save CSV summary")
    p.add_argument("--top-k", type=int, default=0, help="If >0, limit learning-curve grids to the best K runs by val loss; 0 includes all runs")
    args = p.parse_args()

    with open(args.input, "r") as f:
        all_data = json.load(f)

    summaries = summarize_runs(all_data, mode=args.mode)
    if not summaries:
        raise SystemExit("No runs parsed from input. Check filename patterns and data fields.")
    df = to_dataframe(summaries)

    os.makedirs(args.outdir, exist_ok=True)
    title_suffix = f"({args.mode})"

    # Scatter of train vs val with overfit highlighting
    plot_scatter(df, threshold=args.overfit_threshold, outdir=args.outdir, title_suffix=title_suffix)

    # Alternative plots: slopegraph, point/strip, lollipop
    plot_slopegraph(df, outdir=args.outdir, title_suffix=title_suffix)
    plot_point_and_strip(df, outdir=args.outdir, title_suffix=title_suffix)
    plot_gap_lollipop(df, outdir=args.outdir, title_suffix=title_suffix)

    # Heatmaps: gap and ratio (keep for grid view)
    plot_heatmaps(df, metric="gap", outdir=args.outdir, title_suffix=title_suffix)
    plot_heatmaps(df, metric="ratio", outdir=args.outdir, title_suffix=title_suffix)

    if args.csv:
        df.to_csv(os.path.join(args.outdir, f"summary_{args.mode}.csv"), index=False)

    # Learning curves for top runs
    top_k = args.top_k if args.top_k > 0 else None
    plot_learning_curves(all_data, df, outdir=args.outdir, title_suffix=title_suffix, top_k=top_k)

    print(f"Wrote plots to {args.outdir}. Rows summarized: {len(df)}")


if __name__ == "__main__":
    main()

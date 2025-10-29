#!/usr/bin/env python3
"""
Visualization and probing suite for GRU observability analysis.

Consumes outputs produced by scripts/extract_gru_dynamics.py and generates:
  * Gate mean/std trajectories per model
  * Eigenvalue/timescale summaries (heatmaps)
  * PHATE embeddings of hidden states coloured by interpretable features
  * Logistic regression probes measuring how well hidden states encode board features
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import os
from typing import Dict, Iterable, List, Tuple

_DEFAULT_MPL_DIR = Path("diagnostics/mpl_cache").resolve()
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPL_DIR))
_DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False


PROBE_FEATURES: Dict[str, Dict[str, object]] = {
    "current_player": {"type": "binary", "transform": lambda arr: (arr - 1).astype(int)},
    "immediate_win_current": {"type": "binary"},
    "immediate_win_opponent": {"type": "binary"},
    "three_in_row_current": {"type": "binary", "transform": lambda arr: (arr > 0).astype(int)},
    "three_in_row_opponent": {"type": "binary", "transform": lambda arr: (arr > 0).astype(int)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GRU observability plots and probing metrics.")
    parser.add_argument(
        "--analysis-dir",
        default="diagnostics/gru_observability",
        help="Directory containing extract_gru_dynamics outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/gru_observability",
        help="Destination directory for figures and CSV summaries.",
    )
    parser.add_argument(
        "--embedding-epochs",
        nargs="+",
        type=int,
        default=[3, 30, 60, 100],
        help="Epochs to visualise in PHATE embeddings (must exist in hidden_samples).",
    )
    parser.add_argument(
        "--embedding-feature",
        default="move_index",
        help="Feature name used to colour PHATE embeddings.",
    )
    parser.add_argument(
        "--probe-epochs",
        nargs="+",
        type=int,
        default=[30, 60, 100],
        help="Epochs used for logistic regression probing.",
    )
    parser.add_argument(
        "--probe-features",
        nargs="+",
        default=["current_player", "immediate_win_current", "immediate_win_opponent"],
        help="Feature names to evaluate with logistic regression probes.",
    )
    parser.add_argument(
        "--probe-components",
        nargs="+",
        choices=["gru", "cnn"],
        default=["gru"],
        help="Representations to probe (GRU hidden state and/or CNN features).",
    )
    parser.add_argument(
        "--max-hidden-samples",
        type=int,
        default=2000,
        help="Maximum number of hidden samples to use per epoch for embeddings/probing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--palette",
        default="Set2",
        help="Matplotlib/Seaborn palette name for multi-model plots.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip PHATE embedding visualisations.",
    )
    parser.add_argument(
        "--skip-probing",
        action="store_true",
        help="Skip logistic regression probing.",
    )
    return parser.parse_args()


def parse_model_name(name: str) -> Dict[str, int]:
    parts = {}
    for token in name.split("_"):
        if token.startswith("k"):
            parts["kernel"] = int(token[1:])
        elif token.startswith("c"):
            parts["channels"] = int(token[1:])
        elif token.startswith("gru"):
            parts["gru"] = int(token[3:])
    return parts


def load_metrics_table(analysis_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for model_dir in analysis_dir.iterdir():
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        meta = parse_model_name(model_dir.name)
        for key, val in meta.items():
            df[key] = val
        df["model"] = model_dir.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No metrics.csv found under {analysis_dir}")
    combined = pd.concat(rows, ignore_index=True)
    return combined


def plot_gate_trajectories(df: pd.DataFrame, output_dir: Path, palette: str) -> None:
    sns.set_style("whitegrid")
    palette_colors = sns.color_palette(palette, n_colors=df["model"].nunique())
    color_map = {model: palette_colors[idx] for idx, model in enumerate(sorted(df["model"].unique()))}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for model, group in df.groupby("model"):
        axes[0].plot(group["epoch"], group["update_gate_mean"], label=model, color=color_map[model])
        axes[1].plot(group["epoch"], group["reset_gate_mean"], label=model, color=color_map[model])

    axes[0].set_ylabel("Update Gate Mean")
    axes[0].set_title("GRU Update Gate Mean Across Training")
    axes[1].set_ylabel("Reset Gate Mean")
    axes[1].set_xlabel("Epoch")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path = output_dir / "gate_mean_trajectories.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_timescale_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    final_epoch = df.loc[df.groupby("model")["epoch"].idxmax()]
    pivot = final_epoch.pivot_table(
        index="channels",
        columns="gru",
        values="median_timescale",
        aggfunc="mean",
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="mako", cbar_kws={"label": "Median Timescale"})
    plt.title("Median Timescale (final epoch)")
    plt.ylabel("Channels")
    plt.xlabel("GRU Hidden Size")
    output_path = output_dir / "timescale_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_sample_components(
    model_dir: Path,
    epoch: int,
    max_samples: int,
    rng: np.random.Generator,
    keys: Iterable[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str], List[str]]:
    sample_path = model_dir / "hidden_samples" / f"epoch_{epoch:03d}.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Hidden sample not found: {sample_path}")
    data = np.load(sample_path)
    features = data["features"]
    feature_names = [str(name) for name in data["feature_names"]]

    available: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for key in keys:
        if key in data:
            available[key] = data[key]
        else:
            missing.append(str(key))

    if available:
        first_key = next(iter(available))
        base = available[first_key]
        if base.shape[0] > 0:
            max_count = min(max_samples, base.shape[0])
            if base.shape[0] > max_count:
                indices = rng.choice(base.shape[0], size=max_count, replace=False)
            else:
                indices = np.arange(base.shape[0])
            for key in list(available.keys()):
                available[key] = available[key][indices]
            features = features[indices]
        else:
            features = features[: base.shape[0]]
    else:
        if features.shape[0] > max_samples:
            indices = rng.choice(features.shape[0], size=max_samples, replace=False)
            features = features[indices]

    return available, features, feature_names, missing


def load_hidden_samples(
    model_dir: Path,
    epoch: int,
    max_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    arrays, features, feature_names, _ = load_sample_components(
        model_dir, epoch, max_samples, rng, keys=("hidden",)
    )
    hidden = arrays.get("hidden")
    if hidden is None:
        hidden = np.empty((0, 0), dtype=np.float32)
    return hidden, features, feature_names


def clean_hidden_features(
    hidden: np.ndarray, features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if hidden.size == 0:
        return hidden, features

    # Remove rows containing NaNs or infs
    mask_hidden = np.isfinite(hidden).all(axis=1)
    mask_features = np.isfinite(features).all(axis=1)
    mask = mask_hidden & mask_features
    hidden = hidden[mask]
    features = features[mask]

    if hidden.shape[0] == 0:
        return hidden, features

    # Deduplicate exact duplicate hidden states to avoid zero-distance issues in PHATE
    try:
        _, unique_indices = np.unique(hidden, axis=0, return_index=True)
        if unique_indices.size < hidden.shape[0]:
            unique_indices = np.sort(unique_indices)
            hidden = hidden[unique_indices]
            features = features[unique_indices]
    except TypeError:
        # Fallback if axis kw not supported (older numpy)
        seen = {}
        kept_hidden = []
        kept_features = []
        for vec, feat in zip(hidden, features):
            key = tuple(vec.tolist())
            if key in seen:
                continue
            seen[key] = True
            kept_hidden.append(vec)
            kept_features.append(feat)
        if kept_hidden:
            hidden = np.stack(kept_hidden, axis=0)
            features = np.stack(kept_features, axis=0)

    return hidden, features


def plot_phate_embeddings(
    analysis_dir: Path,
    output_dir: Path,
    epochs: Iterable[int],
    feature_name: str,
    max_samples: int,
    rng: np.random.Generator,
) -> None:
    if not HAS_PHATE:
        print("PHATE not installed; skipping embedding plot.")
        return

    models = sorted(p for p in analysis_dir.iterdir() if p.is_dir())
    for epoch in epochs:
        fig, axes = plt.subplots(3, 3, figsize=(15, 14))
        for ax in axes.flat:
            ax.axis("off")

        for idx, model_dir in enumerate(models):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            try:
                hidden, features, feature_names = load_hidden_samples(
                    model_dir, epoch, max_samples, rng
                )
            except FileNotFoundError:
                ax.set_title(model_dir.name)
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                continue

            if hidden.shape[0] == 0:
                ax.set_title(model_dir.name)
                ax.text(0.5, 0.5, "No samples", ha="center", va="center")
                continue

            hidden, features = clean_hidden_features(hidden, features)

            if hidden.shape[0] < 3:
                ax.set_title(model_dir.name)
                ax.text(0.5, 0.5, "Too few samples", ha="center", va="center")
                continue

            knn = min(5, max(2, hidden.shape[0] - 1))
            try:
                reducer = phate.PHATE(n_components=2, knn=knn, random_state=0, verbose=False)
                embedding = reducer.fit_transform(hidden)
            except Exception as exc:
                ax.set_title(model_dir.name)
                ax.text(0.5, 0.5, f"PHATE failed\n{exc}", ha="center", va="center", fontsize=7)
                continue

            if feature_name not in feature_names:
                colour = np.zeros(hidden.shape[0])
            else:
                feature_idx = feature_names.index(feature_name)
                colour = features[:, feature_idx]

            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colour,
                cmap="viridis",
                s=12,
                alpha=0.8,
            )
            ax.set_title(model_dir.name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"PHATE Embedding at Epoch {epoch} (colour: {feature_name})", fontsize=14)
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.6)
        cbar.set_label(feature_name)
        output_path = output_dir / f"phate_epoch_{epoch:03d}_{feature_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def prepare_targets(values: np.ndarray, feature_name: str) -> np.ndarray | None:
    info = PROBE_FEATURES.get(feature_name, {"type": "continuous"})
    arr = values.copy()
    if "transform" in info:
        arr = info["transform"](arr)
    if info.get("type") == "binary":
        arr = arr.astype(int)
        unique = np.unique(arr)
        if unique.size < 2:
            return None
        return arr
    return None  # Unsupported feature type for default probes


def run_probes(
    analysis_dir: Path,
    output_dir: Path,
    epochs: Iterable[int],
    feature_names: Iterable[str],
    components: Iterable[str],
    max_samples: int,
    rng: np.random.Generator,
) -> None:
    component_rows: Dict[str, List[Dict[str, object]]] = {}

    component_sequence = list(dict.fromkeys(components))
    component_map = {"gru": "hidden", "cnn": "cnn"}
    reverse_component_map = {value: key for key, value in component_map.items()}
    requested_keys = tuple(
        dict.fromkeys(
            component_map[name] for name in component_sequence if component_map.get(name) is not None
        ).keys()
    )

    if not requested_keys:
        print("No valid probe components specified; skipping probes.")
        return

    missing_reported: set[Tuple[str, str]] = set()

    for model_dir in sorted(p for p in analysis_dir.iterdir() if p.is_dir()):
        model_meta = parse_model_name(model_dir.name)
        for epoch in epochs:
            try:
                arrays, features, feat_names, missing = load_sample_components(
                    model_dir, epoch, max_samples, rng, keys=requested_keys
                )
            except FileNotFoundError:
                continue
            for missing_key in missing:
                component_label = reverse_component_map.get(missing_key, missing_key)
                marker = (model_dir.name, component_label)
                if marker not in missing_reported:
                    print(
                        f"Skipping {component_label} probes for {model_dir.name}: "
                        f"representation '{missing_key}' not found in epoch {epoch:03d} samples."
                    )
                    missing_reported.add(marker)
            if not arrays:
                continue
            name_to_idx = {name: idx for idx, name in enumerate(feat_names)}

            for component in component_sequence:
                rep_key = component_map.get(component)
                if rep_key is None or rep_key not in arrays:
                    continue
                X = arrays[rep_key]
                if X.shape[0] == 0:
                    continue

                for feature in feature_names:
                    if feature not in name_to_idx:
                        continue
                    y = prepare_targets(features[:, name_to_idx[feature]], feature)
                    if y is None:
                        continue
                    unique, counts = np.unique(y, return_counts=True)
                    if counts.min() < 2:
                        continue
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=0, stratify=y
                    )
                    if np.unique(y_train).size < 2:
                        continue

                    # Standardize features for better convergence
                    # Many hidden dimensions may have very different scales
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Real probe with increased iterations and L2 regularization
                    clf = LogisticRegression(
                        max_iter=5000,
                        solver='lbfgs',
                        C=1.0,  # Regularization strength
                        random_state=0,
                        tol=1e-4
                    )
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="binary")

                    # Control task: permuted labels
                    y_train_permuted = rng.permutation(y_train)
                    y_test_permuted = rng.permutation(y_test)
                    clf_control = LogisticRegression(
                        max_iter=5000,
                        solver='lbfgs',
                        C=1.0,
                        random_state=0,
                        tol=1e-4
                    )
                    clf_control.fit(X_train_scaled, y_train_permuted)
                    y_pred_control = clf_control.predict(X_test_scaled)
                    acc_control = accuracy_score(y_test_permuted, y_pred_control)

                    row = {
                        "model": model_dir.name,
                        "epoch": epoch,
                        "feature": feature,
                        "component": component,
                        "accuracy": float(acc),
                        "f1": float(f1),
                        "control_accuracy": float(acc_control),
                        "signal_over_control": float(acc - acc_control),
                    }
                    row.update(model_meta)
                    component_rows.setdefault(component, []).append(row)

    if not component_rows:
        print("No probe results computed.")
        return

    for component in component_sequence:
        rows = component_rows.get(component)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        use_subdir = component != "gru" or len(component_rows) > 1
        component_dir = output_dir / component if use_subdir else output_dir
        component_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(component_dir / "probe_results.csv", index=False)

        # Plot 1: Regular accuracy
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x="epoch",
            y="accuracy",
            hue="feature",
            style="gru",
            markers=True,
        )
        title = f"{component.upper()} Probe Accuracy over Epochs"
        plt.title(title)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig(component_dir / "probe_accuracy.png", dpi=300)
        plt.close()

        # Plot 2: Signal over control (new)
        if "signal_over_control" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x="epoch",
                y="signal_over_control",
                hue="feature",
                style="gru",
                markers=True,
            )
            plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            title = f"{component.upper()} Signal Over Control Task"
            plt.title(title)
            plt.ylabel("Accuracy - Control Accuracy")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(component_dir / "probe_signal_over_control.png", dpi=300)
            plt.close()

            # Plot 3: Comparison plot (real vs control)
            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Real accuracy
            sns.lineplot(
                data=df,
                x="epoch",
                y="accuracy",
                hue="feature",
                style="gru",
                markers=True,
                ax=ax1,
            )
            ax1.set_title(f"{component.upper()} Real Probe Accuracy")
            ax1.set_ylabel("Accuracy")
            ax1.set_xlabel("Epoch")

            # Control accuracy
            sns.lineplot(
                data=df,
                x="epoch",
                y="control_accuracy",
                hue="feature",
                style="gru",
                markers=True,
                ax=ax2,
            )
            ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
            ax2.set_title(f"{component.upper()} Control Task Accuracy (Permuted Labels)")
            ax2.set_ylabel("Accuracy")
            ax2.set_xlabel("Epoch")

            plt.tight_layout()
            plt.savefig(component_dir / "probe_comparison.png", dpi=300)
            plt.close()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    metrics_df = load_metrics_table(analysis_dir)

    plot_gate_trajectories(metrics_df, output_dir, palette=args.palette)
    plot_timescale_heatmap(metrics_df, output_dir)
    if not args.skip_embedding:
        plot_phate_embeddings(
            analysis_dir,
            output_dir,
            epochs=args.embedding_epochs,
            feature_name=args.embedding_feature,
            max_samples=args.max_hidden_samples,
            rng=rng,
        )
    if not args.skip_probing:
        run_probes(
            analysis_dir,
            output_dir,
            epochs=args.probe_epochs,
            feature_names=args.probe_features,
            components=args.probe_components,
            max_samples=args.max_hidden_samples,
            rng=rng,
        )

    summary_path = output_dir / "metrics_summary.json"
    summary = {
        "analysis_dir": str(analysis_dir),
        "embedding_epochs": args.embedding_epochs,
        "embedding_feature": args.embedding_feature,
        "probe_epochs": args.probe_epochs,
        "probe_features": args.probe_features,
        "skip_embedding": args.skip_embedding,
        "skip_probing": args.skip_probing,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()

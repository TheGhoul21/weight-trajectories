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
import time

_DEFAULT_MPL_DIR = Path("diagnostics/mpl_cache").resolve()
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPL_DIR))
_DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False

# If available, use imageio-ffmpeg to provide a bundled ffmpeg binary for Matplotlib
try:
    import imageio_ffmpeg  # type: ignore
    _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    if _FFMPEG_EXE and os.path.exists(_FFMPEG_EXE):
        matplotlib.rcParams["animation.ffmpeg_path"] = _FFMPEG_EXE
except Exception:
    _FFMPEG_EXE = None


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
        default=list(range(1, 101, 2)),
        help="Epochs to visualise in PHATE embeddings (must exist in hidden_samples).",
    )
    parser.add_argument(
        "--embedding-feature",
        default="move_index",
        help="Feature name used to colour PHATE embeddings.",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=["separate", "joint"],
        default="separate",
        help="How to compute embeddings across epochs for animation: 'separate' fits per-epoch (fast, may flip); 'joint' fits once per model on pooled epochs (stable axes).",
    )
    parser.add_argument(
        "--embedding-joint-samples",
        type=int,
        default=300,
        help="Max samples per epoch per model to pool when --embedding-mode=joint (controls memory).",
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
        "--embedding-animate",
        action="store_true",
        help="Render an animation over embedding-epochs (3x3 grid across models).",
    )
    parser.add_argument(
        "--embedding-fps",
        type=int,
        default=4,
        help="Frames per second for embedding animation.",
    )
    parser.add_argument(
        "--embedding-format",
        choices=["auto", "mp4", "gif"],
        default="auto",
        help="Animation output format (mp4 requires ffmpeg; gif uses PillowWriter).",
    )
    parser.add_argument(
        "--embedding-dpi",
        type=int,
        default=150,
        help="DPI for saved animation frames.",
    )
    parser.add_argument(
        "--embedding-point-size",
        type=float,
        default=12.0,
        help="Marker size for PHATE scatter points (points^2).",
    )
    parser.add_argument(
        "--embedding-alpha",
        type=float,
        default=0.8,
        help="Alpha (transparency) for PHATE scatter points.",
    )
    parser.add_argument(
        "--embedding-dedup",
        choices=["auto", "soft", "off"],
        default="auto",
        help="Control deduplication of identical hidden states before PHATE: 'auto' keeps unique rows only; 'soft' ensures a minimum retained sample size by re-adding duplicates; 'off' disables dedup.",
    )
    parser.add_argument(
        "--embedding-dedup-min-fraction",
        type=float,
        default=0.1,
        help="When --embedding-dedup=soft, keep at least this fraction of the original samples (0-1).",
    )
    parser.add_argument(
        "--embedding-dedup-min-count",
        type=int,
        default=50,
        help="When --embedding-dedup=soft, keep at least this many samples after dedup.",
    )
    parser.add_argument(
        "--embedding-jitter",
        type=float,
        default=0.0,
        help="Optional small Gaussian noise added when re-adding duplicates in 'soft' mode to avoid exact ties (e.g., 1e-7).",
    )
    parser.add_argument(
        "--skip-probing",
        action="store_true",
        help="Skip logistic regression probing.",
    )
    parser.add_argument(
        "--only-confusions",
        action="store_true",
        help="Generate only confusion-matrix plots from probes (skip other probe visualizations).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=20,
        help="Print progress every N fits/steps (>=1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes/threads to use (currently unused; reserved for future parallelism).",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Handle class imbalance in probes by using sklearn's class_weight (default: balanced).",
    )
    parser.add_argument(
        "--balance-train",
        action="store_true",
        help="Under-sample the majority class in the training split to approximate a 1:1 class ratio.",
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


def _get_best_and_final_saved_epoch(model_name: str, analysis_dir: Path) -> tuple[int | None, int | None]:
    """Return (best_saved_epoch, final_saved_epoch) for a model.

    Resolves the model's checkpoints directory even when names include timestamps by
    preferring checkpoints/<model_name>/training_history.json if it exists, otherwise
    the most recent checkpoints/<model_name>_*/training_history.json.

    Uses the "epochs_saved" list to select the epoch with minimal val_loss among those saved.
    Falls back to inferring only the final available epoch from hidden_samples if history is missing.
    """
    try:
        # Primary path: exact match
        hist_path = Path("checkpoints") / model_name / "training_history.json"
        # Fallback: any directory that starts with the model prefix (e.g., has a timestamp)
        if not hist_path.exists():
            candidates = sorted(Path("checkpoints").glob(f"{model_name}_*/training_history.json"))
            if candidates:
                # pick the most recent by modified time; tie-breaker: lexical order
                hist_path = max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))
        if hist_path.exists():
            with hist_path.open("r") as f:
                hist = json.load(f)
            val_loss = hist.get("val_loss", [])
            epochs_saved = hist.get("epochs_saved")
            if not epochs_saved:
                # Fallback to hidden_samples epochs
                samples_dir = analysis_dir / model_name / "hidden_samples"
                if samples_dir.exists():
                    epochs_saved = []
                    for p in samples_dir.glob("epoch_*.npz"):
                        try:
                            epochs_saved.append(int(p.stem.split("_")[-1]))
                        except Exception:
                            pass
            best_e, best_v = None, float("inf")
            for e in epochs_saved or []:
                idx = int(e) - 1
                if 0 <= idx < len(val_loss):
                    v = val_loss[idx]
                    if v < best_v:
                        best_v = v
                        best_e = int(e)
            final_e = int(max(epochs_saved)) if epochs_saved else None
            return best_e, final_e
    except Exception:
        pass
    # Last resort: infer final from hidden_samples only
    samples_dir = analysis_dir / model_name / "hidden_samples"
    if not samples_dir.exists():
        return None, None
    epochs = []
    for p in samples_dir.glob("epoch_*.npz"):
        try:
            epochs.append(int(p.stem.split("_")[-1]))
        except Exception:
            pass
    if not epochs:
        return None, None
    return None, int(max(epochs))


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
    hidden: np.ndarray,
    features: np.ndarray,
    *,
    dedup: bool | str = True,
    min_fraction: float = 0.1,
    min_count: int = 50,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
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

    # Deduplicate identical hidden states to avoid zero-distance issues in PHATE
    # dedup: True/'auto' => keep uniques only; 'soft' => ensure minimum retained size by re-adding duplicates; False/'off' => no dedup
    mode = dedup
    if mode is True:
        mode = 'auto'
    if isinstance(mode, str):
        mode = mode.lower()
    if mode in ('auto', 'soft'):
        try:
            uniq, unique_indices, inverse, counts = np.unique(
                hidden, axis=0, return_index=True, return_inverse=True, return_counts=True
            )
            unique_indices = np.sort(unique_indices)
        except TypeError:
            # Fallback if axis kw not supported (older numpy)
            seen = {}
            unique_indices = []
            inverse_list = []
            for i, vec in enumerate(hidden):
                key = tuple(vec.tolist())
                if key not in seen:
                    seen[key] = len(seen)
                    unique_indices.append(i)
                inverse_list.append(seen[key])
            inverse = np.array(inverse_list, dtype=int)
            uniq_count = len(seen)
            counts = np.zeros(uniq_count, dtype=int)
            for idx in inverse:
                counts[idx] += 1
            unique_indices = np.array(sorted(unique_indices), dtype=int)

        # Start with one representative per duplicate group
        kept_idx = list(unique_indices.tolist())
        if mode == 'soft':
            n0 = hidden.shape[0]
            target = max(3, int(max(min_fraction, 0.0) * n0), int(min_count))
            if len(kept_idx) < target:
                # Build pools of extra indices for each group
                group_to_indices: dict[int, list[int]] = {}
                for orig_idx, g in enumerate(inverse):
                    # Skip the first representative; others are extras
                    if orig_idx == unique_indices[g]:
                        continue
                    group_to_indices.setdefault(int(g), []).append(orig_idx)
                # Round-robin add extras across groups to reach target size
                extras: list[int] = []
                # Deterministic order: iterate groups by descending counts, then by id
                groups_order = [int(g) for g in np.argsort(counts)[::-1].tolist()]
                ptrs = {g: 0 for g in groups_order}
                while len(kept_idx) + len(extras) < min(target, n0):
                    progressed = False
                    for g in groups_order:
                        lst = group_to_indices.get(g, [])
                        p = ptrs[g]
                        if p < len(lst):
                            extras.append(lst[p])
                            ptrs[g] = p + 1
                            progressed = True
                            if len(kept_idx) + len(extras) >= min(target, n0):
                                break
                    if not progressed:
                        break  # no more extras available
                if extras:
                    add_idx = np.array(extras, dtype=int)
                    kept_idx = np.array(kept_idx + extras, dtype=int)
                    # Apply tiny jitter if requested, only to the added duplicates to avoid exact ties
                    if jitter and jitter > 0.0:
                        noise = (rng.normal(0.0, jitter, size=(add_idx.size, hidden.shape[1])) if rng is not None
                                 else np.random.normal(0.0, jitter, size=(add_idx.size, hidden.shape[1])))
                        # Create a copy to avoid mutating the original array outside this function
                        hidden = hidden.copy()
                        hidden[add_idx] = hidden[add_idx] + noise.astype(hidden.dtype, copy=False)
                else:
                    kept_idx = np.array(kept_idx, dtype=int)
            else:
                kept_idx = np.array(kept_idx, dtype=int)
        else:
            kept_idx = np.array(kept_idx, dtype=int)

        hidden = hidden[kept_idx]
        features = features[kept_idx]

    return hidden, features


def _sort_models_grid(models: list[Path]) -> list[Path]:
    def parse_model_name(name: str) -> tuple[int, int, int, str]:
        # Expect tokens like k3_c64_gru32_...
        k = c = g = 10**9
        for tok in name.split("_"):
            if tok.startswith("k") and tok[1:].isdigit():
                k = int(tok[1:])
            elif tok.startswith("c") and tok[1:].isdigit():
                c = int(tok[1:])
            elif tok.startswith("gru") and tok[3:].isdigit():
                g = int(tok[3:])
        return (k, c, g, name)
    return sorted(models, key=lambda p: parse_model_name(p.name))


def plot_phate_embeddings(
    analysis_dir: Path,
    output_dir: Path,
    epochs: Iterable[int],
    feature_name: str,
    max_samples: int,
    rng: np.random.Generator,
    *,
    point_size: float = 12.0,
    alpha: float = 0.8,
    dedup: bool | str = True,
    dedup_min_fraction: float = 0.1,
    dedup_min_count: int = 50,
    jitter: float = 0.0,
) -> None:
    if not HAS_PHATE:
        print("PHATE not installed; skipping embedding plot.")
        return

    models = _sort_models_grid([p for p in analysis_dir.iterdir() if p.is_dir()])[:9]
    for epoch in epochs:
        print(f"[Embed] PHATE embeddings for epoch {epoch} (feature='{feature_name}')", flush=True)
        fig, axes = plt.subplots(3, 3, figsize=(15, 14))
        # Leave room at right for a vertical colorbar
        fig.subplots_adjust(right=0.88)
        for ax in axes.flat:
            # Keep axes on, but hide ticks and spines for consistent text rendering
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

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

            hidden, features = clean_hidden_features(
                hidden, features,
                dedup=dedup,
                min_fraction=dedup_min_fraction,
                min_count=dedup_min_count,
                jitter=jitter,
                rng=rng,
            )

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
                s=float(point_size),
                alpha=float(alpha),
            )
            ax.set_title(model_dir.name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"PHATE Embedding at Epoch {epoch} (colour: {feature_name})", fontsize=14)
        # Layout first, reserving space on the right for the colorbar
        fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
        # Dedicated colorbar axis to avoid overlapping last column
        cbar_ax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(feature_name)
        output_path = output_dir / f"phate_epoch_{epoch:03d}_{feature_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"[Embed] Saved {output_path}", flush=True)


def animate_phate_embeddings(
    analysis_dir: Path,
    output_dir: Path,
    epochs: Iterable[int],
    feature_name: str,
    max_samples: int,
    rng: np.random.Generator,
    fps: int = 4,
    fmt: str = "auto",
    dpi: int = 150,
    *,
    point_size: float = 12.0,
    alpha: float = 0.8,
    dedup: bool | str = True,
    dedup_min_fraction: float = 0.1,
    dedup_min_count: int = 50,
    jitter: float = 0.0,
) -> None:
    """Generate a 3x3 grid animation of PHATE embeddings over epochs.

    Notes:
    - Uses consistent axis limits across frames for stability.
    - Uses a global color normalization across all frames.
    - Only the first 9 models are included to fit a 3x3 grid.
    """
    if not HAS_PHATE:
        print("PHATE not installed; skipping animation.")
        return

    sel_epochs = [int(e) for e in sorted(set(int(e) for e in epochs))]
    models_all = _sort_models_grid([p for p in analysis_dir.iterdir() if p.is_dir()])
    if not models_all:
        print("[Embed] No models found for animation.")
        return
    models = models_all[:9]
    if len(models_all) > 9:
        print(f"[Embed] More than 9 models found; animating first 9: {[m.name for m in models]}")

    # Precompute embeddings and colors for all (epoch, model)
    cache: Dict[tuple[int, str], tuple[np.ndarray | None, np.ndarray | None]] = {}
    # Track global axis limits and color range
    xmin: Dict[str, float] = {m.name: float("inf") for m in models}
    xmax: Dict[str, float] = {m.name: float("-inf") for m in models}
    ymin: Dict[str, float] = {m.name: float("inf") for m in models}
    ymax: Dict[str, float] = {m.name: float("-inf") for m in models}
    cmin = float("inf")
    cmax = float("-inf")
    # Per-model orientation anchoring based on correlation with the chosen feature
    # For each model, pick the PHATE axis (0 or 1) that best correlates with the feature
    # at the first available epoch, and fix its sign to keep correlation positive.
    orientation: Dict[str, Dict[str, int]] = {}

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel(); b = b.ravel()
        if a.size != b.size or a.size < 3:
            return 0.0
        sa, sb = np.std(a), np.std(b)
        if sa <= 1e-12 or sb <= 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    print(f"[Embed] Precomputing PHATE embeddings for animation over {len(sel_epochs)} epochs × {len(models)} models…", flush=True)

    # Decide mode: separate vs joint from outer scope (args)
    # We'll detect via presence on global namespace when calling; fallback to separate
    # Instead, read environment var set by caller (simple hook) — but better: infer from function default not available.
    # We pass mode via a closure variable in main(); emulate by checking attribute set later.
    mode = getattr(animate_phate_embeddings, "_mode", "separate")
    joint_samples = getattr(animate_phate_embeddings, "_joint_samples", 300)

    if mode == "joint":
        # For each model, pool across epochs with per-epoch cap and fit once; then slice per epoch
        for model_dir in models:
            pooled_hidden = []
            pooled_colour = []
            pooled_epoch = []
            first_feat_names = None
            for epoch in sel_epochs:
                try:
                    hidden, features, feature_names = load_hidden_samples(model_dir, epoch, min(joint_samples, max_samples), rng)
                except FileNotFoundError:
                    continue
                if hidden.shape[0] == 0:
                    continue
                hidden, features = clean_hidden_features(
                    hidden, features,
                    dedup=dedup,
                    min_fraction=dedup_min_fraction,
                    min_count=dedup_min_count,
                    jitter=jitter,
                    rng=rng,
                )
                if hidden.shape[0] < 3:
                    continue
                if first_feat_names is None:
                    first_feat_names = feature_names
                # Determine colouring
                if feature_name in feature_names:
                    ci = feature_names.index(feature_name)
                    colour = features[:, ci]
                else:
                    colour = np.zeros(hidden.shape[0], dtype=float)
                pooled_hidden.append(hidden)
                pooled_colour.append(colour)
                pooled_epoch.append(np.full(hidden.shape[0], epoch, dtype=int))
            if not pooled_hidden:
                # Fill cache with None for all epochs for this model
                for epoch in sel_epochs:
                    cache[(epoch, model_dir.name)] = (None, None)
                continue
            H = np.concatenate(pooled_hidden, axis=0)
            C = np.concatenate(pooled_colour, axis=0)
            E = np.concatenate(pooled_epoch, axis=0)
            knn = min(10, max(2, H.shape[0] - 1))
            try:
                reducer = phate.PHATE(n_components=2, knn=knn, random_state=0, verbose=False)
                EMB = reducer.fit_transform(H)
            except Exception as exc:
                print(f"[Embed] PHATE joint fit failed for {model_dir.name}: {exc}")
                for epoch in sel_epochs:
                    cache[(epoch, model_dir.name)] = (None, None)
                continue
            # Orientation based on correlation with feature for this model
            orient_axis = 0
            orient_sign = 1
            c0 = _corr(EMB[:, 0], C)
            c1 = _corr(EMB[:, 1], C)
            if abs(c1) > abs(c0):
                orient_axis = 1
                orient_sign = 1 if c1 >= 0 else -1
            else:
                orient_axis = 0
                orient_sign = 1 if c0 >= 0 else -1
            X = orient_sign * EMB[:, orient_axis]
            Y = EMB[:, 1 - orient_axis]
            EMB2 = np.stack([X, Y], axis=1)
            # Update limits per model
            xmin[model_dir.name] = float(np.min(EMB2[:, 0])); xmax[model_dir.name] = float(np.max(EMB2[:, 0]))
            ymin[model_dir.name] = float(np.min(EMB2[:, 1])); ymax[model_dir.name] = float(np.max(EMB2[:, 1]))
            cmin = min(cmin, float(np.min(C))); cmax = max(cmax, float(np.max(C)))
            # Slice back per epoch
            for epoch in sel_epochs:
                mask = (E == epoch)
                if not np.any(mask):
                    cache[(epoch, model_dir.name)] = (None, None)
                else:
                    cache[(epoch, model_dir.name)] = (EMB2[mask], C[mask])
    else:
        # Separate fits (per-epoch)
        for epoch in sel_epochs:
            for model_dir in models:
                key = (epoch, model_dir.name)
                try:
                    hidden, features, feature_names = load_hidden_samples(model_dir, epoch, max_samples, rng)
                except FileNotFoundError:
                    cache[key] = (None, None)
                    continue
                if hidden.shape[0] == 0:
                    cache[key] = (None, None)
                    continue
                hidden, features = clean_hidden_features(
                    hidden, features,
                    dedup=dedup,
                    min_fraction=dedup_min_fraction,
                    min_count=dedup_min_count,
                    jitter=jitter,
                    rng=rng,
                )
                if hidden.shape[0] < 3:
                    cache[key] = (None, None)
                    continue
                knn = min(5, max(2, hidden.shape[0] - 1))
                try:
                    reducer = phate.PHATE(n_components=2, knn=knn, random_state=0, verbose=False)
                    emb = reducer.fit_transform(hidden)
                except Exception as exc:
                    print(f"[Embed] PHATE failed for {model_dir.name} epoch {epoch}: {exc}")
                    cache[key] = (None, None)
                    continue

                if feature_name in feature_names:
                    ci = feature_names.index(feature_name)
                    colour = features[:, ci]
                else:
                    colour = np.zeros(emb.shape[0], dtype=float)

                # Orientation stabilisation: determine and apply model-wise axis/sign choice
                # Use the first epoch with valid data to anchor which axis correlates best with the feature
                orient = orientation.get(model_dir.name)
                if orient is None and colour.size > 0:
                    c0 = _corr(emb[:, 0], colour)
                    c1 = _corr(emb[:, 1], colour)
                    if abs(c1) > abs(c0):
                        chosen_axis = 1
                        chosen_sign = 1 if c1 >= 0 else -1
                    else:
                        chosen_axis = 0
                        chosen_sign = 1 if c0 >= 0 else -1
                    orientation[model_dir.name] = {"axis": chosen_axis, "sign": chosen_sign}
                    orient = orientation[model_dir.name]
                # Apply orientation if available
                if orient is not None:
                    ax_idx = orient["axis"]
                    sign = orient["sign"]
                    x = sign * emb[:, ax_idx]
                    y = emb[:, 1 - ax_idx]
                    emb = np.stack([x, y], axis=1)

                cache[key] = (emb, colour)
                # Update limits
                if emb.size:
                    mnx, mxx = float(np.min(emb[:, 0])), float(np.max(emb[:, 0]))
                    mny, mxy = float(np.min(emb[:, 1])), float(np.max(emb[:, 1]))
                    xmin[model_dir.name] = min(xmin[model_dir.name], mnx)
                    xmax[model_dir.name] = max(xmax[model_dir.name], mxx)
                    ymin[model_dir.name] = min(ymin[model_dir.name], mny)
                    ymax[model_dir.name] = max(ymax[model_dir.name], mxy)
                if colour.size:
                    cmin = min(cmin, float(np.min(colour)))
                    cmax = max(cmax, float(np.max(colour)))

    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmin == cmax:
        cmin, cmax = 0.0, 1.0

    # Set up figure and initial artists
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.subplots_adjust(right=0.9)
    for ax in axes.flat:
        ax.axis("off")
    scatters: Dict[str, plt.Collection] = {}

    # Draw first frame
    first_epoch = sel_epochs[0]
    norm = plt.Normalize(vmin=cmin, vmax=cmax)
    last_valid_scatter = None
    for idx, model_dir in enumerate(models):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(model_dir.name, fontsize=9)
        emb, colour = cache.get((first_epoch, model_dir.name), (None, None))
        if emb is None or colour is None or emb.shape[0] == 0:
            txt = "Missing" if (first_epoch, model_dir.name) not in cache else "No samples"
            ax.text(0.5, 0.5, txt, ha="center", va="center")
            continue
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=colour, cmap="viridis", s=float(point_size), alpha=float(alpha), norm=norm)
        ax.set_xlim(xmin[model_dir.name], xmax[model_dir.name])
        ax.set_ylim(ymin[model_dir.name], ymax[model_dir.name])
        scatters[model_dir.name] = sc
        last_valid_scatter = sc

    suptitle = fig.suptitle(f"PHATE Embedding — {feature_name} — epoch {first_epoch}", fontsize=14)
    if last_valid_scatter is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
        cbar = fig.colorbar(last_valid_scatter, cax=cbar_ax)
        cbar.set_label(feature_name)

    def update(frame_idx: int):
        epoch = sel_epochs[frame_idx]
        suptitle.set_text(f"PHATE Embedding — {feature_name} — epoch {epoch}")
        for idx, model_dir in enumerate(models):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            emb, colour = cache.get((epoch, model_dir.name), (None, None))
            if model_dir.name not in scatters:
                # Nothing to update on this axis
                continue
            sc = scatters[model_dir.name]
            if emb is None or colour is None or emb.shape[0] == 0:
                # Clear axis and put a message
                ax.cla()
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(model_dir.name, fontsize=9)
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                # Recreate empty scatter to retain colorbar binding
                sc = ax.scatter([], [], c=[], cmap="viridis", s=float(point_size), alpha=float(alpha), norm=norm)
                scatters[model_dir.name] = sc
                ax.set_xlim(xmin[model_dir.name], xmax[model_dir.name])
                ax.set_ylim(ymin[model_dir.name], ymax[model_dir.name])
            else:
                sc.set_offsets(emb)
                sc.set_array(colour)
                ax.set_xlim(xmin[model_dir.name], xmax[model_dir.name])
                ax.set_ylim(ymin[model_dir.name], ymax[model_dir.name])
        return list(scatters.values())

    anim = animation.FuncAnimation(fig, update, frames=len(sel_epochs), interval=1000 // max(1, fps), blit=False)

    # Select writer
    out_base = output_dir / f"phate_animation_{feature_name}"
    output_path = None
    writer = None
    fmt_choice = fmt
    # Prefer mp4 if ffmpeg is available, otherwise fall back to gif
    try:
        import shutil
        has_ffmpeg = (shutil.which("ffmpeg") is not None) or bool(matplotlib.rcParams.get("animation.ffmpeg_path"))
    except Exception:
        has_ffmpeg = bool(matplotlib.rcParams.get("animation.ffmpeg_path"))

    if fmt_choice == "auto":
        if has_ffmpeg:
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
                output_path = out_base.with_suffix(".mp4")
                fmt_choice = "mp4"
            except Exception:
                writer = None
                output_path = None
                fmt_choice = "auto"
        else:
            # No ffmpeg available; prefer gif
            fmt_choice = "gif"
    if writer is None and (fmt_choice == "gif" or fmt_choice == "auto"):
        try:
            writer = animation.PillowWriter(fps=fps)
            output_path = out_base.with_suffix(".gif")
            fmt_choice = "gif"
        except Exception:
            writer = None
            output_path = None
            fmt_choice = "auto"
    if writer is None and fmt_choice == "mp4":
        # Last attempt for mp4 (only if ffmpeg present)
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            output_path = out_base.with_suffix(".mp4")
        except Exception:
            writer = None
            output_path = None

    if writer is None or output_path is None:
        print("[Embed] Could not create animation writer; skipping animation.")
        plt.close(fig)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Embed] Saving animation to {output_path} ({fmt_choice}, {fps} fps)…", flush=True)
    try:
        anim.save(str(output_path), writer=writer, dpi=dpi)
    except FileNotFoundError as exc:
        # Likely ffmpeg not available even though we attempted mp4; try GIF fallback
        print(f"[Embed] Animation writer failed: {exc}")
        try:
            print("[Embed] Retrying with GIF (PillowWriter)...", flush=True)
            writer_gif = animation.PillowWriter(fps=fps)
            output_path_gif = out_base.with_suffix(".gif")
            anim.save(str(output_path_gif), writer=writer_gif, dpi=dpi)
            output_path = output_path_gif
            fmt_choice = "gif"
            print(f"[Embed] Saved animation (gif): {output_path}", flush=True)
        except Exception as exc2:
            print(f"[Embed] Failed to save animation with fallback writer: {exc2}", flush=True)
            print("[Embed] To enable mp4 output, install ffmpeg (e.g., 'sudo apt install ffmpeg' or 'conda install -c conda-forge ffmpeg').", flush=True)
            plt.close(fig)
            return
    except Exception as exc:
        print(f"[Embed] Unexpected error while saving animation: {exc}", flush=True)
        plt.close(fig)
        return
    plt.close(fig)
    print(f"[Embed] Saved animation: {output_path} ({fmt_choice})", flush=True)


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
    progress_interval: int = 20,
    workers: int = 1,
    class_weight: str = "balanced",
    balance_train: bool = False,
    only_confusions: bool = False,
) -> None:
    component_rows: Dict[str, List[Dict[str, object]]] = {}
    # Aggregate confusion counts across models for compact, informative confusion matrices
    # Key: (component, feature, epoch) -> counts dict
    confusion_counts: Dict[Tuple[str, str, int], Dict[str, int]] = {}
    # Also aggregate across models at their respective best-saved epoch (keyed by 'best')
    confusion_counts_best: Dict[Tuple[str, str], Dict[str, int]] = {}
    # Per-model confusion counts (by epoch) for optional split visualisations
    confusion_counts_by_model: Dict[Tuple[str, str, str, int], Dict[str, int]] = {}
    # Per-model best-epoch confusion counts
    confusion_counts_best_by_model: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    # For density plots at the final probed epoch, store scores and labels aggregated across models
    try:
        _final_epoch_target = max(list(epochs))
    except Exception:
        _final_epoch_target = None
    score_store_final: Dict[Tuple[str, str], List[np.ndarray]] = {}
    label_store_final: Dict[Tuple[str, str], List[np.ndarray]] = {}
    score_store_best: Dict[Tuple[str, str], List[np.ndarray]] = {}
    label_store_best: Dict[Tuple[str, str], List[np.ndarray]] = {}

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

    # Progress bookkeeping
    model_dirs = sorted(p for p in analysis_dir.iterdir() if p.is_dir())
    # Each (model, epoch, component, feature) performs two fits: real and control
    total_estimated = max(1, 2 * len(model_dirs) * len(list(epochs)) * len(component_sequence) * len(list(feature_names)))
    completed = 0
    t0 = time.time()

    progress_interval = max(1, int(progress_interval))
    print(f"[Probes] Starting logistic regression probes: up to ~{total_estimated} fits (including controls); interval={progress_interval}", flush=True)

    for model_dir in model_dirs:
        model_meta = parse_model_name(model_dir.name)
        print(f"[Probes] Model: {model_dir.name}", flush=True)
        # Expand epochs for this model to include its best and final saved epochs
        be, fe = _get_best_and_final_saved_epoch(model_dir.name, analysis_dir)
        # Discover available sample epochs for this model (so we can snap to nearest if needed)
        samples_dir = model_dir / "hidden_samples"
        available_epochs = []
        if samples_dir.exists():
            for p in samples_dir.glob("epoch_*.npz"):
                try:
                    available_epochs.append(int(p.stem.split("_")[-1]))
                except Exception:
                    pass
        available_epochs = sorted(set(available_epochs))
        # Snap best epoch to the nearest available epoch if needed
        be_adj = None
        if be is not None and available_epochs:
            if int(be) in available_epochs:
                be_adj = int(be)
            else:
                be_adj = int(min(available_epochs, key=lambda e: abs(e - int(be))))
        epochs_for_model = set(int(e) for e in epochs)
        if be_adj is not None:
            epochs_for_model.add(int(be_adj))
        if fe is not None:
            epochs_for_model.add(int(fe))
        for epoch in sorted(epochs_for_model):
            epoch_t0 = time.time()
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

                    # Optionally under-sample majority class in training split
                    if balance_train:
                        classes, class_counts = np.unique(y_train, return_counts=True)
                        if classes.size == 2:
                            maj_class = classes[np.argmax(class_counts)]
                            min_class = classes[np.argmin(class_counts)]
                            n_min = class_counts.min()
                            idx_min = np.where(y_train == min_class)[0]
                            idx_maj = np.where(y_train == maj_class)[0]
                            if idx_maj.size > n_min:
                                idx_maj = rng.choice(idx_maj, size=n_min, replace=False)
                            keep_idx = np.concatenate([idx_min, idx_maj])
                            X_train = X_train[keep_idx]
                            y_train = y_train[keep_idx]
                    if np.unique(y_train).size < 2:
                        continue

                    # Standardize features for better convergence
                    # Many hidden dimensions may have very different scales
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Progress update before training
                    completed += 1
                    elapsed = time.time() - t0
                    avg = elapsed / max(1, completed)
                    remaining = max(0, total_estimated - completed)
                    eta_min = (remaining * avg) / 60.0
                    if completed <= progress_interval or completed % progress_interval == 0:
                        print(
                        f"[Probes] {completed}/{total_estimated} | {model_dir.name} | epoch={epoch} | component={component} | feature={feature} | ETA~{eta_min:.1f}m",
                        flush=True,
                        )

                    # Real probe with increased iterations and L2 regularization
                    sk_class_weight = None if class_weight == "none" else "balanced"
                    clf = LogisticRegression(
                        max_iter=5000,
                        solver='lbfgs',
                        C=1.0,  # Regularization strength
                        random_state=0,
                        tol=1e-4,
                        class_weight=sk_class_weight,
                    )
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            clf.fit(X_train_scaled, y_train)
                    except Exception as e:
                        # Fallback solver for tough cases
                        try:
                            clf = LogisticRegression(
                                max_iter=10000,
                                solver='liblinear',
                                C=1.0,
                                random_state=0,
                                tol=1e-4,
                                class_weight=sk_class_weight,
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                clf.fit(X_train_scaled, y_train)
                        except Exception as e2:
                            print(f"[Probes] Skipping probe due to error: {e2}")
                            continue
                    y_pred = clf.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="binary")
                    # Update aggregated confusion counts
                    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
                    tn = int(np.sum((y_pred == 0) & (y_test == 0)))
                    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
                    fn = int(np.sum((y_pred == 0) & (y_test == 1)))
                    key_cc = (component, feature, epoch)
                    bucket = confusion_counts.setdefault(key_cc, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
                    bucket["tp"] += tp
                    bucket["tn"] += tn
                    bucket["fp"] += fp
                    bucket["fn"] += fn
                    # Per-model by-epoch counts
                    key_ccm = (model_dir.name, component, feature, epoch)
                    bucket_m = confusion_counts_by_model.setdefault(key_ccm, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
                    bucket_m["tp"] += tp
                    bucket_m["tn"] += tn
                    bucket_m["fp"] += fp
                    bucket_m["fn"] += fn
                    # Also aggregate into best-epoch bucket for this model if this is its best epoch
                    if be_adj is not None and epoch == int(be_adj):
                        key_best = (component, feature)
                        b2 = confusion_counts_best.setdefault(key_best, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
                        b2["tp"] += tp
                        b2["tn"] += tn
                        b2["fp"] += fp
                        b2["fn"] += fn
                        # And store per-model best as well
                        key_bm = (model_dir.name, component, feature)
                        bm = confusion_counts_best_by_model.setdefault(key_bm, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
                        bm["tp"] += tp
                        bm["tn"] += tn
                        bm["fp"] += fp
                        bm["fn"] += fn
                    # Per-class accuracy (recall per class)
                    pos_mask = y_test == 1
                    neg_mask = y_test == 0
                    if np.any(pos_mask):
                        pos_accuracy = (y_pred[pos_mask] == y_test[pos_mask]).mean()
                    else:
                        pos_accuracy = float('nan')
                    if np.any(neg_mask):
                        neg_accuracy = (y_pred[neg_mask] == y_test[neg_mask]).mean()
                    else:
                        neg_accuracy = float('nan')
                    bal_acc = balanced_accuracy_score(y_test, y_pred)
                    try:
                        y_scores = clf.predict_proba(X_test_scaled)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_scores)
                        ap = average_precision_score(y_test, y_scores)
                    except Exception:
                        roc_auc = float('nan')
                        ap = float('nan')
                    # Store scores/labels at final epoch for lean density plots
                    if _final_epoch_target is not None and epoch == _final_epoch_target:
                        key_sf = (component, feature)
                        score_store_final.setdefault(key_sf, []).append(y_scores)
                        label_store_final.setdefault(key_sf, []).append(y_test)
                    # Collect scores for best-epoch aggregation as well
                    if be is not None and epoch == int(be):
                        key_sb = (component, feature)
                        score_store_best.setdefault(key_sb, []).append(y_scores)
                        label_store_best.setdefault(key_sb, []).append(y_test)
                    _, test_counts = np.unique(y_test, return_counts=True)
                    majority_baseline = test_counts.max() / test_counts.sum()
                    adjusted_accuracy = acc - majority_baseline

                    # Control task: permuted labels
                    y_train_permuted = rng.permutation(y_train)
                    y_test_permuted = rng.permutation(y_test)
                    clf_control = LogisticRegression(
                        max_iter=5000,
                        solver='lbfgs',
                        C=1.0,
                        random_state=0,
                        tol=1e-4,
                        class_weight=sk_class_weight,
                    )
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            clf_control.fit(X_train_scaled, y_train_permuted)
                    except Exception:
                        try:
                            clf_control = LogisticRegression(
                                max_iter=10000,
                                solver='liblinear',
                                C=1.0,
                                random_state=0,
                                tol=1e-4,
                                class_weight=sk_class_weight,
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                clf_control.fit(X_train_scaled, y_train_permuted)
                        except Exception as e3:
                            print(f"[Probes] Skipping control probe due to error: {e3}")
                            continue
                    y_pred_control = clf_control.predict(X_test_scaled)
                    acc_control = accuracy_score(y_test_permuted, y_pred_control)
                    # Per-class control accuracies (optional, helpful for diagnostics)
                    pos_mask_ctrl = y_test_permuted == 1
                    neg_mask_ctrl = y_test_permuted == 0
                    if np.any(pos_mask_ctrl):
                        pos_accuracy_control = (y_pred_control[pos_mask_ctrl] == y_test_permuted[pos_mask_ctrl]).mean()
                    else:
                        pos_accuracy_control = float('nan')
                    if np.any(neg_mask_ctrl):
                        neg_accuracy_control = (y_pred_control[neg_mask_ctrl] == y_test_permuted[neg_mask_ctrl]).mean()
                    else:
                        neg_accuracy_control = float('nan')
                    bal_acc_control = balanced_accuracy_score(y_test_permuted, y_pred_control)
                    try:
                        y_scores_ctrl = clf_control.predict_proba(X_test_scaled)[:, 1]
                        roc_auc_control = roc_auc_score(y_test_permuted, y_scores_ctrl)
                        ap_control = average_precision_score(y_test_permuted, y_scores_ctrl)
                    except Exception:
                        roc_auc_control = float('nan')
                        ap_control = float('nan')
                    completed += 1
                    if completed % progress_interval == 0:
                        elapsed = time.time() - t0
                        avg = elapsed / max(1, completed)
                        remaining = max(0, total_estimated - completed)
                        eta_min = (remaining * avg) / 60.0
                        print(
                            f"[Probes] control {completed}/{total_estimated} | {model_dir.name} | epoch={epoch} | component={component} | feature={feature} | ETA~{eta_min:.1f}m",
                            flush=True,
                        )

                    row = {
                        "model": model_dir.name,
                        "epoch": epoch,
                        "feature": feature,
                        "component": component,
                        "accuracy": float(acc),
                        "f1": float(f1),
                        "pos_accuracy": float(pos_accuracy),
                        "neg_accuracy": float(neg_accuracy),
                        "balanced_accuracy": float(bal_acc),
                        "roc_auc": float(roc_auc),
                        "average_precision": float(ap),
                        "majority_baseline": float(majority_baseline),
                        "adjusted_accuracy": float(adjusted_accuracy),
                        "control_accuracy": float(acc_control),
                        "control_pos_accuracy": float(pos_accuracy_control),
                        "control_neg_accuracy": float(neg_accuracy_control),
                        "control_balanced_accuracy": float(bal_acc_control),
                        "control_roc_auc": float(roc_auc_control),
                        "control_average_precision": float(ap_control),
                        "signal_over_control": float(acc - acc_control),
                        "balanced_signal_over_control": float(bal_acc - bal_acc_control),
                    }
                    row.update(model_meta)
                    component_rows.setdefault(component, []).append(row)

            # Model-epoch summary
            epoch_elapsed = time.time() - epoch_t0
            print(f"[Probes] Completed {model_dir.name} epoch {epoch} in {epoch_elapsed:.1f}s", flush=True)

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
        print(f"[Probes] Wrote results for component={component} to {component_dir / 'probe_results.csv'}", flush=True)

        # Determine model order once (used by multiple plots)
        model_order = sorted(df["model"].unique().tolist())

        # Plot 1: Improved readability accuracy plot (facet by feature)
        if not only_confusions:
            unique_feats = sorted(df["feature"].unique().tolist())
            n = len(unique_feats)
            cols = min(3, n) if n else 1
            rows = int(math.ceil(n / cols)) if n else 1
            fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.6 * rows), squeeze=False)
            # consistent palette across models
            palette = sns.color_palette("Set2", n_colors=len(model_order))
            color_map = {m: palette[i] for i, m in enumerate(model_order)}
            for idx, feat in enumerate(unique_feats):
                ax = axes[idx // cols][idx % cols]
                sub = df[df["feature"] == feat]
                if sub.empty:
                    ax.axis('off')
                    continue
                for m, g in sub.groupby("model"):
                    g = g.sort_values("epoch")
                    ax.plot(g["epoch"], g["accuracy"], label=m, color=color_map[m], linewidth=1.8)
                ax.set_title(feat)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
            # turn off extras
            for extra in range(n, rows * cols):
                axes[extra // cols][extra % cols].axis('off')
            # single legend on top
            handles, labels = axes[0][0].get_legend_handles_labels() if n else ([], [])
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                fig.tight_layout()
            plt.savefig(component_dir / "probe_accuracy.png", dpi=300)
            plt.close(fig)

        # Plot 1 (compact): Overall vs Pos vs Neg accuracy in a single figure
        if not only_confusions and set(["pos_accuracy", "neg_accuracy"]).issubset(df.columns):
            acc_long = pd.melt(
                df,
                id_vars=["model", "epoch", "feature", "gru", "channels", "kernel", "component"],
                value_vars=["accuracy", "pos_accuracy", "neg_accuracy"],
                var_name="metric",
                value_name="value",
            )
            metric_labels = {"accuracy": "Overall", "pos_accuracy": "+ (pos)", "neg_accuracy": "− (neg)"}
            acc_long["metric_label"] = acc_long["metric"].map(metric_labels)
            plt.figure(figsize=(11, 6))
            sns.lineplot(
                data=acc_long,
                x="epoch",
                y="value",
                hue="feature",
                style="metric_label",
                markers=True,
            )
            plt.ylim(0, 1)
            title = f"{component.upper()} Accuracy (Overall vs Pos vs Neg)"
            plt.title(title)
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(component_dir / "probe_accuracy_compact.png", dpi=300)
            plt.close()

        # Plot 2: Signal over control (new)
        if not only_confusions and "signal_over_control" in df.columns:
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

        # Plot 4: Balanced accuracy
        if not only_confusions and "balanced_accuracy" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x="epoch",
                y="balanced_accuracy",
                hue="feature",
                style="gru",
                markers=True,
            )
            title = f"{component.upper()} Balanced Accuracy over Epochs"
            plt.title(title)
            plt.ylabel("Balanced Accuracy")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(component_dir / "probe_balanced_accuracy.png", dpi=300)
            plt.close()

        # Plot 5: Balanced signal over control
        if not only_confusions and "balanced_signal_over_control" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df,
                x="epoch",
                y="balanced_signal_over_control",
                hue="feature",
                style="gru",
                markers=True,
            )
            plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            title = f"{component.upper()} Balanced Signal Over Control"
            plt.title(title)
            plt.ylabel("Balanced Acc - Control Balanced Acc")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig(component_dir / "probe_balanced_signal_over_control.png", dpi=300)
            plt.close()

        # Plot 6a: Confusion matrices at final epoch (aggregated across models)
        final_epoch = int(df["epoch"].max()) if not df.empty else None
        if final_epoch is not None:
            features_sorted = sorted(df["feature"].unique().tolist())
            n = len(features_sorted)
            if n > 0:
                fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
                if n == 1:
                    axes = [axes]
                for ax, feat in zip(axes, features_sorted):
                    counts = confusion_counts.get((component, feat, final_epoch))
                    if counts is None:
                        ax.axis('off')
                        ax.set_title(f"{feat}\n(no data)")
                        continue
                    tn = counts.get("tn", 0)
                    fp = counts.get("fp", 0)
                    fn = counts.get("fn", 0)
                    tp = counts.get("tp", 0)
                    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
                    total = mat.sum()
                    # Avoid division by zero
                    pct = mat / total if total > 0 else mat
                    # Annotate with count and percent
                    annot = np.empty_like(mat).astype(object)
                    for i in range(2):
                        for j in range(2):
                            annot[i, j] = f"{int(mat[i, j])}\n{(pct[i, j]*100):.1f}%"
                    sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0)
                    ax.set_title(f"{feat} @ epoch {final_epoch}")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    ax.set_xticks([0.5, 1.5], labels=["0", "1"])
                    ax.set_yticks([0.5, 1.5], labels=["0", "1"])
                plt.tight_layout()
                out_path = component_dir / f"probe_confusion_matrices_epoch_{final_epoch:03d}.png"
                plt.savefig(out_path, dpi=300)
                plt.close()

            # Lean summary figure: bars (overall/pos/neg) + confusion matrices per feature
            if n > 0:
                fig, axes = plt.subplots(2, n, figsize=(4.0*n, 6.0))
                if n == 1:
                    axes = np.array(axes).reshape(2, 1)
                for col, feat in enumerate(features_sorted):
                    sub = df[(df["epoch"] == final_epoch) & (df["feature"] == feat)]
                    overall = float(sub["accuracy"].mean()) if not sub.empty else float('nan')
                    posacc = float(sub["pos_accuracy"].mean()) if "pos_accuracy" in sub and not sub.empty else float('nan')
                    negacc = float(sub["neg_accuracy"].mean()) if "neg_accuracy" in sub and not sub.empty else float('nan')
                    bar_ax = axes[0, col]
                    vals = [overall, posacc, negacc]
                    labels = ["overall", "+", "−"]
                    # Avoid seaborn FutureWarning (palette without hue). Use explicit colors with Matplotlib.
                    colors = sns.color_palette("Set2", n_colors=len(labels))
                    bar_ax.bar(labels, vals, color=colors)
                    bar_ax.set_ylim(0, 1)
                    bar_ax.set_title(f"{feat}")
                    if col == 0:
                        bar_ax.set_ylabel(f"Accuracy @ epoch {final_epoch}")
                    else:
                        bar_ax.set_ylabel("")
                    for p in bar_ax.patches:
                        height = p.get_height()
                        bar_ax.annotate(f"{height:.2f}", (p.get_x()+p.get_width()/2, height), ha='center', va='bottom', fontsize=8)

                    cm_ax = axes[1, col]
                    counts = confusion_counts.get((component, feat, final_epoch))
                    if counts is None:
                        cm_ax.axis('off')
                        cm_ax.set_title("no data")
                    else:
                        tn = counts.get("tn", 0)
                        fp = counts.get("fp", 0)
                        fn = counts.get("fn", 0)
                        tp = counts.get("tp", 0)
                        mat = np.array([[tn, fp], [fn, tp]], dtype=float)
                        total = mat.sum()
                        pct = mat / total if total > 0 else mat
                        annot = np.empty_like(mat).astype(object)
                        for i in range(2):
                            for j in range(2):
                                annot[i, j] = f"{int(mat[i, j])}\n{(pct[i, j]*100):.1f}%"
                        sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=cm_ax, vmin=0)
                        cm_ax.set_xlabel("Predicted")
                        if col == 0:
                            cm_ax.set_ylabel("True")
                        cm_ax.set_xticks([0.5, 1.5], labels=["0", "1"])
                        cm_ax.set_yticks([0.5, 1.5], labels=["0", "1"])
                plt.tight_layout()
                out_path2 = component_dir / f"probe_feature_summary_epoch_{final_epoch:03d}.png"
                plt.savefig(out_path2, dpi=300)
                plt.close()

            # Score density per feature (aggregated across models) at final epoch
            if n > 0:
                fig, axes = plt.subplots(1, n, figsize=(4.0*n, 3.5))
                if n == 1:
                    axes = [axes]
                for ax, feat in zip(axes, features_sorted):
                    key_sf = (component, feat)
                    scores_list = score_store_final.get(key_sf)
                    labels_list = label_store_final.get(key_sf)
                    if not scores_list or not labels_list:
                        ax.axis('off')
                        ax.set_title(f"{feat}\n(no scores)")
                        continue
                    scores = np.concatenate([s for s in scores_list if s.size > 0])
                    labels_np = np.concatenate(labels_list)
                    if scores.size == 0 or labels_np.size == 0:
                        ax.axis('off')
                        ax.set_title(f"{feat}\n(no scores)")
                        continue
                    df_scores = pd.DataFrame({"score": scores, "label": labels_np})
                    sns.kdeplot(data=df_scores[df_scores["label"] == 0], x="score", ax=ax, label="true 0", fill=True, alpha=0.5)
                    sns.kdeplot(data=df_scores[df_scores["label"] == 1], x="score", ax=ax, label="true 1", fill=True, alpha=0.5)
                    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
                    ax.set_xlim(0, 1)
                    ax.set_title(feat)
                    ax.legend(fontsize=8)
                plt.tight_layout()
                out_path3 = component_dir / f"probe_score_density_epoch_{final_epoch:03d}.png"
                plt.savefig(out_path3, dpi=300)
                plt.close()

            # Optional: per-model confusion matrices at each model's final probed epoch
            for m in model_order:
                df_m = df[df["model"] == m]
                if df_m.empty:
                    continue
                m_final = int(df_m["epoch"].max())
                feats_m = sorted(df_m["feature"].unique().tolist())
                fig_m, axes_m = plt.subplots(1, len(feats_m), figsize=(5*len(feats_m), 4))
                if len(feats_m) == 1:
                    axes_m = [axes_m]
                for ax, feat in zip(axes_m, feats_m):
                    counts = confusion_counts_by_model.get((m, component, feat, m_final))
                    if counts is None:
                        ax.axis('off')
                        ax.set_title(f"{feat}\n(no data)")
                        continue
                    tn = counts.get("tn", 0); fp = counts.get("fp", 0); fn = counts.get("fn", 0); tp = counts.get("tp", 0)
                    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
                    total = mat.sum(); pct = mat/total if total>0 else mat
                    annot = np.empty_like(mat).astype(object)
                    for i in range(2):
                        for j in range(2):
                            annot[i, j] = f"{int(mat[i, j])}\n{(pct[i, j]*100):.1f}%"
                    sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0)
                    ax.set_title(f"{feat} @ epoch {m_final}")
                    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                    ax.set_xticks([0.5, 1.5], labels=["0", "1"]); ax.set_yticks([0.5, 1.5], labels=["0", "1"])
                plt.tight_layout()
                out_path_m = component_dir / f"probe_confusion_matrices_epoch_{m_final:03d}_{m}.png"
                plt.savefig(out_path_m, dpi=300)
                plt.close()

        # Plot 6b: Confusion matrices aggregated at each model's best-saved epoch
        features_sorted = sorted(df["feature"].unique().tolist()) if not df.empty else []
        if features_sorted:
            fig, axes = plt.subplots(1, len(features_sorted), figsize=(5*len(features_sorted), 4))
            if len(features_sorted) == 1:
                axes = [axes]
            for ax, feat in zip(axes, features_sorted):
                counts = confusion_counts_best.get((component, feat))
                if counts is None:
                    ax.axis('off')
                    ax.set_title(f"{feat}\n(no data)")
                    continue
                tn = counts.get("tn", 0)
                fp = counts.get("fp", 0)
                fn = counts.get("fn", 0)
                tp = counts.get("tp", 0)
                mat = np.array([[tn, fp], [fn, tp]], dtype=float)
                total = mat.sum()
                pct = mat / total if total > 0 else mat
                annot = np.empty_like(mat).astype(object)
                for i in range(2):
                    for j in range(2):
                        annot[i, j] = f"{int(mat[i, j])}\n{(pct[i, j]*100):.1f}%"
                sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0)
                ax.set_title(f"{feat} @ best-epoch (per model)")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_xticks([0.5, 1.5], labels=["0", "1"])
                ax.set_yticks([0.5, 1.5], labels=["0", "1"])
            plt.tight_layout()
            out_path_best = component_dir / "probe_confusion_matrices_best.png"
            plt.savefig(out_path_best, dpi=300)
            plt.close()

            # Optional: per-model confusion matrices at each model's best epoch
            for m in model_order:
                fig_m, axes_m = plt.subplots(1, len(features_sorted), figsize=(5*len(features_sorted), 4))
                if len(features_sorted) == 1:
                    axes_m = [axes_m]
                any_data = False
                for ax, feat in zip(axes_m, features_sorted):
                    counts = confusion_counts_best_by_model.get((m, component, feat))
                    if counts is None:
                        ax.axis('off')
                        ax.set_title(f"{feat}\n(no data)")
                        continue
                    any_data = True
                    tn = counts.get("tn", 0); fp = counts.get("fp", 0); fn = counts.get("fn", 0); tp = counts.get("tp", 0)
                    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
                    total = mat.sum(); pct = mat/total if total>0 else mat
                    annot = np.empty_like(mat).astype(object)
                    for i in range(2):
                        for j in range(2):
                            annot[i, j] = f"{int(mat[i, j])}\n{(pct[i, j]*100):.1f}%"
                    sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0)
                    ax.set_title(f"{feat} @ best (per {m})")
                    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                    ax.set_xticks([0.5, 1.5], labels=["0", "1"]); ax.set_yticks([0.5, 1.5], labels=["0", "1"])
                if any_data:
                    plt.tight_layout()
                    out_path_mb = component_dir / f"probe_confusion_matrices_best_{m}.png"
                    plt.savefig(out_path_mb, dpi=300)
                    plt.close()
                else:
                    plt.close(fig_m)

            # Lean summary at best
            fig, axes = plt.subplots(2, len(features_sorted), figsize=(4.0*len(features_sorted), 6.0))
            if len(features_sorted) == 1:
                axes = np.array(axes).reshape(2, 1)
            for col, feat in enumerate(features_sorted):
                # overall/pos/neg using the best epoch per model: approximate by taking max per model at its best epoch
                # Here we simply use the df values at each model's best epoch
                # Build per-model selector
                vals_overall = []
                vals_pos = []
                vals_neg = []
                # derive best epoch per model once
                models = sorted(df["model"].unique().tolist())
                for m in models:
                    be, _ = _get_best_and_final_saved_epoch(m, analysis_dir)
                    if be is None:
                        continue
                    sub = df[(df["model"] == m) & (df["epoch"] == int(be)) & (df["feature"] == feat)]
                    if sub.empty:
                        continue
                    vals_overall.append(float(sub["accuracy"].mean()))
                    if "pos_accuracy" in sub:
                        vals_pos.append(float(sub["pos_accuracy"].mean()))
                    if "neg_accuracy" in sub:
                        vals_neg.append(float(sub["neg_accuracy"].mean()))
                bar_ax = axes[0, col]
                yvals = [np.nanmean(vals_overall) if vals_overall else np.nan,
                         np.nanmean(vals_pos) if vals_pos else np.nan,
                         np.nanmean(vals_neg) if vals_neg else np.nan]
                labels3 = ["overall", "+", "−"]
                colors = sns.color_palette("Set2", n_colors=len(labels3))
                bar_ax.bar(labels3, yvals, color=colors)
                bar_ax.set_ylim(0, 1)
                bar_ax.set_title(f"{feat}")
                if col == 0:
                    bar_ax.set_ylabel("Accuracy @ best (per model)")
                for p in bar_ax.patches:
                    h = p.get_height()
                    if np.isfinite(h):
                        bar_ax.annotate(f"{h:.2f}", (p.get_x()+p.get_width()/2, h), ha='center', va='bottom', fontsize=8)
                cm_ax = axes[1, col]
                counts = confusion_counts_best.get((component, feat))
                if counts is None:
                    cm_ax.axis('off')
                    cm_ax.set_title("no data")
                else:
                    tn = counts.get("tn", 0)
                    fp = counts.get("fp", 0)
                    fn = counts.get("fn", 0)
                    tp = counts.get("tp", 0)
                    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
                    total = mat.sum()
                    pct = mat / total if total > 0 else mat
                    annot = np.empty_like(mat).astype(object)
                    for i in range(2):
                        for j in range(2):
                            annot[i, j] = f"{int(mat[i, j])}\n{(pct[i, j]*100):.1f}%"
                    sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=cm_ax, vmin=0)
                    cm_ax.set_xlabel("Predicted")
                    if col == 0:
                        cm_ax.set_ylabel("True")
                    cm_ax.set_xticks([0.5, 1.5], labels=["0", "1"])
                    cm_ax.set_yticks([0.5, 1.5], labels=["0", "1"])
            plt.tight_layout()
            out_path2b = component_dir / "probe_feature_summary_best.png"
            plt.savefig(out_path2b, dpi=300)
            plt.close()

            # Score density at best
            fig, axes = plt.subplots(1, len(features_sorted), figsize=(4.0*len(features_sorted), 3.5))
            if len(features_sorted) == 1:
                axes = [axes]
            for ax, feat in zip(axes, features_sorted):
                key_sf = (component, feat)
                scores_list = score_store_best.get(key_sf)
                labels_list = label_store_best.get(key_sf)
                if not scores_list or not labels_list:
                    ax.axis('off')
                    ax.set_title(f"{feat}\n(no scores)")
                    continue
                scores = np.concatenate([s for s in scores_list if s.size > 0])
                labels_np = np.concatenate(labels_list)
                if scores.size == 0 or labels_np.size == 0:
                    ax.axis('off')
                    ax.set_title(f"{feat}\n(no scores)")
                    continue
                df_scores = pd.DataFrame({"score": scores, "label": labels_np})
                sns.kdeplot(data=df_scores[df_scores["label"] == 0], x="score", ax=ax, label="true 0", fill=True, alpha=0.5)
                sns.kdeplot(data=df_scores[df_scores["label"] == 1], x="score", ax=ax, label="true 1", fill=True, alpha=0.5)
                ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
                ax.set_xlim(0, 1)
                ax.set_title(feat)
                ax.legend(fontsize=8)
            plt.tight_layout()
            out_path3b = component_dir / "probe_score_density_best.png"
            plt.savefig(out_path3b, dpi=300)
            plt.close()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Configuration summary
    print("[Analyze] Config:")
    print(f"  analysis-dir      = {analysis_dir}")
    print(f"  output-dir        = {output_dir}")
    print(f"  embedding-epochs  = {args.embedding_epochs}")
    print(f"  embedding-feature = {args.embedding_feature}")
    print(f"  embedding-mode    = {args.embedding_mode}")
    print(f"  probe-epochs      = {args.probe_epochs}")
    print(f"  probe-features    = {args.probe_features}")
    print(f"  probe-components  = {args.probe_components}")
    print(f"  max-hidden-samples= {args.max_hidden_samples}")
    print(f"  seed              = {args.seed}")
    print(f"  palette           = {args.palette}")
    print(f"  skip-embedding    = {args.skip_embedding}")
    print(f"  embedding-animate = {args.embedding_animate}")
    print(f"  embedding-fps     = {args.embedding_fps}")
    print(f"  embedding-format  = {args.embedding_format}")
    print(f"  embedding-dpi     = {args.embedding_dpi}")
    print(f"  embedding-ptsize  = {args.embedding_point_size}")
    print(f"  embedding-alpha   = {args.embedding_alpha}")
    print(f"  embedding-dedup   = {args.embedding_dedup}")
    print(f"  dedup-min-frac    = {args.embedding_dedup_min_fraction}")
    print(f"  dedup-min-count   = {args.embedding_dedup_min_count}")
    print(f"  embedding-jitter  = {args.embedding_jitter}")
    print(f"  skip-probing      = {args.skip_probing}")
    print(f"  progress-interval = {args.progress_interval}")
    print(f"  workers           = {args.workers}")
    print(f"  class-weight      = {args.class_weight}")
    print(f"  balance-train     = {args.balance_train}")

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
            point_size=args.embedding_point_size,
            alpha=args.embedding_alpha,
            dedup=(args.embedding_dedup if args.embedding_dedup in ("auto", "soft") else False),
            dedup_min_fraction=args.embedding_dedup_min_fraction,
            dedup_min_count=args.embedding_dedup_min_count,
            jitter=args.embedding_jitter,
        )
        if args.embedding_animate:
            # Pass mode and joint-sample cap to the animation function via attributes
            setattr(animate_phate_embeddings, "_mode", args.embedding_mode)
            setattr(animate_phate_embeddings, "_joint_samples", args.embedding_joint_samples)
            animate_phate_embeddings(
                analysis_dir,
                output_dir,
                epochs=args.embedding_epochs,
                feature_name=args.embedding_feature,
                max_samples=args.max_hidden_samples,
                rng=rng,
                fps=args.embedding_fps,
                fmt=args.embedding_format,
                dpi=args.embedding_dpi,
                point_size=args.embedding_point_size,
                alpha=args.embedding_alpha,
                dedup=(args.embedding_dedup if args.embedding_dedup in ("auto", "soft") else False),
                dedup_min_fraction=args.embedding_dedup_min_fraction,
                dedup_min_count=args.embedding_dedup_min_count,
                jitter=args.embedding_jitter,
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
            progress_interval=args.progress_interval,
            workers=args.workers,
            class_weight=args.class_weight,
            balance_train=args.balance_train,
        )

    summary_path = output_dir / "metrics_summary.json"
    summary = {
        "analysis_dir": str(analysis_dir),
        "embedding_epochs": args.embedding_epochs,
        "embedding_feature": args.embedding_feature,
        "embedding_mode": args.embedding_mode,
        "embedding_animate": args.embedding_animate,
        "probe_epochs": args.probe_epochs,
        "probe_features": args.probe_features,
        "skip_embedding": args.skip_embedding,
        "skip_probing": args.skip_probing,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Analysis complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()

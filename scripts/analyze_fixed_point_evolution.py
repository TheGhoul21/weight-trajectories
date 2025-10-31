#!/usr/bin/env python3
"""Analyse evolution of GRU fixed points across training epochs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise GRU fixed-point evolution.")
    parser.add_argument(
        "--fixed-dir",
        default="diagnostics/gru_fixed_points",
        help="Directory produced by find_gru_fixed_points.py",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/gru_fixed_points",
        help="Destination directory for evolution plots.",
    )
    args, _ = parser.parse_known_args()
    return args


def list_models(fixed_dir: Path) -> List[Path]:
    return sorted(p for p in fixed_dir.iterdir() if p.is_dir())


def load_epoch_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def plot_classification_counts(model_name: str, df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="epoch", hue="classification")
    plt.title(f"Fixed-point classification counts — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = output_dir / f"{model_name}_classification_counts.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_context_centroids(model_dir: Path, summary: pd.DataFrame) -> Dict[int, Dict[int, np.ndarray]]:
    """Return mapping context -> epoch -> centroid of stable fixed points."""

    epoch_dir = model_dir / "epochs"
    centroids: Dict[int, Dict[int, np.ndarray]] = {}
    for epoch in sorted(summary["epoch"].unique()):
        npz_path = epoch_dir / f"epoch_{epoch:03d}_fixed_points.npz"
        if not npz_path.exists():
            continue
        data = load_epoch_npz(npz_path)
        classifications = data["classification"]
        hidden = data["hidden"]
        context_index = data["context_index"]
        for ctx in np.unique(context_index):
            mask = (context_index == ctx) & (classifications == "stable")
            if np.count_nonzero(mask) == 0:
                continue
            centroid = hidden[mask].mean(axis=0)
            centroids.setdefault(int(ctx), {})[int(epoch)] = centroid
    return centroids


def plot_centroid_drift(model_name: str, centroids: Dict[int, Dict[int, np.ndarray]], output_dir: Path) -> None:
    if not centroids:
        return

    plt.figure(figsize=(8, 4))
    for ctx, epoch_dict in centroids.items():
        epochs = sorted(epoch_dict.keys())
        if len(epochs) < 2:
            continue
        drifts = []
        drift_epochs = []
        for e1, e2 in zip(epochs[:-1], epochs[1:]):
            c1 = epoch_dict[e1]
            c2 = epoch_dict[e2]
            drift = np.linalg.norm(c2 - c1)
            drifts.append(drift)
            drift_epochs.append(e2)
        if drifts:
            plt.plot(drift_epochs, drifts, marker="o", label=f"context {ctx}")

    if plt.gca().has_data():
        plt.title(f"Stable attractor drift — {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Centroid drift (L2)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        output_path = output_dir / f"{model_name}_attractor_drift.png"
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_centroid_drift_enhanced(model_name: str, centroids: Dict[int, Dict[int, np.ndarray]], output_dir: Path) -> None:
    if not centroids:
        return
    # Build a tidy frame: rows of (epoch, context, drift)
    rows = []
    for ctx, epoch_dict in centroids.items():
        epochs = sorted(epoch_dict.keys())
        for e1, e2 in zip(epochs[:-1], epochs[1:]):
            drift = float(np.linalg.norm(epoch_dict[e2] - epoch_dict[e1]))
            rows.append({"epoch": e2, "context_index": int(ctx), "drift": drift})
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Aggregate
    agg = df.groupby("epoch", as_index=False)["drift"].agg(["mean", "sem"]).reset_index()
    agg.rename(columns={"mean": "mean", "sem": "sem"}, inplace=True)

    plt.figure(figsize=(9.5, 5))
    # Faint per-context lines
    for ctx, g in df.groupby("context_index"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g["drift"], color="#8e9aaf", alpha=0.3, linewidth=1.0)
    # Mean + ribbon
    plt.plot(agg["epoch"], agg["mean"], color="#3a0ca3", linewidth=2.2, label="mean drift")
    upper = agg["mean"] + agg["sem"].fillna(0)
    lower = agg["mean"] - agg["sem"].fillna(0)
    plt.fill_between(agg["epoch"], lower, upper, color="#7209b7", alpha=0.18, label="±1 SEM")
    plt.title(f"Stable attractor drift (centroids) — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Drift (L2 between successive epochs)")
    plt.legend(frameon=False)
    plt.tight_layout()
    out = output_dir / f"{model_name}_attractor_drift_enhanced.png"
    plt.savefig(out, dpi=300)
    plt.close()


def plot_spectral_radius(model_name: str, df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        data=df[df["classification"] == "stable"],
        x="epoch",
        y="spectral_radius",
        hue="context_index",
        marker="o",
    )
    plt.title(f"Spectral radius of stable fixed points — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Spectral radius")
    plt.legend(title="Context", fontsize=8)
    plt.tight_layout()
    output_path = output_dir / f"{model_name}_spectral_radius.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_spectral_radius_enhanced(model_name: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Less cluttered spectral-radius plot with mean±SEM ribbon and faint context lines."""
    data = df[df["classification"] == "stable"].copy()
    if data.empty:
        return
    # Compute mean and SEM across contexts per epoch
    agg = (
        data.groupby(["epoch"], as_index=False)["spectral_radius"]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"mean": "mean", "sem": "sem"})
    )
    # Draw
    plt.figure(figsize=(9.5, 5))
    # Faint per-context lines
    for ctx, g in data.groupby("context_index"):
        g = g.sort_values("epoch")
        plt.plot(g["epoch"], g["spectral_radius"], color="#5c8db8", alpha=0.25, linewidth=1.0)
    # Mean line
    plt.plot(agg["epoch"], agg["mean"], color="#264653", linewidth=2.2, label="mean (stable)")
    # Ribbon
    upper = agg["mean"] + agg["sem"].fillna(0)
    lower = agg["mean"] - agg["sem"].fillna(0)
    plt.fill_between(agg["epoch"], lower, upper, color="#2a9d8f", alpha=0.25, label="±1 SEM")
    plt.title(f"Spectral radius (stable fixed points) — {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Spectral radius")
    plt.legend(frameon=False)
    plt.tight_layout()
    out = output_dir / f"{model_name}_spectral_radius_enhanced.png"
    plt.savefig(out, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    fixed_dir = Path(args.fixed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_dir in list_models(fixed_dir):
        summary_path = model_dir / "fixed_points_summary.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        model_name = model_dir.name

        plot_classification_counts(model_name, summary, output_dir)
        plot_spectral_radius(model_name, summary, output_dir)
        plot_spectral_radius_enhanced(model_name, summary, output_dir)

        centroids = compute_context_centroids(model_dir, summary)
        plot_centroid_drift(model_name, centroids, output_dir)
        plot_centroid_drift_enhanced(model_name, centroids, output_dir)

    print(f"Fixed-point evolution figures saved to {output_dir}")


if __name__ == "__main__":
    main()


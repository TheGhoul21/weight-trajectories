#!/usr/bin/env python3
"""Summarize checkpoint metric CSV files.

This script is a lightweight companion for compute_checkpoint_metrics.py.
Given one or more CSV paths, it prints descriptive statistics and optionally
export plots so you do not have to drop into a notebook/Pandas shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def _collect_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    return [col for col in candidates if col in df.columns]


def summarize_csv(path: Path, plot_dir: Optional[Path] = None) -> None:
    df = pd.read_csv(path)
    print(f"\n=== {path} ===")
    print(df.head())

    numeric_cols = _collect_columns(
        df,
        [
            "weight_norm",
            "step_norm",
            "step_cosine",
            "relative_step",
            "repr_total_variance",
            "repr_top1_ratio",
            "repr_top2_ratio",
            "repr_top3_ratio",
            "repr_top4_ratio",
        ],
    )
    if numeric_cols:
        print("\nDescriptive statistics:")
        print(df[["epoch"] + numeric_cols].describe())
    else:
        print("\nNo numeric diagnostic columns found beyond epoch.")

    if plot_dir is not None:
        if plt is None:
            print("matplotlib not available; skipping plots.")
            return
        plot_dir.mkdir(parents=True, exist_ok=True)

        to_plot = [
            ("weight_norm", "Weight Norm"),
            ("step_norm", "Step Norm"),
            ("step_cosine", "Step Cosine"),
            ("relative_step", "Relative Step"),
            ("repr_top1_ratio", "Top Singular Value Ratio"),
        ]

        for column, label in to_plot:
            if column not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(df["epoch"], df[column], marker="o", linestyle="-", linewidth=1.4, markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs Epoch")
            ax.grid(alpha=0.3)
            save_path = plot_dir / f"{path.stem}_{column}.png"
            fig.tight_layout()
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f"Saved plot: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize checkpoint metrics CSV files.")
    parser.add_argument("csv_paths", nargs="+", type=Path, help="One or more metrics CSV files.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Optional directory to save time-series plots (requires matplotlib).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for csv_path in args.csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        summarize_csv(csv_path, args.plot_dir)


if __name__ == "__main__":
    main()


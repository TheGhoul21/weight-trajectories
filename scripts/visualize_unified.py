#!/usr/bin/env python3
"""
Unified visualization script for weight trajectory analysis.
Generates all plots in one coherent framework.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import json
import argparse

sns.set_style("whitegrid")

CHANNELS = [16, 64, 256]
GRU_SIZES = [8, 32, 128]
CHANNEL_COLORS = {16: '#2E86AB', 64: '#A23B72', 256: '#F18F01'}
GRU_COLORS = {8: '#06A77D', 32: '#D5C67A', 128: '#F1A208'}

def load_all_data(metrics_dir, checkpoint_dir):
    """Load metrics and training history for all models."""
    metrics_data, history_data = {}, {}

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'

            # Load metrics
            csv_path = Path(metrics_dir) / f'{model_name}_metrics.csv'
            if csv_path.exists():
                metrics_data[model_name] = pd.read_csv(csv_path)

            # Load training history
            history_path = Path(checkpoint_dir) / model_name / 'training_history.json'
            if history_path.exists():
                with open(history_path) as f:
                    history_data[model_name] = json.load(f)

    return metrics_data, history_data

def plot_factorial_heatmaps(metrics_data, history_data, output_dir):
    """Create comprehensive factorial heatmaps."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    metrics = [
        ('weight_norm', 'Final Weight Norm', 'final'),
        ('step_norm', 'Mean Step Norm', 'mean'),
        ('step_cosine', 'Final Step Cosine', 'final'),
        ('repr_total_variance', 'Mean Total Variance', 'mean'),
        ('repr_top1_ratio', 'Mean Top-1 Ratio', 'mean'),
        ('val_loss', 'Min Validation Loss', 'min'),
        ('val_loss', 'Final Validation Loss', 'final'),
        ('train_val_gap', 'Final Train/Val Gap', 'final'),
        ('overfit_degree', 'Val Loss Increase', 'final'),
    ]

    for idx, (metric, title, agg) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        matrix = np.zeros((3, 3))
        for i, channels in enumerate(CHANNELS):
            for j, gru_size in enumerate(GRU_SIZES):
                model_name = f'k3_c{channels}_gru{gru_size}'

                if metric in ['val_loss', 'train_val_gap', 'overfit_degree']:
                    if model_name in history_data:
                        hist = history_data[model_name]
                        if metric == 'val_loss':
                            val = np.min(hist['val_loss']) if agg == 'min' else hist['val_loss'][-1]
                        elif metric == 'train_val_gap':
                            val = hist['val_loss'][-1] - hist['train_loss'][-1]
                        else:  # overfit_degree
                            val = hist['val_loss'][-1] - np.min(hist['val_loss'])
                        matrix[i, j] = val
                elif model_name in metrics_data:
                    df = metrics_data[model_name]
                    if metric in df.columns:
                        valid = df[metric].dropna()
                        if len(valid) > 0:
                            matrix[i, j] = valid.iloc[-1] if agg == 'final' else valid.mean()

        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels([f'GRU{g}' for g in GRU_SIZES])
        ax.set_yticklabels([f'c{c}' for c in CHANNELS])
        ax.set_xlabel('GRU Hidden Size')
        ax.set_ylabel('Channels')
        ax.set_title(title, fontweight='bold')

        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                text_val = f'{val:.2f}' if val >= 10 or val < 0.01 else f'{val:.3f}'
                ax.text(j, i, text_val, ha='center', va='center',
                       color='white' if val > matrix.mean() else 'black',
                       fontsize=9, fontweight='bold')

        plt.colorbar(im, ax=ax)

    plt.suptitle('Factorial Analysis: Complete Metric Heatmaps\n' +
                 'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(output_dir) / 'factorial_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_loss_trajectory_grid(metrics_data, history_data, output_dir):
    """Create 3x3 grid of loss + trajectory overlays."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), sharex=True)

    for i, channels in enumerate(CHANNELS):
        for j, gru_size in enumerate(GRU_SIZES):
            ax = axes[i, j]
            ax2 = ax.twinx()

            model_name = f'k3_c{channels}_gru{gru_size}'

            if model_name in metrics_data and model_name in history_data:
                df = metrics_data[model_name]
                hist = history_data[model_name]

                epochs_full = np.arange(1, len(hist['train_loss']) + 1)
                min_val_idx = np.argmin(hist['val_loss'])

                # Weight norm on left axis
                ax.plot(df['epoch'], df['weight_norm'], color=CHANNEL_COLORS[channels],
                       linewidth=2, marker='o', markersize=3, label='Weight Norm', alpha=0.8)

                # Val loss on right axis
                ax2.plot(epochs_full, hist['val_loss'], color='#F18F01',
                        linewidth=2, alpha=0.7, label='Val Loss')
                ax2.plot(epochs_full, hist['train_loss'], color='#2E86AB',
                        linewidth=1.5, alpha=0.5, linestyle='--', label='Train Loss')

                # Mark minimum
                ax.axvline(min_val_idx + 1, color='green', linestyle='--', alpha=0.3, linewidth=2)

                ax.set_ylabel('Weight Norm', color=CHANNEL_COLORS[channels], fontsize=9)
                ax2.set_ylabel('Loss', color='#F18F01', fontsize=9)
                ax.tick_params(axis='y', labelcolor=CHANNEL_COLORS[channels])
                ax2.tick_params(axis='y', labelcolor='#F18F01')

                # Title with key stats
                min_val = hist['val_loss'][min_val_idx]
                final_val = hist['val_loss'][-1]
                weight_growth = df['weight_norm'].iloc[-1] / df['weight_norm'].iloc[0]

                ax.set_title(f'{model_name}\n' +
                            f'Min Val: {min_val:.3f} @E{min_val_idx+1} | ' +
                            f'Final: {final_val:.3f} | Wgt: {weight_growth:.2f}×',
                            fontsize=9, fontweight='bold')

                if j == 0:  # First column
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax2.set_yticks([])

            if i == 2:  # Bottom row
                ax.set_xlabel('Epoch', fontsize=9)

            ax.grid(True, alpha=0.3)

    plt.suptitle('Weight Trajectory vs Validation Loss: Complete 3×3 Grid\n' +
                 'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(output_dir) / 'loss_trajectory_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_representation_grid(metrics_data, output_dir):
    """Create 3x3 grid of representation evolution."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    shared_legend_handles = None
    shared_legend_labels = None

    for i, channels in enumerate(CHANNELS):
        for j, gru_size in enumerate(GRU_SIZES):
            ax = axes[i, j]
            model_name = f'k3_c{channels}_gru{gru_size}'

            if model_name in metrics_data:
                df = metrics_data[model_name]

                for sv_idx in range(1, 5):
                    col = f'repr_top{sv_idx}_ratio'
                    if col in df.columns:
                        valid = df[df[col].notna()]
                        line, = ax.plot(valid['epoch'], valid[col],
                                        label=f'SV{sv_idx}', marker='o', markersize=2, linewidth=1.5)

                mean_top1 = df['repr_top1_ratio'].mean() if 'repr_top1_ratio' in df.columns else np.nan
                ax.set_title(f'{model_name}\nMean Top-1: {mean_top1:.3f}',
                            fontsize=9, fontweight='bold')
                # Defer legends to a single shared figure-level legend
                if shared_legend_handles is None or shared_legend_labels is None:
                    handles, labels = ax.get_legend_handles_labels()
                    shared_legend_handles, shared_legend_labels = handles, labels
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Variance Ratio', fontsize=8)

    # Add shared legend once for SV curves across all subplots
    if shared_legend_handles and shared_legend_labels:
        fig.legend(shared_legend_handles, shared_legend_labels, loc='upper center', ncol=4, frameon=False)
        plt.suptitle('Representation Collapse Analysis: Singular Value Evolution\n' +
                     'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.suptitle('Representation Collapse Analysis: Singular Value Evolution\n' +
                     'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

    output_path = Path(output_dir) / 'representation_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_gru_sweep_comparison(metrics_data, history_data, output_dir):
    """Compare GRU sizes for each channel count (main effect plots)."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    shared_handles = None
    shared_labels = None

    for ch_idx, channels in enumerate(CHANNELS):
        # Weight norm
        ax = axes[ch_idx, 0]
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            if model_name in metrics_data:
                df = metrics_data[model_name]
                ax.plot(
                    df['epoch'], df['weight_norm'],
                    label=f'GRU{gru_size}', color=GRU_COLORS[gru_size],
                    linewidth=2, marker='o', markersize=2
                )
        ax.set_title(f'Weight Norm (c{channels})', fontweight='bold')
        ax.set_ylabel('Weight Norm')
        # Capture legend entries once (GRU sizes) for a shared legend later
        if shared_handles is None and shared_labels is None:
            shared_handles, shared_labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.legend_.remove()
        ax.grid(True, alpha=0.3)

        # Step norm
        ax = axes[ch_idx, 1]
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            if model_name in metrics_data:
                df = metrics_data[model_name]
                valid = df[df['step_norm'].notna()]
                ax.plot(
                    valid['epoch'], valid['step_norm'],
                    label=f'GRU{gru_size}', color=GRU_COLORS[gru_size],
                    linewidth=2, marker='o', markersize=2
                )
        ax.set_title(f'Step Norm (c{channels})', fontweight='bold')
        ax.set_ylabel('Step Norm')
        if ax.get_legend():
            ax.legend_.remove()
        ax.grid(True, alpha=0.3)

        # Validation loss
        ax = axes[ch_idx, 2]
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            if model_name in history_data:
                hist = history_data[model_name]
                epochs = np.arange(1, len(hist['val_loss']) + 1)
                ax.plot(
                    epochs, hist['val_loss'],
                    label=f'GRU{gru_size}', color=GRU_COLORS[gru_size],
                    linewidth=2
                )
                min_idx = np.argmin(hist['val_loss'])
                ax.scatter(
                    [min_idx + 1], [hist['val_loss'][min_idx]],
                    color=GRU_COLORS[gru_size], s=100, zorder=5, marker='*'
                )
        ax.set_title(f'Validation Loss (c{channels})', fontweight='bold')
        ax.set_ylabel('Val Loss')
        if ax.get_legend():
            ax.legend_.remove()
        ax.grid(True, alpha=0.3)

        if ch_idx == 2:
            for ax_bottom in axes[ch_idx, :]:
                ax_bottom.set_xlabel('Epoch')

    # Shared legend for GRU sizes across the figure
    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='upper center', ncol=3, frameon=False, title='GRU Size')
        plt.suptitle('GRU Size Effect: Varying RNN Capacity (Fixed Channel Count)\n' +
                     'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.suptitle('GRU Size Effect: Varying RNN Capacity (Fixed Channel Count)\n' +
                     'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

    output_path = Path(output_dir) / 'gru_sweep_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_channel_sweep_comparison(metrics_data, history_data, output_dir):
    """Compare channel counts for each GRU size (main effect plots)."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    shared_handles = None
    shared_labels = None

    for gru_idx, gru_size in enumerate(GRU_SIZES):
        # Weight norm
        ax = axes[gru_idx, 0]
        for channels in CHANNELS:
            model_name = f'k3_c{channels}_gru{gru_size}'
            if model_name in metrics_data:
                df = metrics_data[model_name]
                ax.plot(
                    df['epoch'], df['weight_norm'],
                    label=f'c{channels}', color=CHANNEL_COLORS[channels],
                    linewidth=2, marker='o', markersize=2
                )
        ax.set_title(f'Weight Norm (GRU{gru_size})', fontweight='bold')
        ax.set_ylabel('Weight Norm')
        if shared_handles is None and shared_labels is None:
            shared_handles, shared_labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.legend_.remove()
        ax.grid(True, alpha=0.3)

        # Step norm
        ax = axes[gru_idx, 1]
        for channels in CHANNELS:
            model_name = f'k3_c{channels}_gru{gru_size}'
            if model_name in metrics_data:
                df = metrics_data[model_name]
                valid = df[df['step_norm'].notna()]
                ax.plot(
                    valid['epoch'], valid['step_norm'],
                    label=f'c{channels}', color=CHANNEL_COLORS[channels],
                    linewidth=2, marker='o', markersize=2
                )
        ax.set_title(f'Step Norm (GRU{gru_size})', fontweight='bold')
        ax.set_ylabel('Step Norm')
        if ax.get_legend():
            ax.legend_.remove()
        ax.grid(True, alpha=0.3)

        # Validation loss
        ax = axes[gru_idx, 2]
        for channels in CHANNELS:
            model_name = f'k3_c{channels}_gru{gru_size}'
            if model_name in history_data:
                hist = history_data[model_name]
                epochs = np.arange(1, len(hist['val_loss']) + 1)
                ax.plot(
                    epochs, hist['val_loss'],
                    label=f'c{channels}', color=CHANNEL_COLORS[channels],
                    linewidth=2
                )
                min_idx = np.argmin(hist['val_loss'])
                ax.scatter(
                    [min_idx + 1], [hist['val_loss'][min_idx]],
                    color=CHANNEL_COLORS[channels], s=100, zorder=5, marker='*'
                )
        ax.set_title(f'Validation Loss (GRU{gru_size})', fontweight='bold')
        ax.set_ylabel('Val Loss')
        if ax.get_legend():
            ax.legend_.remove()
        ax.grid(True, alpha=0.3)

        if gru_idx == 2:
            for ax_bottom in axes[gru_idx, :]:
                ax_bottom.set_xlabel('Epoch')

    if shared_handles and shared_labels:
        fig.legend(shared_handles, shared_labels, loc='upper center', ncol=3, frameon=False, title='Channels')
        plt.suptitle('Channel Count Effect: Varying CNN Capacity (Fixed GRU Size)\n' +
                     'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.suptitle('Channel Count Effect: Varying CNN Capacity (Fixed GRU Size)\n' +
                     'Connect-4 as a Testbed: Analyzing Learning Trajectories in a Solved Game',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

    output_path = Path(output_dir) / 'channel_sweep_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate unified trajectory visualizations')
    parser.add_argument('--metrics-dir', default='diagnostics/trajectory_analysis')
    parser.add_argument('--checkpoint-dir', default='checkpoints/save_every_1')
    parser.add_argument('--output-dir', default='visualizations')
    args = parser.parse_args()

    print("="*80)
    print("Unified Trajectory Visualization")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load all data
    print("\nLoading data...")
    metrics_data, history_data = load_all_data(args.metrics_dir, args.checkpoint_dir)
    print(f"Loaded {len(metrics_data)} models with metrics")
    print(f"Loaded {len(history_data)} models with training history")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_factorial_heatmaps(metrics_data, history_data, output_dir)
    plot_loss_trajectory_grid(metrics_data, history_data, output_dir)
    plot_representation_grid(metrics_data, output_dir)
    plot_gru_sweep_comparison(metrics_data, history_data, output_dir)
    plot_channel_sweep_comparison(metrics_data, history_data, output_dir)

    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Saved 5 comprehensive plots to: {output_dir.absolute()}")
    print("="*80)

if __name__ == '__main__':
    main()

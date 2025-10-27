#!/usr/bin/env python3
"""
Enhanced visualization with training/validation loss overlay.
Shows correlation between weight trajectories and actual model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")

# Define all 9 models
CHANNELS = [16, 64, 256]
GRU_SIZES = [8, 32, 128]

# Color schemes
CHANNEL_COLORS = {16: '#2E86AB', 64: '#A23B72', 256: '#F18F01'}
GRU_COLORS = {8: '#06A77D', 32: '#D5C67A', 128: '#F1A208'}

def get_model_name(channels, gru_size):
    """Generate model name from parameters."""
    return f'k3_c{channels}_gru{gru_size}'

def load_training_history(checkpoint_dir):
    """Load training history JSON file."""
    history_path = Path(checkpoint_dir) / 'training_history.json'
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None

def load_all_data(base_dir='diagnostics/checkpoint_metrics',
                  checkpoint_base='checkpoints/save_every_3'):
    """Load both metrics and training history."""
    metrics_data = {}
    history_data = {}

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = get_model_name(channels, gru_size)

            # Load metrics
            csv_path = Path(base_dir) / f'{model_name}_metrics.csv'
            if csv_path.exists():
                metrics_data[model_name] = pd.read_csv(csv_path)

            # Load training history
            checkpoint_dir = Path(checkpoint_base) / model_name
            history = load_training_history(checkpoint_dir)
            if history:
                history_data[model_name] = history

    print(f"Loaded {len(metrics_data)} metric files")
    print(f"Loaded {len(history_data)} training history files")

    return metrics_data, history_data

def find_overfitting_epoch(history):
    """Find when validation loss starts consistently increasing."""
    if not history or 'val_loss' not in history:
        return None

    val_loss = np.array(history['val_loss'])
    min_idx = np.argmin(val_loss)

    # Look for sustained increase after min (over 5+ epochs)
    window = 5
    for i in range(min_idx + 1, len(val_loss) - window):
        # Check if val loss is consistently higher than min
        if np.all(val_loss[i:i+window] > val_loss[min_idx] * 1.02):  # 2% threshold
            return i

    return None

def plot_trajectory_with_loss_overlay(metrics_data, history_data, output_dir='visusualizations'):
    """Create combined plots showing trajectories and loss curves."""

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = get_model_name(channels, gru_size)

            if model_name not in metrics_data or model_name not in history_data:
                continue

            df = metrics_data[model_name]
            history = history_data[model_name]

            fig, axes = plt.subplots(3, 2, figsize=(16, 14))

            # Prepare epoch arrays
            epochs_full = np.arange(1, len(history['train_loss']) + 1)
            epochs_checkpoints = df['epoch'].values

            # Find key events
            min_val_idx = np.argmin(history['val_loss'])
            min_val_epoch = min_val_idx + 1
            min_val_loss = history['val_loss'][min_val_idx]
            final_val_loss = history['val_loss'][-1]
            overfit_epoch = find_overfitting_epoch(history)

            # Calculate train/val gap
            train_val_gap = np.array(history['val_loss']) - np.array(history['train_loss'])

            # ============= Row 1: Loss Curves =============

            # 1.1: Training and Validation Loss
            ax = axes[0, 0]
            ax.plot(epochs_full, history['train_loss'], label='Train Loss',
                   color='#2E86AB', linewidth=2, alpha=0.8)
            ax.plot(epochs_full, history['val_loss'], label='Val Loss',
                   color='#F18F01', linewidth=2, alpha=0.8)

            # Mark minimum validation loss
            ax.axvline(min_val_epoch, color='green', linestyle='--', alpha=0.5,
                      label=f'Min Val (E{min_val_epoch})')
            ax.scatter([min_val_epoch], [min_val_loss], color='green',
                      s=100, zorder=5, marker='*')

            # Mark overfitting point if detected
            if overfit_epoch:
                ax.axvline(overfit_epoch, color='red', linestyle='--', alpha=0.5,
                          label=f'Overfit Start (E{overfit_epoch})')

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title('Training & Validation Loss', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # 1.2: Train/Val Gap
            ax = axes[0, 1]
            ax.plot(epochs_full, train_val_gap, color='#A23B72', linewidth=2)
            ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax.axvline(min_val_epoch, color='green', linestyle='--', alpha=0.5)
            if overfit_epoch:
                ax.axvline(overfit_epoch, color='red', linestyle='--', alpha=0.5)

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Val Loss - Train Loss', fontsize=10)
            ax.set_title('Train/Val Gap (Overfitting Indicator)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add annotation for gap at key epochs
            gap_at_min = train_val_gap[min_val_idx]
            gap_final = train_val_gap[-1]
            ax.text(0.98, 0.98, f'Gap @ Min Val: {gap_at_min:.3f}\nGap @ Final: {gap_final:.3f}',
                   transform=ax.transAxes, fontsize=9, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # ============= Row 2: Weight Trajectory with Loss Overlay =============

            # 2.1: Weight Norm with Val Loss
            ax = axes[1, 0]
            ax2 = ax.twinx()

            # Weight norm on left y-axis
            ax.plot(epochs_checkpoints, df['weight_norm'], color='#2E86AB',
                   linewidth=2, marker='o', markersize=4, label='Weight Norm')

            # Val loss on right y-axis
            ax2.plot(epochs_full, history['val_loss'], color='#F18F01',
                    linewidth=1.5, alpha=0.6, label='Val Loss')

            # Mark key epochs
            ax.axvline(min_val_epoch, color='green', linestyle='--', alpha=0.3)
            if overfit_epoch:
                ax.axvline(overfit_epoch, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Weight Norm', color='#2E86AB', fontsize=10)
            ax2.set_ylabel('Val Loss', color='#F18F01', fontsize=10)
            ax.set_title('Weight Norm vs Validation Loss', fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='#2E86AB')
            ax2.tick_params(axis='y', labelcolor='#F18F01')
            ax.grid(True, alpha=0.3)

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

            # 2.2: Step Norm with Val Loss
            ax = axes[1, 1]
            ax2 = ax.twinx()

            valid_steps = df[df['step_norm'].notna()]
            ax.plot(valid_steps['epoch'], valid_steps['step_norm'], color='#A23B72',
                   linewidth=2, marker='o', markersize=4, label='Step Norm')

            ax2.plot(epochs_full, history['val_loss'], color='#F18F01',
                    linewidth=1.5, alpha=0.6, label='Val Loss')

            ax.axvline(min_val_epoch, color='green', linestyle='--', alpha=0.3)
            if overfit_epoch:
                ax.axvline(overfit_epoch, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Step Norm', color='#A23B72', fontsize=10)
            ax2.set_ylabel('Val Loss', color='#F18F01', fontsize=10)
            ax.set_title('Step Norm vs Validation Loss', fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='#A23B72')
            ax2.tick_params(axis='y', labelcolor='#F18F01')
            ax.grid(True, alpha=0.3)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

            # ============= Row 3: Representation with Loss Overlay =============

            # 3.1: Total Variance with Val Loss
            ax = axes[2, 0]
            ax2 = ax.twinx()

            if 'repr_total_variance' in df.columns:
                valid_repr = df[df['repr_total_variance'].notna()]
                ax.plot(valid_repr['epoch'], valid_repr['repr_total_variance'],
                       color='#06A77D', linewidth=2, marker='o', markersize=4,
                       label='Total Variance')

            ax2.plot(epochs_full, history['val_loss'], color='#F18F01',
                    linewidth=1.5, alpha=0.6, label='Val Loss')

            ax.axvline(min_val_epoch, color='green', linestyle='--', alpha=0.3)
            if overfit_epoch:
                ax.axvline(overfit_epoch, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Total Repr. Variance', color='#06A77D', fontsize=10)
            ax2.set_ylabel('Val Loss', color='#F18F01', fontsize=10)
            ax.set_title('Representation Variance vs Validation Loss', fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='#06A77D')
            ax2.tick_params(axis='y', labelcolor='#F18F01')
            ax.grid(True, alpha=0.3)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

            # 3.2: Top-1 Ratio with Val Loss
            ax = axes[2, 1]
            ax2 = ax.twinx()

            if 'repr_top1_ratio' in df.columns:
                valid_top1 = df[df['repr_top1_ratio'].notna()]
                ax.plot(valid_top1['epoch'], valid_top1['repr_top1_ratio'],
                       color='#D5C67A', linewidth=2, marker='o', markersize=4,
                       label='Top-1 Ratio')

            ax2.plot(epochs_full, history['val_loss'], color='#F18F01',
                    linewidth=1.5, alpha=0.6, label='Val Loss')

            ax.axvline(min_val_epoch, color='green', linestyle='--', alpha=0.3)
            if overfit_epoch:
                ax.axvline(overfit_epoch, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Top-1 SV Ratio (Collapse)', color='#D5C67A', fontsize=10)
            ax2.set_ylabel('Val Loss', color='#F18F01', fontsize=10)
            ax.set_title('Representation Collapse vs Validation Loss', fontsize=11, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='#D5C67A')
            ax2.tick_params(axis='y', labelcolor='#F18F01')
            ax.grid(True, alpha=0.3)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

            # ============= Overall Title =============
            overfit_status = f"Overfit @ E{overfit_epoch}" if overfit_epoch else "No clear overfitting"
            plt.suptitle(f'{model_name}: Weight Trajectory & Loss Analysis\n' +
                        f'Min Val Loss: {min_val_loss:.4f} @ Epoch {min_val_epoch} | ' +
                        f'Final Val Loss: {final_val_loss:.4f} | {overfit_status}',
                        fontsize=13, fontweight='bold', y=0.995)

            plt.tight_layout()

            output_path = Path(output_dir) / f'loss_overlay_{model_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()

def create_loss_summary_table(history_data, output_dir='visusualizations'):
    """Create summary table of loss metrics across all models."""
    summary_rows = []

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = get_model_name(channels, gru_size)

            if model_name not in history_data:
                continue

            history = history_data[model_name]

            train_loss = np.array(history['train_loss'])
            val_loss = np.array(history['val_loss'])

            min_val_idx = np.argmin(val_loss)
            overfit_epoch = find_overfitting_epoch(history)

            row = {
                'Model': model_name,
                'Channels': channels,
                'GRU_Size': gru_size,
                'Min_Val_Loss': val_loss[min_val_idx],
                'Min_Val_Epoch': min_val_idx + 1,
                'Final_Val_Loss': val_loss[-1],
                'Final_Train_Loss': train_loss[-1],
                'Final_Gap': val_loss[-1] - train_loss[-1],
                'Gap_at_Min_Val': val_loss[min_val_idx] - train_loss[min_val_idx],
                'Overfit_Epoch': overfit_epoch if overfit_epoch else np.nan,
                'Val_Loss_Increase': val_loss[-1] - val_loss[min_val_idx],
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save CSV
    output_path = Path(output_dir) / 'loss_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print pivot tables
    print("\n" + "="*80)
    print("LOSS ANALYSIS SUMMARY")
    print("="*80)

    print("\nMinimum Validation Loss:")
    pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Min_Val_Loss')
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\nEpoch of Minimum Validation Loss:")
    pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Min_Val_Epoch')
    print(pivot.to_string(float_format=lambda x: f'{int(x)}'))

    print("\nFinal Validation Loss:")
    pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Final_Val_Loss')
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\nFinal Train/Val Gap (Overfitting Indicator):")
    pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Final_Gap')
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\nValidation Loss Increase from Min to Final (Overfitting Degree):")
    pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Val_Loss_Increase')
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

    return summary_df

def main():
    """Main execution."""
    print("=" * 80)
    print("Enhanced Visualization with Loss Overlay")
    print("=" * 80)

    # Create output directory
    output_dir = Path('visusualizations')
    output_dir.mkdir(exist_ok=True)

    # Load all data
    print("\nLoading metrics and training history...")
    metrics_data, history_data = load_all_data()

    # Generate per-model plots
    print("\nGenerating loss overlay plots for each model...")
    plot_trajectory_with_loss_overlay(metrics_data, history_data, output_dir)

    # Generate summary table
    print("\nGenerating loss summary table...")
    summary_df = create_loss_summary_table(history_data, output_dir)

    print("\n" + "="*80)
    print("Loss overlay analysis complete!")
    print(f"Generated {len(metrics_data)} detailed plots")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Factorial analysis visualization: 3×3 grid of (channels × GRU sizes).
Analyzes both CNN capacity effects and RNN capacity effects.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

# Define all 9 models in factorial design
CHANNELS = [16, 64, 256]
GRU_SIZES = [8, 32, 128]

# Color schemes
CHANNEL_COLORS = {16: '#2E86AB', 64: '#A23B72', 256: '#F18F01'}  # Blue, Purple, Orange
GRU_COLORS = {8: '#06A77D', 32: '#D5C67A', 128: '#F1A208'}  # Green, Yellow, Orange

def get_model_name(channels, gru_size):
    """Generate model name from parameters."""
    return f'k3_c{channels}_gru{gru_size}'

def load_all_metrics(base_dir='diagnostics/checkpoint_metrics'):
    """Load all 9 model metrics."""
    base_path = Path(base_dir)
    data = {}

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = get_model_name(channels, gru_size)
            csv_path = base_path / f'{model_name}_metrics.csv'

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                data[model_name] = df
                print(f"✓ Loaded {model_name}: {len(df)} checkpoints")
            else:
                print(f"✗ Missing: {model_name}")

    return data

def plot_channel_sweeps(data, output_dir='visusualizations'):
    """Plot metrics varying channels while fixing GRU size (3 rows × 2 cols per GRU)."""
    metrics_to_plot = [
        ('weight_norm', 'Weight Norm'),
        ('step_norm', 'Step Norm'),
        ('relative_step', 'Relative Step'),
        ('step_cosine', 'Step Cosine'),
        ('repr_total_variance', 'Total Repr. Variance'),
        ('repr_top1_ratio', 'Top-1 SV Ratio'),
    ]

    for gru_size in GRU_SIZES:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()

        for metric_idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[metric_idx]

            for channels in CHANNELS:
                model_name = get_model_name(channels, gru_size)
                if model_name in data:
                    df = data[model_name]
                    if metric in df.columns:
                        valid_df = df[df[metric].notna()]
                        ax.plot(valid_df['epoch'], valid_df[metric],
                               label=f'c{channels}', color=CHANNEL_COLORS[channels],
                               marker='o', markersize=2, linewidth=1.5, alpha=0.8)

            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f'{label}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            if metric == 'relative_step':
                ax.set_yscale('log')

        plt.suptitle(f'Channel Sweeps: Effect of CNN Capacity (GRU Size = {gru_size})',
                     fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = Path(output_dir) / f'factorial_channel_sweep_gru{gru_size}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

def plot_gru_sweeps(data, output_dir='visusualizations'):
    """Plot metrics varying GRU size while fixing channels (3 rows × 2 cols per channel count)."""
    metrics_to_plot = [
        ('weight_norm', 'Weight Norm'),
        ('step_norm', 'Step Norm'),
        ('relative_step', 'Relative Step'),
        ('step_cosine', 'Step Cosine'),
        ('repr_total_variance', 'Total Repr. Variance'),
        ('repr_top1_ratio', 'Top-1 SV Ratio'),
    ]

    for channels in CHANNELS:
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()

        for metric_idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[metric_idx]

            for gru_size in GRU_SIZES:
                model_name = get_model_name(channels, gru_size)
                if model_name in data:
                    df = data[model_name]
                    if metric in df.columns:
                        valid_df = df[df[metric].notna()]
                        ax.plot(valid_df['epoch'], valid_df[metric],
                               label=f'GRU{gru_size}', color=GRU_COLORS[gru_size],
                               marker='o', markersize=2, linewidth=1.5, alpha=0.8)

            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f'{label}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            if metric == 'relative_step':
                ax.set_yscale('log')

        plt.suptitle(f'GRU Sweeps: Effect of RNN Capacity (Channels = {channels})',
                     fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = Path(output_dir) / f'factorial_gru_sweep_c{channels}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

def plot_factorial_heatmaps(data, output_dir='visusualizations'):
    """Create 3×3 heatmaps showing final metrics across the factorial design."""
    metrics = [
        ('weight_norm', 'Final Weight Norm', False),
        ('step_norm', 'Final Step Norm', False),
        ('relative_step', 'Final Relative Step', True),
        ('step_cosine', 'Final Step Cosine', False),
        ('repr_total_variance', 'Final Total Variance', False),
        ('repr_top1_ratio', 'Final Top-1 Ratio', False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric, title, is_log) in enumerate(metrics):
        ax = axes[idx]

        # Create 3×3 matrix: rows=channels, cols=GRU sizes
        matrix = np.zeros((3, 3))
        for i, channels in enumerate(CHANNELS):
            for j, gru_size in enumerate(GRU_SIZES):
                model_name = get_model_name(channels, gru_size)
                if model_name in data:
                    df = data[model_name]
                    if metric in df.columns:
                        valid = df[metric].dropna()
                        if len(valid) > 0:
                            matrix[i, j] = valid.iloc[-1]  # Final value

        # Apply log if needed
        if is_log:
            matrix = np.log10(matrix + 1e-10)
            title += ' (log10)'

        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto')

        # Set ticks and labels
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels([f'GRU{g}' for g in GRU_SIZES])
        ax.set_yticklabels([f'c{c}' for c in CHANNELS])
        ax.set_xlabel('GRU Hidden Size')
        ax.set_ylabel('Channels')
        ax.set_title(title, fontweight='bold')

        # Annotate cells with values
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                if is_log:
                    text_val = f'{10**val:.2f}'
                else:
                    text_val = f'{val:.1f}' if val >= 10 else f'{val:.3f}'
                ax.text(j, i, text_val, ha='center', va='center',
                       color='white' if val > matrix.mean() else 'black',
                       fontsize=8, fontweight='bold')

        plt.colorbar(im, ax=ax)

    plt.suptitle('Factorial Design: Final Metrics across Channels × GRU Sizes',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(output_dir) / 'factorial_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_representation_factorial(data, output_dir='visusualizations'):
    """Detailed representation analysis across the 3×3 grid."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for i, channels in enumerate(CHANNELS):
        for j, gru_size in enumerate(GRU_SIZES):
            ax = axes[i, j]
            model_name = get_model_name(channels, gru_size)

            if model_name in data:
                df = data[model_name]

                # Plot all 4 singular values
                for sv_idx in range(1, 5):
                    col = f'repr_top{sv_idx}_ratio'
                    if col in df.columns:
                        valid_df = df[df[col].notna()]
                        ax.plot(valid_df['epoch'], valid_df[col],
                               label=f'SV{sv_idx}', marker='o', markersize=2, linewidth=1.5)

                ax.set_title(f'c{channels}, GRU{gru_size}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Epoch', fontsize=8)
                ax.set_ylabel('Variance Ratio', fontsize=8)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, max(0.6, ax.get_ylim()[1])])
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.suptitle('Representation Analysis: Singular Value Evolution Across Factorial Design',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(output_dir) / 'factorial_representation_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def compute_factorial_summary(data, output_dir='visusualizations'):
    """Compute summary statistics for factorial analysis."""
    summary_rows = []

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = get_model_name(channels, gru_size)

            if model_name in data:
                df = data[model_name]

                stats = {
                    'Channels': channels,
                    'GRU_Size': gru_size,
                    'Model': model_name,
                }

                # Weight metrics
                if 'weight_norm' in df.columns:
                    stats['Initial_Weight_Norm'] = df['weight_norm'].iloc[0]
                    stats['Final_Weight_Norm'] = df['weight_norm'].iloc[-1]
                    stats['Weight_Growth'] = stats['Final_Weight_Norm'] / stats['Initial_Weight_Norm']

                # Step metrics
                if 'step_norm' in df.columns:
                    valid_steps = df['step_norm'].dropna()
                    if len(valid_steps) > 0:
                        stats['Mean_Step_Norm'] = valid_steps.mean()
                        stats['Step_Norm_CV'] = valid_steps.std() / valid_steps.mean()

                # Convergence metrics
                if 'step_cosine' in df.columns:
                    valid_cosine = df['step_cosine'].dropna()
                    if len(valid_cosine) > 0:
                        stats['Final_Cosine'] = valid_cosine.iloc[-1]

                if 'relative_step' in df.columns:
                    valid_rel = df['relative_step'].dropna()
                    if len(valid_rel) > 0:
                        stats['Final_Relative_Step'] = valid_rel.iloc[-1]

                # Representation metrics
                if 'repr_total_variance' in df.columns:
                    valid_repr = df['repr_total_variance'].dropna()
                    if len(valid_repr) > 0:
                        stats['Mean_Total_Variance'] = valid_repr.mean()
                        stats['Final_Total_Variance'] = valid_repr.iloc[-1]

                if 'repr_top1_ratio' in df.columns:
                    valid_top1 = df['repr_top1_ratio'].dropna()
                    if len(valid_top1) > 0:
                        stats['Mean_Top1_Ratio'] = valid_top1.mean()
                        stats['Max_Top1_Ratio'] = valid_top1.max()

                summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)

    # Save full summary
    output_path = Path(output_dir) / 'factorial_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print formatted summary
    print("\n" + "="*80)
    print("FACTORIAL DESIGN SUMMARY: 3×3 Grid (Channels × GRU Size)")
    print("="*80)

    # Pivot table for weight growth
    if 'Weight_Growth' in summary_df.columns:
        print("\nWeight Growth Factor (Final / Initial):")
        pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Weight_Growth')
        print(pivot.to_string(float_format=lambda x: f'{x:.2f}x'))

    # Pivot table for step norm CV
    if 'Step_Norm_CV' in summary_df.columns:
        print("\nStep Norm Coefficient of Variation (Stability):")
        pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Step_Norm_CV')
        print(pivot.to_string(float_format=lambda x: f'{x:.1%}'))

    # Pivot table for representation collapse
    if 'Mean_Top1_Ratio' in summary_df.columns:
        print("\nMean Top-1 SV Ratio (Representation Concentration):")
        pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Mean_Top1_Ratio')
        print(pivot.to_string(float_format=lambda x: f'{x:.3f}'))

    # Pivot table for total variance
    if 'Mean_Total_Variance' in summary_df.columns:
        print("\nMean Total Representation Variance:")
        pivot = summary_df.pivot(index='Channels', columns='GRU_Size', values='Mean_Total_Variance')
        print(pivot.to_string(float_format=lambda x: f'{x:.1f}'))

    return summary_df

def main():
    """Main execution."""
    print("=" * 80)
    print("FACTORIAL ANALYSIS: 3×3 Design (Channels × GRU Sizes)")
    print("=" * 80)

    # Create output directory
    output_dir = Path('visusualizations')
    output_dir.mkdir(exist_ok=True)

    # Load all data
    print("\nLoading metrics for all 9 model configurations...")
    data = load_all_metrics()

    if len(data) != 9:
        print(f"\nWarning: Expected 9 models, found {len(data)}")
        print("Missing models will be skipped in analysis.")

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating factorial visualizations...")
    print("="*80)

    print("\n1. Channel sweeps (CNN capacity effect)...")
    plot_channel_sweeps(data, output_dir)

    print("\n2. GRU sweeps (RNN capacity effect)...")
    plot_gru_sweeps(data, output_dir)

    print("\n3. Factorial heatmaps...")
    plot_factorial_heatmaps(data, output_dir)

    print("\n4. Representation factorial grid...")
    plot_representation_factorial(data, output_dir)

    print("\n5. Computing factorial summary statistics...")
    summary_df = compute_factorial_summary(data, output_dir)

    print("\n" + "="*80)
    print("Factorial analysis complete!")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == '__main__':
    main()

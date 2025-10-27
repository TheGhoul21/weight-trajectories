#!/usr/bin/env python3
"""
Visualization script for checkpoint metrics analysis.
Generates comprehensive plots of weight trajectories, step dynamics, and representation analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Color palette for models
COLORS = {
    'k3_c16_gru128': '#2E86AB',   # Blue
    'k3_c64_gru128': '#A23B72',   # Purple
    'k3_c256_gru128': '#F18F01',  # Orange
}

LABELS = {
    'k3_c16_gru128': 'c16 (16 channels)',
    'k3_c64_gru128': 'c64 (64 channels)',
    'k3_c256_gru128': 'c256 (256 channels)',
}


def load_metrics(base_dir='diagnostics/checkpoint_metrics'):
    """Load all checkpoint metrics CSV files."""
    base_path = Path(base_dir)
    data = {}

    for model_name in ['k3_c16_gru128', 'k3_c64_gru128', 'k3_c256_gru128']:
        csv_path = base_path / f'{model_name}_metrics.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data[model_name] = df
            print(f"Loaded {len(df)} checkpoints for {model_name}")
        else:
            print(f"Warning: {csv_path} not found")

    return data


def compute_derived_metrics(data):
    """Compute additional derived metrics."""
    for model_name, df in data.items():
        # Trajectory curvature (angular change between consecutive steps)
        if 'step_cosine' in df.columns:
            # Convert cosine to angle (radians)
            df['step_angle_rad'] = np.arccos(df['step_cosine'].clip(-1, 1))
            df['step_angle_deg'] = np.degrees(df['step_angle_rad'])

            # Curvature: change in angle per epoch
            df['curvature'] = df['step_angle_deg'].diff() / 3  # per epoch (checkpoints every 3 epochs)

        # Effective learning rate proxy: relative_step change
        if 'relative_step' in df.columns:
            df['relative_step_change'] = df['relative_step'].diff()

        # Weight growth rate
        if 'weight_norm' in df.columns:
            df['weight_growth_rate'] = df['weight_norm'].diff() / 3  # per epoch

        # Representation concentration (entropy-like measure)
        if 'repr_top1_ratio' in df.columns:
            # Use first 4 singular values to compute concentration
            top_cols = [f'repr_top{i}_ratio' for i in range(1, 5)]
            if all(col in df.columns for col in top_cols):
                # Effective rank (inverse participation ratio)
                top4_sum = df[top_cols].sum(axis=1)
                top4_squared = (df[top_cols] ** 2).sum(axis=1)
                df['repr_effective_rank'] = top4_sum ** 2 / top4_squared

    return data


def plot_weight_trajectories(data, output_dir='visusualizations'):
    """Plot 1: Weight norm evolution over training."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: Absolute weight norm
    ax = axes[0, 0]
    for model_name, df in data.items():
        ax.plot(df['epoch'], df['weight_norm'],
                label=LABELS[model_name], color=COLORS[model_name],
                marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Norm')
    ax.set_title('Weight Norm Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b: Normalized weight growth (relative to initial)
    ax = axes[0, 1]
    for model_name, df in data.items():
        initial_norm = df['weight_norm'].iloc[0]
        normalized = df['weight_norm'] / initial_norm
        ax.plot(df['epoch'], normalized,
                label=LABELS[model_name], color=COLORS[model_name],
                marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Weight Norm (ratio to initial)')
    ax.set_title('Normalized Weight Growth', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Initial')

    # 1c: Weight growth rate
    ax = axes[1, 0]
    for model_name, df in data.items():
        if 'weight_growth_rate' in df.columns:
            ax.plot(df['epoch'], df['weight_growth_rate'],
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Growth Rate (norm units / epoch)')
    ax.set_title('Weight Growth Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # 1d: Log-scale weight norm
    ax = axes[1, 1]
    for model_name, df in data.items():
        ax.semilogy(df['epoch'], df['weight_norm'],
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Norm (log scale)')
    ax.set_title('Weight Norm Evolution (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'weight_trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_step_dynamics(data, output_dir='visusualizations'):
    """Plot 2: Step norm and directional dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 2a: Step norm
    ax = axes[0, 0]
    for model_name, df in data.items():
        # Skip first row (NaN)
        valid_df = df[df['step_norm'].notna()]
        ax.plot(valid_df['epoch'], valid_df['step_norm'],
                label=LABELS[model_name], color=COLORS[model_name],
                marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Step Norm (Euclidean distance)')
    ax.set_title('Weight Step Magnitude', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2b: Relative step
    ax = axes[0, 1]
    for model_name, df in data.items():
        valid_df = df[df['relative_step'].notna()]
        ax.plot(valid_df['epoch'], valid_df['relative_step'],
                label=LABELS[model_name], color=COLORS[model_name],
                marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative Step (step_norm / weight_norm)')
    ax.set_title('Scale-Normalized Step Size', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2c: Step cosine similarity
    ax = axes[1, 0]
    for model_name, df in data.items():
        valid_df = df[df['step_cosine'].notna()]
        ax.plot(valid_df['epoch'], valid_df['step_cosine'],
                label=LABELS[model_name], color=COLORS[model_name],
                marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Directional Consistency (Step Cosine)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.8, 1.0])

    # 2d: Angular deviation (degrees)
    ax = axes[1, 1]
    for model_name, df in data.items():
        if 'step_angle_deg' in df.columns:
            valid_df = df[df['step_angle_deg'].notna()]
            ax.plot(valid_df['epoch'], valid_df['step_angle_deg'],
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Angular Deviation (degrees)')
    ax.set_title('Trajectory Angular Change', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    output_path = Path(output_dir) / 'step_dynamics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_representation_analysis(data, output_dir='visusualizations'):
    """Plot 3: Representation variance and singular value analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 3a: Total representation variance
    ax = axes[0, 0]
    for model_name, df in data.items():
        if 'repr_total_variance' in df.columns:
            valid_df = df[df['repr_total_variance'].notna()]
            ax.plot(valid_df['epoch'], valid_df['repr_total_variance'],
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Variance')
    ax.set_title('Representation Total Variance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3b: Top singular value concentration
    ax = axes[0, 1]
    for model_name, df in data.items():
        if 'repr_top1_ratio' in df.columns:
            valid_df = df[df['repr_top1_ratio'].notna()]
            ax.plot(valid_df['epoch'], valid_df['repr_top1_ratio'],
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Top-1 Singular Value Ratio')
    ax.set_title('Representation Concentration (Top SV)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1/128, color='black', linestyle='--', alpha=0.3, label='Uniform (1/128)')

    # 3c: Cumulative top-4 variance
    ax = axes[1, 0]
    for model_name, df in data.items():
        top_cols = [f'repr_top{i}_ratio' for i in range(1, 5)]
        if all(col in df.columns for col in top_cols):
            valid_df = df[df['repr_top1_ratio'].notna()]
            cumulative = valid_df[top_cols].sum(axis=1)
            ax.plot(valid_df['epoch'], cumulative,
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.set_title('Top-4 Cumulative Variance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=4/128, color='black', linestyle='--', alpha=0.3, label='Uniform (4/128)')

    # 3d: Effective rank
    ax = axes[1, 1]
    for model_name, df in data.items():
        if 'repr_effective_rank' in df.columns:
            valid_df = df[df['repr_effective_rank'].notna()]
            ax.plot(valid_df['epoch'], valid_df['repr_effective_rank'],
                    label=LABELS[model_name], color=COLORS[model_name],
                    marker='o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective Rank (IPR)')
    ax.set_title('Representation Effective Rank', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=4, color='black', linestyle='--', alpha=0.3, label='Maximum (4)')

    plt.tight_layout()
    output_path = Path(output_dir) / 'representation_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_phase_analysis(data, output_dir='visusualizations'):
    """Plot 4: Training phase identification via combined metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4a: Step norm vs weight norm (trajectory in weight space)
    ax = axes[0, 0]
    for model_name, df in data.items():
        valid_df = df[df['step_norm'].notna()]
        scatter = ax.scatter(valid_df['weight_norm'], valid_df['step_norm'],
                           c=valid_df['epoch'], cmap='viridis',
                           label=LABELS[model_name], s=50, alpha=0.6,
                           edgecolors=COLORS[model_name], linewidths=2)
    ax.set_xlabel('Weight Norm')
    ax.set_ylabel('Step Norm')
    ax.set_title('Weight-Step Phase Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Epoch')

    # 4b: Relative step vs cosine similarity
    ax = axes[0, 1]
    for model_name, df in data.items():
        valid_df = df[df['relative_step'].notna() & df['step_cosine'].notna()]
        scatter = ax.scatter(valid_df['step_cosine'], valid_df['relative_step'],
                           c=valid_df['epoch'], cmap='viridis',
                           label=LABELS[model_name], s=50, alpha=0.6,
                           edgecolors=COLORS[model_name], linewidths=2)
    ax.set_xlabel('Step Cosine Similarity')
    ax.set_ylabel('Relative Step')
    ax.set_title('Convergence Phase Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.colorbar(scatter, ax=ax, label='Epoch')

    # 4c: Representation variance vs weight norm
    ax = axes[1, 0]
    for model_name, df in data.items():
        if 'repr_total_variance' in df.columns:
            valid_df = df[df['repr_total_variance'].notna()]
            scatter = ax.scatter(valid_df['weight_norm'], valid_df['repr_total_variance'],
                               c=valid_df['epoch'], cmap='viridis',
                               label=LABELS[model_name], s=50, alpha=0.6,
                               edgecolors=COLORS[model_name], linewidths=2)
    ax.set_xlabel('Weight Norm')
    ax.set_ylabel('Representation Total Variance')
    ax.set_title('Weight-Representation Phase Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Epoch')

    # 4d: Timeline view of all key metrics (normalized)
    ax = axes[1, 1]
    for model_name in ['k3_c256_gru128']:  # Focus on most interesting model
        df = data[model_name]

        # Normalize metrics to [0, 1] for visualization
        def normalize(series):
            valid = series.dropna()
            if len(valid) == 0 or valid.max() == valid.min():
                return series
            return (series - valid.min()) / (valid.max() - valid.min())

        ax.plot(df['epoch'], normalize(df['weight_norm']),
                label='Weight Norm', linewidth=2, alpha=0.8)
        if 'step_norm' in df.columns:
            ax.plot(df['epoch'], normalize(df['step_norm']),
                    label='Step Norm', linewidth=2, alpha=0.8)
        if 'step_cosine' in df.columns:
            ax.plot(df['epoch'], normalize(df['step_cosine']),
                    label='Step Cosine', linewidth=2, alpha=0.8)
        if 'repr_total_variance' in df.columns:
            ax.plot(df['epoch'], normalize(df['repr_total_variance']),
                    label='Repr Variance', linewidth=2, alpha=0.8)
        if 'repr_top1_ratio' in df.columns:
            ax.plot(df['epoch'], normalize(df['repr_top1_ratio']),
                    label='Top-1 Ratio', linewidth=2, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Value [0, 1]')
    ax.set_title(f'Multi-Metric Timeline: {LABELS["k3_c256_gru128"]}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'phase_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparative_heatmaps(data, output_dir='visusualizations'):
    """Plot 5: Heatmaps comparing all models across all metrics."""
    metrics = ['weight_norm', 'step_norm', 'relative_step', 'step_cosine',
               'repr_total_variance', 'repr_top1_ratio']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Create matrix: rows=epochs, cols=models
        epochs = None
        matrix_data = []
        model_names = []

        for model_name in ['k3_c16_gru128', 'k3_c64_gru128', 'k3_c256_gru128']:
            df = data[model_name]
            if metric in df.columns:
                if epochs is None:
                    epochs = df['epoch'].values
                matrix_data.append(df[metric].values)
                model_names.append(LABELS[model_name].split('(')[0].strip())

        if matrix_data:
            matrix = np.array(matrix_data).T  # Transpose for epoch rows

            # Create heatmap
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r',
                          interpolation='nearest')

            # Set ticks
            ax.set_yticks(np.arange(0, len(epochs), 5))
            ax.set_yticklabels(epochs[::5])
            ax.set_xticks(np.arange(len(model_names)))
            ax.set_xticklabels(model_names, rotation=0)

            ax.set_ylabel('Epoch')
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')

            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    output_path = Path(output_dir) / 'comparative_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_singular_value_evolution(data, output_dir='visusualizations'):
    """Plot 6: Detailed singular value evolution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model_name in enumerate(['k3_c16_gru128', 'k3_c64_gru128', 'k3_c256_gru128']):
        ax = axes[idx]
        df = data[model_name]

        # Plot each singular value ratio
        for sv_idx in range(1, 5):
            col = f'repr_top{sv_idx}_ratio'
            if col in df.columns:
                valid_df = df[df[col].notna()]
                ax.plot(valid_df['epoch'], valid_df[col],
                       label=f'SV{sv_idx}', marker='o', markersize=3, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Variance Ratio')
        ax.set_title(f'{LABELS[model_name]}\nSingular Value Distribution',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(0.6, ax.get_ylim()[1])])

    plt.tight_layout()
    output_path = Path(output_dir) / 'singular_value_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_statistics(data, output_dir='visusualizations'):
    """Generate summary statistics table."""
    summary = []

    for model_name in ['k3_c16_gru128', 'k3_c64_gru128', 'k3_c256_gru128']:
        df = data[model_name]

        stats = {
            'Model': LABELS[model_name],
            'Initial Weight Norm': f"{df['weight_norm'].iloc[0]:.2f}",
            'Final Weight Norm': f"{df['weight_norm'].iloc[-1]:.2f}",
            'Weight Growth': f"{df['weight_norm'].iloc[-1] / df['weight_norm'].iloc[0]:.2f}x",
        }

        if 'step_norm' in df.columns:
            valid_steps = df['step_norm'].dropna()
            stats['Mean Step Norm'] = f"{valid_steps.mean():.2f}"
            stats['Step Norm CV'] = f"{(valid_steps.std() / valid_steps.mean()):.2%}"

        if 'step_cosine' in df.columns:
            valid_cosine = df['step_cosine'].dropna()
            stats['Mean Cosine'] = f"{valid_cosine.mean():.4f}"
            stats['Final Cosine'] = f"{valid_cosine.iloc[-1]:.4f}"

        if 'repr_total_variance' in df.columns:
            valid_repr = df['repr_total_variance'].dropna()
            stats['Mean Repr Variance'] = f"{valid_repr.mean():.1f}"

        if 'repr_top1_ratio' in df.columns:
            valid_top1 = df['repr_top1_ratio'].dropna()
            stats['Mean Top-1 Ratio'] = f"{valid_top1.mean():.3f}"
            stats['Top-1 Range'] = f"{valid_top1.min():.3f}-{valid_top1.max():.3f}"

        summary.append(stats)

    summary_df = pd.DataFrame(summary)
    output_path = Path(output_dir) / 'summary_statistics.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary Statistics:\n{summary_df.to_string(index=False)}")
    print(f"\nSaved: {output_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Checkpoint Metrics Visualization")
    print("=" * 80)

    # Create output directory
    output_dir = Path('visusualizations')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading metrics data...")
    data = load_metrics()

    if not data:
        print("Error: No metrics data found!")
        return

    # Compute derived metrics
    print("\nComputing derived metrics...")
    data = compute_derived_metrics(data)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_weight_trajectories(data, output_dir)
    plot_step_dynamics(data, output_dir)
    plot_representation_analysis(data, output_dir)
    plot_phase_analysis(data, output_dir)
    plot_comparative_heatmaps(data, output_dir)
    plot_singular_value_evolution(data, output_dir)

    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_statistics(data, output_dir)

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()

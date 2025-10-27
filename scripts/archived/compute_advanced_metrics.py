#!/usr/bin/env python3
"""
Compute advanced derived metrics from checkpoint trajectories.
Includes trajectory curvature, effective learning rates, phase detection, and more.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import linregress


def load_metrics(base_dir='diagnostics/checkpoint_metrics'):
    """Load all checkpoint metrics CSV files."""
    base_path = Path(base_dir)
    data = {}

    # All 9 models in factorial design
    channels = [16, 64, 256]
    gru_sizes = [8, 32, 128]

    for c in channels:
        for g in gru_sizes:
            model_name = f'k3_c{c}_gru{g}'
            csv_path = base_path / f'{model_name}_metrics.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                data[model_name] = df
                print(f"Loaded {len(df)} checkpoints for {model_name}")

    return data


def compute_trajectory_curvature(df):
    """
    Compute trajectory curvature in weight space.
    Curvature measures how much the trajectory "bends" at each point.
    """
    metrics = {}

    if 'step_cosine' in df.columns:
        # Angular curvature: change in direction
        angles_rad = np.arccos(df['step_cosine'].clip(-1, 1))
        angles_deg = np.degrees(angles_rad)

        metrics['step_angle_rad'] = angles_rad
        metrics['step_angle_deg'] = angles_deg

        # Curvature: rate of angular change
        # Using finite differences: d(angle)/d(epoch)
        epoch_diff = df['epoch'].diff()
        angle_diff = pd.Series(angles_deg).diff()
        metrics['angular_curvature'] = angle_diff / epoch_diff

        # Cumulative angular change (total "turning")
        metrics['cumulative_angle'] = angles_deg.cumsum()

    # Geometric curvature using weight positions
    if 'step_norm' in df.columns and 'step_cosine' in df.columns:
        # Approximate curvature: κ ≈ |dθ/ds| where s is arc length
        # Arc length increment ≈ step_norm
        arc_length_increment = df['step_norm']
        angle_change = pd.Series(metrics.get('step_angle_rad', np.zeros(len(df))))

        # Curvature (rad per norm unit)
        metrics['geometric_curvature'] = angle_change / arc_length_increment.replace(0, np.nan)

    return pd.DataFrame(metrics)


def compute_effective_learning_rate(df):
    """
    Estimate effective learning rate from weight trajectory.
    """
    metrics = {}

    # Proxy 1: Relative step size
    if 'relative_step' in df.columns:
        metrics['lr_proxy_relative_step'] = df['relative_step']

        # Smoothed version (5-point Savitzky-Golay filter)
        if len(df) >= 5:
            smoothed = savgol_filter(df['relative_step'].fillna(0), 5, 2)
            metrics['lr_proxy_smoothed'] = smoothed

    # Proxy 2: Step norm normalized by gradient magnitude proxy
    # Assumption: gradient ∝ 1 / (epoch + 1) for typical decay schedules
    if 'step_norm' in df.columns:
        epoch_factor = 1.0 / (df['epoch'] + 1)
        metrics['lr_proxy_step_normalized'] = df['step_norm'] * epoch_factor

    # Proxy 3: Rate of weight norm change
    if 'weight_norm' in df.columns:
        weight_velocity = df['weight_norm'].diff() / df['epoch'].diff()
        metrics['weight_velocity'] = weight_velocity

        # Acceleration
        weight_acceleration = weight_velocity.diff() / df['epoch'].diff()
        metrics['weight_acceleration'] = weight_acceleration

    return pd.DataFrame(metrics)


def detect_training_phases(df):
    """
    Detect distinct training phases using clustering on trajectory metrics.
    """
    phases = {}

    # Phase indicators
    if 'relative_step' in df.columns:
        rel_step = df['relative_step'].fillna(0)

        # Exploration phase: high relative steps
        phases['is_exploration'] = rel_step > 0.15

        # Refinement phase: moderate relative steps, high cosine
        if 'step_cosine' in df.columns:
            cosine = df['step_cosine'].fillna(0)
            phases['is_refinement'] = (rel_step < 0.15) & (rel_step > 0.05) & (cosine > 0.99)

        # Convergence phase: low relative steps
        phases['is_convergence'] = rel_step < 0.05

    # Detect regime changes (sudden increases in step norm)
    if 'step_norm' in df.columns:
        step_norm = df['step_norm'].fillna(0)

        # Rolling mean and std
        window = 5
        if len(df) >= window:
            rolling_mean = step_norm.rolling(window, center=True).mean()
            rolling_std = step_norm.rolling(window, center=True).std()

            # Anomaly: step norm > 2 std above rolling mean
            phases['is_regime_shift'] = step_norm > (rolling_mean + 2 * rolling_std)

    # Representation collapse detection
    if 'repr_top1_ratio' in df.columns:
        top1 = df['repr_top1_ratio'].fillna(0)

        # Collapsed: top-1 captures > 40% variance
        phases['is_repr_collapsed'] = top1 > 0.4

        # Balanced: top-1 captures 10-20%
        phases['is_repr_balanced'] = (top1 > 0.10) & (top1 < 0.20)

    return pd.DataFrame(phases)


def compute_trajectory_statistics(df):
    """
    Compute global statistics for the entire trajectory.
    """
    stats = {}

    # Weight norm statistics
    if 'weight_norm' in df.columns:
        stats['initial_weight_norm'] = df['weight_norm'].iloc[0]
        stats['final_weight_norm'] = df['weight_norm'].iloc[-1]
        stats['weight_growth_factor'] = stats['final_weight_norm'] / stats['initial_weight_norm']
        stats['weight_norm_mean'] = df['weight_norm'].mean()
        stats['weight_norm_std'] = df['weight_norm'].std()

        # Linear fit to weight norm
        valid_idx = df['weight_norm'].notna()
        if valid_idx.sum() > 2:
            slope, intercept, r_value, _, _ = linregress(
                df.loc[valid_idx, 'epoch'],
                df.loc[valid_idx, 'weight_norm']
            )
            stats['weight_norm_slope'] = slope
            stats['weight_norm_r2'] = r_value ** 2

    # Step statistics
    if 'step_norm' in df.columns:
        valid_steps = df['step_norm'].dropna()
        if len(valid_steps) > 0:
            stats['step_norm_mean'] = valid_steps.mean()
            stats['step_norm_std'] = valid_steps.std()
            stats['step_norm_cv'] = stats['step_norm_std'] / stats['step_norm_mean']
            stats['step_norm_min'] = valid_steps.min()
            stats['step_norm_max'] = valid_steps.max()

            # Total trajectory length (arc length)
            stats['total_trajectory_length'] = valid_steps.sum()

    # Directional statistics
    if 'step_cosine' in df.columns:
        valid_cosine = df['step_cosine'].dropna()
        if len(valid_cosine) > 0:
            stats['step_cosine_mean'] = valid_cosine.mean()
            stats['step_cosine_min'] = valid_cosine.min()
            stats['step_cosine_final'] = valid_cosine.iloc[-1]

            # Directional consistency: proportion with cosine > 0.99
            stats['high_alignment_fraction'] = (valid_cosine > 0.99).mean()

    # Relative step statistics
    if 'relative_step' in df.columns:
        valid_rel = df['relative_step'].dropna()
        if len(valid_rel) > 2:
            stats['relative_step_mean'] = valid_rel.mean()
            stats['relative_step_initial'] = valid_rel.iloc[0]
            stats['relative_step_final'] = valid_rel.iloc[-1]

            # Exponential decay fit: y = A * exp(-k * x)
            # Using log transform: log(y) = log(A) - k * x
            log_rel = np.log(valid_rel + 1e-10)
            epochs = df.loc[valid_rel.index, 'epoch']
            slope, intercept, r_value, _, _ = linregress(epochs, log_rel)
            stats['relative_step_decay_constant'] = -slope
            stats['relative_step_decay_r2'] = r_value ** 2
            stats['relative_step_half_life'] = np.log(2) / max(-slope, 1e-10)

    # Representation statistics
    if 'repr_total_variance' in df.columns:
        valid_repr = df['repr_total_variance'].dropna()
        if len(valid_repr) > 0:
            stats['repr_variance_mean'] = valid_repr.mean()
            stats['repr_variance_std'] = valid_repr.std()
            stats['repr_variance_initial'] = valid_repr.iloc[0]
            stats['repr_variance_final'] = valid_repr.iloc[-1]

    if 'repr_top1_ratio' in df.columns:
        valid_top1 = df['repr_top1_ratio'].dropna()
        if len(valid_top1) > 0:
            stats['repr_top1_mean'] = valid_top1.mean()
            stats['repr_top1_min'] = valid_top1.min()
            stats['repr_top1_max'] = valid_top1.max()
            stats['repr_top1_range'] = valid_top1.max() - valid_top1.min()

            # Concentration score (inverse of uniformity)
            uniform_ratio = 1.0 / 128  # Assuming GRU size 128
            stats['repr_concentration_score'] = stats['repr_top1_mean'] / uniform_ratio

    return stats


def compute_convergence_metrics(df):
    """
    Compute metrics specifically related to convergence behavior.
    """
    metrics = {}

    # Find convergence epoch using different criteria
    if 'relative_step' in df.columns:
        valid_rel = df['relative_step'].dropna()
        # Criterion 1: relative_step < 0.08
        converged = valid_rel < 0.08
        if converged.any():
            metrics['convergence_epoch_rel08'] = df.loc[converged.idxmax(), 'epoch']

        # Criterion 2: relative_step < 0.05
        converged = valid_rel < 0.05
        if converged.any():
            metrics['convergence_epoch_rel05'] = df.loc[converged.idxmax(), 'epoch']

    if 'step_cosine' in df.columns:
        valid_cosine = df['step_cosine'].dropna()
        # Criterion 3: cosine > 0.995
        converged = valid_cosine > 0.995
        if converged.any():
            metrics['convergence_epoch_cos995'] = df.loc[converged.idxmax(), 'epoch']

    if 'step_norm' in df.columns:
        valid_step = df['step_norm'].dropna()
        # Criterion 4: step_norm < 10
        converged = valid_step < 10
        if converged.any():
            metrics['convergence_epoch_step10'] = df.loc[converged.idxmax(), 'epoch']

    # Late-stage volatility (last 25% of training)
    last_quarter_idx = int(len(df) * 0.75)
    late_df = df.iloc[last_quarter_idx:]

    if 'step_norm' in late_df.columns:
        late_steps = late_df['step_norm'].dropna()
        if len(late_steps) > 0:
            metrics['late_stage_step_mean'] = late_steps.mean()
            metrics['late_stage_step_std'] = late_steps.std()
            metrics['late_stage_step_cv'] = metrics['late_stage_step_std'] / max(metrics['late_stage_step_mean'], 1e-10)

    # Detect non-monotonic behavior in late stage
    if 'step_norm' in late_df.columns and len(late_df) > 5:
        step_increases = (late_df['step_norm'].diff() > 0).sum()
        total_changes = len(late_df) - 1
        metrics['late_stage_increase_fraction'] = step_increases / max(total_changes, 1)

    # Final convergence quality (last 3 checkpoints)
    final_df = df.iloc[-3:]

    if 'relative_step' in final_df.columns:
        final_rel = final_df['relative_step'].dropna()
        if len(final_rel) > 0:
            metrics['final_relative_step_mean'] = final_rel.mean()

    if 'step_cosine' in final_df.columns:
        final_cosine = final_df['step_cosine'].dropna()
        if len(final_cosine) > 0:
            metrics['final_cosine_mean'] = final_cosine.mean()

    return metrics


def analyze_representation_collapse(df):
    """
    Detailed analysis of representation collapse dynamics.
    """
    metrics = {}

    if 'repr_top1_ratio' not in df.columns:
        return metrics

    # Collapse indicators
    top1 = df['repr_top1_ratio'].fillna(0)

    # Maximum concentration
    metrics['max_repr_concentration'] = top1.max()
    metrics['max_concentration_epoch'] = df.loc[top1.idxmax(), 'epoch']

    # Concentration trajectory
    # Is it increasing (bad), decreasing (good), or stable?
    if len(top1) > 2:
        early_mean = top1.iloc[:len(top1)//3].mean()
        late_mean = top1.iloc[-len(top1)//3:].mean()
        metrics['repr_concentration_change'] = late_mean - early_mean

        if metrics['repr_concentration_change'] > 0.05:
            metrics['repr_collapse_trend'] = 'increasing'
        elif metrics['repr_concentration_change'] < -0.05:
            metrics['repr_collapse_trend'] = 'decreasing'
        else:
            metrics['repr_collapse_trend'] = 'stable'

    # Effective dimensionality using all 4 singular values
    if all(f'repr_top{i}_ratio' in df.columns for i in range(1, 5)):
        top4_df = df[[f'repr_top{i}_ratio' for i in range(1, 5)]].fillna(0)

        # Shannon entropy (normalized)
        # H = -sum(p_i * log(p_i))
        epsilon = 1e-10
        top4_log = np.log(top4_df + epsilon)
        entropy = -(top4_df * top4_log).sum(axis=1)
        metrics['repr_entropy_mean'] = entropy.mean()
        metrics['repr_entropy_final'] = entropy.iloc[-1]

        # Participation ratio: (sum p_i)^2 / sum(p_i^2)
        top4_sum = top4_df.sum(axis=1)
        top4_sum_sq = (top4_df ** 2).sum(axis=1)
        participation = top4_sum ** 2 / (top4_sum_sq + epsilon)
        metrics['repr_participation_mean'] = participation.mean()
        metrics['repr_participation_final'] = participation.iloc[-1]

    # Variance stability
    if 'repr_total_variance' in df.columns:
        total_var = df['repr_total_variance'].fillna(0)
        if len(total_var) > 2:
            var_cv = total_var.std() / max(total_var.mean(), 1e-10)
            metrics['repr_variance_stability'] = 1.0 / (1.0 + var_cv)  # Higher = more stable

    return metrics


def main():
    """Main execution."""
    print("=" * 80)
    print("Advanced Trajectory Metrics Computation")
    print("=" * 80)

    # Load data
    print("\nLoading metrics data...")
    data = load_metrics()

    if not data:
        print("Error: No metrics data found!")
        return

    # Process each model
    all_results = {}

    for model_name, df in data.items():
        print(f"\n{'='*80}")
        print(f"Processing: {model_name}")
        print(f"{'='*80}")

        # Compute all derived metrics
        curvature_df = compute_trajectory_curvature(df)
        lr_df = compute_effective_learning_rate(df)
        phases_df = detect_training_phases(df)

        # Combine with original
        extended_df = pd.concat([df, curvature_df, lr_df, phases_df], axis=1)

        # Save extended metrics
        output_path = Path('diagnostics/checkpoint_metrics') / f'{model_name}_extended_metrics.csv'
        extended_df.to_csv(output_path, index=False)
        print(f"Saved extended metrics: {output_path}")

        # Compute global statistics
        trajectory_stats = compute_trajectory_statistics(df)
        convergence_metrics = compute_convergence_metrics(df)
        repr_analysis = analyze_representation_collapse(df)

        # Combine all statistics
        all_stats = {**trajectory_stats, **convergence_metrics, **repr_analysis}
        all_results[model_name] = all_stats

        # Print key findings
        print(f"\nKey Statistics:")
        print(f"  Weight growth: {all_stats.get('weight_growth_factor', 'N/A'):.3f}x")
        print(f"  Trajectory length: {all_stats.get('total_trajectory_length', 'N/A'):.1f}")
        print(f"  Step norm CV: {all_stats.get('step_norm_cv', 'N/A'):.2%}")
        print(f"  Relative step half-life: {all_stats.get('relative_step_half_life', 'N/A'):.1f} epochs")
        print(f"  Convergence (rel<0.08): epoch {all_stats.get('convergence_epoch_rel08', 'N/A')}")
        print(f"  Repr concentration: {all_stats.get('repr_concentration_score', 'N/A'):.2f}x uniform")
        print(f"  Repr collapse trend: {all_stats.get('repr_collapse_trend', 'N/A')}")

    # Save comparative summary
    summary_df = pd.DataFrame(all_results).T
    summary_path = Path('diagnostics/checkpoint_metrics') / 'advanced_metrics_summary.csv'
    summary_df.to_csv(summary_path)
    print(f"\n{'='*80}")
    print(f"Saved comparative summary: {summary_path}")
    print(f"{'='*80}")

    # Print comparative table
    print("\nComparative Summary:")
    print(summary_df[['weight_growth_factor', 'step_norm_cv', 'relative_step_half_life',
                      'repr_concentration_score', 'repr_collapse_trend']].to_string())


if __name__ == '__main__':
    main()

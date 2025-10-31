#!/usr/bin/env python3
"""
Metric-space embedding of learning trajectories using UMAP, t-SNE, or PHATE.

Embeds all 9 models' trajectories into a shared 2D space using architecture-agnostic metrics.
This allows direct comparison across different model configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

# Try to import UMAP, fall back to other methods if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    UMAP = None

try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

CHANNELS = [16, 64, 256]
GRU_SIZES = [8, 32, 128]
CHANNEL_COLORS = {16: '#2E86AB', 64: '#A23B72', 256: '#F18F01'}
GRU_COLORS = {8: '#06A77D', 32: '#D5C67A', 128: '#F1A208'}

def load_all_trajectories(metrics_dir, checkpoint_dir):
    """Load metrics and training history for all models."""
    all_data = []

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'

            # Load metrics
            csv_path = Path(metrics_dir) / f'{model_name}_metrics.csv'
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)

            # Load training history
            history_path = Path(checkpoint_dir) / model_name / 'training_history.json'
            history = None
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)

            # Combine metrics with loss data
            for idx, row in df.iterrows():
                epoch = int(row['epoch'])

                point = {
                    'model': model_name,
                    'channels': channels,
                    'gru_size': gru_size,
                    'epoch': epoch,
                    'checkpoint_idx': idx,
                }

                # Add trajectory metrics
                for col in ['weight_norm', 'step_norm', 'step_cosine', 'relative_step',
                           'repr_total_variance', 'repr_top1_ratio', 'repr_top2_ratio',
                           'repr_top3_ratio', 'repr_top4_ratio']:
                    if col in row:
                        point[col] = row[col]

                # Add loss data if available
                if history and epoch <= len(history['train_loss']):
                    epoch_idx = epoch - 1
                    point['train_loss'] = history['train_loss'][epoch_idx]
                    point['val_loss'] = history['val_loss'][epoch_idx]
                    point['train_val_gap'] = point['val_loss'] - point['train_loss']

                all_data.append(point)

    return pd.DataFrame(all_data)

def create_feature_matrix(df):
    """Create feature matrix for embedding."""

    # Select features (architecture-agnostic metrics)
    feature_cols = [
        'epoch',                    # Temporal position
        'weight_norm',             # Trajectory position
        'step_norm',               # Movement magnitude
        'step_cosine',             # Movement direction consistency
        'relative_step',           # Scale-free movement
        'repr_total_variance',     # Representation capacity
        'repr_top1_ratio',         # Collapse measure
        'repr_top2_ratio',
        'repr_top3_ratio',
        'repr_top4_ratio',
        'train_loss',              # Training performance
        'val_loss',                # Generalization performance
        'train_val_gap',           # Overfitting measure
    ]

    # Extract available features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    # Create feature matrix (fill NaN with column mean)
    X = df[available_features].values
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, available_features

def embed_trajectories(
    X,
    method='umap',
    n_neighbors=15,
    min_dist=0.1,
    time_alpha=None,
    feature_names=None,
    n_components: int = 2,
):
    """Embed trajectories using UMAP, t-SNE, PHATE, or T-PHATE.

    Args:
        X: standardized feature matrix
        method: 'umap' | 'tsne' | 'phate' | 'tphate'
        n_neighbors: neighborhood size for UMAP/PHATE
        min_dist: UMAP min_dist
        time_alpha: when method='tphate', scales the epoch feature by this factor
        feature_names: optional list of column names for X
        n_components: 2 or 3
    """

    method = (method or 'umap').lower()

    if method == 'tphate':
        if not HAS_PHATE:
            print("PHATE not available, falling back to UMAP/t-SNE")
            method = 'umap'
        else:
            # Identify epoch column if present and scale it
            X_t = X
            if feature_names and 'epoch' in feature_names:
                idx = feature_names.index('epoch')
                scale = float(time_alpha) if time_alpha is not None else 3.0
                X_t = X.copy()
                X_t[:, idx] = X_t[:, idx] * scale
                print(f"Using T-PHATE (knn={max(2, n_neighbors)}, time_alpha={scale})")
            else:
                print("Using T-PHATE without explicit epoch feature; proceeding as PHATE")
                X_t = X
            reducer = phate.PHATE(n_components=n_components, knn=max(2, n_neighbors), random_state=42, verbose=False)
            return reducer.fit_transform(X_t)

    if method == 'phate':
        if HAS_PHATE:
            knn = max(2, n_neighbors)
            print(f"Using PHATE (knn={knn})")
            reducer = phate.PHATE(n_components=n_components, knn=knn, random_state=42, verbose=False)
            return reducer.fit_transform(X)
        else:
            print("PHATE not available, falling back to UMAP/t-SNE")
            method = 'umap'

    if method == 'umap':
        if HAS_UMAP:
            print(f"Using UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
            reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, n_components=n_components)
            return reducer.fit_transform(X)
        else:
            print("UMAP not available, falling back to t-SNE")
            method = 'tsne'

    if method == 'tsne':
        print("Using t-SNE (perplexity=30)")
    return TSNE(n_components=n_components, perplexity=30, random_state=42).fit_transform(X)

    raise ValueError(f"Unsupported embedding method '{method}'.")

def _pretty_method_name(method: str) -> str:
    mapping = {
        'umap': 'UMAP',
        'tsne': 't-SNE',
        'phate': 'PHATE',
        'tphate': 'T-PHATE',
    }
    return mapping.get((method or '').lower(), method.upper())


def plot_embedding_by_model(df, embedding, output_dir, method):
    """Plot embedding colored by model."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot each model separately
    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            mask = df['model'] == model_name

            if mask.sum() == 0:
                continue

            # Get model data
            x = embedding[mask, 0]
            y = embedding[mask, 1]
            epochs = df.loc[mask, 'epoch'].values

            # Color by channel, marker size by epoch
            color = CHANNEL_COLORS[channels]
            alpha = 0.3 if gru_size == 8 else (0.5 if gru_size == 32 else 0.7)
            marker = 'o' if gru_size == 8 else ('s' if gru_size == 32 else '^')

            # Plot trajectory
            ax.plot(x, y, color=color, alpha=0.3, linewidth=1, zorder=1)

            # Plot points
            scatter = ax.scatter(x, y, c=epochs, cmap='viridis',
                               s=(epochs / 100 * 200) + 20,  # Size grows with epoch
                               alpha=alpha, edgecolors=color,
                               linewidths=2, marker=marker,
                               label=f'{model_name}', zorder=2)

            # Annotate start and end
            ax.annotate(f'{model_name}\nE{int(epochs[0])}', (x[0], y[0]),
                       fontsize=7, alpha=0.7, ha='center')
            ax.annotate(f'E{int(epochs[-1])}', (x[-1], y[-1]),
                       fontsize=7, alpha=0.7, ha='center')

    # Add colorbar for epoch
    cbar = plt.colorbar(scatter, ax=ax, label='Epoch')

    method_name = _pretty_method_name(method)
    ax.set_xlabel(f'{method_name} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Dimension 2', fontsize=12)
    ax.set_title(f'Learning Trajectories in Metric Space ({method_name})\n' +
                 'Connect-4 as a Testbed: 9 Model Configurations',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='GRU8', alpha=0.7),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, label='GRU32', alpha=0.7),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, label='GRU128', alpha=0.7),
        Line2D([0], [0], color=CHANNEL_COLORS[16], linewidth=3, label='c16'),
        Line2D([0], [0], color=CHANNEL_COLORS[64], linewidth=3, label='c64'),
        Line2D([0], [0], color=CHANNEL_COLORS[256], linewidth=3, label='c256'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_dir) / 'trajectory_embedding_all.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_embedding_faceted(df, embedding, output_dir, method):
    """Plot embedding faceted by GRU size."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
    method_name = _pretty_method_name(method)

    for idx, gru_size in enumerate(GRU_SIZES):
        ax = axes[idx]

        for channels in CHANNELS:
            model_name = f'k3_c{channels}_gru{gru_size}'
            mask = df['model'] == model_name

            if mask.sum() == 0:
                continue

            x = embedding[mask, 0]
            y = embedding[mask, 1]
            epochs = df.loc[mask, 'epoch'].values

            color = CHANNEL_COLORS[channels]

            # Plot trajectory line
            ax.plot(x, y, color=color, alpha=0.5, linewidth=2, zorder=1)

            # Plot points
            scatter = ax.scatter(x, y, c=epochs, cmap='plasma',
                               s=(epochs / 100 * 150) + 30,
                               alpha=0.7, edgecolors=color,
                               linewidths=2, zorder=2)

            # Annotate endpoints
            ax.annotate(f'c{channels}', (x[-1], y[-1]),
                       fontsize=9, fontweight='bold',
                       color=color, ha='center')

        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=11)
        if idx == 0:
            ax.set_ylabel(f'{method_name} Dimension 2', fontsize=11)
        ax.set_title(f'GRU{gru_size}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=axes, label='Epoch', pad=0.01)

    plt.suptitle(f'Learning Trajectories by GRU Size ({method_name})\n' +
                 'Connect-4 as a Testbed: Channel Count Effects',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = Path(output_dir) / 'trajectory_embedding_by_gru.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def _plot_3d_matplotlib(df, embedding_3d, method):
    """Build a 3D matplotlib figure and axes with trajectory lines and points."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - side-effect import for 3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            mask = df['model'] == model_name

            if mask.sum() == 0:
                continue

            x = embedding_3d[mask, 0]
            y = embedding_3d[mask, 1]
            z = embedding_3d[mask, 2]
            epochs = df.loc[mask, 'epoch'].values

            color = CHANNEL_COLORS[channels]
            marker = 'o' if gru_size == 8 else ('s' if gru_size == 32 else '^')

            # Trajectory
            ax.plot(x, y, z, color=color, alpha=0.35, linewidth=1.5)
            # Points
            ax.scatter(x, y, z, c=epochs, cmap='viridis', s=50, alpha=0.7,
                       edgecolors=color, linewidths=1.0, marker=marker)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    method_name = _pretty_method_name(method)
    ax.set_title(
        f'3D Learning Trajectory Embedding ({method_name})\nConnect-4 as a Testbed',
        fontsize=13, fontweight='bold'
    )
    return fig, ax


def save_3d_static_png(df, embedding_3d, output_dir, method):
    fig, ax = _plot_3d_matplotlib(df, embedding_3d, method)
    output_path = Path(output_dir) / 'trajectory_embedding_3d.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_3d_rotation_gif(df, embedding_3d, output_dir, method, frames=180, seconds=12.0, elev=20.0):
    """Save a simple camera-orbit GIF around the 3D plot."""
    from matplotlib.animation import FuncAnimation

    fig, ax = _plot_3d_matplotlib(df, embedding_3d, method)
    fps = max(1, int(frames / max(0.5, float(seconds))))
    interval = int(1000 / fps)

    def update(i):
        azim = (360.0 * (i / float(frames))) % 360.0
        ax.view_init(elev=elev, azim=azim)
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    output_path = Path(output_dir) / 'trajectory_embedding_3d_rotate.gif'
    try:
        anim.save(output_path, dpi=120, writer='pillow', fps=fps)
        print(f"Saved: {output_path}")
    finally:
        plt.close(fig)


def export_plotly_3d_html(df, embedding_3d, output_html):
    """Export an interactive 3D HTML plot using Plotly (if available)."""
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print(f"Plotly not available ({e}); skipping interactive HTML export.")
        return

    fig = go.Figure()

    # Build traces per model
    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            mask = (df['model'] == model_name).values
            if not mask.any():
                continue
            x = embedding_3d[mask, 0]
            y = embedding_3d[mask, 1]
            z = embedding_3d[mask, 2]
            epochs = df.loc[mask, 'epoch'].values
            color = CHANNEL_COLORS[channels]

            # Lines
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=color, width=3),
                name=f"{model_name} path",
                showlegend=False,
                hoverinfo='skip',
                opacity=0.6,
            ))

            # Points with epoch colorscale
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=4, color=epochs, colorscale='Viridis', showscale=False,
                            line=dict(color='black', width=0.3)),
                name=model_name,
                hovertemplate=(
                    'Model: '+model_name+'<br>'+
                    'Epoch %{customdata}<br>'+
                    'x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>'
                ),
                customdata=epochs,
                opacity=0.9,
            ))

    fig.update_layout(
        title=dict(text='3D Learning Trajectory Embedding', x=0.5),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html), include_plotlyjs='cdn', full_html=True)
    print(f"Saved interactive HTML: {output_html}")

def analyze_clusters(df, embedding):
    """Analyze clustering in the embedding."""
    from sklearn.cluster import KMeans

    # Cluster trajectories
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embedding)

    df['cluster'] = clusters

    print("\n" + "="*80)
    print("TRAJECTORY CLUSTERING ANALYSIS")
    print("="*80)

    for cluster_id in range(n_clusters):
        cluster_models = df[df['cluster'] == cluster_id]['model'].unique()
        cluster_epochs = df[df['cluster'] == cluster_id]['epoch']

        print(f"\nCluster {cluster_id}:")
        print(f"  Models: {list(cluster_models)}")
        print(f"  Epoch range: {cluster_epochs.min():.0f} - {cluster_epochs.max():.0f}")
        print(f"  Size: {(df['cluster'] == cluster_id).sum()} checkpoints")

def main():
    parser = argparse.ArgumentParser(description='Embed learning trajectories in metric space')
    parser.add_argument('--metrics-dir', default='diagnostics/trajectory_analysis')
    parser.add_argument('--checkpoint-dir', default='checkpoints/save_every_3')
    parser.add_argument('--output-dir', default='visualizations')
    parser.add_argument('--method', choices=['umap', 'tsne', 'phate', 'tphate'], default='umap')
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--time-alpha', type=float, default=3.0, help='Time scaling for T-PHATE (epoch feature multiplier)')
    parser.add_argument('--dims', type=int, choices=[2, 3], default=2, help='Embedding dimensionality for main plots (default: 2)')
    parser.add_argument('--animate-3d', action='store_true', help='Create a rotating 3D GIF')
    parser.add_argument('--anim-frames', type=int, default=180, help='Number of frames in the 3D rotation GIF')
    parser.add_argument('--anim-seconds', type=float, default=12.0, help='Duration (seconds) of the 3D rotation GIF')
    parser.add_argument('--plotly-html', type=str, help='Optional path to write an interactive 3D HTML (Plotly)')
    args = parser.parse_args()

    print("="*80)
    print("Metric-Space Trajectory Embedding")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load all trajectory data
    print("\nLoading trajectory data...")
    df = load_all_trajectories(args.metrics_dir, args.checkpoint_dir)
    print(f"Loaded {len(df)} checkpoints from {df['model'].nunique()} models")

    # Create feature matrix
    print("\nCreating feature matrix...")
    X, feature_names = create_feature_matrix(df)
    print(f"Feature matrix shape: {X.shape}")

    # Embed trajectories
    print("\nEmbedding trajectories...")
    embedding = embed_trajectories(
        X,
        method=args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        time_alpha=args.time_alpha,
        feature_names=feature_names,
        n_components=2,
    )
    print(f"Embedding shape: {embedding.shape}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_embedding_by_model(df, embedding, output_dir, args.method)
    plot_embedding_faceted(df, embedding, output_dir, args.method)

    # 3D: compute on-demand and export static + optional animation/interactive
    if args.dims == 3 or args.animate_3d or args.plotly_html:
        embedding_3d = embed_trajectories(
            X,
            method=args.method,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            time_alpha=args.time_alpha,
            feature_names=feature_names,
            n_components=3,
        )
        print(f"3D embedding shape: {embedding_3d.shape}")
        save_3d_static_png(df, embedding_3d, output_dir, args.method)
        if args.animate_3d:
            save_3d_rotation_gif(
                df, embedding_3d, output_dir, args.method,
                frames=max(30, args.anim_frames),
                seconds=max(1.0, args.anim_seconds),
            )
        if args.plotly_html:
            export_plotly_3d_html(df, embedding_3d, args.plotly_html)

    # Analyze clusters
    analyze_clusters(df, embedding)

    print("\n" + "="*80)
    print("Embedding complete!")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == '__main__':
    main()

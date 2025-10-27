#!/usr/bin/env python3
"""
Metric-Space UMAP Embedding of Learning Trajectories.

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

# Try to import UMAP, fall back to t-SNE if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
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

def embed_trajectories(X, method='umap', n_neighbors=15, min_dist=0.1):
    """Embed trajectories using UMAP or t-SNE."""

    if method == 'umap' and HAS_UMAP:
        print(f"Using UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    else:
        if method == 'umap' and not HAS_UMAP:
            print("UMAP not available, falling back to t-SNE")
        print("Using t-SNE (perplexity=30)")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)

    embedding = reducer.fit_transform(X)
    return embedding

def plot_embedding_by_model(df, embedding, output_dir):
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

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('Learning Trajectories in Metric Space\n' +
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

def plot_embedding_faceted(df, embedding, output_dir):
    """Plot embedding faceted by GRU size."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

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

        ax.set_xlabel('UMAP Dimension 1', fontsize=11)
        if idx == 0:
            ax.set_ylabel('UMAP Dimension 2', fontsize=11)
        ax.set_title(f'GRU{gru_size}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=axes, label='Epoch', pad=0.01)

    plt.suptitle('Learning Trajectories by GRU Size\n' +
                 'Connect-4 as a Testbed: Channel Count Effects',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = Path(output_dir) / 'trajectory_embedding_by_gru.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_embedding_3d_animation(df, embedding, output_dir):
    """Create animated 3D embedding (optional, if matplotlib 3D available)."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from sklearn.decomposition import PCA

        # Create 3D embedding
        if HAS_UMAP:
            reducer = UMAP(n_components=3, n_neighbors=15, random_state=42)
        else:
            reducer = PCA(n_components=3)

        # Get feature matrix
        X, _ = create_feature_matrix(df)
        embedding_3d = reducer.fit_transform(X)

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

                # Plot trajectory
                ax.plot(x, y, z, color=color, alpha=0.3, linewidth=1.5)

                # Plot points
                ax.scatter(x, y, z, c=epochs, cmap='viridis',
                          s=50, alpha=0.6, edgecolors=color,
                          linewidths=1.5, marker=marker)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D Learning Trajectory Embedding\n' +
                    'Connect-4 as a Testbed',
                    fontsize=13, fontweight='bold')

        output_path = Path(output_dir) / 'trajectory_embedding_3d.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Could not create 3D embedding: {e}")

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
    parser.add_argument('--method', choices=['umap', 'tsne'], default='umap')
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
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
    embedding = embed_trajectories(X, method=args.method,
                                   n_neighbors=args.n_neighbors,
                                   min_dist=args.min_dist)
    print(f"Embedding shape: {embedding.shape}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_embedding_by_model(df, embedding, output_dir)
    plot_embedding_faceted(df, embedding, output_dir)
    plot_embedding_3d_animation(df, embedding, output_dir)

    # Analyze clusters
    analyze_clusters(df, embedding)

    print("\n" + "="*80)
    print("Embedding complete!")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print("="*80)

if __name__ == '__main__':
    main()

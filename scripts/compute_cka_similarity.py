#!/usr/bin/env python3
"""
CKA (Centered Kernel Alignment) Representation Similarity Analysis.

Compares learned representations across all 9 model configurations using CKA,
the industry-standard metric from Kornblith et al. (2019).

References:
    Kornblith et al. (2019): "Similarity of Neural Network Representations Revisited"
    https://arxiv.org/abs/1905.00414
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import sys
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Updated imports to match current project structure
from src.model import create_model
from src.play_game import Connect4Game

sns.set_style("whitegrid")

CHANNELS = [16, 64, 256]
GRU_SIZES = [8, 32, 128]
BOARD_SIZE = 64  # Number of test positions to use


def gram_linear(X):
    """
    Compute Gram matrix for linear kernel.

    Args:
        X: (n_samples, n_features) activation matrix

    Returns:
        (n_samples, n_samples) Gram matrix
    """
    return X @ X.T


def center_gram(K):
    """
    Center a Gram matrix.

    Args:
        K: (n, n) Gram matrix

    Returns:
        Centered Gram matrix
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka(X, Y):
    """
    Compute linear CKA between two sets of representations.

    Args:
        X: (n_samples, n_features_x) activations from model 1
        Y: (n_samples, n_features_y) activations from model 2

    Returns:
        CKA similarity score (0 to 1)
    """
    # Compute centered Gram matrices
    K = center_gram(gram_linear(X))
    L = center_gram(gram_linear(Y))

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = np.trace(K @ L)

    # Normalize
    normalization = np.sqrt(np.trace(K @ K) * np.trace(L @ L))

    if normalization == 0:
        return 0.0

    return hsic / normalization


def load_model_checkpoint(checkpoint_path, channels, gru_size, device='cpu', kernel_size=3):
    """Load a model from checkpoint (compatible with current training saves)."""
    model = create_model(cnn_channels=channels, gru_hidden_size=gru_size, kernel_size=kernel_size).to(device)

    # Training saves use key 'state_dict'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    return model


def extract_gru_representations(model, boards, device='cpu'):
    """
    Extract GRU hidden state representations from model.

    Args:
        model: AlphaZeroNet instance
        boards: (batch_size, 3, 6, 7) board states

    Returns:
        (batch_size, gru_hidden_size) GRU hidden states
    """
    with torch.no_grad():
        boards = boards.to(device)
        # Use model forward pass to obtain GRU hidden state
        _, _, hidden = model(boards)
        # hidden shape: (1, batch, gru_hidden)
        return hidden.squeeze(0).cpu().numpy()


def extract_cnn_representations(model, boards, device='cpu'):
    """
    Extract flattened CNN feature maps from the residual stack output.

    Args:
        model: ResNetGRUConnect4 instance
        boards: (batch_size, 3, 6, 7) board states

    Returns:
        (batch_size, cnn_channels * 6 * 7) feature matrix
    """
    with torch.no_grad():
        boards = boards.to(device)
        features = model.resnet(boards)
        return features.view(features.size(0), -1).cpu().numpy()


def generate_test_boards(game, num_boards=64, seed=42):
    """Generate random test board positions as (N, 3, 6, 7) numpy array.

    Channels: [yellow_pieces, red_pieces, current_turn]
    current_turn is 0 for player 1 (yellow), 1 for player 2 (red).
    """
    np.random.seed(seed)
    boards = []

    for _ in range(num_boards):
        game.reset()
        # Play random moves (between 0 and 20 moves)
        num_moves = np.random.randint(0, 21)
        for _ in range(num_moves):
            valid_moves = game.get_valid_moves()
            if len(valid_moves) == 0:
                break
            move = np.random.choice(valid_moves)
            game.make_move(move)

        # Convert current board to (3, 6, 7) state
        yellow = (game.board == 1).astype(np.float32)
        red = (game.board == 2).astype(np.float32)
        turn = np.full((6, 7), game.current_player - 1, dtype=np.float32)  # 0 or 1
        state = np.stack([yellow, red, turn], axis=0)
        boards.append(state)

    return np.stack(boards, axis=0)


def compute_similarity_matrix_for_epoch(checkpoint_dir, epoch, test_boards, device='cpu', representation_type='gru'):
    """
    Compute 9×9 CKA similarity matrix for a specific epoch.

    Args:
        checkpoint_dir: Base checkpoint directory
        epoch: Epoch number
        test_boards: Fixed set of test board positions

    Returns:
        (9, 9) similarity matrix, list of model names
    """
    model_names = []
    representations = []

    if representation_type == 'gru':
        extractor = extract_gru_representations
    elif representation_type == 'cnn':
        extractor = extract_cnn_representations
    else:
        raise ValueError(f"Unsupported representation_type '{representation_type}'")

    # Load all 9 models and extract representations
    for channels in CHANNELS:
        for gru_size in GRU_SIZES:
            model_name = f'k3_c{channels}_gru{gru_size}'
            # Checkpoints saved as weights_epoch_XXXX.pt under save_every_* directories
            checkpoint_path = Path(checkpoint_dir) / model_name / f'weights_epoch_{epoch:04d}.pt'

            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found: {checkpoint_path}")
                continue

            # Load model
            model = load_model_checkpoint(checkpoint_path, channels, gru_size, device, kernel_size=3)

            # Extract representations
            reprs = extractor(model, test_boards, device)

            model_names.append(model_name)
            representations.append(reprs)

            # Clean up
            del model
            torch.cuda.empty_cache()

    # Compute pairwise CKA
    n_models = len(model_names)
    if n_models == 0:
        return np.zeros((0, 0)), model_names

    similarity_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim = cka(representations[i], representations[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric

    return similarity_matrix, model_names


def plot_cka_heatmap(similarity_matrix, model_names, epoch, output_dir, rep_label):
    """Plot CKA similarity heatmap."""
    if len(model_names) == 0:
        print(f"No models for epoch {epoch}; skipping heatmap.")
        return
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha='center', va='center',
                          color='black' if similarity_matrix[i, j] > 0.5 else 'white',
                          fontsize=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='CKA Similarity')

    ax.set_title(f'CKA ({rep_label.upper()}) Representation Similarity - Epoch {epoch}\n' +
                 'Connect-4 as a Testbed: Cross-Architecture Comparison',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(output_dir) / f'cka_{rep_label}_similarity_epoch_{epoch}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cka_evolution(similarity_matrices, epochs, model_names, output_dir, rep_label):
    """
    Plot evolution of CKA similarity over training.
    Shows how representation similarity changes for key model pairs.
    """
    # Guard: need at least 2 models to plot evolution
    if not model_names or len(model_names) < 2:
        print("Not enough models to plot evolution; skipping evolution plot.")
        return

    # Select interesting model pairs to track
    pairs = [
        ('k3_c16_gru8', 'k3_c256_gru8'),    # Same GRU, different channels
        ('k3_c64_gru8', 'k3_c64_gru128'),   # Same channels, different GRU
        ('k3_c64_gru32', 'k3_c64_gru128'),  # Best vs overfit
        ('k3_c16_gru128', 'k3_c256_gru128'), # Same GRU, extreme channels
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (model1, model2) in enumerate(pairs):
        ax = axes[idx]

        if model1 not in model_names or model2 not in model_names:
            continue

        idx1 = model_names.index(model1)
        idx2 = model_names.index(model2)

        # Extract similarity over epochs
        similarities = [sim_matrix[idx1, idx2] for sim_matrix in similarity_matrices]

        ax.plot(epochs, similarities, marker='o', linewidth=2, markersize=8,
               color='#2E86AB', alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('CKA Similarity', fontsize=11)
        ax.set_title(f'{model1} vs {model2}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Add reference line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate similarity')
        ax.legend(fontsize=9)

    plt.suptitle(f'CKA ({rep_label.upper()}) Similarity Evolution Over Training\n' +
                 'Connect-4 as a Testbed: Representation Convergence',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(output_dir) / f'cka_{rep_label}_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def animate_cka_heatmaps(similarity_matrices: List[np.ndarray], model_names: List[str], epochs: List[int], output_dir: Path, rep_label: str, fps: int = 3, fmt: str = 'gif'):
    """
    Create an animated heatmap of CKA similarity across epochs.
    Saves as GIF by default, or MP4 if requested and supported.
    """
    if not similarity_matrices or not model_names or len(model_names) < 2:
        print("Not enough data to animate heatmaps; skipping.")
        return

    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrices[0], cmap='RdYlGn', vmin=0, vmax=1, animated=True)

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, label='CKA Similarity')

    title = ax.set_title(f'CKA ({rep_label.upper()}) Representation Similarity - Epoch {epochs[0]}', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()

    def init():
        im.set_data(similarity_matrices[0])
        title.set_text(f'CKA ({rep_label.upper()}) Representation Similarity - Epoch {epochs[0]}')
        return [im, title]

    def update(frame_idx):
        im.set_data(similarity_matrices[frame_idx])
        title.set_text(f'CKA ({rep_label.upper()}) Representation Similarity - Epoch {epochs[frame_idx]}')
        return [im, title]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(similarity_matrices), interval=1000/max(fps,1), blit=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == 'mp4':
        try:
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps, bitrate=1800)
            output_path = output_dir / f'cka_{rep_label}_heatmap_animation.mp4'
            anim.save(str(output_path), writer=writer)
            print(f"Saved: {output_path}")
            plt.close(fig)
            return
        except Exception as e:
            print(f"FFMpeg not available or failed ({e}); falling back to GIF.")

    try:
        Writer = animation.PillowWriter
        writer = Writer(fps=fps)
        output_path = output_dir / f'cka_{rep_label}_heatmap_animation.gif'
        anim.save(str(output_path), writer=writer)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Pillow writer not available; skipping animation. ({e})")
    finally:
        plt.close(fig)


def plot_cka_clustered(similarity_matrix, model_names, epoch, output_dir, rep_label):
    """
    Plot CKA heatmap with hierarchical clustering.
    Groups similar models together. If SciPy is unavailable, skip gracefully.
    """
    if len(model_names) < 2:
        print(f"Not enough models for clustered plot at epoch {epoch}; skipping.")
        return

    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
    except Exception:
        print("SciPy not installed; skipping clustered CKA plot.")
        return

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Create figure with dendrogram
    fig = plt.figure(figsize=(14, 12))

    # Dendrogram on top
    ax_dendro = fig.add_axes([0.15, 0.71, 0.65, 0.2])
    dendro = dendrogram(linkage_matrix, labels=model_names, ax=ax_dendro,
                       color_threshold=0.5, above_threshold_color='gray')
    ax_dendro.set_title(f'{rep_label.upper()} Representation Similarity Clustering - Epoch {epoch}',
                       fontsize=13, fontweight='bold')
    ax_dendro.axis('off')

    # Reorder similarity matrix by dendrogram
    order = dendro['leaves']
    ordered_names = [model_names[i] for i in order]
    ordered_matrix = similarity_matrix[order, :][:, order]

    # Heatmap
    ax_heatmap = fig.add_axes([0.15, 0.1, 0.65, 0.6])
    im = ax_heatmap.imshow(ordered_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax_heatmap.set_xticks(np.arange(len(ordered_names)))
    ax_heatmap.set_yticks(np.arange(len(ordered_names)))
    ax_heatmap.set_xticklabels(ordered_names, rotation=45, ha='right', fontsize=9)
    ax_heatmap.set_yticklabels(ordered_names, fontsize=9)

    # Colorbar
    cbar_ax = fig.add_axes([0.82, 0.1, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, label='CKA Similarity')

    output_path = Path(output_dir) / f'cka_{rep_label}_clustered_epoch_{epoch}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def analyze_similarity_patterns(similarity_matrix, model_names, epoch, rep_label):
    """Analyze and print insights from similarity matrix."""
    print(f"\n{'='*80}")
    print(f"CKA ({rep_label.upper()}) SIMILARITY ANALYSIS - EPOCH {epoch}")
    print(f"{'='*80}")

    # Find most similar pairs (excluding self-similarity)
    most_similar_pairs = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            most_similar_pairs.append((similarity_matrix[i, j], model_names[i], model_names[j]))
    most_similar_pairs.sort(reverse=True)

    print(f"\nTop 5 Most Similar Model Pairs:")
    for sim, m1, m2 in most_similar_pairs[:5]:
        print(f"  {m1} ↔ {m2}: CKA = {sim:.3f}")

    # Find least similar pairs
    print(f"\nTop 5 Least Similar Model Pairs:")
    for sim, m1, m2 in most_similar_pairs[-5:]:
        print(f"  {m1} ↔ {m2}: CKA = {sim:.3f}")

    # Analyze by GRU size groups
    print(f"\nSimilarity Within GRU Size Groups:")
    for gru_size in GRU_SIZES:
        models_in_group = [m for m in model_names if f'gru{gru_size}' in m]
        if len(models_in_group) < 2:
            continue

        indices = [model_names.index(m) for m in models_in_group]
        within_group_sims = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_group_sims.append(similarity_matrix[indices[i], indices[j]])

        avg_sim = np.mean(within_group_sims)
        print(f"  GRU{gru_size}: Average CKA = {avg_sim:.3f} (across {len(models_in_group)} models)")

    # Analyze by channel count groups
    print(f"\nSimilarity Within Channel Count Groups:")
    for channels in CHANNELS:
        models_in_group = [m for m in model_names if f'c{channels}_' in m]
        if len(models_in_group) < 2:
            continue

        indices = [model_names.index(m) for m in models_in_group]
        within_group_sims = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_group_sims.append(similarity_matrix[indices[i], indices[j]])

        avg_sim = np.mean(within_group_sims)
        print(f"  c{channels}: Average CKA = {avg_sim:.3f} (across {len(models_in_group)} models)")


def main():
    parser = argparse.ArgumentParser(description='CKA representation similarity analysis')
    parser.add_argument('--checkpoint-dir', default='checkpoints/save_every_3',
                       help='Base checkpoint directory')
    parser.add_argument('--epochs', nargs='+', type=int, default=[3, 10, 30, 60, 100],
                       help='Epochs to analyze')
    parser.add_argument('--epoch-step', type=int, default=None,
                       help='If provided, overrides --epochs with [3, 3+step, ..., 99] and appends 100')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--num-boards', type=int, default=64,
                       help='Number of test board positions')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for test board generation')
    parser.add_argument('--device', default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--animate', action='store_true',
                        help='Generate animated heatmap across epochs')
    parser.add_argument('--animate-fps', type=int, default=3,
                        help='Frames per second for heatmap animation (default: 3)')
    parser.add_argument('--animate-format', choices=['gif','mp4'], default='gif',
                        help='Animation format to save (default: gif)')
    parser.add_argument('--representation', choices=['gru', 'cnn'], default='gru',
                        help='Which representation to compare (default: gru hidden state)')
    args = parser.parse_args()

    # Expand epochs if epoch-step is provided
    if args.epoch_step is not None:
        step = max(1, args.epoch_step)
        generated = list(range(3, 100, step))
        if 100 not in generated:
            generated.append(100)
        args.epochs = generated

    print("="*80)
    print(f"CKA Representation Similarity Analysis ({args.representation.upper()})")
    print("="*80)
    print(f"\nAnalyzing epochs: {args.epochs}")
    print(f"Using {args.num_boards} test board positions")

    # Create output directory
    output_dir = Path(args.output_dir) / args.representation
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate fixed test set
    print("\nGenerating test board positions...")
    game = Connect4Game()
    test_boards = generate_test_boards(game, num_boards=args.num_boards, seed=args.seed)
    test_boards_tensor = torch.FloatTensor(test_boards)
    print(f"Test set shape: {test_boards_tensor.shape}")

    # Compute similarity matrices for each epoch
    similarity_matrices = []
    processed_epochs = []
    all_model_names = None

    for epoch in tqdm(args.epochs, desc="Computing CKA similarities"):
        similarity_matrix, model_names = compute_similarity_matrix_for_epoch(
            args.checkpoint_dir, epoch, test_boards_tensor, args.device, args.representation
        )

        # Skip epochs with no models
        if len(model_names) == 0:
            print(f"No checkpoints found for epoch {epoch}; skipping.")
            continue

        if all_model_names is None:
            all_model_names = model_names

        similarity_matrices.append(similarity_matrix)
        processed_epochs.append(epoch)

        # Generate visualizations for this epoch
        plot_cka_heatmap(similarity_matrix, model_names, epoch, output_dir, args.representation)
        plot_cka_clustered(similarity_matrix, model_names, epoch, output_dir, args.representation)

        # Print analysis (only if at least 2 models)
        if len(model_names) >= 2:
            analyze_similarity_patterns(similarity_matrix, model_names, epoch, args.representation)

    # Generate evolution plot
    if similarity_matrices:
        print("\nGenerating evolution plot...")
        plot_cka_evolution(similarity_matrices, processed_epochs, all_model_names, output_dir, args.representation)
    else:
        print("\nNo epochs processed; skipping evolution plot.")

    # Save similarity matrices to CSV
    print("\nSaving similarity matrices...")
    for epoch, sim_matrix in zip(processed_epochs, similarity_matrices):
        df = pd.DataFrame(sim_matrix, columns=all_model_names, index=all_model_names)
        csv_path = output_dir / f'cka_{args.representation}_similarity_epoch_{epoch}.csv'
        df.to_csv(csv_path)
        print(f"Saved: {csv_path}")

    print("\n" + "="*80)
    print("CKA analysis complete!")
    print(f"Visualizations saved to: {output_dir.absolute()}")
    print("="*80)

    # Optional animation of heatmaps
    if args.animate and similarity_matrices:
        print("\nGenerating animated heatmap...")
        try:
            animate_cka_heatmaps(similarity_matrices, all_model_names, processed_epochs, output_dir, args.representation, fps=args.animate_fps, fmt=args.animate_format)
        except Exception as e:
            print(f"Failed to generate animation: {e}")


if __name__ == '__main__':
    main()

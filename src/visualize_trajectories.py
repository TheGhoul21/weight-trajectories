"""
Visualize weight trajectories and representations using PHATE

Provides visualizations for:
1. CNN weight evolution across training epochs
2. GRU weight evolution across training epochs
3. Board state representations (how model sees positions)
4. Game trajectory evolution (how representations change during a game)
"""

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import phate

from model import create_model


class TrajectoryVisualizer:
    """Visualize weight and representation trajectories using PHATE"""

    def __init__(self, checkpoint_dir, device='cpu'):
        """
        Args:
            checkpoint_dir: Directory containing saved checkpoints
            device: torch device
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        # Load training history
        history_path = self.checkpoint_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = None

        # Extract model config from directory name
        # Format: k{kernel}_c{channels}_gru{gru}_{timestamp}
        import re
        match = re.search(r'k(\d+)_c(\d+)_gru(\d+)', str(checkpoint_dir))
        if match:
            self.kernel_size = int(match.group(1))
            self.cnn_channels = int(match.group(2))
            self.gru_hidden = int(match.group(3))
        else:
            raise ValueError("Cannot parse model config from checkpoint directory name")

        print(f"Model config: k={self.kernel_size}, c={self.cnn_channels}, gru={self.gru_hidden}")

    def load_checkpoints(self):
        """Load all saved checkpoints"""
        checkpoint_files = sorted(self.checkpoint_dir.glob("weights_epoch_*.pt"))

        if len(checkpoint_files) == 0:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")

        print(f"Found {len(checkpoint_files)} checkpoints")

        checkpoints = []
        for cp_file in checkpoint_files:
            cp = torch.load(cp_file, map_location=self.device, weights_only=False)
            checkpoints.append(cp)

        return checkpoints

    def extract_cnn_weights(self, checkpoints):
        """
        Extract CNN weights from all checkpoints
        Returns: (n_epochs, n_weights) array
        """
        all_weights = []

        for cp in checkpoints:
            state_dict = cp['state_dict']

            # Extract all ResNet convolutional weights (Conv2d layers)
            cnn_weights = []
            for key, value in state_dict.items():
                # Match resnet layers that are Conv2d (exclude BatchNorm)
                if 'resnet' in key and 'weight' in key:
                    # Conv2d weights have 4 dimensions, BatchNorm has 1
                    if len(value.shape) == 4:
                        cnn_weights.append(value.cpu().numpy().flatten())

            # Concatenate all CNN weights
            if len(cnn_weights) > 0:
                weights_flat = np.concatenate(cnn_weights)
                all_weights.append(weights_flat)
            else:
                raise ValueError("No CNN weights found in checkpoint")

        return np.array(all_weights)

    def extract_gru_weights(self, checkpoints):
        """
        Extract GRU weights from all checkpoints
        Returns: (n_epochs, n_weights) array
        """
        all_weights = []

        for cp in checkpoints:
            state_dict = cp['state_dict']

            # Extract all GRU weights
            gru_weights = []
            for key, value in state_dict.items():
                if 'gru' in key and 'weight' in key:
                    gru_weights.append(value.cpu().numpy().flatten())

            weights_flat = np.concatenate(gru_weights)
            all_weights.append(weights_flat)

        return np.array(all_weights)

    def visualize_weight_trajectory(self, weights, title, save_path=None):
        """
        Visualize weight trajectory using PHATE

        Args:
            weights: (n_epochs, n_weights) array
            title: Plot title
            save_path: Optional path to save figure
        """
        print(f"\nComputing PHATE embedding for {title}...")
        print(f"  Input shape: {weights.shape}")

        # Apply PHATE with adaptive knn for small datasets
        n_samples = weights.shape[0]
        knn = min(3, max(2, n_samples - 2))  # Ensure knn is valid

        phate_op = phate.PHATE(n_components=2, knn=knn, t=10, verbose=False)
        weights_phate = phate_op.fit_transform(weights)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory
        ax.plot(weights_phate[:, 0], weights_phate[:, 1],
                'o-', alpha=0.6, linewidth=2, markersize=8)

        # Mark start and end
        ax.scatter(weights_phate[0, 0], weights_phate[0, 1],
                  c='green', s=200, marker='o', edgecolors='black',
                  linewidths=2, label='Start', zorder=10)
        ax.scatter(weights_phate[-1, 0], weights_phate[-1, 1],
                  c='red', s=200, marker='*', edgecolors='black',
                  linewidths=2, label='End', zorder=10)

        # Add epoch labels
        for i, (x, y) in enumerate(weights_phate):
            if i % max(1, len(weights_phate) // 10) == 0:  # Label every ~10%
                ax.annotate(f'E{i+1}', (x, y), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('PHATE 1', fontsize=12)
        ax.set_ylabel('PHATE 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to {save_path}")

        return fig, weights_phate

    def visualize_cnn_trajectory(self, save_path=None):
        """Visualize CNN weight evolution"""
        checkpoints = self.load_checkpoints()
        cnn_weights = self.extract_cnn_weights(checkpoints)

        epochs = [cp['epoch'] for cp in checkpoints]
        title = f"CNN Weight Trajectory (k={self.kernel_size}, c={self.cnn_channels})"

        return self.visualize_weight_trajectory(cnn_weights, title, save_path)

    def visualize_gru_trajectory(self, save_path=None):
        """Visualize GRU weight evolution"""
        checkpoints = self.load_checkpoints()
        gru_weights = self.extract_gru_weights(checkpoints)

        title = f"GRU Weight Trajectory (hidden={self.gru_hidden})"

        return self.visualize_weight_trajectory(gru_weights, title, save_path)

    def visualize_board_representations(self, board_states, checkpoint_path, save_path=None):
        """
        Visualize how the model represents different board states

        Args:
            board_states: List of board state tensors (N, 3, 6, 7)
            checkpoint_path: Path to checkpoint to use
            save_path: Optional save path
        """
        print("\nVisualizing board state representations...")

        # Load model
        model = create_model(self.cnn_channels, self.gru_hidden, self.kernel_size)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()

        # Extract representations (GRU hidden states)
        representations = []
        with torch.no_grad():
            for state in board_states:
                state = state.to(self.device)
                _, _, hidden = model(state)
                representations.append(hidden.squeeze().cpu().numpy())

        representations = np.array(representations)
        print(f"  Representation shape: {representations.shape}")

        # Apply PHATE
        phate_op = phate.PHATE(n_components=2, knn=min(5, len(representations)-1),
                              t=10, verbose=False)
        repr_phate = phate_op.fit_transform(representations)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(repr_phate[:, 0], repr_phate[:, 1],
                           c=np.arange(len(repr_phate)), cmap='viridis',
                           s=100, alpha=0.7, edgecolors='black', linewidths=1)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Position Index', fontsize=10)

        ax.set_xlabel('PHATE 1', fontsize=12)
        ax.set_ylabel('PHATE 2', fontsize=12)
        ax.set_title('Board State Representations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to {save_path}")

        return fig, repr_phate

    def visualize_game_trajectory(self, game_states, checkpoint_path, save_path=None):
        """
        Visualize how board representations evolve during a game

        Args:
            game_states: List of board states from a single game (chronological order)
            checkpoint_path: Path to checkpoint to use
            save_path: Optional save path
        """
        print("\nVisualizing game trajectory...")

        # Load model
        model = create_model(self.cnn_channels, self.gru_hidden, self.kernel_size)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()

        # Extract representations
        representations = []
        with torch.no_grad():
            for state in game_states:
                state = state.to(self.device)
                _, _, hidden = model(state)
                representations.append(hidden.squeeze().cpu().numpy())

        representations = np.array(representations)
        print(f"  Game length: {len(representations)} moves")

        # Apply PHATE
        phate_op = phate.PHATE(n_components=2, knn=min(5, len(representations)-1),
                              t=10, verbose=False)
        repr_phate = phate_op.fit_transform(representations)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory
        ax.plot(repr_phate[:, 0], repr_phate[:, 1],
               'o-', alpha=0.6, linewidth=2, markersize=8, color='blue')

        # Mark start and end
        ax.scatter(repr_phate[0, 0], repr_phate[0, 1],
                  c='green', s=300, marker='o', edgecolors='black',
                  linewidths=2, label='Start', zorder=10)
        ax.scatter(repr_phate[-1, 0], repr_phate[-1, 1],
                  c='red', s=300, marker='*', edgecolors='black',
                  linewidths=2, label='End', zorder=10)

        # Add move numbers
        for i, (x, y) in enumerate(repr_phate):
            if i % max(1, len(repr_phate) // 10) == 0:
                ax.annotate(f'{i}', (x, y), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('PHATE 1', fontsize=12)
        ax.set_ylabel('PHATE 2', fontsize=12)
        ax.set_title('Game Trajectory in Representation Space', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to {save_path}")

        return fig, repr_phate

    def create_summary_plot(self):
        """Create a 2x2 summary plot with all visualizations"""
        print("\nCreating summary visualization...")

        checkpoints = self.load_checkpoints()
        cnn_weights = self.extract_cnn_weights(checkpoints)
        gru_weights = self.extract_gru_weights(checkpoints)

        # Compute PHATE embeddings with adaptive knn
        n_samples = len(checkpoints)
        knn = min(3, max(2, n_samples - 2))

        print("  Computing CNN PHATE...")
        phate_op_cnn = phate.PHATE(n_components=2, knn=knn, t=10, verbose=False)
        cnn_phate = phate_op_cnn.fit_transform(cnn_weights)

        print("  Computing GRU PHATE...")
        phate_op_gru = phate.PHATE(n_components=2, knn=knn, t=10, verbose=False)
        gru_phate = phate_op_gru.fit_transform(gru_weights)

        # Create 2x2 plot
        fig = plt.figure(figsize=(16, 12))

        # CNN trajectory
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(cnn_phate[:, 0], cnn_phate[:, 1], 'o-', alpha=0.6, linewidth=2, markersize=6)
        ax1.scatter(cnn_phate[0, 0], cnn_phate[0, 1], c='green', s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)
        ax1.scatter(cnn_phate[-1, 0], cnn_phate[-1, 1], c='red', s=150, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)
        ax1.set_xlabel('PHATE 1')
        ax1.set_ylabel('PHATE 2')
        ax1.set_title(f'CNN Trajectory (k={self.kernel_size}, c={self.cnn_channels})', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # GRU trajectory
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(gru_phate[:, 0], gru_phate[:, 1], 'o-', alpha=0.6, linewidth=2, markersize=6, color='orange')
        ax2.scatter(gru_phate[0, 0], gru_phate[0, 1], c='green', s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10)
        ax2.scatter(gru_phate[-1, 0], gru_phate[-1, 1], c='red', s=150, marker='*',
                   edgecolors='black', linewidths=2, zorder=10)
        ax2.set_xlabel('PHATE 1')
        ax2.set_ylabel('PHATE 2')
        ax2.set_title(f'GRU Trajectory (hidden={self.gru_hidden})', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Loss curves
        if self.history:
            ax3 = plt.subplot(2, 2, 3)
            epochs = range(1, len(self.history['train_loss']) + 1)
            ax3.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
            ax3.plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Progress', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Component losses
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(epochs, self.history['val_policy_loss'], label='Policy Loss', linewidth=2)
            ax4.plot(epochs, self.history['val_value_loss'], label='Value Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Validation Component Losses', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Weight Trajectory Summary: k={self.kernel_size}, c={self.cnn_channels}, gru={self.gru_hidden}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig


def generate_random_boards(n_boards=100):
    """Generate random Connect-4 board states for visualization"""
    boards = []
    for _ in range(n_boards):
        # Random board with 0-20 pieces
        n_pieces = np.random.randint(0, 21)
        board = np.zeros((3, 6, 7), dtype=np.float32)

        for _ in range(n_pieces):
            player = np.random.randint(0, 2)  # 0=yellow, 1=red
            col = np.random.randint(0, 7)

            # Find lowest empty row
            for row in range(5, -1, -1):
                if board[0, row, col] == 0 and board[1, row, col] == 0:
                    board[player, row, col] = 1
                    break

        # Add turn plane
        board[2, :, :] = np.random.randint(0, 2)

        boards.append(torch.from_numpy(board).unsqueeze(0))

    return boards


def main():
    parser = argparse.ArgumentParser(description="Visualize weight trajectories using PHATE")

    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory containing checkpoints")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--viz-type", type=str, default="all",
                       choices=["all", "cnn", "gru", "boards", "summary"],
                       help="Type of visualization to create")
    parser.add_argument("--n-boards", type=int, default=100,
                       help="Number of random boards for representation viz")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz = TrajectoryVisualizer(args.checkpoint_dir, device)

    # Generate visualizations
    if args.viz_type in ["all", "cnn"]:
        print("\n" + "="*60)
        print("CNN Weight Trajectory")
        print("="*60)
        viz.visualize_cnn_trajectory(save_path=output_dir / "cnn_trajectory.png")
        plt.close()

    if args.viz_type in ["all", "gru"]:
        print("\n" + "="*60)
        print("GRU Weight Trajectory")
        print("="*60)
        viz.visualize_gru_trajectory(save_path=output_dir / "gru_trajectory.png")
        plt.close()

    if args.viz_type in ["all", "boards"]:
        print("\n" + "="*60)
        print("Board State Representations")
        print("="*60)
        boards = generate_random_boards(args.n_boards)
        checkpoint_path = sorted(viz.checkpoint_dir.glob("weights_epoch_*.pt"))[-1]
        viz.visualize_board_representations(boards, checkpoint_path,
                                           save_path=output_dir / "board_representations.png")
        plt.close()

    if args.viz_type in ["all", "summary"]:
        print("\n" + "="*60)
        print("Summary Visualization")
        print("="*60)
        fig = viz.create_summary_plot()
        fig.savefig(output_dir / "summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("\n" + "="*60)
    print(f"✓ Visualizations saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

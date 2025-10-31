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
import re

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import phate

try:
    from .model import create_model
except ImportError:  # Fallback for direct script execution
    from model import create_model


class TrajectoryVisualizer:
    """Visualize weight and representation trajectories using PHATE"""

    def __init__(self, checkpoint_dir, device='cpu'):
        """Initialize helper for a specific checkpoint directory."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        # Load training history if available
        history_path = self.checkpoint_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = None

        # Extract model config from directory name
        # Expected format: k{kernel}_c{channels}_gru{gru}_<timestamp>
        match = re.search(r'k(\d+)_c(\d+)_gru(\d+)', str(self.checkpoint_dir))
        if match:
            self.kernel_size = int(match.group(1))
            self.cnn_channels = int(match.group(2))
            self.gru_hidden = int(match.group(3))
        else:
            raise ValueError("Cannot parse model config from checkpoint directory name")

        print(f"Model config: k={self.kernel_size}, c={self.cnn_channels}, gru={self.gru_hidden}")

    @staticmethod
    def _closest_epoch_index(epochs, target_epoch):
        """Return index of epoch closest to target."""
        if not epochs:
            raise ValueError("Epoch list is empty")
        diffs = [abs(epoch - target_epoch) for epoch in epochs]
        return int(np.argmin(diffs))

    def _collect_validation_markers(self, epochs):
        """Return indices for minimal and final validation loss if history is available."""
        if not self.history or 'val_loss' not in self.history:
            return {}

        val_losses = self.history['val_loss']
        if not val_losses:
            return {}

        markers = {}
        # Minimal validation loss epoch
        min_epoch = int(np.argmin(val_losses)) + 1
        min_idx = self._closest_epoch_index(epochs, min_epoch)
        markers['min'] = {
            'epoch': min_epoch,
            'loss': float(val_losses[min_epoch - 1]),
            'index': min_idx,
        }

        # Final epoch in history (may not be saved)
        final_epoch = int(len(val_losses))
        final_idx = self._closest_epoch_index(epochs, final_epoch)
        markers['final'] = {
            'epoch': final_epoch,
            'loss': float(val_losses[final_epoch - 1]),
            'index': final_idx,
        }
        return markers

    def _gather_checkpoint_files(self):
        """Collect (epoch, path) pairs sorted by epoch from the checkpoint directory."""
        pairs = []
        for path in self.checkpoint_dir.glob("weights_epoch_*.pt"):
            name = path.stem
            try:
                epoch = int(name.replace("weights_epoch_", ""))
            except ValueError:
                continue
            pairs.append((epoch, path))
        pairs.sort(key=lambda item: item[0])
        return pairs

    def load_checkpoints(self):
        """Load all checkpoints as a list of dicts with epoch, path, and state_dict."""
        pairs = self._gather_checkpoint_files()
        if not pairs:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        out = []
        for epoch, path in pairs:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            out.append({
                'epoch': int(checkpoint.get('epoch', epoch)),
                'path': path,
                'state_dict': state_dict,
            })
        return out

    def get_latest_checkpoint_path(self):
        """Return the path to the most recent checkpoint, sorted by epoch."""
        return self._gather_checkpoint_files()[-1][1]

    def _filter_checkpoints(self, checkpoints, epoch_min=None, epoch_max=None, epoch_stride=1):
        """Filter checkpoints by epoch range and stride."""
        stride = max(1, int(epoch_stride or 1))
        pairs = []
        for idx, cp in enumerate(checkpoints):
            epoch_val = cp.get('epoch', idx + 1)
            if epoch_min is not None and epoch_val < epoch_min:
                continue
            if epoch_max is not None and epoch_val > epoch_max:
                continue
            pairs.append((cp, epoch_val))

        if not pairs:
            raise ValueError("No checkpoints remain after applying epoch filters")

        if stride > 1:
            pairs = pairs[::stride]

        filtered_checkpoints = [item[0] for item in pairs]
        epochs = [item[1] for item in pairs]
        return filtered_checkpoints, epochs

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

    def _select_knn(self, n_samples, max_knn=5):
        """Select a PHATE knn parameter that works for small sample counts."""
        if n_samples < 2:
            raise ValueError("Need at least two checkpoints to compute a trajectory embedding")
        return max(2, min(max_knn, n_samples - 1))

    @staticmethod
    def _resolve_phate_n_pca(n_features, n_samples, requested=None):
        """Resolve a safe PCA dimension before PHATE given feature/sample counts."""
        if n_features < 2 or n_samples < 2:
            return None, None

        max_valid = min(n_features - 1, n_samples - 1)
        if max_valid < 1:
            return None, None

        if requested is not None:
            if requested <= 0:
                raise ValueError("PHATE n_pca must be a positive integer")
            resolved = min(requested, max_valid)
            message = None
            if requested > max_valid:
                message = ("Requested n_pca exceeds what the data supports; "
                           f"using {resolved} instead of {requested}.")
            return resolved, message

        if n_features >= 5000:
            resolved = min(32, max_valid)
            message = (f"Auto-selecting n_pca={resolved} for high-dimensional weights "
                       f"({n_features} features, {n_samples} checkpoints).")
            return resolved, message

        return None, None

    @staticmethod
    def _resolve_phate_knn(n_samples, default_max, requested=None):
        """Resolve a PHATE kNN value, respecting sample count and optional override."""
        if n_samples < 2:
            raise ValueError("Need at least two checkpoints to compute a trajectory embedding")

        auto_knn = max(2, min(default_max, n_samples - 1))
        if requested is None:
            return auto_knn, None

        if requested < 2:
            raise ValueError("PHATE knn must be at least 2")

        resolved = min(requested, n_samples - 1)
        message = None
        if resolved != requested:
            message = ("Requested knn exceeds available checkpoints; "
                       f"using {resolved} instead of {requested}.")
        return resolved, message

    @staticmethod
    def _delay_embed(series: np.ndarray, tau: int, lags: int) -> np.ndarray:
        """Delay-embed a sequence with window defined by tau and lags.

        Args:
            series: (n_samples, n_features)
            tau: step between delays (>=1)
            lags: number of past lags to include (>=1 means include x_{t}, x_{t-τ}, ..., x_{t-l*τ})

        Returns:
            (n_samples, n_features * (lags+1)) delay-embedded matrix with edge replication.
        """
        n, d = series.shape
        tau = max(1, int(tau))
        lags = max(0, int(lags))
        if lags == 0:
            return series.astype(np.float32, copy=False)
        out = np.empty((n, d * (lags + 1)), dtype=np.float32)
        for t in range(n):
            cols = []
            for k in range(0, lags + 1):
                idx = max(0, t - k * tau)
                cols.append(series[idx])
            out[t] = np.concatenate(cols, axis=0)
        return out

    @staticmethod
    def _temporal_blended_distance(X: np.ndarray, group_ids: np.ndarray, alpha: float, tau: float) -> np.ndarray:
        """Compute a blended distance matrix combining feature and temporal distances.

        D_eff(i,j)^2 = ||x_i - x_j||^2 + alpha * (|t_i - t_j| / tau)^2 for i,j in same group;
                        large penalty if groups differ.

        Args:
            X: (n_samples, n_features) feature matrix (after optional PCA/delay).
            group_ids: (n_samples,) int labels mapping each row to a sequence/group.
            alpha: weight on temporal term.
            tau: temporal scale (in steps).

        Returns:
            (n_samples, n_samples) symmetric distance matrix.
        """
        X = X.astype(np.float32, copy=False)
        n = X.shape[0]
        # Pairwise squared Euclidean via Gram trick
        norms = np.sum(X * X, axis=1, keepdims=True)
        Dx2 = norms + norms.T - 2.0 * (X @ X.T)
        Dx2 = np.maximum(Dx2, 0.0)

        # Temporal distance per group
        idx = np.arange(n, dtype=np.int32)
        T = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
        same = (group_ids[:, None] == group_ids[None, :]).astype(np.float32)
        # If different groups, use a large temporal separation to null affinity
        large = 1e6
        T_scaled2 = np.where(same > 0, (T / max(1e-6, float(tau))) ** 2, large)

        Deff2 = Dx2 + float(alpha) * T_scaled2
        Deff = np.sqrt(np.maximum(Deff2, 0.0)).astype(np.float32)
        # Zero diagonal
        np.fill_diagonal(Deff, 0.0)
        return Deff

    def visualize_weight_trajectory(self, weights, title, save_path=None, epochs=None,
                                     phate_n_pca=None, phate_knn=None, phate_t=None,
                                     phate_decay=None, use_tphate=False, tphate_alpha=1.0,
                                     tphate_mode='time-feature', tphate_delay=None, tphate_lags=None,
                                     tphate_kernel=False, tphate_kernel_alpha=1.0, tphate_kernel_tau=3.0):
        """
        Visualize weight trajectory using PHATE

        Args:
            weights: (n_epochs, n_weights) array
            title: Plot title
            save_path: Optional path to save figure
            epochs: Optional list of epoch indices aligned with weights
            phate_n_pca: Optional PCA dimension; defaults to auto-thresholding
            phate_knn: Optional override for PHATE knn neighbourhood size
            phate_t: Optional PHATE diffusion time
            phate_decay: Optional PHATE decay parameter (float, or 0 to disable)
        """
        print(f"\nComputing PHATE embedding for {title}...")
        print(f"  Input shape: {weights.shape}")

        if epochs is None:
            epochs = list(range(1, len(weights) + 1))

        if len(epochs) != len(weights):
            raise ValueError("Length of epochs must match number of weight snapshots")

        # Apply PHATE with adaptive knn for small datasets
        n_samples = weights.shape[0]
        knn, knn_msg = self._resolve_phate_knn(n_samples, default_max=3,
                                               requested=phate_knn)
        if knn_msg:
            print(f"  {knn_msg}")

        resolved_n_pca, info_msg = self._resolve_phate_n_pca(weights.shape[1], n_samples, phate_n_pca)
        weights_for_phate = weights
        if resolved_n_pca:
            if info_msg:
                print(f"  {info_msg}")
            weights_for_phate, manual_msg = _manual_pca_projection(weights, resolved_n_pca)
            if manual_msg:
                print(f"  {manual_msg}")
        else:
            weights_for_phate = weights.astype(np.float32, copy=False)

        # Optional T-PHATE components
        if use_tphate and n_samples >= 2:
            # 1) Delay embedding for time-series structure
            if tphate_delay and tphate_lags and tphate_lags > 0:
                weights_for_phate = self._delay_embed(weights_for_phate, tphate_delay, tphate_lags)
            # 2) Append a scaled temporal feature unless kernel blending is used
            if not tphate_kernel:
                time_idx = np.linspace(0.0, 1.0, num=n_samples, dtype=np.float32).reshape(-1, 1)
                time_feat = (tphate_alpha if tphate_alpha is not None else 1.0) * time_idx
                weights_for_phate = np.hstack([weights_for_phate, time_feat]).astype(np.float32, copy=False)

        if tphate_kernel and n_samples >= 2:
            # Build blended precomputed distance
            group_ids = np.zeros(n_samples, dtype=np.int32)
            D = self._temporal_blended_distance(weights_for_phate, group_ids,
                                                alpha=(tphate_kernel_alpha or 1.0),
                                                tau=(tphate_kernel_tau or 3.0))
            try:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                                verbose=False, n_pca=None, knn_dist='precomputed')
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op = phate.PHATE(**_kwargs)
                weights_phate = phate_op.fit_transform(D)
            except Exception:
                # Fallback to feature-based PHATE
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                                verbose=False, n_pca=None)
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op = phate.PHATE(**_kwargs)
                weights_phate = phate_op.fit_transform(weights_for_phate)
        else:
            # Sanitize input to avoid duplicate rows causing zero-distance issues in graph kernel
            weights_for_phate = _sanitize_phate_input(weights_for_phate)
            _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                            verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op = phate.PHATE(**_kwargs)
            weights_phate = phate_op.fit_transform(weights_for_phate)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory backbone
        ax.plot(weights_phate[:, 0], weights_phate[:, 1],
                '-', color='0.7', alpha=0.8, linewidth=2)

        # Color points by epoch to convey temporal progression
        cmap = plt.cm.viridis
        norm = colors.Normalize(vmin=min(epochs), vmax=max(epochs))
        scatter = ax.scatter(weights_phate[:, 0], weights_phate[:, 1],
                             c=epochs, cmap=cmap, norm=norm, s=120, alpha=0.9,
                             edgecolors='black', linewidths=1.0, zorder=5)

        # Mark start and end
        ax.scatter(weights_phate[0, 0], weights_phate[0, 1],
                  c='green', s=220, marker='o', edgecolors='black',
                  linewidths=2, label=f'Start (epoch {epochs[0]})', zorder=10)

        markers = self._collect_validation_markers(epochs)
        if 'min' in markers:
            min_data = markers['min']
            idx = min_data['index']
            ax.scatter(weights_phate[idx, 0], weights_phate[idx, 1],
                      c='orange', s=200, marker='D', edgecolors='black', linewidths=1.5,
                      label=(f"Min val loss (epoch {min_data['epoch']}, "
                             f"loss {min_data['loss']:.4f})"), zorder=12)

        if 'final' in markers:
            final_data = markers['final']
            idx = final_data['index']
            ax.scatter(weights_phate[idx, 0], weights_phate[idx, 1],
                      c='red', s=260, marker='*', edgecolors='black', linewidths=2,
                      label=(f"Final val loss (epoch {final_data['epoch']}, "
                             f"loss {final_data['loss']:.4f})"), zorder=12)
        else:
            ax.scatter(weights_phate[-1, 0], weights_phate[-1, 1],
                      c='red', s=240, marker='*', edgecolors='black',
                      linewidths=2, label=f'End (epoch {epochs[-1]})', zorder=10)

        # Add epoch labels
        step = max(1, len(weights_phate) // 10)
        for i, (x, y) in enumerate(weights_phate):
            if i % step == 0 or i == len(weights_phate) - 1:
                ax.annotate(f'Ep {epochs[i]}', (x, y), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('PHATE 1', fontsize=12)
        ax.set_ylabel('PHATE 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Colorbar to reinforce temporal ordering
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Epoch', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to {save_path}")

        return fig, weights_phate

    def visualize_cnn_trajectory(self, save_path=None, epoch_min=None, epoch_max=None,
                                 epoch_stride=1, phate_n_pca=None, phate_knn=None,
                                 phate_t=None, phate_decay=None, use_tphate=False,
                                 tphate_alpha=1.0):
        """Visualize CNN weight evolution."""
        checkpoints = self.load_checkpoints()
        checkpoints, epochs = self._filter_checkpoints(checkpoints, epoch_min, epoch_max, epoch_stride)
        cnn_weights = self.extract_cnn_weights(checkpoints)
        title = f"CNN Weight Trajectory (k={self.kernel_size}, c={self.cnn_channels})"

        return self.visualize_weight_trajectory(cnn_weights, title, save_path, epochs,
                                                phate_n_pca=phate_n_pca,
                                                phate_knn=phate_knn,
                                                phate_t=phate_t,
                                                phate_decay=phate_decay,
                                                use_tphate=use_tphate,
                                                tphate_alpha=tphate_alpha)

    def visualize_gru_trajectory(self, save_path=None, epoch_min=None, epoch_max=None,
                                 epoch_stride=1, phate_n_pca=None, phate_knn=None,
                                 phate_t=None, phate_decay=None, use_tphate=False,
                                 tphate_alpha=1.0):
        """Visualize GRU weight evolution."""
        checkpoints = self.load_checkpoints()
        checkpoints, epochs = self._filter_checkpoints(checkpoints, epoch_min, epoch_max, epoch_stride)
        gru_weights = self.extract_gru_weights(checkpoints)
        title = f"GRU Weight Trajectory (hidden={self.gru_hidden})"
        return self.visualize_weight_trajectory(gru_weights, title, save_path, epochs,
                                                phate_n_pca=phate_n_pca,
                                                phate_knn=phate_knn,
                                                phate_t=phate_t,
                                                phate_decay=phate_decay,
                                                use_tphate=use_tphate,
                                                tphate_alpha=tphate_alpha)

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
        if len(representations) < 2:
            raise ValueError("Need at least two board states to compute a PHATE embedding")

        knn = self._select_knn(len(representations))
        phate_op = phate.PHATE(n_components=2, knn=knn, t=10, verbose=False)
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
        if len(representations) < 2:
            raise ValueError("Need at least two game states to compute a PHATE embedding")

        knn = self._select_knn(len(representations))
        phate_op = phate.PHATE(n_components=2, knn=knn, t=10, verbose=False)
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

    def create_summary_plot(self, epoch_min=None, epoch_max=None, epoch_stride=1,
                            phate_n_pca=None, phate_knn=None, phate_t=None,
                            phate_decay=None, use_tphate=False, tphate_alpha=1.0,
                            tphate_mode='time-feature', tphate_delay=None, tphate_lags=None,
                            tphate_kernel=False, tphate_kernel_alpha=1.0, tphate_kernel_tau=3.0):
        # Create a two-by-two summary plot (CNN, GRU, losses, component losses).
        # phate_n_pca: Optional PCA dimension applied before each PHATE fit.
        # phate_knn: Optional PHATE knn override shared across subplots.
        # phate_t: Optional diffusion time parameter.
        # phate_decay: Optional decay parameter.
        print("\nCreating summary visualization...")

        checkpoints = self.load_checkpoints()
        checkpoints, checkpoint_epochs = self._filter_checkpoints(
            checkpoints, epoch_min, epoch_max, epoch_stride)
        cnn_weights = self.extract_cnn_weights(checkpoints)
        gru_weights = self.extract_gru_weights(checkpoints)

        # Compute PHATE embeddings with adaptive knn
        n_samples = len(checkpoints)
        knn, knn_msg = self._resolve_phate_knn(n_samples, default_max=3, requested=phate_knn)
        if knn_msg:
            print(f"  {knn_msg}")

        print("  Computing CNN PHATE...")
        resolved_n_pca_cnn, info_cnn = self._resolve_phate_n_pca(cnn_weights.shape[1], n_samples,
                                                                  phate_n_pca)
        cnn_input = cnn_weights.astype(np.float32, copy=False)
        if resolved_n_pca_cnn:
            if info_cnn:
                print(f"  {info_cnn}")
            cnn_input, manual_msg = _manual_pca_projection(cnn_input, resolved_n_pca_cnn)
            if manual_msg:
                print(f"  {manual_msg}")

        # Optional experimental T-PHATE: append per-sample time feature
        if use_tphate and n_samples >= 2:
            if tphate_delay and tphate_lags and tphate_lags > 0:
                cnn_input = self._delay_embed(cnn_input, tphate_delay, tphate_lags)
            if not tphate_kernel:
                time_idx = np.linspace(0.0, 1.0, num=n_samples, dtype=np.float32).reshape(-1, 1)
                cnn_input = np.hstack([cnn_input, (tphate_alpha if tphate_alpha is not None else 1.0) * time_idx]).astype(np.float32, copy=False)

        if tphate_kernel and n_samples >= 2:
            groups = np.zeros(n_samples, dtype=np.int32)
            D = self._temporal_blended_distance(cnn_input, groups, tphate_kernel_alpha or 1.0, tphate_kernel_tau or 3.0)
            try:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                                verbose=False, n_pca=None, knn_dist='precomputed')
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op_cnn = phate.PHATE(**_kwargs)
                cnn_phate = phate_op_cnn.fit_transform(D)
            except Exception:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                                verbose=False, n_pca=None)
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op_cnn = phate.PHATE(**_kwargs)
                cnn_phate = phate_op_cnn.fit_transform(_sanitize_phate_input(cnn_input))
        else:
            _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                            verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op_cnn = phate.PHATE(**_kwargs)
            cnn_phate = phate_op_cnn.fit_transform(_sanitize_phate_input(cnn_input))

        print("  Computing GRU PHATE...")
        resolved_n_pca_gru, info_gru = self._resolve_phate_n_pca(gru_weights.shape[1], n_samples,
                                                                  phate_n_pca)
        gru_input = gru_weights.astype(np.float32, copy=False)
        if resolved_n_pca_gru:
            if info_gru:
                print(f"  {info_gru}")
            gru_input, manual_msg = _manual_pca_projection(gru_input, resolved_n_pca_gru)
            if manual_msg:
                print(f"  {manual_msg}")

        # Optional experimental T-PHATE for GRU
        if use_tphate and n_samples >= 2:
            if tphate_delay and tphate_lags and tphate_lags > 0:
                gru_input = self._delay_embed(gru_input, tphate_delay, tphate_lags)
            if not tphate_kernel:
                time_idx = np.linspace(0.0, 1.0, num=n_samples, dtype=np.float32).reshape(-1, 1)
                gru_input = np.hstack([gru_input, (tphate_alpha if tphate_alpha is not None else 1.0) * time_idx]).astype(np.float32, copy=False)

        if tphate_kernel and n_samples >= 2:
            groups = np.zeros(n_samples, dtype=np.int32)
            D = self._temporal_blended_distance(gru_input, groups, tphate_kernel_alpha or 1.0, tphate_kernel_tau or 3.0)
            try:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                                verbose=False, n_pca=None, knn_dist='precomputed')
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op_gru = phate.PHATE(**_kwargs)
                gru_phate = phate_op_gru.fit_transform(D)
            except Exception:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                                verbose=False, n_pca=None)
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op_gru = phate.PHATE(**_kwargs)
                gru_phate = phate_op_gru.fit_transform(_sanitize_phate_input(gru_input))
        else:
            _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                            verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op_gru = phate.PHATE(**_kwargs)
            gru_phate = phate_op_gru.fit_transform(_sanitize_phate_input(gru_input))

        # Create 2x2 plot
        fig = plt.figure(figsize=(16, 12))
        norm_epochs = colors.Normalize(vmin=min(checkpoint_epochs), vmax=max(checkpoint_epochs))

        # CNN trajectory
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(cnn_phate[:, 0], cnn_phate[:, 1], '-', color='0.7', alpha=0.8, linewidth=2)
        scatter_cnn = ax1.scatter(cnn_phate[:, 0], cnn_phate[:, 1],
                                  c=checkpoint_epochs, cmap=plt.cm.viridis, norm=norm_epochs, s=80,
                                  edgecolors='black', linewidths=1.0, zorder=5)
        ax1.scatter(cnn_phate[0, 0], cnn_phate[0, 1], c='green', s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10, label=f'Start (epoch {checkpoint_epochs[0]})')

        cnn_markers = self._collect_validation_markers(checkpoint_epochs)
        if 'min' in cnn_markers:
            m = cnn_markers['min']
            idx = m['index']
            ax1.scatter(cnn_phate[idx, 0], cnn_phate[idx, 1], c='orange', s=170, marker='D',
                        edgecolors='black', linewidths=1.3,
                        label=f"Min val loss (epoch {m['epoch']}, loss {m['loss']:.4f})", zorder=12)
        if 'final' in cnn_markers:
            m = cnn_markers['final']
            idx = m['index']
            ax1.scatter(cnn_phate[idx, 0], cnn_phate[idx, 1], c='red', s=200, marker='*',
                        edgecolors='black', linewidths=1.6,
                        label=f"Final val loss (epoch {m['epoch']}, loss {m['loss']:.4f})", zorder=12)
        else:
            ax1.scatter(cnn_phate[-1, 0], cnn_phate[-1, 1], c='red', s=150, marker='*',
                        edgecolors='black', linewidths=2, zorder=10, label=f'End (epoch {checkpoint_epochs[-1]})')

        ax1.legend(fontsize=9, loc='best')
        ax1.set_xlabel('PHATE 1')
        ax1.set_ylabel('PHATE 2')
        ax1.set_title(f'CNN Trajectory (k={self.kernel_size}, c={self.cnn_channels})', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # GRU trajectory
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(gru_phate[:, 0], gru_phate[:, 1], '-', color='0.7', alpha=0.8, linewidth=2)
        scatter_gru = ax2.scatter(gru_phate[:, 0], gru_phate[:, 1],
                                  c=checkpoint_epochs, cmap=plt.cm.viridis, norm=norm_epochs, s=80,
                                  edgecolors='black', linewidths=1.0, zorder=5)
        ax2.scatter(gru_phate[0, 0], gru_phate[0, 1], c='green', s=150, marker='o',
                   edgecolors='black', linewidths=2, zorder=10, label=f'Start (epoch {checkpoint_epochs[0]})')

        gru_markers = self._collect_validation_markers(checkpoint_epochs)
        if 'min' in gru_markers:
            m = gru_markers['min']
            idx = m['index']
            ax2.scatter(gru_phate[idx, 0], gru_phate[idx, 1], c='orange', s=170, marker='D',
                        edgecolors='black', linewidths=1.3,
                        label=f"Min val loss (epoch {m['epoch']}, loss {m['loss']:.4f})", zorder=12)
        if 'final' in gru_markers:
            m = gru_markers['final']
            idx = m['index']
            ax2.scatter(gru_phate[idx, 0], gru_phate[idx, 1], c='red', s=200, marker='*',
                        edgecolors='black', linewidths=1.6,
                        label=f"Final val loss (epoch {m['epoch']}, loss {m['loss']:.4f})", zorder=12)
        else:
            ax2.scatter(gru_phate[-1, 0], gru_phate[-1, 1], c='red', s=150, marker='*',
                        edgecolors='black', linewidths=2, zorder=10, label=f'End (epoch {checkpoint_epochs[-1]})')

        ax2.legend(fontsize=9, loc='best')
        ax2.set_xlabel('PHATE 1')
        ax2.set_ylabel('PHATE 2')
        ax2.set_title(f'GRU Trajectory (hidden={self.gru_hidden})', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Shared colorbar for epoch progression
        sm = plt.cm.ScalarMappable(norm=norm_epochs, cmap=plt.cm.viridis)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.046, pad=0.04)
        cbar.set_label('Epoch')

        # Loss curves
        if self.history:
            ax3 = plt.subplot(2, 2, 3)
            epoch_range = range(1, len(self.history['train_loss']) + 1)
            ax3.plot(epoch_range, self.history['train_loss'], label='Train Loss', linewidth=2)
            ax3.plot(epoch_range, self.history['val_loss'], label='Val Loss', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Training Progress', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Component losses
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(epoch_range, self.history['val_policy_loss'], label='Policy Loss', linewidth=2)
            ax4.plot(epoch_range, self.history['val_value_loss'], label='Value Loss', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Validation Component Losses', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Weight Trajectory Summary: k={self.kernel_size}, c={self.cnn_channels}, gru={self.gru_hidden}',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig

    @staticmethod
    def _board_from_tensor(board_tensor):
        """Convert board tensor to occupancy grid and turn indicator."""
        arr = board_tensor.squeeze(0).cpu().detach().numpy()
        occupancy = np.zeros((6, 7), dtype=np.int8)
        occupancy[arr[0] == 1] = 1  # Player one pieces
        occupancy[arr[1] == 1] = -1  # Player two pieces
        turn = int(arr[2].mean().round())
        return occupancy, turn

    def _compute_grad_cam(self, model, board_tensor, target='policy', move=None):
        """Compute Grad-CAM heatmap for a single board state."""
        activations = {}
        gradients = {}

        target_layer = model.resnet[-1]

        def forward_hook(_, __, output):
            activations['value'] = output.detach()

        def backward_hook(_, grad_input, grad_output):
            gradients['value'] = grad_output[0].detach()

        handle_f = target_layer.register_forward_hook(forward_hook)
        if hasattr(target_layer, 'register_full_backward_hook'):
            handle_b = target_layer.register_full_backward_hook(backward_hook)
        else:
            handle_b = target_layer.register_backward_hook(backward_hook)

        model.zero_grad(set_to_none=True)

        board_tensor = board_tensor.to(self.device)
        board_tensor = board_tensor.requires_grad_(True)
        policy_logits, value, _ = model(board_tensor)

        predicted_move = torch.argmax(policy_logits, dim=-1).item()

        if target == 'policy':
            focus_move = move if move is not None else predicted_move
            target_logit = policy_logits[0, focus_move]
        else:
            focus_move = None
            target_logit = value.view(-1)[0]

        target_logit.backward(retain_graph=True)

        activation = activations['value']
        gradient = gradients['value']

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(6, 7), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        policy_probs = torch.softmax(policy_logits, dim=-1).detach().cpu().numpy().squeeze()
        value_pred = value.detach().cpu().numpy().squeeze().item()

        handle_f.remove()
        handle_b.remove()
        model.zero_grad(set_to_none=True)

        return cam, {
            'focus_move': focus_move,
            'predicted_move': predicted_move,
            'policy_probs': policy_probs,
            'value': value_pred,
            'target': target
        }

    def visualize_cnn_activations(self, board_states, checkpoint_path, save_dir,
                                   target='policy', move=None, max_examples=4):
        """Generate Grad-CAM heatmaps for CNN activations on supplied boards."""
        print("\nComputing CNN Grad-CAM activations...")

        model = create_model(self.cnn_channels, self.gru_hidden, self.kernel_size)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        board_states = board_states[:max_examples]
        results = []

        for idx, state in enumerate(board_states):
            cam, meta = self._compute_grad_cam(
                model, state, target=target, move=move)

            occupancy, turn = self._board_from_tensor(state)

            fig, ax = plt.subplots(figsize=(5, 4))
            board_cmap = colors.ListedColormap(['#d73027', '#f7f7f7', '#4575b4'])
            board_norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], board_cmap.N)
            ax.imshow(occupancy, cmap=board_cmap, norm=board_norm, origin='upper')
            heat = ax.imshow(cam, cmap='inferno', alpha=0.55, origin='upper')

            ax.set_xticks(range(7))
            ax.set_yticks(range(6))
            ax.set_xticklabels(range(7))
            ax.set_yticklabels(range(5, -1, -1))
            ax.set_xlabel('Column')
            ax.set_ylabel('Row (bottom to top)')
            move_text = "N/A"
            move_prob = None
            if meta['focus_move'] is not None:
                move_text = str(meta['focus_move'])
                move_prob = meta['policy_probs'][meta['focus_move']]
            elif meta['predicted_move'] is not None:
                move_text = str(meta['predicted_move'])
                move_prob = meta['policy_probs'][meta['predicted_move']]

            subtitle = f"Grad-CAM ({target.title()})"
            subtitle += f"\nFocus move {move_text}"
            if move_prob is not None:
                subtitle += f" prob {move_prob:.2f}"
            subtitle += f", value {meta['value']:.2f}"
            ax.set_title(subtitle)
            ax.grid(which='both', color='black', linestyle='-', linewidth=0.5, alpha=0.2)

            cbar = plt.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Activation intensity')

            turn_text = 'Red to move' if turn else 'Yellow to move'
            ax.text(0.02, 0.02, turn_text, transform=ax.transAxes,
                    fontsize=9, fontweight='bold', color='black', bbox=dict(facecolor='white', alpha=0.6))

            fname = save_dir / f"activation_{idx:03d}.png"
            fig.tight_layout()
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)

            results.append({
                'path': fname,
                'focus_move': meta['focus_move'],
                'predicted_move': meta['predicted_move'],
                'value': meta['value'],
                'target': meta['target']
            })

            print(f"  Saved {fname}")

        return results

    def visualize_joint_cnn_gru(self, save_path=None, epoch_min=None, epoch_max=None,
                                 epoch_stride=1, center_strategy='none',
                                 phate_n_pca=None, phate_knn=None, phate_t=None,
                                 phate_decay=None, use_tphate=False, tphate_alpha=1.0,
                                 tphate_mode='time-feature', tphate_delay=None, tphate_lags=None,
                                 tphate_kernel=False, tphate_kernel_alpha=1.0, tphate_kernel_tau=3.0):
        """Plot CNN and GRU trajectories together in a shared PHATE embedding.

        Args:
            phate_n_pca: Optional PCA dimension applied to the stacked weights before PHATE.
            phate_knn: Optional PHATE knn override for the joint embedding.
            phate_t: Optional diffusion time parameter.
            phate_decay: Optional decay parameter.
        """
        checkpoints = self.load_checkpoints()
        checkpoints, epochs = self._filter_checkpoints(checkpoints, epoch_min, epoch_max, epoch_stride)

        cnn_weights = self.extract_cnn_weights(checkpoints)
        gru_weights = self.extract_gru_weights(checkpoints)

        max_dim = max(cnn_weights.shape[1], gru_weights.shape[1])
        cnn_weights = _pad_weights_to_length(cnn_weights, max_dim)
        gru_weights = _pad_weights_to_length(gru_weights, max_dim)

        combined = np.vstack([cnn_weights, gru_weights]).astype(np.float32, copy=False)
        knn, knn_msg = self._resolve_phate_knn(combined.shape[0], default_max=3,
                                               requested=phate_knn)
        if knn_msg:
            print(f"  {knn_msg}")
        resolved_n_pca, info_msg = self._resolve_phate_n_pca(combined.shape[1], combined.shape[0],
                                                             phate_n_pca)
        phate_input = combined
        if resolved_n_pca:
            if info_msg:
                print(f"  {info_msg}")
            phate_input, manual_msg = _manual_pca_projection(combined, resolved_n_pca)
            if manual_msg:
                print(f"  {manual_msg}")

        # Optional experimental T-PHATE: append a time feature that resets per block (CNN/GRU)
        if use_tphate:
            n = len(cnn_weights)
            m = len(gru_weights)
            if n >= 2:
                time_cnn = np.linspace(0.0, 1.0, num=n, dtype=np.float32).reshape(-1, 1)
            else:
                time_cnn = np.zeros((n, 1), dtype=np.float32)
            if m >= 2:
                time_gru = np.linspace(0.0, 1.0, num=m, dtype=np.float32).reshape(-1, 1)
            else:
                time_gru = np.zeros((m, 1), dtype=np.float32)
            time_feat = np.vstack([time_cnn, time_gru]) * (tphate_alpha if tphate_alpha is not None else 1.0)
            # Optional delay embedding per block before stacking time feature
            if tphate_delay and tphate_lags and tphate_lags > 0:
                combined_cnn = self._delay_embed(cnn_weights.astype(np.float32, copy=False), tphate_delay, tphate_lags)
                combined_gru = self._delay_embed(gru_weights.astype(np.float32, copy=False), tphate_delay, tphate_lags)
                combined = np.vstack([combined_cnn, combined_gru]).astype(np.float32, copy=False)
                phate_input = combined
            phate_input = np.hstack([phate_input, time_feat]).astype(np.float32, copy=False)

        if tphate_kernel:
            # Group ids: 0 for CNN rows, 1 for GRU rows
            groups = np.array([0] * len(cnn_weights) + [1] * len(gru_weights), dtype=np.int32)
            D = self._temporal_blended_distance(phate_input, groups, tphate_kernel_alpha or 1.0, tphate_kernel_tau or 3.0)
            try:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                               verbose=False, n_pca=None, knn_dist='precomputed')
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op = phate.PHATE(**_kwargs)
                embedding = phate_op.fit_transform(D)
            except Exception:
                _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                               verbose=False, n_pca=None)
                if phate_decay is not None:
                    _kwargs['decay'] = phate_decay
                phate_op = phate.PHATE(**_kwargs)
                embedding = phate_op.fit_transform(phate_input)
        else:
            _kwargs = dict(n_components=2, knn=knn, t=phate_t if phate_t is not None else 10,
                           verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op = phate.PHATE(**_kwargs)
            embedding = phate_op.fit_transform(phate_input)

        cnn_embed = embedding[:len(cnn_weights)]
        gru_embed = embedding[len(cnn_weights):]

        cnn_embed = _adjust_embedding(cnn_embed, center_strategy)
        gru_embed = _adjust_embedding(gru_embed, center_strategy)

        fig, ax = plt.subplots(figsize=(10, 8))
        cnn_color = '#1f77b4'
        gru_color = '#d62728'

        ax.plot(cnn_embed[:, 0], cnn_embed[:, 1], '-', color=cnn_color, linewidth=2, alpha=0.85)
        ax.scatter(cnn_embed[:, 0], cnn_embed[:, 1], c=cnn_color, s=70, alpha=0.7,
                   edgecolors='black', linewidths=0.6)
        ax.scatter(cnn_embed[0, 0], cnn_embed[0, 1], c=cnn_color, s=160, marker='o',
                   edgecolors='black', linewidths=1.2, zorder=10)

        ax.plot(gru_embed[:, 0], gru_embed[:, 1], '-', color=gru_color, linewidth=2, alpha=0.85)
        ax.scatter(gru_embed[:, 0], gru_embed[:, 1], c=gru_color, s=70, alpha=0.7,
                   edgecolors='black', linewidths=0.6)
        ax.scatter(gru_embed[0, 0], gru_embed[0, 1], c=gru_color, s=160, marker='s',
                   edgecolors='black', linewidths=1.2, zorder=10)

        markers = self._collect_validation_markers(epochs)
        if 'min' in markers:
            m = markers['min']
            idx = m['index']
            ax.scatter(cnn_embed[idx, 0], cnn_embed[idx, 1], c='orange', marker='D', s=210,
                       edgecolors='black', linewidths=1.4, zorder=12)
            ax.scatter(gru_embed[idx, 0], gru_embed[idx, 1], facecolors='none', marker='D', s=210,
                       edgecolors='orange', linewidths=1.6, zorder=12)

        if 'final' in markers:
            m = markers['final']
            idx = m['index']
            ax.scatter(cnn_embed[idx, 0], cnn_embed[idx, 1], c='red', marker='*', s=230,
                       edgecolors='black', linewidths=1.6, zorder=12)
            ax.scatter(gru_embed[idx, 0], gru_embed[idx, 1], facecolors='none', marker='*', s=230,
                       edgecolors='red', linewidths=1.8, zorder=12)
        else:
            ax.scatter(cnn_embed[-1, 0], cnn_embed[-1, 1], c='red', marker='*', s=230,
                       edgecolors='black', linewidths=1.6, zorder=12)
            ax.scatter(gru_embed[-1, 0], gru_embed[-1, 1], facecolors='none', marker='*', s=230,
                       edgecolors='red', linewidths=1.8, zorder=12)

        step = max(1, len(epochs) // 10)
        for i in range(0, len(epochs), step):
            ax.annotate(f'Ep {epochs[i]}', (cnn_embed[i, 0], cnn_embed[i, 1]), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')
            ax.annotate(f'Ep {epochs[i]}', (gru_embed[i, 0], gru_embed[i, 1]), fontsize=8,
                        xytext=(-30, -10), textcoords='offset points')
        ax.annotate(f'Ep {epochs[-1]}', (cnn_embed[-1, 0], cnn_embed[-1, 1]), fontsize=8,
                    xytext=(5, 5), textcoords='offset points')
        ax.annotate(f'Ep {epochs[-1]}', (gru_embed[-1, 0], gru_embed[-1, 1]), fontsize=8,
                    xytext=(-30, -10), textcoords='offset points')

        ax.set_xlabel('PHATE 1', fontsize=12)
        ax.set_ylabel('PHATE 2', fontsize=12)
        ax.set_title(f'CNN vs GRU Trajectories (k={self.kernel_size}, c={self.cnn_channels}, '
                     f'gru={self.gru_hidden})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        legend_elements = [
            Line2D([0], [0], color=cnn_color, lw=2, label='CNN trajectory'),
            Line2D([0], [0], color=gru_color, lw=2, label='GRU trajectory'),
            Line2D([0], [0], marker='o', color='black', markerfacecolor=cnn_color,
                   markersize=8, linestyle='None', label='CNN start'),
            Line2D([0], [0], marker='s', color='black', markerfacecolor=gru_color,
                   markersize=8, linestyle='None', label='GRU start'),
            Line2D([0], [0], marker='D', color='black', markerfacecolor='orange',
                   markersize=8, linestyle='None', label='Min val loss (filled=CNN, outline=GRU)'),
            Line2D([0], [0], marker='*', color='black', markerfacecolor='red',
                   markersize=10, linestyle='None', label='Final val loss (filled=CNN, outline=GRU)')
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved to {save_path}")

        return fig, {'cnn': cnn_embed, 'gru': gru_embed, 'epochs': epochs}

def _pad_weights_to_length(weight_matrix, target_size):
    """Pad flattened weight matrix with zeros so shapes align across runs."""
    if weight_matrix.shape[1] == target_size:
        return weight_matrix
    pad_amount = target_size - weight_matrix.shape[1]
    if pad_amount < 0:
        raise ValueError("target_size must be at least as large as the current weight dimension")
    return np.pad(weight_matrix, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0.0)


def _manual_pca_projection(matrix, target_dim):
    """Project samples into a PCA subspace without constructing huge feature matrices."""
    n_samples, n_features = matrix.shape
    target_dim = int(max(1, min(target_dim, n_samples - 1, n_features)))
    if target_dim <= 0:
        raise ValueError("target_dim must be positive after adjustment")

    data = np.array(matrix, dtype=np.float32, copy=True)
    mean = data.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    data -= mean

    gram = data @ data.T
    gram = gram.astype(np.float64)

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-9
    keep = int(min(target_dim, np.count_nonzero(positive)))
    if keep == 0:
        reduced = np.zeros((n_samples, 1), dtype=np.float32)
        return reduced, "PCA spectrum collapsed; returning zero embedding."

    eigvals = eigvals[:keep]
    eigvecs = eigvecs[:, :keep]
    scales = np.sqrt(np.clip(eigvals, a_min=0.0, a_max=None))
    reduced = eigvecs * scales
    return reduced.astype(np.float32), f"Manual PCA reduced weights to {keep} dimensions."


def _adjust_embedding(run_embed, strategy):
    """Optionally center or normalize trajectory embedding for readability."""
    if strategy == 'anchor':
        return run_embed - run_embed[0]
    if strategy == 'normalize':
        anchored = run_embed - run_embed[0]
        max_dist = np.linalg.norm(anchored, axis=1).max()
        if max_dist > 0:
            anchored = anchored / max_dist
        return anchored
    return run_embed


def _sanitize_phate_input(X: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """Make PHATE input robust by removing NaNs/Infs and jittering duplicate rows.

    - Replaces NaN/Inf with 0.0 (safe neutral fill after mean-centering/PCA).
    - Adds tiny deterministic jitter to repeated rows to avoid zero-distance degeneracies
      that can produce NaNs in graphtools kernel normalization.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    # Ensure finite values
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Jitter exact duplicates
    try:
        uniq, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            rng = np.random.default_rng(rng_seed)
            scale = float(np.std(X))
            eps = 1e-5 * (scale if scale > 0 else 1.0)
            seen = {}
            for i, g in enumerate(inverse):
                c = seen.get(int(g), 0)
                if c > 0:
                    X[i] += (eps * rng.standard_normal(X.shape[1])).astype(np.float32)
                seen[int(g)] = c + 1
            # Second pass: if duplicates remain due to numerical ties, add tiny deterministic per-row offset
            uniq2, inverse2, counts2 = np.unique(X, axis=0, return_inverse=True, return_counts=True)
            if np.any(counts2 > 1):
                eps2 = 1e-6 * (scale if scale > 0 else 1.0)
                for i in range(X.shape[0]):
                    X[i, 0] += np.float32(eps2 * ((i % 97) / 97.0))
    except Exception:
        # Be conservative: add a minuscule global jitter if uniqueness fails
        scale = float(np.std(X))
        eps = 1e-7 * (scale if scale > 0 else 1.0)
        X = X + eps
    return X


def _create_ablation_animation(ax, run_data, component, animation_path, fps=2):
    """Create a simple epoch-sweep animation highlighting each run's current checkpoint."""
    if not animation_path:
        return

    unique_epochs = sorted({epoch for entry in run_data for epoch in entry['epochs']})
    if len(unique_epochs) < 2:
        print("Not enough epochs to build an animation; skipping.")
        return

    fig = ax.figure
    highlight_markers = []
    for entry in run_data:
        marker = ax.scatter([], [], s=260, marker='o', edgecolors='black', linewidths=1.2,
                            facecolors=[entry['color']], zorder=15, alpha=0.95)
        marker.set_visible(False)
        highlight_markers.append(marker)

    epoch_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                         fontsize=12, fontweight='bold', va='top')

    def update(frame_idx):
        epoch = unique_epochs[frame_idx]
        epoch_text.set_text(f'Epoch {epoch}')
        for entry, marker in zip(run_data, highlight_markers):
            available = [i for i, ep in enumerate(entry['epochs']) if ep <= epoch]
            if not available:
                marker.set_visible(False)
                continue
            idx = available[-1]
            position = entry['embedding'][idx]
            marker.set_offsets(np.array([[position[0], position[1]]]))
            marker.set_sizes([240 if idx == len(entry['epochs']) - 1 else 200])
            marker.set_visible(True)
        return highlight_markers + [epoch_text]

    interval = max(200, int(1000 / max(1, fps)))
    anim = FuncAnimation(fig, update, frames=len(unique_epochs), interval=interval, blit=False)
    animation_path = Path(animation_path)
    animation_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(animation_path, dpi=150, writer='pillow')
    print(f"  Saved animation to {animation_path}")


def visualize_ablation_weight_trajectories(checkpoint_dirs, component='cnn', device='cpu',
                                           save_path=None, animation_path=None, max_knn=8,
                                           center_strategy='none', epoch_min=None,
                                           epoch_max=None, epoch_stride=1, phate_n_pca=None,
                                           phate_knn=None, phate_t=None, phate_decay=None,
                                           use_tphate=False, tphate_alpha=1.0,
                                           tphate_mode='time-feature', tphate_delay=None,
                                           tphate_lags=None, tphate_kernel=False,
                                           tphate_kernel_alpha=1.0, tphate_kernel_tau=3.0):
    """Compare weight trajectories from multiple checkpoint directories in one PHATE embedding.

    Args:
        phate_n_pca: Optional PCA dimension applied before PHATE; defaults to auto-selection.
        phate_knn: Optional PHATE knn override.
        phate_t: Optional diffusion time.
        phate_decay: Optional decay parameter.
    """
    if component not in {'cnn', 'gru'}:
        raise ValueError("component must be 'cnn' or 'gru'")

    path_list = [Path(cp).expanduser() for cp in checkpoint_dirs]
    if len(path_list) < 2:
        raise ValueError("Provide at least two checkpoint directories for ablation comparison")

    run_data = []
    max_dim = 0
    for cp_path in path_list:
        viz = TrajectoryVisualizer(cp_path, device)
        checkpoints = viz.load_checkpoints()
        checkpoints, epochs = viz._filter_checkpoints(checkpoints, epoch_min, epoch_max, epoch_stride)
        if component == 'cnn':
            weights = viz.extract_cnn_weights(checkpoints)
        else:
            weights = viz.extract_gru_weights(checkpoints)
        descriptor = f"k={viz.kernel_size}, c={viz.cnn_channels}, gru={viz.gru_hidden}"
        run_label = f"{cp_path.name} ({descriptor})"

        max_dim = max(max_dim, weights.shape[1])
        run_data.append({
            'path': cp_path,
            'weights': weights,
            'epochs': epochs,
            'label': run_label,
            'markers': viz._collect_validation_markers(epochs)
        })

    if max_dim == 0:
        raise ValueError("Unable to determine weight dimensionality for ablation embedding")

    padded_weights = []
    for entry in run_data:
        padded = _pad_weights_to_length(entry['weights'], max_dim)
        entry['weights'] = padded
        padded_weights.append(padded)

    stacked = np.vstack(padded_weights).astype(np.float32, copy=False)
    if stacked.shape[0] < 2:
        raise ValueError("Need at least two weight snapshots for PHATE embedding")

    knn, knn_msg = TrajectoryVisualizer._resolve_phate_knn(
        stacked.shape[0], default_max=max_knn, requested=phate_knn)
    if knn_msg:
        print(f"  {knn_msg}")
    resolved_n_pca, info_msg = TrajectoryVisualizer._resolve_phate_n_pca(
        stacked.shape[1], stacked.shape[0], phate_n_pca)
    phate_input = stacked
    if resolved_n_pca:
        if info_msg:
            print(f"  {info_msg}")
        phate_input, manual_msg = _manual_pca_projection(stacked, resolved_n_pca)
        if manual_msg:
            print(f"  {manual_msg}")

    # Optional T-PHATE: delay-embed and append per-run time feature (resets for each run)
    if use_tphate:
        if tphate_delay and tphate_lags and tphate_lags > 0:
            phate_input = TrajectoryVisualizer._delay_embed(phate_input, tphate_delay, tphate_lags)
        time_cols = []
        for entry in run_data:
            count = entry['weights'].shape[0]
            if count >= 2:
                t = np.linspace(0.0, 1.0, num=count, dtype=np.float32).reshape(-1, 1)
            else:
                t = np.zeros((count, 1), dtype=np.float32)
            time_cols.append(t)
        time_feat = np.vstack(time_cols) * (tphate_alpha if tphate_alpha is not None else 1.0)
        phate_input = np.hstack([phate_input, time_feat]).astype(np.float32, copy=False)

    if tphate_kernel:
        counts = [entry['weights'].shape[0] for entry in run_data]
        groups = np.concatenate([np.full(c, i, dtype=np.int32) for i, c in enumerate(counts)])
        D = TrajectoryVisualizer._temporal_blended_distance(phate_input, groups,
                                                            tphate_kernel_alpha or 1.0,
                                                            tphate_kernel_tau or 3.0)
        try:
            _kwargs = dict(n_components=2, knn=knn,
                           t=phate_t if phate_t is not None else 10,
                           verbose=False, n_pca=None, knn_dist='precomputed')
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op = phate.PHATE(**_kwargs)
            embedding = phate_op.fit_transform(D)
        except Exception:
            _kwargs = dict(n_components=2, knn=knn,
                           t=phate_t if phate_t is not None else 10,
                           verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op = phate.PHATE(**_kwargs)
            embedding = phate_op.fit_transform(phate_input)
    else:
        # Sanitize input to avoid zero-distance duplicates causing NaNs in graphtools normalization
        phate_input = _sanitize_phate_input(phate_input)
        try:
            _kwargs = dict(n_components=2, knn=knn,
                           t=phate_t if phate_t is not None else 10,
                           verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op = phate.PHATE(**_kwargs)
            embedding = phate_op.fit_transform(phate_input)
        except Exception as e:
            # Retry with slightly larger neighbourhood and fresh sanitization
            print(f"  PHATE failed ({e}); retrying with knn={min(knn+2, phate_input.shape[0]-1)}.")
            knn_retry = max(2, min(knn + 2, phate_input.shape[0] - 1))
            phate_input = _sanitize_phate_input(phate_input, rng_seed=1)
            _kwargs = dict(n_components=2, knn=knn_retry,
                           t=phate_t if phate_t is not None else 10,
                           verbose=False, n_pca=None)
            if phate_decay is not None:
                _kwargs['decay'] = phate_decay
            phate_op = phate.PHATE(**_kwargs)
            embedding = phate_op.fit_transform(phate_input)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.colormaps['tab20'].resampled(max(len(run_data), 1))
    color_samples = cmap(np.linspace(0, 1, len(run_data))) if run_data else np.empty((0, 4))
    min_label_used = False
    final_label_used = False

    cursor = 0
    for idx, entry in enumerate(run_data):
        count = entry['weights'].shape[0]
        run_embed = embedding[cursor:cursor + count]
        cursor += count

        run_embed = _adjust_embedding(run_embed, center_strategy)
        entry['embedding'] = run_embed
        entry['color'] = (tuple(color_samples[idx])
                          if len(color_samples) else '#1f77b4')

        line_color = entry['color']
        ax.plot(run_embed[:, 0], run_embed[:, 1], '-', color=line_color, linewidth=2,
                alpha=0.75, label=entry['label'])
        ax.scatter(run_embed[:, 0], run_embed[:, 1], c=[line_color], s=70, alpha=0.65,
                   edgecolors='black', linewidths=0.6)

        ax.scatter(run_embed[0, 0], run_embed[0, 1], c=[line_color], s=160, marker='o',
                   edgecolors='black', linewidths=1.2, zorder=10)

        markers = entry.get('markers', {})
        if 'min' in markers:
            m = markers['min']
            idx_min = m['index']
            label = None
            if not min_label_used:
                label = f"Min val loss (epoch {m['epoch']}, loss {m['loss']:.4f})"
                min_label_used = True
            ax.scatter(run_embed[idx_min, 0], run_embed[idx_min, 1], c='orange', s=200, marker='D',
                       edgecolors='black', linewidths=1.4, zorder=12, label=label)

        if 'final' in markers:
            m = markers['final']
            idx_final = m['index']
            label = None
            if not final_label_used:
                label = f"Final val loss (epoch {m['epoch']}, loss {m['loss']:.4f})"
                final_label_used = True
            ax.scatter(run_embed[idx_final, 0], run_embed[idx_final, 1], c='red', s=220, marker='*',
                       edgecolors='black', linewidths=1.6, zorder=12, label=label)
        else:
            ax.scatter(run_embed[-1, 0], run_embed[-1, 1], c=[line_color], s=180, marker='*',
                       edgecolors='black', linewidths=1.2, zorder=10)

        epochs = entry['epochs']
        step = max(1, len(epochs) // 5)
        for j in range(0, len(run_embed), step):
            ax.annotate(f"Ep {epochs[j]}", (run_embed[j, 0], run_embed[j, 1]), fontsize=8,
                        xytext=(4, 4), textcoords='offset points')
        ax.annotate(f"Ep {epochs[-1]}", (run_embed[-1, 0], run_embed[-1, 1]), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')

    comp_title = 'CNN' if component == 'cnn' else 'GRU'
    ax.set_xlabel('PHATE 1', fontsize=12)
    ax.set_ylabel('PHATE 2', fontsize=12)
    ax.set_title(f'{comp_title} Weight Trajectories Across Ablations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    _create_ablation_animation(ax, run_data, component, animation_path)

    return fig, embedding, run_data


def generate_random_boards(n_boards=100, seed=None):
    """Generate random Connect-4 board states for visualization"""
    boards = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boards):
        # Random board with 0-20 pieces
        n_pieces = int(rng.integers(0, 21))
        board = np.zeros((3, 6, 7), dtype=np.float32)

        for _ in range(n_pieces):
            player = int(rng.integers(0, 2))  # 0=yellow, 1=red
            col = int(rng.integers(0, 7))

            # Find lowest empty row
            for row in range(5, -1, -1):
                if board[0, row, col] == 0 and board[1, row, col] == 0:
                    board[player, row, col] = 1
                    break

        # Add turn plane
        board[2, :, :] = rng.integers(0, 2, size=(6, 7))

        boards.append(torch.from_numpy(board).unsqueeze(0))

    return boards


def generate_random_game(n_moves=20, seed=None):
    """Generate a simple random legal Connect-4 game sequence as board tensors."""
    rng = np.random.default_rng(seed)
    board = np.zeros((3, 6, 7), dtype=np.float32)
    seq = []
    turn = 0  # 0=yellow, 1=red
    for _ in range(max(1, int(n_moves))):
        seq.append(torch.from_numpy(board.copy()).unsqueeze(0))
        col = int(rng.integers(0, 7))
        placed = False
        for row in range(5, -1, -1):
            if board[0, row, col] == 0 and board[1, row, col] == 0:
                board[turn, row, col] = 1.0
                placed = True
                break
        if not placed:
            continue
        board[2, :, :] = 1.0 if turn == 1 else 0.0
        turn = 1 - turn
    return seq


def main():
    parser = argparse.ArgumentParser(description="Visualize weight trajectories using PHATE")

    parser.add_argument("--checkpoint-dir", type=str,
                       help="Directory containing checkpoints")
    parser.add_argument("--ablation-dirs", type=str, nargs='+',
                       help="List of checkpoint directories for ablation comparison")
    parser.add_argument("--output-dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--viz-type", type=str, default="all",
                       choices=["all", "cnn", "gru", "boards", "summary", "joint", "ablation-cnn", "ablation-gru", "activations", "temporal"],
                       help="Type of visualization to create")
    parser.add_argument("--n-boards", type=int, default=100,
                       help="Number of random boards for representation viz")
    parser.add_argument("--board-seed", type=int, default=0,
                       help="Seed for random board generation (default: 0)")
    parser.add_argument("--ablation-animate", action='store_true',
                       help="Produce a GIF animation for ablation visualizations")
    parser.add_argument("--ablation-center", choices=["none", "anchor", "normalize"], default="none",
                       help="Post-process ablation embeddings to emphasise relative paths")
    parser.add_argument("--joint-center", choices=["none", "anchor", "normalize"], default="none",
                       help="Post-process joint CNN/GRU embeddings")
    parser.add_argument("--activation-target", choices=['policy', 'value'], default='policy',
                       help="Grad-CAM target for activation maps")
    parser.add_argument("--activation-move", type=int,
                       help="Optional policy move index to focus Grad-CAM on (0-6)")
    parser.add_argument("--activation-max-examples", type=int, default=4,
                       help="Maximum number of activation maps to export")
    parser.add_argument("--epoch-min", type=int,
                       help="Minimum epoch to include (inclusive)")
    parser.add_argument("--epoch-max", type=int,
                       help="Maximum epoch to include (inclusive)")
    parser.add_argument("--epoch-step", type=int, default=1,
                       help="Stride when sampling checkpoints (1 = use every checkpoint)")
    parser.add_argument("--phate-n-pca", type=int,
                       help="Optional PCA dimension before PHATE (auto-selects for high-dimensional weights)")
    parser.add_argument("--phate-knn", type=int,
                       help="Optional override for PHATE knn (neighbourhood size)")
    parser.add_argument("--phate-t", type=int,
                       help="Optional PHATE diffusion time (higher emphasises global structure)")
    parser.add_argument("--phate-decay", type=float,
                       help="Optional PHATE decay parameter controlling kernel tail (default 'auto')")
    parser.add_argument("--t-phate", action='store_true',
                       help="Experimental: enable T-PHATE by appending a temporal feature to each sample")
    parser.add_argument("--t-phate-alpha", type=float, default=1.0,
                       help="Temporal feature weight for T-PHATE (higher enforces stronger chronology; default: 1.0)")
    parser.add_argument("--t-phate-delay", type=int,
                       help="T-PHATE delay embedding step (tau). Requires --t-phate-lags > 0")
    parser.add_argument("--t-phate-lags", type=int,
                       help="T-PHATE number of past lags to include in delay embedding")
    parser.add_argument("--t-phate-kernel", action='store_true',
                       help="Use temporal kernel blending (precomputed distances) instead of feature augmentation")
    parser.add_argument("--t-phate-kernel-alpha", type=float, default=1.0,
                       help="Temporal kernel weight (alpha) for blended distance")
    parser.add_argument("--t-phate-kernel-tau", type=float, default=3.0,
                       help="Temporal kernel scale (tau) measured in steps")
    parser.add_argument("--temporal-mode", choices=['training', 'game'], default='training',
                       help="Temporal axis: training=across epochs at fixed game step; game=across game at fixed epoch")
    parser.add_argument("--time-steps", type=int, default=20,
                       help="Game: number of moves to simulate; Training: used to pick mid step if --time-step not set")
    parser.add_argument("--time-step", type=int,
                       help="Training mode: which game step index to hold while traversing epochs")
    parser.add_argument("--time-epoch", type=str, default='min',
                       help="Game mode: which epoch to hold (min|final|latest or a number)")
    parser.add_argument("--sequential-data", type=str,
                       help="Path to sequential dataset (.pt) with real game trajectories for temporal viz")

    args = parser.parse_args()

    if args.viz_type in {"ablation-cnn", "ablation-gru"}:
        if not args.ablation_dirs or len(args.ablation_dirs) < 2:
            parser.error("--ablation-dirs requires at least two directories for ablation visualizations")
    elif args.viz_type == "activations":
        if not args.checkpoint_dir:
            parser.error("--checkpoint-dir is required for activation visualizations")
    else:
        if not args.checkpoint_dir:
            parser.error("--checkpoint-dir is required for this viz-type")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_min = args.epoch_min
    epoch_max = args.epoch_max
    epoch_stride = args.epoch_step if args.epoch_step and args.epoch_step > 0 else 1

    if args.viz_type in {"ablation-cnn", "ablation-gru"}:
        component = 'cnn' if args.viz_type == "ablation-cnn" else 'gru'
        print("\n" + "="*60)
        print(f"Ablation {component.upper()} Weight Trajectories")
        print("="*60)
        animation_path = None
        if args.ablation_animate:
            animation_path = output_dir / f"ablation_{component}_trajectories.gif"
        fig, _, run_data = visualize_ablation_weight_trajectories(
            args.ablation_dirs,
            component=component,
            device=device,
            save_path=output_dir / f"ablation_{component}_trajectories.png",
            animation_path=animation_path,
            center_strategy=args.ablation_center,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            epoch_stride=epoch_stride,
            phate_n_pca=args.phate_n_pca,
            phate_knn=args.phate_knn,
            phate_t=args.phate_t,
            phate_decay=args.phate_decay,
            use_tphate=args.t_phate,
            tphate_alpha=args.t_phate_alpha,
            tphate_delay=args.t_phate_delay,
            tphate_lags=args.t_phate_lags,
            tphate_kernel=args.t_phate_kernel,
            tphate_kernel_alpha=args.t_phate_kernel_alpha,
            tphate_kernel_tau=args.t_phate_kernel_tau
        )
        plt.close(fig)

        print("Compared runs:")
        for entry in run_data:
            print(f"  - {entry['label']}")

        print("\n" + "="*60)
        print(f"✓ Visualizations saved to {output_dir}")
        print("="*60)
        return

    viz = TrajectoryVisualizer(args.checkpoint_dir, device)

    if args.viz_type == "activations":
        print("\n" + "="*60)
        print("CNN Activation Maps (Grad-CAM)")
        print("="*60)
        boards = generate_random_boards(args.n_boards, seed=args.board_seed)
        checkpoint_path = viz.get_latest_checkpoint_path()
        viz.visualize_cnn_activations(
            boards,
            checkpoint_path,
            save_dir=output_dir / "activations",
            target=args.activation_target,
            move=args.activation_move,
            max_examples=args.activation_max_examples
        )
        print("\n" + "="*60)
        print(f"✓ Activation maps saved to {output_dir / 'activations'}")
        print("="*60)
        return

    if args.viz_type == "temporal":
        print("\n" + "="*60)
        print("Temporal Trajectory Visualization")
        print("="*60)
        viz = TrajectoryVisualizer(args.checkpoint_dir, device)
        checkpoints = viz.load_checkpoints()
        checkpoints, epochs = viz._filter_checkpoints(checkpoints, epoch_min, epoch_max, epoch_stride)

        # Resolve epoch for game mode
        def _resolve_epoch_index(tag: str) -> int:
            if not epochs:
                return len(checkpoints) - 1
            if tag == 'min' and viz.history and viz.history.get('val_loss'):
                idx = np.argmin(viz.history['val_loss'])
                return viz._closest_epoch_index(epochs, idx + 1)
            if tag == 'final' and viz.history and viz.history.get('val_loss'):
                return viz._closest_epoch_index(epochs, len(viz.history['val_loss']))
            if tag == 'latest':
                return len(checkpoints) - 1
            if tag.isdigit():
                return viz._closest_epoch_index(epochs, int(tag))
            return len(checkpoints) - 1

        if not args.sequential_data:
            parser.error("--sequential-data is required for --viz-type temporal to use real game trajectories")
        ds_path = Path(args.sequential_data)
        if not ds_path.exists():
            parser.error(f"Sequential dataset not found: {ds_path}")
        seq_data = torch.load(ds_path, map_location='cpu')
        games = seq_data.get('games', [])
        if not games:
            parser.error(f"No games found in sequential dataset: {ds_path}")
        rng = np.random.default_rng(args.board_seed)
        game_idx = int(rng.integers(0, len(games)))
        game_entry = games[game_idx]
        states_tensor = game_entry['states']  # (seq_len, 3, 6, 7)
        # Cap by --time-steps if provided
        seq_len = int(states_tensor.shape[0])
        use_len = int(min(seq_len, max(1, args.time_steps))) if args.time_steps else seq_len
        # Build sequence of 1x3x6x7 tensors
        game_seq = [states_tensor[i].unsqueeze(0) for i in range(use_len)]

        if args.temporal_mode == 'training':
            # Fixed game step across epochs
            step_idx = args.time_step if args.time_step is not None else min(len(game_seq) - 1, max(0, args.time_steps // 2))
            board_t = game_seq[step_idx]
            reps = []
            for cp in checkpoints:
                model = create_model(viz.cnn_channels, viz.gru_hidden, viz.kernel_size)
                model.load_state_dict(cp['state_dict'])
                model.to(device)
                model.eval()
                with torch.no_grad():
                    _, _, h = model(board_t.to(device))
                reps.append(h.squeeze().cpu().numpy())
            reps = np.array(reps, dtype=np.float32)
            title = f"Hidden Trajectory across Training (fixed game step {step_idx})"
            fig, _ = viz.visualize_weight_trajectory(reps, title, save_path=output_dir / "temporal_training.png",
                                                     epochs=epochs,
                                                     phate_n_pca=None,
                                                     phate_knn=args.phate_knn,
                                                     phate_t=args.phate_t,
                                                     phate_decay=args.phate_decay,
                                                     use_tphate=args.t_phate,
                                                     tphate_alpha=args.t_phate_alpha,
                                                     tphate_delay=args.t_phate_delay,
                                                     tphate_lags=args.t_phate_lags,
                                                     tphate_kernel=args.t_phate_kernel,
                                                     tphate_kernel_alpha=args.t_phate_kernel_alpha,
                                                     tphate_kernel_tau=args.t_phate_kernel_tau)
            plt.close(fig)
        else:
            # Fixed epoch across game steps
            ep_idx = _resolve_epoch_index(str(args.time_epoch).lower())
            checkpoint = checkpoints[ep_idx]
            model = create_model(viz.cnn_channels, viz.gru_hidden, viz.kernel_size)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()
            reps = []
            with torch.no_grad():
                for bt in game_seq:
                    _, _, h = model(bt.to(device))
                    reps.append(h.squeeze().cpu().numpy())
            reps = np.array(reps, dtype=np.float32)
            steps = list(range(1, len(reps) + 1))
            title = f"Hidden Trajectory across Game (fixed epoch {epochs[ep_idx] if epochs else ep_idx+1})"
            fig, _ = viz.visualize_weight_trajectory(reps, title, save_path=output_dir / "temporal_game.png",
                                                     epochs=steps,
                                                     phate_n_pca=None,
                                                     phate_knn=args.phate_knn,
                                                     phate_t=args.phate_t,
                                                     phate_decay=args.phate_decay,
                                                     use_tphate=args.t_phate,
                                                     tphate_alpha=args.t_phate_alpha,
                                                     tphate_delay=args.t_phate_delay,
                                                     tphate_lags=args.t_phate_lags,
                                                     tphate_kernel=args.t_phate_kernel,
                                                     tphate_kernel_alpha=args.t_phate_kernel_alpha,
                                                     tphate_kernel_tau=args.t_phate_kernel_tau)
            plt.close(fig)

        print("\n" + "="*60)
        print(f"✓ Visualizations saved to {output_dir}")
        print("="*60)
        return

    if args.viz_type == "joint":
        print("\n" + "="*60)
        print("Joint CNN/GRU Trajectory")
        print("="*60)
        fig, _ = viz.visualize_joint_cnn_gru(
            save_path=output_dir / "cnn_gru_joint.png",
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            epoch_stride=epoch_stride,
            center_strategy=args.joint_center,
            phate_n_pca=args.phate_n_pca,
            phate_knn=args.phate_knn,
            phate_t=args.phate_t,
            phate_decay=args.phate_decay,
            use_tphate=args.t_phate,
            tphate_alpha=args.t_phate_alpha,
            tphate_delay=args.t_phate_delay,
            tphate_lags=args.t_phate_lags,
            tphate_kernel=args.t_phate_kernel,
            tphate_kernel_alpha=args.t_phate_kernel_alpha,
            tphate_kernel_tau=args.t_phate_kernel_tau
        )
        plt.close(fig)
        print("\n" + "="*60)
        print(f"✓ Visualizations saved to {output_dir}")
        print("="*60)
        return

    if args.viz_type in ["all", "cnn"]:
        print("\n" + "="*60)
        print("CNN Weight Trajectory")
        print("="*60)
        viz.visualize_cnn_trajectory(save_path=output_dir / "cnn_trajectory.png",
                                     epoch_min=epoch_min,
                                     epoch_max=epoch_max,
                                     epoch_stride=epoch_stride,
                                     phate_n_pca=args.phate_n_pca,
                                     phate_knn=args.phate_knn,
                                     phate_t=args.phate_t,
                                     phate_decay=args.phate_decay,
                                     use_tphate=args.t_phate,
                                     tphate_alpha=args.t_phate_alpha)
        plt.close()

    if args.viz_type in ["all", "gru"]:
        print("\n" + "="*60)
        print("GRU Weight Trajectory")
        print("="*60)
        viz.visualize_gru_trajectory(save_path=output_dir / "gru_trajectory.png",
                                     epoch_min=epoch_min,
                                     epoch_max=epoch_max,
                                     epoch_stride=epoch_stride,
                                     phate_n_pca=args.phate_n_pca,
                                     phate_knn=args.phate_knn,
                                     phate_t=args.phate_t,
                                     phate_decay=args.phate_decay,
                                     use_tphate=args.t_phate,
                                     tphate_alpha=args.t_phate_alpha)
        plt.close()

    if args.viz_type in ["all", "boards"]:
        print("\n" + "="*60)
        print("Board State Representations")
        print("="*60)
        boards = generate_random_boards(args.n_boards, seed=args.board_seed)
        checkpoint_path = viz.get_latest_checkpoint_path()
        viz.visualize_board_representations(boards, checkpoint_path,
                                           save_path=output_dir / "board_representations.png")
        plt.close()

    if args.viz_type in ["all", "summary"]:
        print("\n" + "="*60)
        print("Summary Visualization")
        print("="*60)
        fig = viz.create_summary_plot(epoch_min=epoch_min,
                                      epoch_max=epoch_max,
                                      epoch_stride=epoch_stride,
                                      phate_n_pca=args.phate_n_pca,
                                      phate_knn=args.phate_knn,
                                      phate_t=args.phate_t,
                                      phate_decay=args.phate_decay,
                                      use_tphate=args.t_phate,
                                      tphate_alpha=args.t_phate_alpha,
                                      tphate_delay=args.t_phate_delay,
                                      tphate_lags=args.t_phate_lags,
                                      tphate_kernel=args.t_phate_kernel,
                                      tphate_kernel_alpha=args.t_phate_kernel_alpha,
                                      tphate_kernel_tau=args.t_phate_kernel_tau)
        fig.savefig(output_dir / "summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("\n" + "="*60)
    print(f"✓ Visualizations saved to {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

"""
Simple ResNet + GRU model for Connect-4
Reads board states and predicts policy (moves) and value (win probability)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResNetGRUConnect4(nn.Module):
    """
    Connect-4 player: ResNet feature extractor + GRU for temporal modeling

    Architecture:
        Input: (batch, 3, 6, 7) - [yellow, red, turn] planes
        -> ResNet (single conv layer with configurable kernel) -> features
        -> Flatten + GRU -> hidden state
        -> Policy head: 7 move probabilities
        -> Value head: 1 win probability
    """

    def __init__(self, cnn_channels, gru_hidden_size, kernel_size=3):
        """
        Args:
            cnn_channels: Number of CNN output channels (16, 64, or 256)
            gru_hidden_size: GRU hidden dimension (8, 32, or 128)
            kernel_size: Conv kernel size (3 or 6)
        """
        super().__init__()

        self.cnn_channels = cnn_channels
        self.gru_hidden_size = gru_hidden_size
        self.kernel_size = kernel_size

        # Single ResNet block with 'same' padding to preserve spatial dimensions
        self.resnet = self._make_resnet_block(
            in_channels=3,  # Input: yellow, red, turn
            out_channels=cnn_channels,
            kernel_size=kernel_size
        )

        # Calculate flattened feature size: channels * height * width
        # Board is 6x7, padding preserves dimensions
        self.feature_size = cnn_channels * 6 * 7

        # GRU for temporal modeling (single layer)
        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=gru_hidden_size,
            batch_first=True
        )

        # Policy head: predicts move probabilities (7 columns)
        self.policy_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

        # Value head: predicts win probability [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(gru_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def _make_resnet_block(self, in_channels, out_channels, kernel_size):
        """Create a residual block with configurable kernel size (uses 'same' padding)"""
        return nn.Sequential(
            # First conv
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # Second conv
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, lengths=None, hidden=None):
        """
        Forward pass

        Args:
            x: Either (batch, 3, 6, 7) or (batch, seq_len, 3, 6, 7)
            lengths: Optional lengths tensor for packed GRU execution
            hidden: Optional GRU hidden state (for sequential prediction)

        Returns:
            policy_logits: (batch, seq_len, 7) or (batch, 7)
            value: (batch, seq_len, 1) or (batch, 1)
            hidden: GRU hidden state
        """
        single_timestep_input = False

        if x.dim() == 4:
            # Backward-compatibility path: treat as single timestep sequence
            single_timestep_input = True
            batch_size = x.shape[0]
            x = x.unsqueeze(1)
            lengths = None
        elif x.dim() == 5:
            batch_size = x.shape[0]
        else:
            raise ValueError("Expected input of shape (batch, 3, 6, 7) or (batch, seq_len, 3, 6, 7)")

        seq_len = x.shape[1]

        # Merge batch and time dimensions for CNN processing
        x_reshaped = x.view(batch_size * seq_len, 3, 6, 7)
        features = self.resnet(x_reshaped)  # (batch * seq, channels, 6, 7)
        features_flat = features.view(batch_size, seq_len, -1)

        # Optional packed GRU to skip padded timesteps
        if lengths is not None:
            packed = pack_padded_sequence(
                features_flat,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            packed_out, hidden = self.gru(packed, hidden)
            gru_out, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=seq_len
            )
        else:
            gru_out, hidden = self.gru(features_flat, hidden)

        # Heads operate per timestep; flatten then restore
        head_input = gru_out.contiguous().view(batch_size * seq_len, -1)
        policy_logits = self.policy_head(head_input).view(batch_size, seq_len, -1)
        value = self.value_head(head_input).view(batch_size, seq_len, -1)

        if single_timestep_input:
            policy_logits = policy_logits.squeeze(1)
            value = value.squeeze(1)

        return policy_logits, value, hidden

    def predict(self, x):
        """Convenience method for inference (no hidden state management)"""
        policy_logits, value, _ = self.forward(x)

        if policy_logits.dim() == 3:
            # Squeeze sequence dimension for single-step inference
            policy_logits = policy_logits.squeeze(1)
        policy_probs = torch.softmax(policy_logits, dim=-1)
        return policy_probs, value


def create_model(cnn_channels, gru_hidden_size, kernel_size=3):
    """Factory function to create model"""
    return ResNetGRUConnect4(cnn_channels, gru_hidden_size, kernel_size)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model with all 18 ablation configurations
    print("Model Architecture Tests - 18 Ablation Configurations")
    print("=" * 70)

    for kernel in [3, 6]:
        for channels in [16, 64, 256]:
            for gru_hidden in [8, 32, 128]:
                model = create_model(channels, gru_hidden, kernel)
                params = count_parameters(model)

                # Test forward pass
                batch_size = 4
                x = torch.randn(batch_size, 3, 6, 7)
                policy, value, hidden = model(x)

                print(f"\nKernel={kernel}x{kernel}, Channels={channels}, GRU={gru_hidden}")
                print(f"  Parameters: {params:,}")
                print(f"  Output shapes: policy={policy.shape}, value={value.shape}")

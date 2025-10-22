"""
Training script for Connect-4 ResNet+GRU model
Tracks weights every N epochs for analysis
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from model import create_model, count_parameters


class Connect4Dataset(Dataset):
    """PyTorch dataset for Connect-4 positions"""

    def __init__(self, data_path):
        """Load dataset from .pt file"""
        print(f"Loading dataset from {data_path}...")
        data = torch.load(data_path)

        self.states = data['states']
        self.policies = data['policies']
        self.values = data['values']
        self.metadata = data['metadata']

        print(f"  Loaded {len(self.states)} positions")
        print(f"  Metadata: {self.metadata.get('total_games', 'N/A')} games")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


def create_dataloaders(data_path, batch_size, train_split=0.9):
    """Create train and validation dataloaders"""
    dataset = Connect4Dataset(data_path)

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"Split: {train_size} train, {val_size} val")
    return train_loader, val_loader


def save_weights(model, epoch, save_dir):
    """Save model weights for trajectory analysis"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    weights_path = save_dir / f"weights_epoch_{epoch:04d}.pt"

    # Save full state dict
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }, weights_path)

    return weights_path


def train_epoch(model, train_loader, optimizer, device, policy_weight=1.0, value_weight=1.0):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    for states, target_policies, target_values in train_loader:
        states = states.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)

        # Forward pass
        pred_policy_logits, pred_values, _ = model(states)

        # Compute losses
        # Policy: cross-entropy with target distribution
        policy_loss = policy_criterion(pred_policy_logits, target_policies)

        # Value: MSE with target values
        value_loss = value_criterion(pred_values, target_values)

        # Combined loss
        loss = policy_weight * policy_loss + value_weight * value_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    # Average losses
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'policy_loss': total_policy_loss / n_batches,
        'value_loss': total_value_loss / n_batches,
    }


def validate(model, val_loader, device, policy_weight=1.0, value_weight=1.0):
    """Validate on validation set"""
    model.eval()

    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    with torch.no_grad():
        for states, target_policies, target_values in val_loader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            # Forward pass
            pred_policy_logits, pred_values, _ = model(states)

            # Compute losses
            policy_loss = policy_criterion(pred_policy_logits, target_policies)
            value_loss = value_criterion(pred_values, target_values)
            loss = policy_weight * policy_loss + value_weight * value_loss

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'policy_loss': total_policy_loss / n_batches,
        'value_loss': total_value_loss / n_batches,
    }


def train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs, save_every, checkpoint_dir,
    policy_weight=1.0, value_weight=1.0
):
    """Main training loop"""

    print(f"\nStarting training for {num_epochs} epochs")
    print(f"Saving weights every {save_every} epochs to {checkpoint_dir}")
    print("=" * 60)

    history = {
        'train_loss': [],
        'train_policy_loss': [],
        'train_value_loss': [],
        'val_loss': [],
        'val_policy_loss': [],
        'val_value_loss': [],
        'epochs_saved': []
    }

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            policy_weight, value_weight
        )

        # Validate
        val_metrics = validate(
            model, val_loader, device,
            policy_weight, value_weight
        )

        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_policy_loss'].append(train_metrics['policy_loss'])
        history['train_value_loss'].append(train_metrics['value_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_policy_loss'].append(val_metrics['policy_loss'])
        history['val_value_loss'].append(val_metrics['value_loss'])

        # Print progress
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Policy: {val_metrics['policy_loss']:.4f} | "
              f"Val Value: {val_metrics['value_loss']:.4f}")

        # Save weights every N epochs
        if epoch % save_every == 0:
            weights_path = save_weights(model, epoch, checkpoint_dir)
            history['epochs_saved'].append(epoch)
            print(f"  -> Saved weights to {weights_path.name}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = Path(checkpoint_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_path)

    # Save final weights
    final_path = save_weights(model, num_epochs, checkpoint_dir)
    history['epochs_saved'].append(num_epochs)
    print(f"\n✓ Final weights saved to {final_path.name}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train Connect-4 ResNet+GRU model")

    # Data
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")

    # Model architecture (for ablation study: 3x3x2=18 configs)
    parser.add_argument("--cnn-channels", type=int, default=64,
                        help="CNN channel size: 16, 64, or 256 (default: 64)")
    parser.add_argument("--gru-hidden", type=int, default=32,
                        help="GRU hidden size: 8, 32, or 128 (default: 32)")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Conv kernel size: 3 or 6 (default: 3)")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--policy-weight", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Value loss weight")

    # Weight tracking
    parser.add_argument("--save-every", type=int, default=10, help="Save weights every N epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = create_model(args.cnn_channels, args.gru_hidden, args.kernel_size)
    model = model.to(device)

    print(f"\nModel: Kernel={args.kernel_size}x{args.kernel_size}, Channels={args.cnn_channels}, GRU={args.gru_hidden}")
    print(f"Parameters: {count_parameters(model):,}")

    # Load data
    train_loader, val_loader = create_dataloaders(
        args.data, args.batch_size
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create checkpoint directory with config name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / f"k{args.kernel_size}_c{args.cnn_channels}_gru{args.gru_hidden}_{timestamp}"

    # Train
    history = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs,
        save_every=args.save_every,
        checkpoint_dir=checkpoint_dir,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight
    )

    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  History: {history_path}")


if __name__ == "__main__":
    main()

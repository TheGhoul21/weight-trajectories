"""
Training script for Connect-4 ResNet+GRU model
Tracks weights every N epochs for analysis
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import yaml
except ImportError:  # pragma: no cover -- optional dependency for config mode
    yaml = None

from model import create_model, count_parameters
from utils.repro import child_seed, seed_everything


class Connect4Dataset(Dataset):
    """PyTorch dataset for Connect-4 positions or full game sequences."""

    def __init__(self, data_path):
        """Load dataset from .pt file and detect format."""
        print(f"Loading dataset from {data_path}...")
        data = torch.load(data_path, weights_only=False)

        self.metadata = data.get('metadata', {})
        self.format = self.metadata.get('format', 'flat')

        if 'games' in data or self.format == 'sequential':
            self.format = 'sequential'
            raw_games = data['games']
            self.games = []

            for game in raw_games:
                states = game['states'].float()
                policies = game['policies'].float()
                values = game['values'].float()
                self.games.append((states, policies, values))

            print(f"  Loaded {len(self.games)} games (sequential format)")
            total_positions = sum(states.shape[0] for states, _, _ in self.games)
            print(f"  Total positions: {total_positions}")
        else:
            self.format = 'flat'
            self.states = data['states'].float()
            policies = data['policies']
            if policies.dim() > 1:
                policies = policies.argmax(dim=-1)
            self.policies = policies.long()
            self.values = data['values'].float()

            print(f"  Loaded {len(self.states)} positions (flat format)")
            print(f"  Metadata: {self.metadata.get('total_games', 'N/A')} games")

    def __len__(self):
        if self.format == 'sequential':
            return len(self.games)
        return len(self.states)

    def __getitem__(self, idx):
        if self.format == 'sequential':
            return self.games[idx]
        return self.states[idx], self.policies[idx], self.values[idx]


def sequential_collate_fn(batch):
    """Pad variable-length game sequences for batching."""
    states, policies, values = zip(*batch)

    lengths = torch.tensor([item.shape[0] for item in states], dtype=torch.long)
    batch_size = len(states)
    max_len = lengths.max().item()

    state_tensor = torch.zeros(batch_size, max_len, 3, 6, 7, dtype=states[0].dtype)
    policy_tensor = torch.zeros(batch_size, max_len, 7, dtype=policies[0].dtype)
    value_tensor = torch.zeros(batch_size, max_len, 1, dtype=values[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i in range(batch_size):
        seq_len = lengths[i].item()
        state_tensor[i, :seq_len] = states[i]
        policy_tensor[i, :seq_len] = policies[i]
        value_tensor[i, :seq_len] = values[i]
        mask[i, :seq_len] = True

    policy_targets = policy_tensor.argmax(dim=-1).long()
    value_targets = value_tensor.squeeze(-1)

    return state_tensor, policy_targets, value_targets, mask, lengths


def create_dataloaders(data_path, batch_size, train_split=0.9, seed=0, num_workers=2):
    """Create train and validation dataloaders"""
    dataset = Connect4Dataset(data_path)

    collate_fn = sequential_collate_fn if dataset.format == 'sequential' else None

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    split_seed = seed if seed is not None else 0
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed)
    )

    def _make_worker_init(base_seed: int | None):
        def _worker_init(worker_id: int) -> None:
            if base_seed is None:
                return
            worker_seed = base_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        return _worker_init

    train_gen = torch.Generator().manual_seed(child_seed(seed, 1) or 0)
    val_gen = torch.Generator().manual_seed(child_seed(seed, 2) or 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_make_worker_init(child_seed(seed, 10)),
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_make_worker_init(child_seed(seed, 20)),
        generator=val_gen,
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

    for batch in train_loader:
        if len(batch) == 5:
            states, target_policies, target_values, mask, lengths = batch
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)

            pred_policy_logits, pred_values, _ = model(states, lengths=lengths)
            if pred_values.dim() == 3:
                pred_values = pred_values.squeeze(-1)

            mask_flat = mask.view(-1)
            policy_logits_flat = pred_policy_logits.view(-1, pred_policy_logits.size(-1))
            policy_targets_flat = target_policies.view(-1)
            value_preds_flat = pred_values.view(-1)
            value_targets_flat = target_values.view(-1)

            policy_loss = policy_criterion(policy_logits_flat[mask_flat], policy_targets_flat[mask_flat])
            value_loss = value_criterion(value_preds_flat[mask_flat], value_targets_flat[mask_flat])
        else:
            states, target_policies, target_values = batch
            states = states.to(device)

            if target_policies.dtype != torch.long:
                if target_policies.dim() > 1:
                    target_policies = target_policies.argmax(dim=-1)
                target_policies = target_policies.long()
            target_policies = target_policies.to(device)

            target_values = target_values.to(device)

            pred_policy_logits, pred_values, _ = model(states)
            pred_values = pred_values.squeeze(-1)
            target_values = target_values.squeeze(-1)

            policy_loss = policy_criterion(pred_policy_logits, target_policies)
            value_loss = value_criterion(pred_values, target_values)

        loss = policy_weight * policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        for batch in val_loader:
            if len(batch) == 5:
                states, target_policies, target_values, mask, lengths = batch
                states = states.to(device)
                target_policies = target_policies.to(device)
                target_values = target_values.to(device)
                mask = mask.to(device)
                lengths = lengths.to(device)

                pred_policy_logits, pred_values, _ = model(states, lengths=lengths)
                if pred_values.dim() == 3:
                    pred_values = pred_values.squeeze(-1)

                mask_flat = mask.view(-1)
                policy_logits_flat = pred_policy_logits.view(-1, pred_policy_logits.size(-1))
                policy_targets_flat = target_policies.view(-1)
                value_preds_flat = pred_values.view(-1)
                value_targets_flat = target_values.view(-1)

                policy_loss = policy_criterion(policy_logits_flat[mask_flat], policy_targets_flat[mask_flat])
                value_loss = value_criterion(value_preds_flat[mask_flat], value_targets_flat[mask_flat])
            else:
                states, target_policies, target_values = batch
                states = states.to(device)

                if target_policies.dtype != torch.long:
                    if target_policies.dim() > 1:
                        target_policies = target_policies.argmax(dim=-1)
                    target_policies = target_policies.long()
                target_policies = target_policies.to(device)

                target_values = target_values.to(device)

                pred_policy_logits, pred_values, _ = model(states)
                pred_values = pred_values.squeeze(-1)
                target_values = target_values.squeeze(-1)

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
            best_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--data", type=str, help="Path to dataset .pt file")
    parser.add_argument("--config", type=str, help="YAML config file with training presets")
    parser.add_argument("--model", type=str, help="Model key inside the config file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")

    # Model architecture (for ablation study: 3x3x2=18 configs)
    parser.add_argument("--cnn-channels", type=int, nargs='+', default=[64],
                        help="CNN channel sizes to train (default: 64)")
    parser.add_argument("--gru-hidden", type=int, default=32,
                        help="GRU hidden size: 8, 32, or 128 (default: 32)")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Conv kernel size: 3 or 6 (default: 3)")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2) for optimizer (default: 0.0)")
    parser.add_argument("--policy-weight", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Value loss weight")

    # Weight tracking
    parser.add_argument("--save-every", type=int, default=10, help="Save weights every N epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducible training (default: 0)")

    args = parser.parse_args()

    if args.config:
        if yaml is None:
            parser.error("PyYAML is required for --config support. Install it with `pip install pyyaml`.")
        if not args.model:
            parser.error("--model must be specified when using --config")

        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}

        if args.model not in config_data:
            parser.error(f"Model '{args.model}' not found in {config_path}")

        def resolve_model_configs(model_key):
            if model_key not in config_data:
                parser.error(f"Model '{model_key}' not found in {config_path}")

            raw_cfg = config_data[model_key] or {}

            if isinstance(raw_cfg, dict) and 'includes' in raw_cfg:
                overrides = {k: v for k, v in raw_cfg.items() if k not in {'includes', 'name'}}
                resolved = []
                for child_key in raw_cfg['includes']:
                    for child_label, child_cfg in resolve_model_configs(child_key):
                        combined_cfg = dict(child_cfg)
                        combined_cfg.update(overrides)
                        resolved.append((child_label, combined_cfg))
                return resolved

            cfg_copy = dict(raw_cfg)
            label = cfg_copy.pop('name', model_key)
            return [(label, cfg_copy)]

        resolved_configs = resolve_model_configs(args.model)
        if not resolved_configs:
            parser.error(f"Model '{args.model}' did not resolve to any configurations")

        scheduled_runs = []

        default_data = args.data or config_data.get('data')

        for label, cfg in resolved_configs:
            cfg = dict(cfg)  # shallow copy to avoid accidental reuse

            data_path = cfg.pop('data', default_data)
            if data_path is None:
                parser.error(f"Dataset path must be provided via --data or the config file (missing for '{label}')")

            cnn_channels_list = cfg.pop('cnn_channels', args.cnn_channels)
            if isinstance(cnn_channels_list, int):
                cnn_channels_list = [cnn_channels_list]
            else:
                cnn_channels_list = list(cnn_channels_list)

            batch_size = cfg.pop('batch_size', args.batch_size)
            gru_hidden = cfg.pop('gru_hidden', args.gru_hidden)
            epochs = cfg.pop('epochs', args.epochs)
            lr = cfg.pop('lr', args.lr)
            weight_decay = cfg.pop('weight_decay', args.weight_decay)
            save_every = cfg.pop('save_every', args.save_every)
            kernel_size = cfg.pop('kernel_size', args.kernel_size)
            policy_weight = cfg.pop('policy_weight', args.policy_weight)
            value_weight = cfg.pop('value_weight', args.value_weight)

            if cfg:
                unknown = ', '.join(cfg.keys())
                print(f"[Warning] Unused config keys for '{label}': {unknown}")

            for channels in cnn_channels_list:
                scheduled_runs.append({
                    'label': label,
                    'data_path': data_path,
                    'cnn_channels': int(channels),
                    'batch_size': batch_size,
                    'gru_hidden': int(gru_hidden),
                    'epochs': int(epochs),
                    'lr': float(lr),
                    'weight_decay': float(weight_decay),
                    'save_every': int(save_every),
                    'kernel_size': int(kernel_size),
                    'policy_weight': float(policy_weight),
                    'value_weight': float(value_weight)
                })
    else:
        if args.data is None:
            parser.error("--data is required when not using --config")

        scheduled_runs = [{
            'label': 'default',
            'data_path': args.data,
            'cnn_channels': int(channels),
            'batch_size': args.batch_size,
            'gru_hidden': args.gru_hidden,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'save_every': args.save_every,
            'kernel_size': args.kernel_size,
            'policy_weight': args.policy_weight,
            'value_weight': args.value_weight
        } for channels in args.cnn_channels]

    base_seed = args.seed
    seed_everything(base_seed)

    for idx, run in enumerate(scheduled_runs):
        run_seed = child_seed(base_seed, idx)
        run['seed'] = run_seed if run_seed is not None else base_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    total_runs = len(scheduled_runs)

    for idx, run in enumerate(scheduled_runs, start=1):
        label = run['label']
        kernel_size = run['kernel_size']
        channels = run['cnn_channels']
        gru_hidden = run['gru_hidden']
        batch_size = run['batch_size']
        epochs = run['epochs']
        lr = run['lr']
        weight_decay = run['weight_decay']
        save_every = run['save_every']
        policy_weight = run['policy_weight']
        value_weight = run['value_weight']
        data_path = run['data_path']

        print("\n" + "=" * 70)
        print(f"Training configuration {idx}/{total_runs} ({label}): Kernel={kernel_size}x{kernel_size}, Channels={channels}, GRU={gru_hidden}")
        print("=" * 70)

        run_seed = int(run.get('seed', base_seed))
        seed_everything(run_seed)
        print(f"Seed: {run_seed}")

        model = create_model(channels, gru_hidden, kernel_size).to(device)
        print(f"Parameters: {count_parameters(model):,}")

        loader_seed = child_seed(run_seed, 100)
        train_loader, val_loader = create_dataloaders(
            data_path,
            batch_size,
            seed=loader_seed if loader_seed is not None else run_seed,
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(args.checkpoint_dir) / f"k{kernel_size}_c{channels}_gru{gru_hidden}_{timestamp}"

        history = train_model(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=epochs,
            save_every=save_every,
            checkpoint_dir=checkpoint_dir,
            policy_weight=policy_weight,
            value_weight=value_weight
        )

        history['seed'] = run_seed

        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n✓ Training complete!")
        print(f"  Checkpoints: {checkpoint_dir}")
        print(f"  History: {history_path}")

        results.append({
            'label': label,
            'kernel_size': kernel_size,
            'cnn_channels': channels,
            'gru_hidden': gru_hidden,
            'checkpoint_dir': str(checkpoint_dir),
            'history_path': str(history_path),
            'seed': run_seed,
        })

    if len(results) > 1:
        print("\nSummary of runs:")
        for res in results:
            print(f"  - [{res['label']}] k{res['kernel_size']}_c{res['cnn_channels']}_gru{res['gru_hidden']}: {res['checkpoint_dir']}")


if __name__ == "__main__":
    main()

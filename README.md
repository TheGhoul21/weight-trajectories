# Weight Trajectories: Connect-4 Neural Network Training

Training ResNet+GRU models on Connect-4 and tracking weight evolution during training.

## Project Structure

```
weight-trajectories/
├── dataset/
│   └── Alpha-Zero-algorithm-for-Connect-4-game/  # External AlphaZero repo
├── data/                                          # Generated datasets
│   ├── connect4_10k_games.pt                     # Full dataset
│   └── samples/                                  # Test datasets
├── src/                                           # Source code
│   ├── model.py                                   # ResNet+GRU architecture
│   └── train.py                                   # Training script
├── scripts/                                       # Utility scripts
│   ├── generate_connect4_dataset.py              # Dataset generation
│   └── train_all_models.sh                       # Train all configs
├── configs/                                       # Training configurations
│   └── train_configs.yaml
├── checkpoints/                                   # Saved models & weights
├── models/                                        # Final trained models
└── README.md
```

## Quick Start

### 0. Explore the CLI

All tooling now hangs off `wt.sh`. Run `./wt.sh help` for the command menu. Set `PYTHON_BIN` if you need a specific interpreter.

```bash
./wt.sh help
```

### 1. Generate Dataset

**Test run** (20 games, 2 CPUs, ~20 seconds):
```bash
./wt.sh dataset flat --test-run
```

**Full dataset** (10,000 games, 4 CPUs, ~1.5 hours):
```bash
./wt.sh dataset flat \
  --num-games 10000 \
  --cpus 4 \
  --simulations 200
```

### 2. Test Model Architecture

```bash
./wt.sh model
```

### 3. Train Models

**Single model:**
```bash
./wt.sh train \
  --data data/connect4_10k_games.pt \
  --cnn-channels 16 64 256 \
  --gru-hidden 32 \
  --epochs 100 \
  --save-every 10
```

**All three configurations:**
```bash
./wt.sh train-all --data data/connect4_10k_games.pt
```

## Model Architecture

**Input:** (batch, 3, 6, 7) - Board state [yellow_pieces, red_pieces, turn_indicator]

**ResNet Backbone:**
- Single ResNet block with configurable kernel size (3×3 or 6×6)
- Configurable output channels (16, 64, or 256)
- BatchNorm + ReLU, 'same' padding (preserves 6×7 spatial structure)

**GRU:**
- Single-layer GRU with configurable hidden size (8, 32, or 128)
- Processes flattened CNN features

**Heads:**
- **Policy Head:** Predicts move probabilities (7 columns)
- **Value Head:** Predicts win probability [-1, 1]

## Ablation Study: 18 Configurations

**Parameters varied:**
- Kernel size: 3×3, 6×6 (2 options)
- CNN channels: 16, 64, 256 (3 options)
- GRU hidden: 8, 32, 128 (3 options)
- **Total: 2 × 3 × 3 = 18 models**

**Parameter counts:**
| Kernel | Channels | GRU | Parameters |
|--------|----------|-----|------------|
| 3×3    | 16       | 8   | 20K        |
| 3×3    | 256      | 128 | 4.8M       |
| 6×6    | 16       | 8   | 29K        |
| 6×6    | 256      | 128 | 6.6M       |

## Weight Tracking

Weights are saved every 10 epochs by default:
```
checkpoints/k3_c64_gru32_20251022_033045/
├── weights_epoch_0010.pt
├── weights_epoch_0020.pt
├── ...
├── weights_epoch_0100.pt
├── best_model.pt
└── training_history.json
```

Each checkpoint contains the full model state dict for trajectory analysis.

**Training all 18 ablations:**
```bash
./scripts/train_all_18_ablations.sh
```

This will create 18 separate checkpoint directories, one for each configuration.

## Dataset Format

PyTorch `.pt` files containing:
```python
{
    'states': torch.FloatTensor,    # (N, 3, 6, 7) board states
    'policies': torch.FloatTensor,  # (N, 7) MCTS policy targets
    'values': torch.FloatTensor,    # (N, 1) game outcome values
    'metadata': dict                # Generation info
}
```

**Generation Parameters:**
- MCTS Simulations: 200-250 per move (high quality)
- Dirichlet Noise: Enabled for exploration
- Data Augmentation: Horizontal board flipping
- Source Model: Trained AlphaZero ResNet (ELO 1800+)

**Expected for 10,000 games:**
- Total positions: ~400K-500K (with augmentation)
- File size: ~200-300 MB
- Generation time: ~1.5 hours (5 CPUs, 250 sims)

## Training Output

Training logs show:
- Train/Val loss (combined policy + value)
- Policy loss (cross-entropy with MCTS targets)
- Value loss (MSE with game outcomes)

Example:
```
Epoch  10/100 | Train Loss: 1.2345 | Val Loss: 1.3456 | Val Policy: 1.1000 | Val Value: 0.2456
  -> Saved weights to weights_epoch_0010.pt
```

## Requirements

- Python 3.13+
- PyTorch 2.9+
- NumPy 2.3+
- See `dataset/Alpha-Zero-algorithm-for-Connect-4-game/pyproject.toml` for dependencies

## Citation

AlphaZero implementation:
- [Alpha-Zero-algorithm-for-Connect-4-game](https://github.com/Bruneton/Alpha-Zero-algorithm-for-Connect-4-game)
- Authors: Jean-Philippe Bruneton, Adèle Douin, Vincent Reverdy
- License: BSD 3-Clause

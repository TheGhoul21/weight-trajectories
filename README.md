# Weight Trajectories

A research framework for analyzing the learning dynamics of neural networks through weight trajectory analysis. Train ResNet+GRU models on Connect-4 and study how network representations evolve during learning.

## What is this?

This project trains neural networks on Connect-4 using AlphaZero-style self-play data, then analyzes how the network's internal representations change during training. We track:

- **Weight trajectories**: How parameters evolve through training
- **Representation dynamics**: How hidden states organize information over time
- **Observability analysis**: What information is accessible at different network layers
- **Mutual information**: How much game state information is preserved in representations

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/TheGhoul21/weight-trajectories.git
cd weight-trajectories

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Basic Workflow

```bash
# 1. See all available commands
./wt.sh help

# 2. Generate a small test dataset (20 games, ~20 seconds)
./wt.sh dataset flat --test-run

# 3. Test the model architecture
./wt.sh model

# 4. Train all model configurations
./wt.sh train-all --data data/connect4_10k_games.pt

# 5. Analyze the results
./wt.sh visualize --checkpoint checkpoints/k3_c16_gru8_*/
```

## Model Architecture

**ResNet+GRU hybrid:**
- ResNet backbone processes Connect-4 board states (6×7 grid)
- GRU layer maintains temporal/sequential information
- Dual heads predict move policy and position value

**Configurable parameters:**
- Kernel sizes: 3×3, 6×6
- CNN channels: 16, 64, 256
- GRU hidden size: 8, 32, 128

**Total configurations:** 18 models (2 kernels × 3 channels × 3 GRU sizes)

## Key Features

### Training & Data
- Generate datasets from AlphaZero self-play
- Train multiple model configurations in parallel
- Track weights at regular intervals for trajectory analysis
- Reproducible training with seed control

### Analysis Tools
- **Trajectory Embedding**: Visualize weight evolution in low-dimensional space (UMAP, t-SNE, PHATE)
- **CKA Similarity**: Compare representations across models and training stages
- **Observability Analysis**: Measure information flow through GRU hidden states
- **Mutual Information**: Quantify what information representations capture
- **Fixed-Point Analysis**: Study attractor dynamics in recurrent networks

### Visualization
- Training curves and metrics
- Weight trajectory plots
- Representation similarity matrices
- Gate activation patterns
- Temporal dynamics with T-PHATE

## Documentation

Comprehensive documentation is available in [`docs/`](docs/):

- [Getting Started Guide](docs/manual/getting_started.md)
- [Command Reference](docs/manual/commands/README.md)
- [Analysis Workflows](docs/manual/workflows/README.md)
- [Plots & Visualizations](docs/manual/plots/README.md)
- [Scientific Background](docs/scientific/README.md)

## Project Structure

```
weight-trajectories/
├── src/                    # Core source code
│   ├── model.py           # Neural network architectures
│   ├── train.py           # Training loop
│   └── utils/             # Utility modules
├── scripts/               # Analysis & visualization scripts
├── docs/                  # Full documentation
├── data/                  # Generated datasets
├── checkpoints/           # Training checkpoints & weight trajectories
├── wt.sh                  # Unified CLI entrypoint
└── pyproject.toml        # Project dependencies
```

## Example Commands

```bash
# Generate a full dataset (10k games, ~1.5 hours)
./wt.sh dataset flat --num-games 10000 --cpus 4

# Train a single model
./wt.sh train --data data/connect4_10k_games.pt \
  --cnn-channels 16 64 256 --gru-hidden 32 \
  --epochs 100 --save-every 10

# Compute checkpoint metrics
./wt.sh metrics --checkpoint checkpoints/k3_c64_gru32_*/

# Run GRU observability analysis
./wt.sh observability extract --checkpoint checkpoints/k3_c64_gru32_*/
./wt.sh observability analyze

# Compare models with CKA
./wt.sh cka wizard

# Generate analysis report
./wt.sh report --checkpoint checkpoints/k3_c64_gru32_*/
```

## Research Applications

This framework is designed for studying:

- **Learning dynamics**: How do representations emerge during training?
- **Architecture comparison**: Which designs learn more efficiently?
- **Information theory**: What information do networks preserve?
- **Recurrent dynamics**: How do GRUs process sequential game states?
- **Observability**: Can we predict final performance from early training?

## Requirements

- Python 3.10 or later
- PyTorch 2.2+
- See [pyproject.toml](pyproject.toml) for full dependency list

## Citation

AlphaZero implementation for dataset generation:
- [Alpha-Zero-algorithm-for-Connect-4-game](https://github.com/Bruneton/Alpha-Zero-algorithm-for-Connect-4-game)
- Authors: Jean-Philippe Bruneton, Adèle Douin, Vincent Reverdy
- License: BSD 3-Clause

## License

MIT License - see [LICENSE](LICENSE) for details.

This project is completely open and free. You can use, modify, and distribute it however you want.

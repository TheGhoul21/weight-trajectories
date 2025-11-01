# Weight Trajectories: Connect Four Neural Network Analysis
Version 1.0

> Comprehensive analysis framework for training and interpreting ResNet+GRU models on Connect Four, with focus on dynamical systems analysis of recurrent networks.

## Documentation Overview

Complete coverage of the **wt.sh** toolchain for:
- **Training** ResNet+GRU models on sequential Connect Four games
- **Analyzing** GRU dynamics using dynamical systems theory
- **Visualizing** weight trajectories and representation evolution
- **Interpreting** what the network learns through information theory and probing

## Documentation Sections

### [User Manual](manual/)
Complete practical guide to all commands, plots, and workflows
- [Commands Reference](manual/commands/) - All `./wt.sh` commands documented
- [Plot Interpretation](manual/plots/) - How to read every visualization
- [Conceptual Guides](manual/guides/) - Understanding analysis methods
  - [Reading Dimensionality Reduction](manual/guides/reading_dimensionality_reduction.md) - Interpreting PCA, PHATE, t-SNE, UMAP
- [Workflows](manual/workflows/) - End-to-end analysis pipelines

### [Scientific Background](scientific/)
Theoretical foundations and literature review
- [Theoretical Foundations](scientific/theoretical_foundations.md) - Dynamical systems, information theory, neuroscience
- [Case Studies](scientific/case_studies.md) - Examples from RNN interpretability research
- [References](scientific/references.md) - Comprehensive bibliography (50+ papers)

### [Technical Reference](reference/)
Architecture and shared methods
- [Architecture Diagrams](reference/architecture_diagrams.md) - How to generate model visualizations
- [Methods Reference](reference/methods.md) - Algorithms and library links

---

## Project Structure

```
weight-trajectories/
├── wt.sh                     # Unified CLI for training, analysis, and plotting
├── pyproject.toml            # Project dependencies and tooling configuration
├── src/                      # Core library: model, training loop, helpers
│   ├── model.py
│   ├── train.py
│   └── utils/
├── scripts/                  # Batch utilities for data generation and studies
│   ├── generate_connect4_dataset.py
│   ├── extract_gru_dynamics.py
│   ├── train_all_18_ablations.sh
│   └── run_visualization_suite.py
├── configs/                  # YAML/JSON configs for training and visualization
├── data/                     # Generated Connect Four datasets (`.pt` + metadata)
├── checkpoints/              # Saved model checkpoints and training logs
├── diagnostics/              # Derived metrics (MI sweeps, fixed points, etc.)
├── visualizations/           # Rendered plots and embedding exports
├── ablation_study/           # Ablation outputs and experiment manifests
├── dataset/                  # External AlphaZero Connect Four reference repository
├── docs/                     # User manual, scientific handbook, and references
├── README.md                 # Project overview (root)
├── USAGE.md                  # CLI quick reference
└── report.tex                # Research report manuscript
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

## Key Features

### Core Capabilities
- **18-configuration ablation study**: Systematic sweep over kernel sizes, CNN channels, and GRU hidden dimensions
- **Comprehensive observability**: Gate statistics, eigenvalue analysis, hidden state sampling
- **Advanced visualization**: PHATE trajectories, CKA similarity, Grad-CAM attention
- **Interpretability toolkit**: Mutual information analysis, linear probing, fixed-point finding

### Analysis Methods
- **Weight space**: Norms, SVD, step statistics, trajectory embeddings
- **GRU dynamics**: Gates, timescales, attractors, fixed points
- **Feature encoding**: MI per dimension, neuron specialization, probing
- **Cross-model**: CKA similarity, metric-space embedding, factorial analysis

### Documentation Depth
- **User manual**: 30+ command/plot guides with examples
- **Scientific theory**: 350+ lines on dynamical systems, information theory
- **Case studies**: 6 detailed examples from RNN interpretability literature
- **References**: 50+ papers with annotations

---

## Getting Started

### For Users: Run Your First Analysis

```bash
# 1. Generate dataset (test run: 20 games, ~20 seconds)
./wt.sh dataset flat --test-run

# 2. Train a model
./wt.sh train --data data/connect4_test.pt --epochs 30

# 3. Analyze GRU dynamics
./wt.sh observability extract    # Collect data
./wt.sh observability analyze    # Generate plots
```

See [User Manual](manual/) for complete guide.

### For Researchers: Understand the Theory

Start with [Scientific Background](scientific/):
1. [Theoretical Foundations](scientific/theoretical_foundations.md) - Core concepts
2. [Case Studies](scientific/case_studies.md) - Concrete examples
3. [GRU Observability Literature](scientific/gru_observability_literature.md) - Gap analysis and priorities

### For Developers: Architecture Details

Check [Technical Reference](reference/):
- [Architecture Diagrams](reference/architecture_diagrams.md) - Model structure
- [Methods Reference](reference/methods.md) - Algorithms and library links

---

## What Makes This Project Unique

### Research-Grade Interpretability
Unlike typical RL projects that stop at training, we provide:
- **Dynamical systems analysis**: Fixed points, attractors, stability
- **Information-theoretic probing**: What and how features are encoded
- **Mechanistic understanding**: Not just "it works" but "here's how"

### Comprehensive Documentation
Every plot explained, every algorithm detailed, every paper cited:
- **342-line workflow guide** for GRU interpretability
- **403-line reference** on mutual information analysis
- **500-line case study collection** from literature

### Bridging AI and Neuroscience
Methods inspired by computational neuroscience:
- Attractor network theory from brain research
- Fixed-point analysis from neural dynamics
- Information-theoretic principles from sensory coding

---

## Citation

### AlphaZero Implementation
- [Alpha-Zero-algorithm-for-Connect-4-game](https://github.com/Bruneton/Alpha-Zero-algorithm-for-Connect-4-game)
- Authors: Jean-Philippe Bruneton, Adèle Douin, Vincent Reverdy
- License: BSD 3-Clause

### Key References for Methods
- **Fixed points**: Sussillo & Barak (2013) - Neural Computation
- **PHATE**: Moon et al. (2019) - Nature Biotechnology
- **Mutual information**: Kraskov et al. (2004) - Physical Review E
- **Attractor networks**: Khona & Fiete (2022) - Nature Reviews Neuroscience

See [complete bibliography](scientific/references.md) for 50+ citations.

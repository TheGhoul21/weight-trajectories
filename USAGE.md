# Usage Guide

## Virtual Environment Setup

This project now has its own virtual environment with all dependencies installed.

**Activate the environment:**
```bash
source .venv/bin/activate
```

**Or run directly without activating:**
```bash
.venv/bin/python3 <script>
```

**Installed packages:**
- PyTorch 2.8.0
- NumPy 1.26.4
- Matplotlib 3.9.4
- PHATE 1.0.11 (for trajectory visualization)
- SciPy 1.13.1
- scikit-learn 1.6.1

## 1. Playing Against the AI

After training completes, play Connect-4 against your model:

```bash
# Using the virtual environment
.venv/bin/python3 src/play_game.py \
  --checkpoint checkpoints/k3_c16_gru8_20251023_052314/best_model.pt

# Or if activated
python src/play_game.py \
  --checkpoint checkpoints/k3_c16_gru8_20251023_052314/best_model.pt
```

**Options:**
- `--checkpoint`: Path to model checkpoint (required)
- `--ai-first`: Let AI play first (default: human plays first)

**During gameplay:**
- You play as YELLOW (Y) by default
- AI shows its thinking: win probability and move probabilities
- Enter column number 0-6 to drop your piece
- Game ends on win or draw

## 2. Visualizing Weight Trajectories

Create PHATE visualizations of how weights evolved during training:

### All Visualizations (Recommended)
```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --checkpoint-dir checkpoints/k3_c16_gru8_20251023_052314 \
  --output-dir visualizations/smallest_net
```

This creates:
- `cnn_trajectory.png` - CNN weight evolution through epochs
- `gru_trajectory.png` - GRU weight evolution through epochs
- `board_representations.png` - How model represents 100 random board states
- `summary.png` - 2x2 grid with trajectories + loss curves

### Individual Visualizations

**CNN weights only:**
```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --checkpoint-dir checkpoints/k3_c16_gru8_20251023_052314 \
  --viz-type cnn
```

**GRU weights only:**
```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --checkpoint-dir checkpoints/k3_c16_gru8_20251023_052314 \
  --viz-type gru
```

**Board representations:**
```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --checkpoint-dir checkpoints/k3_c16_gru8_20251023_052314 \
  --viz-type boards \
  --n-boards 200  # Increase number of random boards
```

**Summary plot:**
```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --checkpoint-dir checkpoints/k3_c16_gru8_20251023_052314 \
  --viz-type summary
```

## 3. Training Models

Training is done with the dataset venv:

```bash
dataset/Alpha-Zero-algorithm-for-Connect-4-game/.venv/bin/python3 src/train.py \
  --data data/connect4_10k_games.pt \
  --epochs 20 \
  --cnn-channels 16 \
  --gru-hidden 8 \
  --kernel-size 3 \
  --save-every 5
```

## Understanding PHATE Visualizations

**PHATE** (Potential of Heat-diffusion for Affinity-based Trajectory Embedding) is perfect for visualizing weight evolution because:

1. **Preserves trajectories** - Unlike t-SNE, it maintains temporal ordering
2. **Shows local + global structure** - See both fine details and overall progression
3. **Designed for high-dimensional data** - Your weight spaces have thousands of dimensions

**What to look for:**
- **Smooth trajectories** = Stable, consistent learning
- **Sharp turns** = Sudden changes in learning dynamics
- **Clustered endpoints** = Convergence to similar solutions
- **Long paths** = Exploring many different weight configurations
- **Short paths** = Quick convergence to a solution

## Tips

1. **Compare different architectures** by visualizing multiple checkpoint directories
2. **Look at early vs late training** - does the model explore more early on?
3. **Check if CNN and GRU evolve in sync** - do their trajectories have similar shapes?
4. **Board representations** show if the model learns meaningful position encodings

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

### Handling Large Checkpoints

High-capacity CNN ablations produce very wide weight vectors. The visualizer now auto-applies a compact PCA (capped by the number of checkpoints, defaulting to at most 32 dims) before PHATE when the flattened dimension exceeds 5k features. You can override it explicitly:

```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --checkpoint-dir checkpoints/k3_c256_gru8_20251024_011942 \
  --viz-type cnn \
  --phate-n-pca 16
```

Use a smaller `--phate-n-pca` (it will automatically clip to the maximum supported by the checkpoint count) if you still hit memory pressure. Setting `--phate-n-pca 0` is not allowed; omit the flag to stay on the auto setting.

### Making Crowded Plots Readable

When several runs collapse onto the same region, tweak PHATE's geometry directly:

- `--phate-knn 6` widens the local neighbourhood before diffusion (default adapts to checkpoint count).
- `--phate-t 20` lets diffusion run longer, smoothing noise and separating large-scale trends.
- `--phate-decay 100` softens the kernel tail; smaller values emphasise local structure.

Example:

```bash
.venv/bin/python3 src/visualize_trajectories.py \
  --viz-type ablation-cnn \
  --ablation-dirs checkpoints/k6_c16_gru8_20251024_085919 \
                    checkpoints/k6_c64_gru8_20251024_092656 \
                    checkpoints/k6_c256_gru8_20251024_104719 \
  --ablation-animate \
  --ablation-center normalize \
  --phate-n-pca 20 \
  --phate-knn 6 \
  --phate-t 18 \
  --phate-decay 80
```

Pair the tuning with `--ablation-center normalize` to anchor each curve at epoch zero and scale by path length so trajectories stop overlapping right at the origin.

### Batch Runs + Markdown Reports

Use `scripts/run_visualization_suite.py` to execute multiple visualization commands and stitch the outputs into a single report:

```bash
python scripts/run_visualization_suite.py \
  --config configs/visualization_suite_example.json \
  --report visualizations/latest_report.md
```

Each experiment entry in the JSON config supplies the exact CLI arguments (same format as above) and lists the images the script should embed. After running, the report file contains command summaries and inline previews so you can compare sweeps quickly.

## 3. Training Models

Training is done with the dataset venv:

```bash
dataset/Alpha-Zero-algorithm-for-Connect-4-game/.venv/bin/python3 src/train.py \
  --data data/connect4_10k_games.pt \
  --epochs 20 \
  --cnn-channels 16 \
  --gru-hidden 8 \
  --kernel-size 3 \
  --save-every 5 \
  --weight-decay 1e-4   # Optional L2 regularization via optimizer
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

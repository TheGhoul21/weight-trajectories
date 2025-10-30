# Weight and Representation Embeddings

Overview
- Embeds high‑dimensional weight snapshots and optional hidden‑state representations to visualize training dynamics and representation geometry.
- Useful for sanity checks and sensitivity analyses alongside PHATE‑based trajectory plots.

How to Generate
- `./wt.sh embeddings` (uses `scripts/analyze_weight_embeddings.py`)

Related Guides
- [Reading Dimensionality Reduction Plots](../guides/reading_dimensionality_reduction.md) - Comprehensive guide to interpreting PCA, PHATE, t-SNE, and UMAP visualizations with theoretical background and detailed explanations

## Purpose

Visualize training dynamics by embedding high-dimensional weight snapshots into 2D space. This reveals how parameters evolve during training, shows convergence patterns, and enables comparison across different training runs or architectures.

## Outputs

### Per-Run Visualizations
Saved under `visualizations/simple_embeddings/<model_name>/`

**Weight Trajectories**:
- `cnn_{method}.png` - 2D embedding of CNN (ResNet) weight snapshots
- `gru_{method}.png` - 2D embedding of GRU weight snapshots
- `all_{method}.png` - 2D embedding of all model weights combined
- Where `{method}` ∈ {`pca`, `tsne`, `umap`, `phate`}

**Board Representations** (when `--board-representations` enabled):
- `repr_{method}.png` - 2D embedding of GRU hidden states for sampled board positions
- Shows how the trained model clusters different game states in representation space

**Optional Exports**:
- `{component}_{method}.csv` - Embedding coordinates (with `--export-csv`)
- `{component}_{method}_anim.gif` - Sequential checkpoint reveal animation (with `--animate`)

### Cross-Run Comparisons
Saved under `visualizations/simple_embeddings/comparisons/` (when `--compare` enabled)

- `{component}_{method}_comparison.png` - Overlay of all selected runs
- `{component}_{method}_comparison_anim.gif` - Animated overlay (with `--animate`)

## Embedding Methods

Four dimensionality reduction techniques, each revealing different aspects of training dynamics:

### PCA (Principal Component Analysis)
- **What it shows**: Linear projection capturing maximum variance
- **Best for**: Overall drift direction, steady training progression
- **Characteristics**: Preserves global structure; straight trajectories indicate simple parameter movement
- **Limitation**: Can hide nonlinear manifolds and local clusters

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **What it shows**: Local neighborhood relationships
- **Best for**: Detecting abrupt transitions, regime changes
- **Characteristics**: Emphasizes local structure; sharp turns visible; stochastic (varies with seed)
- **Limitation**: Distorts global distances; less reliable for multi-run comparisons
- **Auto-tuning**: Perplexity adapts to sample count: `min(30, max(5, (n_samples-1)/3))`

### UMAP (Uniform Manifold Approximation and Projection)
- **What it shows**: Balance between local and global structure
- **Best for**: Multi-run comparisons, preserving trajectory topology
- **Characteristics**: Faster than t-SNE; more global structure preserved
- **Auto-tuning**: `n_neighbors = min(15, max(2, n_samples-1))`

### PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)
- **What it shows**: Progressive, smooth changes over time
- **Best for**: Sparse checkpoint sampling (few epochs), temporal continuity
- **Characteristics**: Smoothest trajectories; handles sparse data well
- **Auto-tuning**: `knn = max(2, min(5, n_samples-1))`
- **Why this choice**: See [Weight Trajectory Analysis with PHATE](../../scientific/weight_embeddings_theory.md)

## Visual Elements

### Weight Trajectory Plots
**Markers**:
- **Green circle** (●) - Starting checkpoint (earliest epoch)
- **Red star** (★) - Final checkpoint (latest epoch)
- **Colorbar** - Viridis colormap encoding epoch number (blue→yellow)

**Trajectory Line**:
- Connects checkpoints in temporal order
- Smooth gradient → steady training
- Sharp turns → learning phase transitions
- Loops → oscillatory dynamics or instability

**Optional Annotations** (with `--annotate`):
- Epoch labels appear every ~10 checkpoints
- Format: "Ep {epoch_number}"

### Board Representation Plots
**Markers**:
- Each point = one board position's GRU hidden state
- **Colorbar** - Plasma colormap encoding sample index (#1, #2, ...)

**Interpretation**:
- **Tight clusters** - Model groups similar positions coherently
- **Dispersed clouds** - Entangled representations, poor separation
- **Distinct islands** - Clear game-state categories (e.g., winning vs losing positions)

### Comparison Plots
**Multi-run overlay**:
- Each run gets a unique color from Tab20 colormap
- Trajectories plotted with distinct colors for easy differentiation
- Start/end markers shown for each run
- Legend identifies runs by model configuration (e.g., "k3 c64 gru32")

## Command Options

### Basic Usage
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --component cnn \
  --methods pca phate
```

### Required
- `--checkpoint-dirs DIR [DIR ...]` - One or more checkpoint directories
  - Multiple dirs enable `--compare` mode

### Component Selection
- `--component {cnn|gru|all}` - Which weights to analyze [default: cnn]
  - `cnn`: Only ResNet backbone parameters
  - `gru`: Only GRU parameters
  - `all`: All model parameters concatenated

### Embedding Methods
- `--methods METHOD [METHOD ...]` - Which embedding algorithms [default: pca tsne umap phate]
  - Choices: `pca`, `tsne`, `umap`, `phate`
  - Run multiple methods to compare visualizations

### Checkpoint Filtering
- `--epoch-min N` - Earliest epoch to include (inclusive)
- `--epoch-max N` - Latest epoch to include (inclusive)
- `--epoch-step N` - Stride for subsampling [default: 1]
  - Example: `--epoch-step 3` keeps every 3rd checkpoint
  - Useful for large checkpoint collections

### Architecture Filtering
- `--require-channel N` - Only analyze models with N CNN channels
- `--require-gru N` - Only analyze models with GRU hidden size N
- `--require-kernel N` - Only analyze models with kernel size N
- **Use case**: Compare same-architecture models with different seeds

### Visualization Options
- `--annotate` - Add epoch labels to plots
- `--export-csv` - Write embedding coordinates to CSV
- `--output-dir DIR` - Where to save outputs [default: visualizations/simple_embeddings]
- `--random-seed N` - Seed for stochastic methods (t-SNE, UMAP) [default: 0]

### Comparison Mode
- `--compare` - Create overlay plots comparing all runs
  - Requires multiple `--checkpoint-dirs`
  - Saves to `comparisons/` subdirectory

### Animation
- `--animate` - Generate GIF animations showing checkpoint-by-checkpoint reveal
- `--animate-fps N` - Animation frame rate [default: 4]
  - Higher FPS = faster animation

### Board Representations
- `--board-representations` - Also embed GRU hidden states for board positions
- `--board-source {random|dataset}` - Where to get boards [default: random]
  - `random`: Generate random valid Connect Four positions
  - `dataset`: Load from existing dataset file
- `--board-count N` - Number of boards to sample [default: 120]
- `--board-dataset PATH` - Dataset path when `--board-source=dataset`
  - Supports `.pt` (PyTorch) or `.json` formats
- `--board-seed N` - Random seed for board sampling [default: 0]
- `--representation-methods METHOD [...]` - Methods for board embeddings
  - Defaults to `--methods` if not specified

## Examples

### Quick sanity check (PCA only)
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --component all \
  --methods pca
```

### Compare multiple training runs
```bash
./wt.sh embeddings \
  --checkpoint-dirs \
    checkpoints/save_every_3/k3_c64_gru32 \
    checkpoints/save_every_3/k3_c64_gru64 \
    checkpoints/save_every_3/k3_c64_gru128 \
  --component gru \
  --methods phate \
  --compare \
  --annotate
```

### Generate presentation-ready animation
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --component cnn \
  --methods phate \
  --animate \
  --animate-fps 3 \
  --epoch-step 3
```

### Analyze board representations from dataset
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --board-representations \
  --board-source dataset \
  --board-dataset data/connect4_sequential_10k_games.pt \
  --board-count 200 \
  --methods umap phate
```

### Subsample checkpoints and export data
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --component all \
  --epoch-min 30 \
  --epoch-max 100 \
  --epoch-step 5 \
  --methods pca tsne \
  --export-csv \
  --annotate
```

## Interpretation Guide

### Weight Trajectories

**Steady Progress** (smooth, continuous path):
- Gradual color progression (blue → cyan → yellow)
- Indicates stable, monotonic training
- Example: Straight line in PCA = simple linear drift

**Phase Transitions** (sharp turns):
- Abrupt direction change in trajectory
- Possible causes:
  - Learning rate schedule change
  - Hitting validation minimum, then overfitting
  - Escaping local minimum
  - Mode collapse in adversarial training

**Oscillations** (loops, back-and-forth):
- Indicates:
  - High learning rate causing instability
  - Cycling between multiple attractors
  - Gradient noise dominating signal
- **Fix**: Reduce learning rate or increase batch size

**Convergence** (trajectory endpoints cluster):
- Final checkpoint position stabilizes
- Small movements near endpoint = converged
- Large movements = still learning or unstable

**Divergence** (endpoints far apart in multi-run comparison):
- Different random seeds lead to different solutions
- Common in non-convex optimization
- Check if performance differs (may converge to different-quality solutions)

### Component Comparison

**CNN vs GRU movement**:
- Measure start-to-end distance in embedding space
- Larger distance = more parameter change (more "learning")
- **Common pattern**: GRU changes more in early epochs (learning representation), CNN changes more later (fine-tuning features)

**Example interpretation**:
```
CNN trajectory: Short, straight line → minimal change, already well-initialized
GRU trajectory: Long, curved path → significant learning, adapting to task
→ Conclusion: Transfer learning working (pre-trained CNN, fresh GRU)
```

### Method-Specific Patterns

**PCA**:
- Straight line → low-dimensional parameter subspace (good!)
- Curved line → nonlinear trajectory (UMAP/PHATE reveal more)
- Overlapping trajectories → runs differ only in orthogonal subspace

**t-SNE**:
- Clusters → discrete training phases
- Disconnected regions → abrupt jumps (check what epoch)
- Varies with `--random-seed` → repeat with different seeds to verify

**UMAP/PHATE**:
- Smooth continuous path → gradual optimization
- Branches → alternative paths (multi-run comparison)
- Endpoint separation → converged to different solutions

### Board Representations

**Tight clusters**:
- Model has learned structured representation
- Similar positions mapped to nearby points
- Good sign of generalization

**Islands**:
- Distinct game-state categories (e.g., winning/losing, early/late game)
- Count islands: more = finer-grained state distinctions

**Dispersed cloud**:
- Poor representation learning
- Hidden states don't meaningfully encode board features
- **Possible issues**:
  - Insufficient training
  - Too small GRU (needs more capacity)
  - Task doesn't require recurrence (board fully observable)

**Gradients** (color progression):
- Continuous transition (e.g., opening → midgame → endgame)
- Model encodes temporal game phase

### Comparison Plots

**Convergent runs** (endpoints close):
- Different initializations reach same solution region
- Good sign of robust training

**Divergent runs** (endpoints far):
- Check performance: if similar accuracy, multiple good solutions exist
- If performance differs, some runs failed to converge

**Parallel trajectories** (same shape, different positions):
- Runs follow similar learning dynamics
- Differ only in initialization or random seed
- Indicates consistent training procedure

**Different shapes** (one curved, one straight):
- Different learning rates or schedules
- Different hyperparameters
- Compare training speed: which reaches endpoint faster?

## Common Use Cases

### 1. Sanity Check Training
**Goal**: Verify model is learning (not stuck)

**Method**:
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --methods pca \
  --component all
```

**What to check**:
- Trajectory moves from start (green) to end (red)
- Color progression is monotonic (blue → yellow)
- If trajectory is a single point → weights not updating (bug!)

### 2. Compare Training Runs
**Goal**: See if different architectures converge similarly

**Method**:
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru* \
  --methods phate \
  --component gru \
  --compare
```

**What to check**:
- Do larger GRUs (gru64, gru128) move further?
- Do all runs converge to similar region?
- Which architecture converges fastest (shortest trajectory)?

### 3. Diagnose Training Issues
**Goal**: Find when/why training went wrong

**Method**:
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/failed_run \
  --methods phate \
  --annotate \
  --animate
```

**What to check**:
- Annotated epochs: when did trajectory turn?
- Cross-reference with loss curves
- Animation: see exactly when dynamics changed

### 4. Presentation/Paper Figures
**Goal**: Create publication-quality trajectory visualizations

**Method**:
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --methods phate \
  --component all \
  --annotate \
  --animate \
  --animate-fps 3 \
  --epoch-step 2
```

**Output**: Use PNG for static figures, GIF for talks/posters

### 5. Validate Representation Learning
**Goal**: Check if GRU learns meaningful board encodings

**Method**:
```bash
./wt.sh embeddings \
  --checkpoint-dirs checkpoints/save_every_3/k3_c64_gru32 \
  --board-representations \
  --board-source dataset \
  --board-dataset data/connect4_test_positions.pt \
  --methods umap phate
```

**What to check**:
- Do winning positions cluster separately from losing?
- Are there distinct opening/midgame/endgame regions?
- Compare early vs late training checkpoints (representations improve?)

## Performance Notes

- **Fastest**: PCA (~seconds even for 100+ checkpoints)
- **Medium**: UMAP (~10-30 seconds)
- **Slower**: PHATE (~30-60 seconds), t-SNE (~1-2 minutes)
- **Memory**: ~500MB for 100 checkpoints, 10K-dimensional weights

**Optimization tips**:
- Use `--epoch-step N` to subsample if you have many checkpoints
- Run PCA first for quick feedback
- Use `--methods pca` only for experiments, full suite for final analysis
- `--board-count`: 100-200 is sufficient; more takes longer without much benefit

## Troubleshooting

**"t-SNE requires at least three snapshots"**:
- Need ≥3 checkpoints for t-SNE
- Use `--methods pca phate` if you only have 2

**"No checkpoints found"**:
- Check `--checkpoint-dirs` path is correct
- Ensure directory contains `weights_epoch_*.pt` files

**"UMAP import error"**:
- Install: `pip install umap-learn`
- Or use `--methods pca tsne phate`

**Trajectories overlap completely**:
- Runs are very similar (good!)
- Or: PCA not revealing structure (try PHATE/UMAP)

**Animations fail to save**:
- Check Pillow installed: `pip install pillow`
- Check disk space for GIF files (~1-5 MB each)

**Board representations don't cluster**:
- Model may not have learned structured representation
- Try later checkpoints (more training)
- Increase `--board-count` for better sampling
- Check if dataset has diverse positions

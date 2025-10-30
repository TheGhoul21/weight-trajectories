# Metric-Space Trajectory Embeddings

Overview
- Embeds training checkpoints into a shared 2D space using architecture‑agnostic metrics (loss, weight norms, representation quality) rather than raw weights.
- Reveals architectural families, regime changes, and overfitting/underfitting patterns across models.

How to Generate
- `./wt.sh trajectory-embedding [--method umap|tsne|phate|tphate]` (runs `scripts/visualize_trajectory_embedding.py`)
- T‑PHATE variant: `--method tphate --time-alpha 3` scales the epoch feature to emphasize temporal continuity

Related Guides
- [Reading Dimensionality Reduction Plots](../guides/reading_dimensionality_reduction.md) - Comprehensive guide to interpreting PCA, PHATE, t-SNE, and UMAP visualizations with theoretical background and detailed explanations

Why metric space vs raw weights
- Different architectures have incompatible parameter spaces; raw weights are not directly comparable.
- Shared metrics (norms, step statistics, losses, representation variance) are architecture‑agnostic and support joint embeddings across models.

Pros and cons
- Pros: Comparable across architectures; easy to compute from CSVs; reveals regime shifts and overfitting arcs.
- Cons: Abstracts away full representational content; embedding depends on feature selection and scaling.

Recommended usage
- Use UMAP as default; color by model, shape by GRU size, and size by epoch.
- Validate findings with PHATE/PCA sensitivity checks; inspect specific regimes by subsetting epochs.
 
See also: [Methods Reference](../../reference/methods.md).

## Purpose

Embed training trajectories into 2D space based on **metrics** (loss, weight norms, etc.) rather than raw weights. Reveals how different architectures explore the training dynamics landscape and whether they converge to similar metric profiles.

## Feature engineering pipeline

### 1. Data loading

For each of 9 models:
- Load `diagnostics/trajectory_analysis/<model>_metrics.csv` (weight/representation metrics)
- Load `checkpoints/save_every_3/<model>/training_history.json` (loss curves)

### 2. Feature matrix construction

For each checkpoint, construct feature vector from available metrics:

**Weight-space features** (from metrics CSV):
- `epoch`: Training epoch number
- `weight_norm`: L2 norm of weights
- `step_norm`: L2 distance from previous checkpoint
- `step_cosine`: Cosine similarity with previous checkpoint
- `relative_step`: Normalized step size

**Representation features** (if available):
- `repr_total_variance`: Sum of squared singular values
- `repr_top1_ratio`, `repr_top2_ratio`, ...: Variance concentration in top components

**Loss features** (from training history):
- `train_loss`: Combined training loss
- `val_loss`: Combined validation loss
- `train_val_gap`: `train_loss - val_loss` (overfitting signal)

**Missing values**: If a feature is unavailable, the entire column is dropped (architecture-agnostic)

**Result**: Feature matrix `F` of shape `(total_checkpoints, num_features)` where rows span all models/epochs

### 3. Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
F_scaled = scaler.fit_transform(F)  # Mean=0, Std=1 per feature
```

**Why standardize?**
- Features have different units (loss in [0,10], norms in [100,1000])
- Prevents high-magnitude features from dominating distance metrics
- Required for meaningful Euclidean distances in UMAP/t-SNE

### 4. Dimensionality reduction

Apply chosen embedding method to `F_scaled`:

**UMAP** (default):
```python
from umap import UMAP

reducer = UMAP(n_components=2, random_state=0)
embedding = reducer.fit_transform(F_scaled)
```
- Fast, preserves global structure
- Good for identifying architectural families

**t-SNE**:
```python
from sklearn.manifold import TSNE

reducer = TSNE(n_components=2, perplexity=30, random_state=0)
embedding = reducer.fit_transform(F_scaled)
```
- Emphasizes local neighborhoods
- Better for finding fine-grained training regimes

**PHATE**:
```python
import phate

reducer = phate.PHATE(n_components=2, knn=5, random_state=0)
embedding = reducer.fit_transform(F_scaled)
```
- Smoothest trajectories
- Best for temporal continuity

**T‑PHATE (metric‑space)**:
```bash
./wt.sh trajectory-embedding -- --method tphate --time-alpha 3
```
- Epoch feature scaled by `--time-alpha` before PHATE
- Emphasizes chronological smoothness across checkpoints

**Result**: Embedding matrix of shape `(total_checkpoints, 2)` with x/y coordinates

### 5. Visualization encoding

Each checkpoint plotted as a point with:
- **X/Y position**: Embedding coordinates
- **Color**: Epoch number (colormap: viridis)
- **Marker shape**: GRU size (○=32, □=64, △=128)
- **Marker edge color**: CNN channels (encoded as hue)
- **Point size**: Scales with epoch (larger = later training)
- **Lines**: Connect consecutive checkpoints within same model

## What it does
- Loads per-run metrics CSVs and training histories, builds an architecture-agnostic feature vector per checkpoint, standardizes features, then embeds all checkpoints from all 9 models into a shared 2D space.

## Inputs
- diagnostics/trajectory_analysis/k3_c{channels}_gru{hidden}_metrics.csv
- checkpoints/save_every_3/k3_c{channels}_gru{hidden}/training_history.json

## Outputs
- trajectory_embedding_all.png: All models overlaid; color is epoch, marker encodes GRU size (o,s,^), edge color encodes channels
- trajectory_embedding_by_gru.png: 1×3 facet by GRU size; lines/points per channel; colorbar = epoch
- trajectory_embedding_3d.png: Optional 3D embedding when available (PCA fallback)

Axes/encodings
- UMAP/t-SNE/PHATE component 1/2
- Point size scales with epoch to emphasize later training
- Model labels annotate start/end; facets show within-GRU channel effects

Features used (subset if missing)
- epoch, weight_norm, step_norm, step_cosine, relative_step
- repr_total_variance, repr_top1..4_ratio
- train_loss, val_loss, train_val_gap

Reading the plots
Interpreting the panels
- **All models overlay**
  - Colour = epoch (darker→lighter). Use it to spot which architectures lag or race ahead in feature space.
  - Marker shapes separate GRU sizes; edge colour encodes CNN channels. Look for grouped paths to see main effects.
- **Faceted by GRU**
  - Within each subplot, directly compare CNN capacities while holding GRU fixed. If paths align, CNN width has little effect on these metrics; divergence means the CNN is driving behaviour.
- **3D rendering**
  - Adds depth for non-linear methods; rotate in an external viewer if you need to inspect crossings.

Reading patterns
- **Compact clusters** → models share similar training dynamics; good when you expect invariance.
- **Outliers** → architecture-specific behaviour (e.g. GRU128 taking a long exploratory loop, signalling overfitting risk).
- **Loops/backtracking** → validation dips or LR schedule kicks; cross-reference with loss curves.
- **Straight-shot trajectories** → smooth convergence; often the best generalisers.

Tips
- You can limit epochs with `--epoch-min/--epoch-max/--epoch-step` to focus on salient phases (e.g. early vs late training).
- Try alternative metrics (add/remove columns in `create_feature_matrix`) when experimenting; the script auto-drops missing ones.
- Combine with `visualizations/gru_observability/` plots to explain why a particular trajectory segment behaves strangely.

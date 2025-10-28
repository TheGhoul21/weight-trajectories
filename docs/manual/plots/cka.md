# CKA similarity

Produced by: `./wt.sh cka [--representation gru|cnn]`
Backed by: `scripts/compute_cka_similarity.py`

## Purpose

Measure representational similarity between different model architectures using Centered Kernel Alignment (CKA). Determines whether models with different capacities learn similar or divergent representations of the same board positions.

## CKA computation

### 1. Data collection

For each model and epoch:
1. Load checkpoint weights
2. Sample `num_boards` (default: 64) random Connect Four positions
3. Forward-pass each board through the model
4. Extract representations:
   - **GRU**: Hidden state vectors (shape: `num_boards × gru_size`)
   - **CNN**: ResNet output features (shape: `num_boards × cnn_output_dim`)

**Result**: Representation matrices `X` and `Y` for each model pair

### 2. Linear CKA formula

**Centered Kernel Alignment** (Kornblith et al., 2019):

```python
# For representation matrices X (n×p) and Y (n×q)
# where n = num_boards, p/q = representation dimensions

# 1. Compute Gram matrices (linear kernel)
K = X @ X.T  # (n×n) — inner products between all pairs of boards in X space
L = Y @ Y.T  # (n×n) — inner products in Y space

# 2. Center the Gram matrices
H = I - (1/n) * ones(n, n)  # Centering matrix
K_c = H @ K @ H
L_c = H @ L @ H

# 3. Compute CKA
CKA(X, Y) = ||K_c ⊙ L_c||_F^2 / (||K_c||_F * ||L_c||_F)
          = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
```

Where:
- `⊙`: Element-wise product (Frobenius inner product)
- `||·||_F`: Frobenius norm `sqrt(sum(A^2))`

**Properties**:
- Range: [0, 1]
- CKA(X, X) = 1 (self-similarity)
- CKA(X, Y) = 0 if representations are orthogonal
- Invariant to:
  - Invertible linear transforms (rotation, scaling)
  - Orthonormal basis choice
- Sensitive to:
  - Feature alignment (do models encode same concepts?)
  - Representational geometry (similar clustering structure?)

### 3. Pairwise similarity matrix

Compute CKA for all 9×9 model pairs:
```python
for model_i in models:
    for model_j in models:
        X_i = extract_representations(model_i, boards)
        X_j = extract_representations(model_j, boards)
        similarity[i, j] = linear_CKA(X_i, X_j)
```

**Diagonal**: Always 1.0 (model compared to itself)
**Off-diagonal**: Cross-model similarity

### 4. Hierarchical clustering (optional)

Apply agglomerative clustering to reorder models:
```python
from scipy.cluster.hierarchy import linkage, dendrogram

# Compute linkage from dissimilarity (1 - CKA)
Z = linkage(1 - similarity, method='average')

# Reorder rows/columns to group similar models
```

**Result**: Heatmap with dendrogram showing architectural families

## What it does
- For chosen epochs, loads all 9 models, extracts either GRU hidden vectors or CNN feature maps on a fixed set of test boards, and computes linear CKA pairwise across models. Emits per-epoch heatmaps, clustered heatmaps, evolution curves, and CSV matrices. Optional animated heatmap across epochs.

Outputs (under `visualizations/<representation>/`)
- cka_<rep>_similarity_epoch_<E>.png: 9×9 heatmap (0..1), annotated per-cell
- cka_<rep>_clustered_epoch_<E>.png: Hierarchically clustered heatmap with dendrogram (requires SciPy)
- cka_<rep>_evolution.png: Line plots of selected model pairs across epochs
- cka_<rep>_heatmap_animation.{gif|mp4}: Optional animated heatmap (if --animate)
- cka_<rep>_similarity_epoch_<E>.csv: Numeric matrix for the heatmap

Axes/encodings
- Heatmaps: rows/cols are model ids (k3_c{channels}_gru{hidden}); color encodes CKA similarity
- Evolution: x-axis epoch; y-axis CKA similarity (0..1) for a set of illustrative pairs

Key options
- --epochs E1 E2 ... or --epoch-step to auto-generate [3,3+step,...,99,100]
- --representation [gru|cnn]
- --num-boards [default 64], --seed
- --device [cpu|cuda]
- --animate, --animate-fps, --animate-format [gif|mp4]

Reading the plots
- **Heatmaps**
  - Values in [0, 1]; 1 = identical representations on the sampled boards, 0 = orthogonal.
  - Block structure reveals factor effects: e.g. a GRU-size block with high CKA means different channel counts behave similarly when GRU size matches.
  - Off-diagonal lows highlight architectural combos that learn genuinely different features.
- **Clustered view**
  - Dendrogram groups models with similar representations; use it to summarise “families” of solutions.
  - Large branch distance between two leaves = representations diverge strongly.
- **Evolution plot**
  - Track specific pairs (e.g., best baseline vs ablation). Rising curves indicate convergence during training; falling curves show divergence.
- **Animation**
  - Helps spot phase changes: sudden brighten/dim blocks often align with architectural turning points (e.g., post-min-loss fine-tuning).

Tips
- Compare GRU vs CNN CKA: GRU similarity reflects memory dynamics; CNN similarity focuses on spatial feature extractors.
- Use --num-boards to balance compute vs stability—more boards smooths the estimate.
- Keep epochs aligned with checkpoints from `save_every_3`; for a quick scan, use `--epoch-step 9` to capture 3–30–60–90–100.

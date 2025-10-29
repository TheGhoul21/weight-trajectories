# Checkpoint Metrics (CSV)

Overview
- Computes weight/step/representation metrics per checkpoint to track training stability and capacity use.
- Supports quick comparisons across runs and feeds shared metric embeddings.

How to Generate
- `./wt.sh metrics` (writes `diagnostics/checkpoint_metrics/<run>_metrics.csv` via `scripts/compute_checkpoint_metrics.py`)

## Purpose

Compute diagnostic statistics across training checkpoints to track weight-space dynamics and representation quality. Produces tabular metrics for quick analysis of training stability, parameter drift, and hidden state capacity utilization.

## Data collection pipeline

### 1. Checkpoint loading
For each epoch:
1. Load `weights_epoch_XXX.pt` checkpoint file
2. Extract state_dict (model weights)
3. Parse epoch number from checkpoint metadata or filename

### 2. Weight extraction and aggregation
**Component filtering**:
```python
# Flatten weights by filtering state_dict keys
for name, tensor in state_dict.items():
    if "weight" not in name:
        continue  # Skip biases, norms, etc.
    if component == "cnn" and "resnet" not in name:
        continue
    if component == "gru" and "gru" not in name:
        continue
    weights.append(tensor.ravel())  # Flatten to 1D

weight_vector = np.concatenate(weights)  # Single vector per checkpoint
```

**Result**: Matrix of shape `(num_epochs, weight_dim)` where each row is one checkpoint's flattened weights

### 3. Weight-space metrics computation

**Weight norms** (L2 norm per checkpoint):
```python
weight_norm[i] = ||w_i||_2 = sqrt(sum(w_i^2))
```

**Step norms** (distance between consecutive checkpoints):
```python
delta[i] = w_{i+1} - w_i
step_norm[i] = ||delta[i]||_2
```

**Step cosine** (alignment between consecutive weight vectors):
```python
step_cosine[i] = (w_i · w_{i+1}) / (||w_i|| * ||w_{i+1}||)
```
- Values: [-1, +1]
- +1: perfectly aligned (moving in same direction)
- 0: orthogonal (exploring perpendicular subspace)
- -1: reversal (rare, indicates oscillation)

**Relative step** (scale-free step size):
```python
relative_step[i] = step_norm[i] / ||w_i||
```
- Normalizes by current weight magnitude
- Detects when steps become small relative to parameter scale

### 4. Representation SVD analysis (optional)

When `--board-source` is enabled, for each checkpoint:

**Board collection**:
- `random`: Generate N random Connect Four positions
- `dataset`: Sample N positions from provided dataset

**Forward pass**:
```python
for each board in boards:
    cnn_features = model.resnet(board)  # CNN output
    policy, value, hidden = model(board)  # Full forward
    representations.append(hidden.flatten())  # GRU hidden state
```

**Representation matrix**: `R` of shape `(num_boards, gru_hidden_size)`

**SVD computation**:
```python
# 1. Center the data
R_centered = R - R.mean(axis=0)  # Remove mean per dimension

# 2. Compute singular value decomposition
U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
# S contains singular values in descending order

# 3. Compute variance explained
total_variance = sum(S^2)  # Total sum of squared singular values
variance_ratio[k] = S[k]^2 / total_variance  # Fraction explained by k-th component
```

**Exported metrics**:
- `repr_total_variance`: `sum(S^2)` — total variance in hidden space
- `repr_top1_ratio`: `S[0]^2 / sum(S^2)` — fraction explained by top component
- `repr_top2_ratio`: `(S[0]^2 + S[1]^2) / sum(S^2)` — cumulative for top 2
- ... up to `--top-singular-values` components

**Interpretation**:
- `total_variance ↑`: Model using more representational capacity
- `top1_ratio → 1`: Rank-1 collapse (all hidden states lie on single line)
- `top1_ratio → 1/gru_size`: Uniformly distributed (healthy)
- `top1_ratio > 0.5`: Severe collapse, most information in one direction

## Columns
- epoch: Training epoch number (from checkpoint metadata or filename)
- weight_norm: L2 norm of the flattened weight vector for the selected component (cnn|gru|all)
- step_norm: L2 norm of the weight delta from previous checkpoint; blank for the first row
- step_cosine: Cosine similarity between consecutive weight vectors; blank for the first row
  - +1: perfectly aligned steps, 0: orthogonal, −1: reversal
- relative_step: step_norm / previous weight_norm (scale-free step size); blank for the first row
- repr_total_variance: Sum of squared singular values of hidden representations over a fixed board set (if enabled via --board-source)
- repr_topK_ratio columns (repr_top1_ratio, repr_top2_ratio, ...): Proportion of total variance captured by top-k singular components (k up to --top-singular-values)

How it’s computed
- Weights are flattened by filtering state_dict keys:
  - cnn: keys containing 'resnet' and 'weight'
  - gru: keys containing 'gru' and 'weight'
  - all: any key containing 'weight'
- If `--board-source` is random or dataset, representation stats are computed by forwarding a fixed set of boards through each checkpoint to collect GRU hidden vectors, then computing SVD on the centered matrix.

Typical expectations
- weight_norm usually grows over training, more for larger GRU; excessive growth often correlates with overfitting
- step_norm tends to be larger early, then decay
- step_cosine usually hovers positive; sharp sign flips can indicate oscillations
- relative_step declines as weights grow or learning rates decay
- repr_total_variance increases as the model uses more hidden capacity; declines or instability can indicate collapse
- repr_top1_ratio near 0.5+ implies severe representation collapse (dominant dimension)

Upstream knobs
- --component [cnn|gru|all]
- --epoch-min/--epoch-max/--epoch-step to sub-sample epochs
- --board-source [none|random|dataset], --board-count, --board-dataset, --board-seed
- --top-singular-values (how many repr_topK_ratio columns to include)

How to use it
- **Weight norms**
  - Plot `weight_norm` over epochs to check capacity growth; steep late increases often precede validation loss spikes.
  - Compare CNN vs GRU components: diverging slopes highlight which block is still changing.
- **Step statistics**
  - `step_norm` + `relative_step` show how aggressively weights move. Sudden spikes usually coincide with learning-rate restarts or instability.
  - `step_cosine` near +1 signals consistent updates; oscillations between + and − mean the optimiser is fighting the curvature.
- **Representation SVD metrics**
  - `repr_total_variance` ↑ means the hidden space is expanding; persistent drops can indicate collapse or saturation.
  - `repr_top1_ratio`/`repr_topK_ratio` close to 1 denote a low-rank embedding (bad); healthy models spread variance across many components.

Typical workflows
- **Early overfitting detection**: monitor `step_norm` and `relative_step`—if they flatten while `weight_norm` still rises, the optimiser is taking tiny steps but drifting into sharper minima.
- **Architecture comparisons**: aggregate `repr_total_variance` across models to quantify which GRU size uses more representational capacity.
- **Checkpoint selection**: pick epochs where `repr_top1_ratio` is low (diverse representations) and `step_cosine` remains stable.

# GRU Fixed Points and Evolution

Overview
- Identifies fixed points (equilibria) of the GRU for selected contexts, classifies stability, and tracks evolution across training.
- Connects learned computation to dynamical mechanisms (stable modes, decision boundaries) and their emergence during learning.

How to Generate
- Discovery: `./wt.sh observability fixed` (runs `scripts/find_gru_fixed_points.py`)
- Evolution: `./wt.sh observability evolve` (runs `scripts/analyze_fixed_point_evolution.py`)

## Purpose

Find and track GRU fixed points (attractors) across training. A fixed point h* satisfies `GRU(f, h*) ≈ h*` for a frozen CNN feature vector f. This reveals:
- **Attractor structure**: How many stable states exist per board context
- **Stability**: Whether attractors are strong (fast convergence) or weak (slow)
- **Evolution**: How attractors shift/appear/disappear during training

## Fixed-point finding algorithm (Discovery stage)

### 1. Context selection
Sample `max_contexts` board states from the dataset (default: 12). For each:
1. Extract board position from sequential games
2. Forward through CNN to get feature vector `f = resnet(board)`
3. Freeze `f` for all fixed-point searches on this context

**Why freeze CNN features?**
- Isolates GRU dynamics from board representation
- Makes fixed-point equation well-defined: `h = GRU(f, h)`

### 2. Optimization loop (per context, per restart)

**Goal**: Find hidden state h* such that `h_{t+1} = GRU(f, h_t) ≈ h_t`

**Method**: Adam optimization on residual loss
```python
h = Parameter(h_init)  # Learnable hidden state
optimizer = Adam([h], lr=0.05)

for iteration in range(max_iter):
    h_next = GRU(f, h)  # One GRU step
    loss = mean((h_next - h)^2)  # MSE residual
    loss.backward()
    optimizer.step()

    if loss < tolerance:
        break  # Converged to fixed point
```

**Initialization strategies** (default: 8 restarts):
- **Restart 0**: `h_init = zeros(gru_size)` — most common attractor
- **Restarts 1-7**: `h_init ~ N(0, 0.1^2 I)` — explore hidden space

**Convergence criteria**:
- Residual < `tolerance` (default: 1e-5)
- Max iterations reached (default: 400)
- Keep best h found across iterations

### 3. Deduplication
Merge fixed points that are near-duplicates (L2 distance < 1e-3):
```python
for candidate in candidates:
    if min(||candidate - existing|| for existing in unique) >= 1e-3:
        unique.append(candidate)
```

### 4. Stability classification

**Compute Jacobian** at fixed point h*:
```python
J = ∂GRU(f, h) / ∂h |_{h=h*}  # (gru_size × gru_size) matrix
```

**Eigenvalue analysis**:
```python
eigenvalues = eigvals(J)
spectral_radius = max(|λ|)  # Largest eigenvalue magnitude
```

**Classification**:
- **Stable** (`spectral_radius < 0.997`): Perturbations decay, attractor basin exists
- **Unstable** (`spectral_radius > 1.003`): Perturbations grow, repeller
- **Marginal** (`0.997 ≤ spectral_radius ≤ 1.003`): Neutral stability, boundary case

**Spectral radius interpretation**:
- `ρ = 0.5`: Very stable (perturbations decay by 50% per step)
- `ρ = 0.9`: Moderately stable (slow return to attractor)
- `ρ ≈ 1.0`: Marginally stable (long transients)
- `ρ > 1.0`: Unstable (diverges from fixed point)

### 5. Storage

For each epoch, save:
- `epoch_XXX_fixed_points.npz`:
  - `hidden`: Array of shape `(num_points, gru_size)` — fixed-point locations
  - `residual`: Final MSE residual for each point
  - `spectral_radius`: ρ(J) for each point
  - `classification`: ["stable", "unstable", "marginal"]
  - `context_index`: Which board context this point belongs to
  - `eigvals_real`, `eigvals_imag`: Full Jacobian eigenvalues

## Evolution analysis (Evolution stage)

Tracks how fixed points change across epochs.

### Metrics computed

**Classification counts**:
- Per context, per epoch: count of stable/marginal/unstable points
- Reveals when attractors emerge or vanish

**Spectral radius trends**:
- Track ρ over epochs for stable points
- Shows whether attractors strengthen (ρ decreases) or weaken (ρ → 1)

**Attractor drift**:
```python
# For each context, compute centroid of stable points
centroid[epoch] = mean(h* for all stable h* in context)

# L2 drift between consecutive epochs
drift[epoch] = ||centroid[epoch] - centroid[epoch-1]||
```
- Large drift → network reorganizing strategy for that context
- Small drift → converged attractor structure

## Inputs
- diagnostics/gru_fixed_points/<model>/epochs/epoch_XXX_fixed_points.npz
  - Arrays: hidden, residual, spectral_radius, classification, context_index, eigvals_real/imag
- diagnostics/gru_fixed_points/<model>/fixed_points_summary.csv
- diagnostics/gru_fixed_points/contexts_metadata.json (and contexts.pt)

Outputs (evolution)
- `<model>_classification_counts.png` – bars per epoch counting {stable, marginal, unstable} fixed points for each board context.
- `<model>_spectral_radius.png` – spectral radius over epochs for stable points; hue = context id.
- `<model>_attractor_drift.png` – L2 drift of stable-centroid locations per context between consecutive epochs.

What “contexts” mean
- During discovery we freeze a particular board position (taken from the sequential dataset) and feed its CNN feature vector into the GRU. Each sampled position is a **context**.
- Context indices are stable across runs: e.g. context 0 might be an empty board, context 4 a mid-game with a tactical threat, etc. See `contexts_metadata.json` for per-context features (piece counts, immediate wins, …).
- Comparing contexts shows how the GRU reacts to different strategic situations—opening, forcing win, defensive posture, etc.—without recomputing the entire dataset.

How to read the plots
- **Classification counts**
  - Rising stable bars ⇒ the network is carving out more attractors for that context (richer internal model).
  - Persistent unstable bars ⇒ the hidden dynamics are still bifurcating; often happens early in training or when tolerance is loose.
- **Spectral radius**
  - Values ≪ 1 mean the attractor is strongly stable (fast convergence).
  - Values creeping toward 1 indicate a “soft” attractor—the GRU retains information longer but is more sensitive to perturbations.
- **Attractor drift**
  - Drift ≈ 0 ⇒ the attractor has settled; no major internal reorganisation.
  - Spikes highlight epochs where the network reshapes its strategy for that context (often correlates with validation improvements or sudden overfitting).

Practical tips
- Use `--epoch-step` to sample fewer checkpoints when iterating takes too long (e.g. analyse every 3rd epoch first).
- Loosen `--tolerance` (e.g. `1e-2`) for quick exploratory runs, then tighten it for publication-quality numbers once you know interesting regions.
- Combine these plots with PHATE embeddings: contexts whose attractors move a lot usually correspond to regions where the hidden manifold reshapes.

Knobs (discovery)
- --max-contexts, --restarts, --max-iter, --tolerance
- --epoch-min/--epoch-max/--epoch-step to sub-sample checkpoints
- --device, --seed

Notes
- Fixed points are computed in GRU hidden space given a static CNN feature vector for a board context
- Deduplication removes nearby duplicates; residual threshold enforces quality

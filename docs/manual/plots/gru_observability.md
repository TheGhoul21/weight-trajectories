# GRU observability + probes

Produced by: `./wt.sh observability analyze` → `scripts/analyze_gru_observability_results.py`
Inputs from: `./wt.sh observability extract` → `diagnostics/gru_observability/<model>/`

## Purpose

Analyze GRU internal dynamics during training by extracting gate statistics, eigenvalue-based timescales, hidden state embeddings, and linear probing classifiers. This reveals how the recurrent component learns temporal structure, what features it encodes, and how memory mechanisms evolve over epochs.

## Two-stage pipeline

### Stage 1: Extract (data collection)
**Command**: `./wt.sh observability extract`
**Script**: `scripts/extract_gru_dynamics.py`

For each checkpoint, this stage:
1. **Loads the model** and extracts GRU weight matrices (W_hr, W_hz, W_hn for reset/update/new gates)
2. **Forward-passes game trajectories** from the dataset through the model
3. **Manually computes gate activations** at each timestep using raw GRU equations
4. **Accumulates gate statistics** (mean/std for update and reset gates)
5. **Computes eigenvalues** of the candidate recurrence matrix W_hn to derive timescales
6. **Reservoir samples** hidden states for later PHATE/probing (default: 1500 per epoch)
7. **Extracts board features** for each sampled state (12 features: move_index, piece counts, threats, etc.)

**Outputs per model**:
- `diagnostics/gru_observability/<model>/metrics.csv` — per-epoch gate means, timescales
- `diagnostics/gru_observability/<model>/unit_gate_stats.csv` — per-unit gate statistics
- `diagnostics/gru_observability/<model>/hidden_samples/epoch_XXX.npz` — hidden vectors + features
- `diagnostics/gru_observability/<model>/epoch_XXX_eigenvalues.npz` — eigenvalue arrays

### Stage 2: Analyze (aggregation and visualization)
**Command**: `./wt.sh observability analyze`
**Script**: `scripts/analyze_gru_observability_results.py`

Loads extracted data and generates:
1. **Gate trajectory plots** — mean update/reset over epochs for all models
2. **Timescale heatmap** — median τ at final epoch, aggregated by architecture
3. **PHATE embeddings** — 2D projection of hidden states colored by features
4. **Logistic regression probes** — train classifiers on hidden states to predict board features

## Outputs

### Figures
- **gate_mean_trajectories.png** – two stacked line plots (update mean, reset mean) per model across epochs; legend lists the nine architectures.
- **timescale_heatmap.png** – heatmap of the median GRU integration timescale at the final epoch, indexed by channels (rows) and GRU size (columns).
- **phate_epoch_XXX_<feature>.png** – 3×3 grid (one panel per model) of PHATE embeddings of hidden samples at epoch `XXX`; colour encodes the requested board feature (e.g. `move_index`).
- **probe_accuracy.png** – logistic-probe accuracy over epochs; hue differentiates features, line style differentiates GRU size.
- **probe_results.csv** – long-form table `[model, epoch, feature, accuracy, f1, channels, gru, kernel]` for spreadsheet work.

What to look for
- **Gate trajectories**
  - Update mean climbing toward 1 ⇒ the hidden state leans more on its previous value (longer memories).
  - Update mean collapsing toward 0 ⇒ rapid overwriting; expect this early in training or for under-capacity models.
  - Reset mean near 0 ⇒ gates shut off historical contribution; near 1 ⇒ the model keeps injecting past information.
  - Convergence/flattening pinpoints the epoch where dynamics stabilise.
- **Timescale heatmap**
  - Darker cells (higher values) indicate eigenvalues closer to the unit circle ⇒ slower dynamics / longer integration.
  - Compare across GRU sizes to see where extra capacity really buys you longer memory.
- **PHATE grids**
  - Smooth gradients across the embedding reveal hidden-state progression with the coloured feature (useful for `move_index`).
  - Distinct lobes or clusters for tactical features (`immediate_win_*`) imply the GRU has carved separate attractors for those situations.
  - Late-epoch spirals or sharp turns often coincide with overfitting.
- **Probe accuracy**
  - Upward trajectories show increasingly linearly-decodable features.
  - Plateaus signal saturation; if accuracy later drops, suspect representation drift/overfitting.

Axes & encodings
- Gate plots: x = epoch, y = mean gate value; one line per model.
- Heatmap: colour = median `1 / |log(|λ|)|` of the GRU candidate recurrence matrix.
- PHATE: axes are the first two PHATE components; colourbar = feature value.
- Probe accuracy: x = epoch, y = accuracy; hue = feature choice; linestyle = GRU size.

Interpretation tips
- Update mean ≈ 1 → “keep” behaviour; ≈ 0 → “write” behaviour.
- Median timescale ↑ → the GRU sustains information longer (good for planning ahead in Connect Four).
- PHATE clusters for discrete features (e.g. `current_player`) indicate clear separability in hidden space.
- Probe accuracy rising with epoch → hidden state grows more informative; sudden drops warn of degradation.

## Computational details

### Gate activation computation

Gates are computed manually to enable per-timestep aggregation. For each forward pass:

**GRU equations** (PyTorch convention):
```python
r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_t + b_hr)  # Reset gate
z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_t + b_hz)  # Update gate
n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_t + b_hn))  # Candidate
h_{t+1} = (1 - z_t) * n_t + z_t * h_t  # New hidden state
```

Where:
- `x_t`: CNN feature vector for board at timestep t (shape: cnn_output_dim)
- `h_t`: GRU hidden state (shape: gru_hidden_size)
- `W_*`: Weight matrices extracted from `gru.weight_{ih|hh}_l0`
- Gates are accumulated across all timesteps using `GateAccumulator` class

**Aggregation**:
- Mean: `sum(gate_values) / (n_steps * hidden_size)`
- Std: `sqrt(sum(gate^2) / count - mean^2)` (online variance formula)
- Per-unit stats computed separately for each hidden dimension

**Reservoir sampling** (Vitter's Algorithm R):
- Maintain pool of `sample_hidden` states (default 1500)
- First 1500 states: keep all
- After that: each new state has probability `1500/total_steps` of replacing a random existing sample
- Ensures uniform sampling without storing all states

### Timescale computation

**Goal**: Measure how long the GRU integrates information (memory depth)

**Method**: Eigenvalue analysis of candidate recurrence matrix W_hn
```python
eigenvalues = np.linalg.eigvals(W_hn)  # Complex eigenvalues
abs_eig = np.abs(eigenvalues)  # Magnitude in complex plane

# Timescale τ: number of steps for eigenmode to decay by factor e
# Derived from continuous-time RNN theory (Jordan et al. 2019)
tau = 1 / |log(|λ|)|  # For eigenvalues λ with 0 < |λ| < 1
```

**Filtering**:
- Exclude eigenvalues near 0 (|λ| < 1e-6): irrelevant modes
- Exclude eigenvalues near 1 (|λ| > 1 - 1e-6): to avoid log(0) instability
- Only stable modes (|λ| < 1) contribute to timescale

**Statistics exported**:
- `max_abs_eigen`: Largest eigenvalue magnitude (should be <1 for stability)
- `median_abs_eigen`: Median eigenvalue magnitude
- `median_tau`: Median timescale across all valid eigenvalues
- `max_tau`: Maximum timescale (slowest mode)

**Interpretation**:
- τ ≈ 2: hidden state forgets in ~2 steps (short memory)
- τ ≈ 10: integrates ~10 moves (medium memory)
- τ > 20: long-term dependencies retained

### PHATE embedding

**Goal**: Visualize how hidden states cluster by board features

**Method**: PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)
- Constructs k-nearest-neighbor graph (adaptive knn = min(5, max(2, n_samples-1)))
- Computes diffusion operator via graph Laplacian
- Embeds into 2D preserving smooth manifold structure

**Preprocessing**:
1. Load hidden states from epoch_XXX.npz (N × gru_size array)
2. Remove NaN/Inf rows
3. Deduplicate exact duplicate states (avoids zero-distance issues)
4. Subsample to `--max-hidden-samples` (default 2000) if needed

**Color encoding**:
- Extract feature vector (e.g., move_index) for each hidden state
- Map feature value to viridis colormap
- Smooth gradients indicate continuous feature encoding
- Distinct clusters indicate categorical feature separation

**Why PHATE over t-SNE/UMAP**:
- Better preserves global structure for sparse samples
- Smoother embeddings (less sensitive to noise)
- Designed for trajectory/sequential data

### Logistic regression probing

**Goal**: Measure how **linearly decodable** features are from hidden states

**Method**: Train binary logistic classifier to predict feature from h_t
```python
X_train, X_test = hidden_states, hidden_states  # (N, gru_size)
y_train, y_test = feature_values  # (N,) binary labels

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
f1 = f1_score(y_test, clf.predict(X_test))
```

**Feature preparation**:
- `current_player`: Transform from [1,2] to [0,1]
- `immediate_win_*`: Binary [0,1] as-is
- `three_in_row_*`: Binarize (0 vs ≥1 threats)
- Features must have ≥2 samples per class
- 70/30 train/test split with stratification

**Interpretation**:
- **Accuracy > 0.9**: Feature is strongly encoded (linear separator exists)
- **Accuracy 0.7-0.9**: Moderate encoding (weakly entangled)
- **Accuracy ≈ 0.5**: No linear encoding (random chance for binary)
- **Rising accuracy over epochs**: Model learning to represent feature
- **Plateau then drop**: Overfitting or representation collapse

## Knobs

### Extract stage
- `--checkpoint-dir`: Base directory with model subdirectories [default: checkpoints/save_every_3]
- `--dataset`: Sequential games dataset path [default: data/connect4_sequential_10k_games.pt]
- `--max-games`: Number of games to forward-pass per checkpoint [default: 256]
- `--sample-hidden`: Reservoir sample size per epoch [default: 1500]
- `--device`: cpu or cuda [default: cpu]
- `--seed`: Random seed for reproducible sampling [default: 0]

### Analyze stage
- `--embedding-epochs`: Epochs to visualize in PHATE [default: 3 30 60 100]
- `--embedding-feature`: Feature name for PHATE coloring [default: move_index]
- `--probe-epochs`: Epochs for logistic regression [default: 30 60 100]
- `--probe-features`: Features to probe [default: current_player immediate_win_current immediate_win_opponent]
- `--max-hidden-samples`: Subsample hidden states for embeddings/probes [default: 2000]
- `--palette`: Seaborn color palette for multi-model plots [default: Set2]
- `--skip-embedding`: Skip PHATE plots (faster iteration)
- `--skip-probing`: Skip logistic regression probes

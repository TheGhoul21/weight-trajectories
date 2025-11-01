# GRU Observability

Overview
- Characterizes GRU computation over training via gate behavior, memory timescales, hidden‑state geometry, and linear decodability of task variables.
- Answers: What is remembered, how long, and which features are encoded; when do strategic modes emerge.

How to Generate
- Data collection: `./wt.sh observability extract` (writes to `diagnostics/gru_observability/<model>/`)
- Analysis and plots: `./wt.sh observability analyze` (runs `scripts/analyze_gru_observability_results.py`)
 
See also: [Methods Reference](../../reference/methods.md) for mutual information, probes, PHATE, and Grad‑CAM library links.

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
3. **PHATE embeddings** — 2D projection of hidden states colored by features. Static per‑epoch PNGs are produced by default; optionally, an animation over epochs can be rendered with `--embedding-animate`. Use `--embedding-mode joint` to pool samples and fit a single PHATE per model for stable axes across epochs (resource‑capped), or `--embedding-mode separate` (default) for faster per‑epoch fits with feature‑anchored orientation alignment. The grid is sorted by (kernel, channels, GRU).
4. **Logistic regression probes with control tasks** — train classifiers on hidden states to predict board features, with permuted-label controls to validate signal significance; metrics include accuracy, F1, balanced accuracy, ROC-AUC, Average Precision, and signal-over-control variants.

Best vs final epoch handling
- The analysis automatically detects each model’s best validation-loss epoch from `checkpoints/<model>/training_history.json` (using `epochs_saved` indices).
- Probe visualizations include the requested probe epochs, plus each model’s own best and final saved epochs.
- Confusion matrices are produced for: (a) the final probed epoch, and (b) an aggregate across models at their individual best epochs.

## Outputs

### Figures
- **gate_mean_trajectories.png** – two stacked line plots (update mean, reset mean) per model across epochs; legend lists the nine architectures.
- **timescale_heatmap.png** – heatmap of the median GRU integration timescale at the final epoch, indexed by channels (rows) and GRU size (columns).
- **phate_epoch_XXX_<feature>.png** – 3×3 grid (one panel per model) of PHATE embeddings of hidden samples at epoch `XXX`; colour encodes the requested board feature (e.g. `move_index`).
   - When `--embedding-animate` is set, also writes **phate_animation_<feature>.mp4** (or .gif). Colorbar is placed to the right to avoid overlap with the last column.
### PHATE rendering controls and sparse panels

It’s common for very early epochs to show few points in some panels. This is expected: many hidden states can be identical when the GRU is still untrained. The analyzer deduplicates identical rows by default (`--embedding-dedup auto`) to avoid numerical issues in PHATE with zero distances, which can make early panels appear sparse. If you specifically want to visualize all samples (including duplicates), run with `--embedding-dedup off`.

You can adjust scatter appearance:
- `--embedding-point-size 24` (larger markers; default 12)
- `--embedding-alpha 0.9` (more opaque; default 0.8)

Axes stability across epochs:
- `--embedding-mode joint` pools a capped number of samples per epoch and fits a single PHATE per model; this ensures consistent axes/signs across the animation.
- `--embedding-mode separate` fits per-epoch; the script stabilizes orientation by correlating one PHATE axis with the colouring feature and fixing sign accordingly, but residual flips are still theoretically possible when the feature is weakly correlated.

Animation format:
- The tool prefers MP4 if ffmpeg is available; otherwise it falls back to GIF automatically.
- If you use uv (pip), installing a bundled ffmpeg is easy: `uv pip install imageio-ffmpeg`. The script auto-detects it.
- Force a specific format with `--embedding-format mp4|gif`.

- **probe_accuracy.png** – facet-by-feature plot; each subplot shows accuracy over epochs for all models (one line per model) with a shared legend on top. Much more readable than a single crowded panel.
- **probe_signal_over_control.png** – difference between real and control task accuracy; positive values indicate genuine signal above chance.
- **probe_comparison.png** – side-by-side comparison of real probe accuracy (left) vs control task accuracy with permuted labels (right).
- **probe_balanced_accuracy.png** – balanced (macro-averaged) accuracy over epochs; robust to heavy class imbalance (e.g., rare immediate wins).
- **probe_balanced_signal_over_control.png** – balanced accuracy minus control balanced accuracy; highlights genuine signal under skew.
- **probe_confusion_matrices_epoch_XXX.png** – confusion matrices aggregated across models at the final probed epoch `XXX` (per feature panels).
- **probe_feature_summary_epoch_XXX.png** – for the same final epoch: bar summaries (overall/+/-) above the per-feature confusion matrices.
- **probe_score_density_epoch_XXX.png** – KDE score densities (positive vs negative) at the final probed epoch.
- **probe_confusion_matrices_best.png** – confusion matrices aggregated across models, each at its own best validation-loss epoch (per feature panels).
- **probe_feature_summary_best.png** – bar summaries (overall/+/-) and confusion matrices for the best-epoch aggregation.
- **probe_score_density_best.png** – KDE score densities (positive vs negative) for the best-epoch aggregation.
- **probe_results.csv** – long-form table `[model, epoch, feature, component, accuracy, f1, control_accuracy, signal_over_control, channels, gru, kernel]`; GRU results write to the output directory (or `gru/` when multiple components are requested), while CNN-specific probes land under `cnn/`.
   - Additional columns: `balanced_accuracy`, `roc_auc`, `average_precision`, `majority_baseline`, `adjusted_accuracy`, `control_balanced_accuracy`, `control_roc_auc`, `control_average_precision`, `balanced_signal_over_control`.

## Interpretation guide

### Gate trajectories (gate_mean_trajectories.png)

**What it shows**: Two stacked line plots showing mean update gate (top) and reset gate (bottom) values over training for all 9 models.

**Axes**:
- X: Epoch number
- Y: Mean gate activation (0 to 1)
- Lines: One per model (9 total), color-coded by palette

**Reading gate values**:

**Update gate (z_t)** - Controls "keep vs write":
- **z ≈ 1**: "Keep mode" - hidden state retains previous value
  - `h_next = z * h_prev + (1-z) * candidate ≈ h_prev`
  - Long memory, slow adaptation
- **z ≈ 0**: "Write mode" - hidden state replaced by new candidate
  - `h_next ≈ candidate`
  - Short memory, fast adaptation
- **Typical evolution**:
  - Early training (epochs 0-30): z ≈ 0.3-0.5 (learning from scratch)
  - Mid training (epochs 30-70): z ≈ 0.5-0.7 (stabilizing)
  - Late training (epochs 70-100): z ≈ 0.6-0.8 (converged)

**Reset gate (r_t)** - Controls access to previous hidden state:
- **r ≈ 1**: Full access to h_prev for computing candidate
- **r ≈ 0**: Shuts off h_prev, compute candidate from input only
- **Typical range**: 0.4-0.7 (partial mixing)

**Common patterns**:

1. **Healthy training** (expected):
   ```
   Update gate: Gradual rise from 0.4 → 0.7 over 100 epochs
   Reset gate: Stable around 0.5-0.6
   ```
   - Indicates: GRU learning to integrate temporal context

2. **Memoryless GRU** (problem):
   ```
   Update gate: Stuck near 0.2-0.3 throughout training
   Reset gate: Drops to 0.1-0.2
   ```
   - Diagnosis: GRU not learning recurrence (behaving like feedforward)
   - Possible causes: Task doesn't require memory, GRU too small, learning rate too high
   - Fix: Check if task actually needs recurrence; try larger GRU; reduce LR

3. **Over-memorization** (problem):
   ```
   Update gate: Climbs to 0.9+ by epoch 50
   Reset gate: Near 1.0
   ```
   - Diagnosis: GRU holding onto stale information too long
   - Possible causes: Overfitting, insufficient regularization
   - Fix: Add dropout, reduce capacity, check validation performance

4. **Divergent architectures** (observation):
   ```
   GRU32 models: Update gate plateaus at 0.55
   GRU128 models: Update gate reaches 0.75
   ```
   - Indicates: Larger GRUs develop longer memory (expected)
   - Cross-check: Does GRU128 actually perform better? (training_history.json)

**Convergence indicators**:
- **Flattening curves**: Gates stabilize → training converged
- **Epoch of flattening**: When dynamics settle (typically epoch 60-80)
- **Continued drift**: Gates still changing at epoch 100 → not fully converged

### Timescale heatmap (timescale_heatmap.png)

**What it shows**: Heatmap of median integration timescale (τ) at final epoch, organized by architecture.

**Axes**:
- Rows: CNN channels (32, 64, 128)
- Columns: GRU hidden size (32, 64, 128)
- Color: Median timescale τ (darker = longer memory)

**Timescale interpretation**:
- **τ = 2-5**: Very short memory (forgets in 2-5 steps)
  - Suitable for: Reactive policies, immediate tactics
  - Example: Only remembers last 2-5 moves
- **τ = 5-10**: Medium memory
  - Suitable for: Multi-move planning, short sequences
  - Example: Remembers ~10 moves (early/mid game in Connect Four)
- **τ = 10-20**: Long memory
  - Suitable for: Strategic planning, long dependencies
  - Example: Integrates full game context (~20+ moves)
- **τ > 20**: Very long memory (potentially unstable if τ → ∞)

**Reading the heatmap**:

**Expected pattern**:
```
         GRU32   GRU64   GRU128
C32      5.2     6.8     8.1
C64      5.5     7.2     9.3
C128     5.8     7.6     10.5
```
- Timescale increases with GRU size (more capacity → longer memory)
- Timescale slightly increases with CNN size (richer inputs → more to remember)

**Diagnostic patterns**:

1. **Uniform low timescales** (τ < 3 everywhere):
   ```
         GRU32   GRU64   GRU128
   C64   2.1     2.3     2.5
   ```
   - Diagnosis: GRUs not learning temporal integration
   - Causes: Task doesn't need memory OR training issue
   - Action: Check if Connect Four task actually requires recurrence

2. **GRU size doesn't help** (flat across columns):
   ```
         GRU32   GRU64   GRU128
   C64   6.5     6.6     6.7
   ```
   - Diagnosis: Extra capacity not utilized for longer memory
   - Interpretation: GRU32 is sufficient for this task
   - Action: Use GRU32 for deployment (smaller, faster)

3. **Unstable large GRU** (τ > 50 for GRU128):
   ```
         GRU32   GRU64   GRU128
   C64   5.5     7.2     127.3
   ```
   - Diagnosis: Eigenvalues very close to 1 (near-marginally stable)
   - Risk: Sensitive to perturbations, long transients
   - Action: Check training stability, consider gradient clipping

**Cross-checking with performance**:
- Load `training_history.json` for each model
- Compare: Does higher τ correlate with better validation performance?
- If yes: Task benefits from longer memory
- If no: Extra memory capacity is wasted

### Probe accuracy (probe_accuracy.png)

**What it shows**: Logistic regression accuracy for predicting board features from hidden states.

**Axes**:
- X: Epoch number
- Y: Test accuracy (0 to 1)
- Facets: one subplot per feature (easier to read); within each subplot, each model is a line with its own color and a shared legend across the figure

**Baseline performance**:
- Binary features: Random chance = 0.5, but heavy class imbalance can make naive accuracy misleading (e.g., always predicting 0 for `immediate_win_*` can yield very high accuracy because wins are rare).
- Prefer balanced metrics under skew: check `balanced_accuracy`, `average_precision` (PR-AUC), and `signal_over_control`.

**Reading accuracy levels**:

**Strong encoding** (accuracy 0.85-1.0):
```
immediate_win_current: 0.92 at epoch 100
immediate_win_opponent: 0.89 at epoch 100
```
- Interpretation: Critical tactical features are clearly encoded
- Linear probe can reliably extract them
- Expected for: Features central to task (winning threats)

**Moderate encoding** (accuracy 0.65-0.85):
```
current_player: 0.72 at epoch 100
three_in_row_current: 0.68 at epoch 100
```
- Interpretation: Features are encoded but entangled with others
- Useful but not perfectly separable
- Expected for: Secondary features (piece counts, control)

**Weak encoding** (accuracy 0.50-0.65):
```
piece_diff: 0.53 at epoch 100
```
- Interpretation: Barely encoded linearly (may be nonlinear or unused)
- Expected for: Features irrelevant to task (in Connect Four, piece count matters less than position)

Note on class imbalance (important for `immediate_win_*`):
- Wins at the next move are rare for most positions; the negative class dominates.
- A classifier that always predicts 0 can achieve deceptively high accuracy. Use:
   - `balanced_accuracy` to weigh classes equally,
   - `average_precision` to evaluate the positive class under skew,
   - `signal_over_control` and `balanced_signal_over_control` to discount spurious majority baselines.
   - Training uses class_weight='balanced' by default; you can also under-sample with `--balance-train`.

**Common patterns**:

1. **Healthy learning** (expected):
   ```
   Epoch 0:   immediate_win = 0.52 (random)
   Epoch 30:  immediate_win = 0.78 (learning)
   Epoch 60:  immediate_win = 0.91 (strong)
   Epoch 100: immediate_win = 0.93 (converged)
   ```
   - Rising curves indicate progressive feature learning
   - Plateau at high accuracy = converged representation

2. **Feature never learned** (problem):
   ```
   immediate_win accuracy stays at 0.50-0.52 throughout training
   ```
   - Diagnosis: Critical feature not encoded in hidden state
   - Possible causes: Imbalanced dataset (too few threat positions), insufficient capacity
   - Action: Check dataset distribution, increase GRU size, inspect PHATE embeddings

3. **Representation collapse** (problem):
   ```
   Epoch 60:  immediate_win = 0.89
   Epoch 100: immediate_win = 0.61  (drops!)
   ```
   - Diagnosis: Overfitting or representation drift
   - Hidden state quality degrading despite low loss
   - Action: Early stopping at epoch 60, add regularization

4. **GRU size comparison**:
   ```
   GRU32:  immediate_win = 0.87
   GRU64:  immediate_win = 0.91
   GRU128: immediate_win = 0.92
   ```
   - Larger GRUs encode features more clearly (expected)
   - Diminishing returns: 0.87→0.91→0.92 (GRU64 may be sweet spot)

**Diagnostic use cases**:

**Use case 1: Model plays poorly despite low loss**
```
Check probe_accuracy.png:
  immediate_win_current: 0.54 (barely above chance)
  immediate_win_opponent: 0.52 (random)

Diagnosis: Model not learning threat detection
Action: Inspect dataset for class imbalance, visualize PHATE for separation
```

**Use case 2: Choosing GRU size**
```
Compare GRU32 vs GRU64 probe accuracy:
  GRU32: All features 0.75-0.85
  GRU64: All features 0.78-0.88 (marginal improvement)

Decision: Use GRU32 (sufficient encoding, faster inference)
```

**Use case 3: Verifying CNN vs GRU encoding**
```
Pass --probe-components gru cnn:
  immediate_win from GRU: 0.91
  immediate_win from CNN: 0.68

Interpretation: GRU adds significant temporal reasoning beyond CNN spatial features
```

### Control task validation (probe_signal_over_control.png, probe_comparison.png)

**What it shows**: Control tasks with permuted labels validate that probe accuracy reflects genuine signal rather than memorization or overfitting artifacts.

**Methodology**: For each probe, a control classifier is trained on the same hidden states but with randomly shuffled labels. This control should perform near chance (0.5 for binary features).

**Reading signal over control**:

**Strong signal** (signal > 0.3):
```
Real accuracy: 0.88
Control accuracy: 0.52
Signal over control: 0.36
```
- Interpretation: Hidden state genuinely encodes feature
- Control near chance confirms no spurious patterns
- Expected for well-learned features

**Weak signal** (signal 0.1-0.3):
```
Real accuracy: 0.64
Control accuracy: 0.51
Signal over control: 0.13
```
- Interpretation: Some encoding present but weak or noisy
- May indicate partially learned or distributed representation
- Consider whether feature is truly task-relevant

**No signal** (signal ≈ 0):
```
Real accuracy: 0.53
Control accuracy: 0.51
Signal over control: 0.02
```
- Interpretation: No meaningful encoding (both near chance)
- Hidden state does not contain feature information
- Expected for task-irrelevant features

**Red flag** (control > 0.6):
```
Real accuracy: 0.78
Control accuracy: 0.68
Signal over control: 0.10
```
- Diagnosis: Probe may be exploiting spurious correlations
- Dataset may have hidden structure or imbalance
- Action: Inspect data distribution, try different train/test split

**Comparison plot interpretation**:
- **Left panel** (real accuracy): Shows what the network learned
- **Right panel** (control accuracy): Should stay near 0.5 (dashed red line)
- **Wide gap**: Strong, validated encoding
- **Narrow gap**: Weak or questionable signal
- **Control rising over epochs**: Warning sign of overfitting to noise

### Confusion matrices: final vs best epochs

- Final-epoch matrices (files ending with `epoch_XXX`) summarize performance at the last probed epoch across models.
- Best-epoch matrices (`probe_confusion_matrices_best.png`) aggregate confusion counts by taking, for each model, the epoch with the minimum validation loss; this avoids mixing different training stages and produces a fairer “best-achieved” snapshot across architectures.
- Use the “best” variants when you want cross-model comparability at their strongest checkpoint; use the “epoch_XXX” variants to inspect a specific shared epoch.

### Cross-plot analysis

**Combining insights from all three plots**:

**Scenario 1: High probe accuracy but low timescale**
```
Probe accuracy: immediate_win = 0.90
Timescale: τ = 2.5
```
- Interpretation: Features are encoded, but memory is short
- GRU is reactive (immediate pattern matching) not strategic (long planning)
- For Connect Four: May struggle in complex multi-move sequences

**Scenario 2: High timescale but low probe accuracy**
```
Timescale: τ = 12
Probe accuracy: immediate_win = 0.58
```
- Interpretation: GRU has long memory but not using it for task-relevant features
- Memory capacity is wasted or encoding wrong information
- Action: Check PHATE embeddings for what GRU actually encodes

**Scenario 3: Gates converge early, probes keep improving**
```
Gate plateau: Epoch 50 (update = 0.7, reset = 0.6)
Probe accuracy still rising at epoch 100: 0.65 → 0.88
```
- Interpretation: Gate values stabilize early, but hidden state quality improves
- GRU dynamics are set, but representation fine-tuning continues
- This is healthy (gates ≠ features)

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

# Standardize features for better convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=5000, solver='lbfgs', C=1.0, tol=1e-4)
clf.fit(X_train_scaled, y_train)
accuracy = clf.score(X_test_scaled, y_test)
f1 = f1_score(y_test, clf.predict(X_test_scaled))
```

**Why standardize?** GRU hidden dimensions often have very different scales (some near 0, others in [-5, 5]). Standardizing (zero mean, unit variance) ensures:
- Faster convergence of optimization
- Balanced contribution from all dimensions
- Stable gradients in logistic regression
- More reliable coefficient interpretation

**Feature preparation**:
- `current_player`: Transform from [1,2] to [0,1]
- `immediate_win_*`: Binary [0,1] as-is
- `three_in_row_*`: Binarize (0 vs ≥1 threats)
- Features must have ≥2 samples per class
- 70/30 train/test split with stratification

Pass `--probe-components cnn` to train probes directly on the CNN feature vectors instead of (or alongside) the GRU hidden state; when both are requested, each component writes its own `probe_results.csv` and `probe_accuracy.png` under `<output-dir>/<component>/`.

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
- `--probe-components`: Representations to probe [default: gru] (choose from `gru`, `cnn`, or both)
- `--max-hidden-samples`: Subsample hidden states for embeddings/probes [default: 2000]
- `--palette`: Seaborn color palette for multi-model plots [default: Set2]
- `--skip-embedding`: Skip PHATE plots (faster iteration)
- `--skip-probing`: Skip logistic regression probes
- `--class-weight`: Class weighting for probes (`none` or `balanced`, default: `balanced`)
- `--balance-train`: Under-sample the majority class in the training split to 1:1 (optional)

---

## Methodological Notes

### Probing Methodology

**Sequential Processing**: Hidden states are collected during proper sequential trajectory processing where each hidden state at timestep `t` depends on all previous timesteps through recurrent connections. This follows standard practices in RNN interpretability research [Maheswaranathan et al., 2019; Lambrechts et al., 2022].

**Why This Is Correct**: When probing a hidden state `h_t`, that state already contains the result of all recurrent processing from timesteps 0 to t. The recurrence is not "re-run" during probing—the temporal information is already encoded in `h_t` through the sequential forward pass. Probing measures *what information is encoded in the hidden state* at a given point, not whether the network can use recurrence (which is tested by task performance).

**Data Collection Process** (scripts/extract_gru_dynamics.py:346-395):
1. Load full game trajectory (sequence of board states)
2. Initialize hidden state `h_0 = 0`
3. For each timestep t:
   - Extract CNN features `x_t` from board state
   - Compute GRU step: `h_{t+1} = GRU(x_t, h_t)` with full recurrence
   - Store `h_{t+1}` for probing (reservoir sampling)
4. Each collected hidden state incorporates full sequential history

**Probing**: Train linear classifiers (logistic regression) on collected hidden states to predict board features. High accuracy indicates the feature is linearly decodable from the hidden representation [Belinkov, 2022].

**Control Tasks**: To validate that probe accuracy reflects genuine encoding rather than artifacts, we train control classifiers on the same hidden states but with randomly permuted labels. The control should perform near chance (0.5 for binary features). The "signal over control" (real accuracy - control accuracy) quantifies true encoding strength [Belinkov, 2022].

**Methodological Safeguards**:
- Stratified train/test splits ensure class balance
- Control tasks with permuted labels detect spurious patterns
- Multiple epochs tested to track representation evolution
- Comparison with CNN-only probes isolates GRU contribution
- Signal over control metric validates statistical significance

---

## Theoretical Foundations

### Dynamical Systems View

**Fixed Points and Attractors**: RNN hidden states evolve according to deterministic dynamics. Trained RNNs often exhibit low-dimensional structure with identifiable fixed points and attractors [Sussillo & Barak, 2013]. For GRUs in games:
- Stable attractors may represent game states (e.g., "winning position")
- Line attractors enable continuous integration (e.g., advantage accumulation)
- Multiple attractors indicate discrete behavioral modes [Jordan et al., 2019]

**Eigenvalue-Based Timescales**: The integration timescale τ measures how many steps the GRU retains information. Derived from eigenvalues λ of the recurrent weight matrix:
```
τ_i = 1 / |log(|λ_i|)|
```
Larger τ indicates longer memory capacity [Maheswaranathan et al., 2019].

### Learning Dynamics

**Representation Evolution**: During training, RNN hidden states progressively align with task-relevant state variables. Mutual information between hidden states and critical features (e.g., winning threats) increases as learning proceeds [Lambrechts et al., 2022].

**Attractor Formation**: Fixed points emerge and sharpen during training. The movement of attractors in hidden state space correlates with performance improvements—they converge toward task-optimal representations [Huang et al., 2024].

**Gate Behavior**: GRU gates learn to balance memory retention (update gate) and input integration (reset gate). Update gate values typically increase during training as the network learns which information to preserve [Chung et al., 2014].

---

## References

**Core Methodology**:
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1):207-219. [Probing methodology and best practices]

- Maheswaranathan, N., Williams, A.H., Golub, M.D., Ganguli, S., & Sussillo, D. (2019). Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. *NeurIPS* 32:15696-15705. [Line attractors, eigenvalue-based timescales, probing RNN hidden states]

- Lambrechts, G., Bolland, A., & Ernst, D. (2022). Recurrent networks, hidden states and beliefs in partially observable environments. *Transactions on Machine Learning Research*. [Mutual information between hidden states and beliefs, sequential processing for POMDPs]

**Dynamical Systems Analysis**:
- Sussillo, D. & Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. *Neural Computation*, 25(3):626-649. [Fixed point finding, dynamical systems methodology]

- Jordan, I., Sokol, P.A., Park, I.M., & Park, M. (2021). Gated Recurrent Units viewed through the lens of continuous time dynamical systems. *Frontiers in Computational Neuroscience*, 15:678158. [GRU dynamics, attractors, bifurcations]

- Huang, A., Wang, Y., Chen, Y., Zhang, L. & Fei-Fei, L. (2024). Learning Dynamics and Geometry in Recurrent Neural Controllers. *CCN 2024*. [Attractor evolution during learning, correlation with performance]

**GRU Architecture & Gates**:
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS 2014 Workshop*. [GRU introduction and gate behavior]

- Tang, Z., Wang, D., & Zhang, Z. (2017). Memory visualization for gated recurrent neural networks in speech recognition. *arXiv:1609.08789*. [Gate activation patterns and memory analysis]

**Game-Specific Applications**:
- Lei, S., Xu, S., Luo, S., Wang, K., & Xu, Z. (2024). Learning Strategy Representation for Imitation Learning in Multi-Agent Games. *arXiv:2409.19363*. [GRU-based strategy embeddings in Connect Four, separation of skill levels]

**Visualization & Interpretability**:
- Strobelt, H., Gehrmann, S., Pfister, H., & Rush, A.M. (2018). LSTMVis: A Tool for Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks. *IEEE TVCG*, 24(1):667-676. [Hidden state trajectory visualization]

- Yao, L., Chu, Z., Li, S., Li, Y., Gao, J., & Zhang, A. (2018). RNNVis: A Tool for Visualizing and Understanding RNN Hidden Dynamics. *IEEE Trans. Vis. Comput. Graph.*, 25(1):267-276. [RNN interpretability tools]

- Moon, K.R., van Dijk, D., Wang, Z., Gigante, S., Burkhardt, D.B., Chen, W.S., Yim, K., van den Elzen, A., Hirn, M.J., Coifman, R.R., Ivanova, N.B., Wolf, G., & Krishnaswamy, S. (2019). Visualizing structure and transitions in high-dimensional biological data. *Nature Biotechnology*, 37:1482–1492. [PHATE embedding methodology]

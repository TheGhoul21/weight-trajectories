# GRU Interpretability Workflow

Complete workflow for analyzing GRU internal representations and dynamics in Connect Four models.

## Quick Start

```bash
# 1. Extract GRU dynamics from checkpoints
./wt.sh observability extract \
  --checkpoint-dir checkpoints/save_every_3 \
  --dataset data/connect4_sequential_10k_games.pt \
  --max-games 256 \
  --sample-hidden 1500

# 2. Generate all visualizations and analyses
./wt.sh observability analyze \
  --analysis-dir diagnostics/gru_observability \
  --output-dir visualizations/gru_observability \
  --embedding-epochs 3 30 60 100 \
  --probe-epochs 30 60 100

# 3. Find fixed points and attractor dynamics
./wt.sh observability fixed \
  --checkpoint-dir checkpoints/save_every_3 \
  --dataset data/connect4_sequential_10k_games.pt \
  --max-contexts 12 \
  --restarts 8

# 4. Visualize attractor evolution
./wt.sh observability evolve \
  --fixed-dir diagnostics/gru_fixed_points \
  --output-dir visualizations/gru_fixed_points
```

**Total time**: ~20-30 minutes for 9 models × 35 epochs

---

## What You Get

### Phase 1: Extract (`diagnostics/gru_observability/<model>/`)
- `metrics.csv` — gate statistics and eigenvalue-based timescales per epoch
- `unit_gate_stats.csv` — per-neuron gate behavior
- `hidden_samples/epoch_XXX.npz` — sampled hidden states + board features (for probing)
- `epoch_XXX_eigenvalues.npz` — eigenvalue spectra of GRU recurrence matrix

### Phase 2: Analyze (`visualizations/gru_observability/`)
**Gate & Dynamics:**
- `gate_mean_trajectories.png` — update/reset gate evolution
- `timescale_heatmap.png` — memory capacity (median timescales) by architecture

**Representation Geometry:**
- `phate_epoch_XXX_<feature>.png` — 2D embeddings of hidden states (3×3 grid per epoch)
- `probe_results.csv` + `probe_accuracy.png` — GRU logistic probe performance with control task validation
- `probe_signal_over_control.png` — Signal strength above chance (real - control accuracy)
- `probe_comparison.png` — Side-by-side real vs control task accuracy
- (CNN probes, when enabled via `--probe-components cnn`, write under `cnn/`)

**Mutual Information:**
- `mi_results.csv` — mean MI per model/epoch/feature
- `mi_heatmap_final.png` — cross-model comparison (final epoch)
- `mi_trends.png` — MI evolution over training
- `mi_per_dimension_<model>.png` — which neurons encode which features (★ marks best)
- `mi_dimension_values_<model>.png` — HOW neurons encode features (violin/scatter)

### Phase 3+4: Fixed Points (`diagnostics/gru_fixed_points/`, `visualizations/gru_fixed_points/`)
- `epoch_XXX_fixed_points.npz` — hidden vectors, stability classification, eigenvalues
- `fixed_points_summary.csv` — counts and spectral radii by epoch
- `<model>_classification_counts.png` — stable/marginal/unstable attractor counts vs epoch
- `<model>_spectral_radius.png` — spectral radius trends
- `<model>_attractor_drift.png` — L2 drift of attractor centroids (learning dynamics)

---

## Analysis Flow

### Step 1: High-level model comparison
**Goal**: Which architecture learns best representations?

**Look at**:
1. `mi_heatmap_final.png` — Which models have highest MI for critical features?
   - Expected: `immediate_win_current`, `immediate_win_opponent` should have MI >0.5
2. `probe_accuracy.png` — Which models have best linear decodability?
3. `timescale_heatmap.png` — Do larger GRUs actually use their capacity? (longer timescales)

**Key question**: Does GRU64 outperform GRU32, or is it redundant?

---

### Step 2: Per-model neuron specialization
**Goal**: Understand what individual neurons do

**For your best model (e.g., k3_c64_gru32)**:

1. Open `mi_per_dimension_k3_c64_gru32.png`
   - Look for **sparse patterns** (bright spots) → specialized neurons
   - Look for **vertical clusters** (adjacent dims bright) → functional modules
   - Count ★ symbols in each column → some neurons encode multiple features

2. Open `mi_dimension_values_k3_c64_gru32.png`
   - **Binary features**: Check violin separation
     - Clean separation → reliable encoding
     - Example: "Dim 17 at ±2.5 for immediate_win"
   - **Continuous features**: Check scatter trends
     - Linear → simple counter (e.g., move_index)
     - Nonlinear → complex computation

3. Create neuron labels:
   - Dim 17 → "Winning threat detector" (MI=0.84 with immediate_win_current)
   - Dim 42 → "Game phase counter" (MI=0.61 with move_index)
   - Dim 23 → "Strategic planner" (MI=0.52 with three_in_row_current)

---

### Step 3: Learning dynamics
**Goal**: When does the model learn tactical awareness?

**Look at**:
1. `mi_trends.png` — Find inflection points
   - Example: "immediate_win_current MI jumps from 0.2 → 0.7 between epochs 15-30"
   - Cross-reference with loss curves to find learning phases

2. `gate_mean_trajectories.png` — When do gates stabilize?
   - Update gate near 1.0 → preserving memory
   - Abrupt changes → regime shift in training

3. `phate_epoch_XXX_immediate_win_current.png` sequence
   - Early epochs: Diffuse point cloud
   - Late epochs: Clear clusters (win vs no-win positions)
   - Tracks when "concepts" crystallize in hidden space

---

### Step 4: Dynamical systems view
**Goal**: Identify attractor structure and evolution

**Look at**:
1. `<model>_classification_counts.png`
   - How many stable attractors at final epoch?
   - Example: "3 stable + 2 unstable → model has 3 behavioral modes"

2. `<model>_attractor_drift.png`
   - When do attractors stop moving? → learning has converged
   - Large drift → representation is still being refined

3. Fixed points NPZ files (programmatic inspection):
   ```python
   import numpy as np
   data = np.load('diagnostics/gru_fixed_points/k3_c64_gru32/epochs/epoch_100_fixed_points.npz')
   stable_fps = data['hidden'][data['classification'] == 0]  # Stable attractors
   # Project onto PHATE space for visualization
   ```

---

## Diagnostic Scenarios

### Scenario 1: Model plays poorly despite low loss
**Symptom**: Test loss is low but agent loses to random player

**Check**:
1. `mi_per_dimension` → MI for `immediate_win_current`
   - **If <0.3**: Model isn't learning to detect winning threats
   - **Root cause**: Dataset imbalance (few endgame positions)
   - **Fix**: Rebalance dataset or add auxiliary loss

2. `probe_accuracy.png` → Accuracy for tactical features
   - **If low**: Hidden state doesn't encode game-critical info
   - **Root cause**: Architecture bottleneck (GRU too small)
   - **Fix**: Increase GRU size or use attention

---

### Scenario 2: Large model (GRU128) no better than GRU32
**Symptom**: Similar win rates despite 4× parameters

**Check**:
1. `mi_per_dimension` for both models
   - Count active neurons (MI > 0.1)
   - GRU32: 18/32 active (56%)
   - GRU128: 22/128 active (17%) → **underutilized!**

2. `timescale_heatmap.png`
   - If timescales are similar → extra capacity unused
   - **Conclusion**: GRU32 is more efficient

3. **Decision**: Use GRU32 for deployment (faster, smaller)

---

### Scenario 3: Preparing paper figures
**Goal**: Show interpretable neurons for publication

**Workflow**:
1. Run full analysis on best checkpoint
2. Identify top-3 neurons from `mi_per_dimension` heatmap
3. Extract neuron labels (e.g., "threat detector")
4. Create per-neuron visualizations:
   - Activation trajectories over game (custom script)
   - Saliency maps (backprop from decision to neuron)
   - Ablation study (zero out neuron, measure performance drop)

**Paper figures**:
- Figure 1: `mi_per_dimension` with arrows pointing to interpretable neurons
- Figure 2: `mi_dimension_values` for top-3 neurons (clean separation)
- Figure 3: `phate_epoch_100` colored by neuron activation (not feature)
- Figure 4: Neuron activation over game trajectory (custom)

---

## Advanced: Custom Analysis

### Inspect specific neurons
```python
import numpy as np
import matplotlib.pyplot as plt

# Load final epoch hidden samples
data = np.load('diagnostics/gru_observability/k3_c64_gru32/hidden_samples/epoch_100.npz')
hidden = data['hidden']  # (N, 32)
features = data['features']  # (N, 12)
feature_names = data['feature_names']

# Find immediate_win feature index
feat_idx = list(feature_names).index('immediate_win_current')
y = features[:, feat_idx]

# Plot dimension 17 (identified as threat detector)
dim17 = hidden[:, 17]

plt.figure()
plt.scatter(y, dim17, alpha=0.3)
plt.xlabel('Immediate Win (0 or 1)')
plt.ylabel('Dimension 17 activation')
plt.title('Threat Detector Neuron')
plt.savefig('custom_dim17_analysis.png')
```

### Track neuron specialization over epochs
```python
import pandas as pd
import seaborn as sns

# Load MI results
df = pd.read_csv('visualizations/gru_observability/mi_results.csv')

# Filter to one model and one feature
subset = df[(df['model'] == 'k3_c64_gru32') & (df['feature'] == 'immediate_win_current')]

# Plot MI evolution
plt.figure()
plt.plot(subset['epoch'], subset['mi'])
plt.xlabel('Epoch')
plt.ylabel('MI (immediate_win_current)')
plt.title('When does the model learn threat detection?')
plt.savefig('mi_evolution_threat.png')
```

### Neuron ablation study
```python
import torch
from src.model import create_model

# Load model
model = create_model(channels=64, gru_hidden_size=32, kernel_size=3)
checkpoint = torch.load('checkpoints/save_every_3/k3_c64_gru32/weights_epoch_100.pt')
model.load_state_dict(checkpoint['state_dict'])

# Ablate dimension 17 (threat detector)
# Set W_hn[:, 17] = 0 and W_hz[:, 17] = 0
with torch.no_grad():
    model.gru.weight_hh_l0[:, 17] = 0  # Zero out recurrent connections

# Re-evaluate on test set
# ... (run evaluation loop)
# Expected: Performance drop on endgame positions
```

---

## Interpretation Guide

### Mutual Information Thresholds
- **MI > 0.7**: Strong encoding (feature is reliably decodable)
- **MI 0.3-0.7**: Moderate encoding (partial information)
- **MI < 0.3**: Weak encoding (distributed or absent)

### Neuron Specialization Patterns
- **Single bright spot**: Dedicated feature detector (interpretable!)
- **Row of bright spots**: Redundant encoding (robustness)
- **Vertical cluster**: Functional module (co-encoding)
- **Uniform medium**: Distributed representation (hard to interpret)

### Encoding Quality (Violin Plots)
- **Clean separation**: High-quality encoding
- **Bimodal with gap**: Threshold detector
- **Overlapping**: Noisy encoding (still usable if MI is decent)
- **Uniform blob**: No encoding (shouldn't happen if best dim)

### Temporal Trends (MI over epochs)
- **Monotonic increase**: Successful learning
- **Plateau**: Converged (good!)
- **Early spike then drop**: Temporary representation (later pruned)
- **Oscillations**: Training instability

---

## Performance Tips

### Speed up extraction
- Reduce `--max-games` (256 → 128) for quick iteration
- Reduce `--sample-hidden` (1500 → 1000) if memory-constrained
- Use `--verbose` to track progress

### Speed up analysis
- Use `--skip-embedding` if you don't need PHATE plots (saves ~40% time)
- Use `--skip-probing` if you only care about MI
- Reduce `--max-hidden-samples` for MI (4000 → 2000) for faster estimates

### Focus on specific models
```bash
# Only analyze one model architecture
./wt.sh observability extract --checkpoint-dir checkpoints/save_every_3/k3_c64_gru32
./wt.sh observability analyze
```

---

## Troubleshooting

### "No valid MI" in dimension value plot
**Cause**: All samples have same feature value (e.g., all zeros)
**Solution**: Check data diversity; ensure dataset has varied board positions

### PHATE warning: "knn > samples"
**Cause**: Too few hidden samples for embedding
**Solution**: Reduce `--max-hidden-samples` or increase `--sample-hidden` in extract step

### Empty per-dimension heatmap
**Cause**: Final epoch not found in hidden_samples
**Solution**: Check `diagnostics/gru_observability/<model>/hidden_samples/` has epoch_XXX.npz files

### MI computation very slow
**Cause**: Large hidden_size × many samples
**Solution**: Reduce `--max-samples` in MI script (4000 → 2000)

---

## Cross-References

- Command reference: [observability](manual/commands/observability)
- Plot explainers: [gru_mutual_info](manual/plots/gru_mutual_info), [gru_observability](manual/plots/gru_observability), [fixed_points](manual/plots/fixed_points)
- Research context: `docs/gru_observability_gap_analysis.md` (comparison with literature)
- Per-dimension MI theory: `docs/mi_dimension_analysis.md`

# GRU mutual information

Produced by: `./wt.sh observability analyze` (invokes `scripts/compute_hidden_mutual_info.py`)
Inputs from: `diagnostics/gru_observability/<model>/hidden_samples/epoch_XXX.npz`

## Artifacts (under `visualizations/gru_observability/`)

### Overview plots (all models, all epochs)
- **mi_results.csv**: Long-form table with columns `[model, epoch, feature, mi, type, kernel, channels, gru]`
  - `mi`: mean mutual information across all hidden dimensions
  - `type`: "classification" or "regression" (determines sklearn MI estimator)
- **mi_heatmap_final.png**: Cross-model comparison at final epoch
  - Rows: 12 board features
  - Columns: model architectures (k3_c64_gru32, etc.)
  - Color: MI magnitude (darker = higher dependence)
- **mi_trends.png**: 12-subplot grid showing MI evolution over training
  - Each subplot: one feature
  - X-axis: epoch
  - Y-axis: mean MI
  - Hue: model (each architecture gets a line)
- **mi_metadata.json**: Run parameters (features, max_samples, seed, paths)

### Per-dimension analysis (per model, final epoch only)
- **mi_per_dimension_<model>.png**: Heatmap identifying specialized neurons
  - Rows: 12 board features
  - Columns: hidden dimension indices (0..31 for GRU32, 0..127 for GRU128, etc.)
  - Color: per-dimension MI (brighter = dimension encodes feature more)
  - **★ symbol**: marks the single most informative dimension for each feature
  - Example: ★ at (immediate_win_current, dim17) means dimension 17 has highest MI with immediate wins

- **mi_dimension_values_<model>.png**: Shows HOW the best dimension encodes each feature
  - 12 subplots in 3×4 grid (one per feature)
  - Title format: `<feature>\nBest dim: X (MI=Y.YYY)`
  - **Binary features** (current_player, immediate_win_*, three_in_row_*):
    - Violin plot: distribution of dimension values for each class
    - Well-separated violins → clean encoding
    - Overlapping violins → ambiguous/poor encoding
  - **Continuous features** (move_index, piece counts, center control):
    - Scatter plot: dimension value vs feature value
    - Linear trend → simple accumulator/counter
    - Nonlinear/clustered → complex encoding

## Reading the plots

### Overall MI (mi_heatmap_final.png, mi_trends.png)
- **High MI (>0.5)**: Hidden state strongly encodes this feature
  - Expected for: `immediate_win_current`, `immediate_win_opponent` (critical tactical info)
- **Medium MI (0.2-0.5)**: Partial encoding
  - Common for: `three_in_row_*`, `center_control_*` (strategic features)
- **Low MI (<0.2)**: Weakly encoded or distributed
  - Possible for: `piece_diff`, `valid_moves` (less critical in Connect Four)

- **Increasing MI over epochs**: network learning to represent the feature
- **Plateaus**: converged representation
- **Decreases**: rare; may indicate overfitting or feature becoming redundant

### Neuron specialization (mi_per_dimension_<model>.png)
- **Sparse pattern (few bright spots per row)**: Dedicated feature detectors
  - Example: Single bright spot at dim17 for immediate_win → specialized neuron
- **Dense pattern (many medium-bright spots)**: Distributed encoding
  - Example: Uniform medium MI across dims → ensemble representation
- **Vertical clusters (adjacent dims bright for same feature)**: Co-encoding
  - Example: Dims 23-27 all encode threat-related features → functional module
- **Empty rows (all dark)**: Feature not well-encoded in hidden state
  - Diagnostic: May need more capacity or different architecture

### Encoding quality (mi_dimension_values_<model>.png)
**Binary features:**
- **Clean separation** (two non-overlapping violins):
  - Example: h[17]=+2.5 when win=1, h[17]=-2.1 when win=0
  - Interpretation: Dimension acts as binary threshold detector
- **Partial overlap**: Noisy but usable encoding
- **Complete overlap**: Feature not decodable from this dimension (shouldn't happen if it's the best dim)

**Continuous features:**
- **Strong linear correlation** (tight scatter line):
  - Example: h[42] increases linearly with move_index
  - Interpretation: Simple integrator/counter (common for game phase)
- **Monotonic nonlinear** (curved trend):
  - Example: Saturating function for piece counts
- **Multi-modal clusters**: Complex encoding
  - Example: Different regimes for opening/midgame/endgame

## Interpretation examples

### Example 1: Strong tactical encoding
```
mi_per_dimension_k3_c64_gru32.png:
  Row "immediate_win_current": ★ at dim 17, MI=0.84 (bright yellow)

mi_dimension_values_k3_c64_gru32.png:
  Subplot "immediate_win_current":
    Two violins: class 0 centered at -1.8, class 1 centered at +2.7
    Minimal overlap

→ Dimension 17 is a reliable "winning threat detector"
→ Model has learned critical tactical awareness
```

### Example 2: Game phase counter
```
mi_per_dimension_k3_c64_gru64.png:
  Row "move_index": ★ at dim 42, MI=0.61

mi_dimension_values_k3_c64_gru64.png:
  Subplot "move_index":
    Scatter with strong linear trend (r≈0.78)
    h[42] ranges from -2.0 (move 0) to +1.5 (move 40)

→ Dimension 42 acts as a move counter / game phase indicator
→ Monotonic encoding (no reset between games in batch)
```

### Example 3: Distributed encoding
```
mi_per_dimension_k3_c128_gru128.png:
  Row "three_in_row_current": Multiple medium-bright cells (dims 8,23,56,91)
  No single dominant ★ (max MI only 0.32)

→ Threat detection is distributed across multiple dimensions
→ Redundancy/robustness, but harder to interpret
```

### Example 4: Missing feature
```
mi_per_dimension_k3_c32_gru32.png:
  Row "piece_diff": All dark (max MI=0.08)

mi_dimension_values_k3_c32_gru32.png:
  Subplot "piece_diff": Scattered blob, no pattern

→ Piece advantage is NOT encoded in hidden state
→ In Connect Four, this is expected (positions matter more than counts)
```

## Diagnostic use cases

### Use case 1: Debugging poor gameplay
**Problem**: Model loses to simple heuristic player

**Check**:
1. Open `mi_per_dimension_<model>.png`
2. Check MI for `immediate_win_current` and `immediate_win_opponent`
3. **If low (<0.3)**: Model isn't learning to detect winning threats
4. **Root cause**: Likely imbalanced dataset (few end-game positions)
5. **Fix**: Rebalance training data or add auxiliary loss for threat detection

### Use case 2: Architecture selection
**Question**: GRU32 vs GRU64 - which is better?

**Check**:
1. Compare `mi_per_dimension` for both models
2. Count dimensions with MI > 0.1 (active neurons)
   - GRU32: 18/32 active (56% utilization)
   - GRU64: 22/64 active (34% utilization)
3. Compare max MI per feature
   - GRU32: immediate_win MI=0.84
   - GRU64: immediate_win MI=0.86 (marginal gain)
4. **Conclusion**: GRU32 is more efficient; GRU64 has redundancy but minimal quality improvement
5. **Decision**: Use GRU32 for deployment (faster, smaller, sufficient)

### Use case 3: Identifying interpretable neurons
**Goal**: Visualize/ablate specific neurons for paper

**Method**:
1. Find ★-marked dimensions in `mi_per_dimension` heatmap
2. Priority order by MI score
3. Top candidates for interpretation:
   - Dim 17 (immediate_win_current, MI=0.84) → "threat detector"
   - Dim 42 (move_index, MI=0.61) → "game phase counter"
   - Dim 23 (three_in_row_current, MI=0.52) → "strategic planner"
4. Use these for:
   - Saliency analysis (backprop from decision to neuron)
   - Ablation study (zero out neuron, measure performance drop)
   - Visualization (plot activation over game trajectory)

### Use case 4: Feature engineering
**Question**: Should we add more board features?

**Check**:
1. Look at existing MI scores in `mi_heatmap_final.png`
2. **All features >0.3**: Current features are well-encoded; adding more may be redundant
3. **Some features <0.1**: Either useless OR not learned
4. **Experiment**: Remove low-MI features, retrain
   - If performance unchanged → features were useless
   - If performance drops → features were important but poorly learned (need different architecture/training)

## Computational details

### MI estimation
- **Classification** (binary/categorical features):
  - Uses `sklearn.feature_selection.mutual_info_classif`
  - Based on k-NN density estimation
  - Returns array of shape `(hidden_size,)` with MI per dimension
  - We take mean for overall score, full array for per-dimension analysis

- **Regression** (continuous features):
  - Uses `sklearn.feature_selection.mutual_info_regression`
  - Same k-NN approach but for continuous targets
  - Note: Sensitive to outliers; we apply outlier removal in preprocessing

### Per-dimension vs mean MI
- **Mean MI**: `mean(MI(h_i, feature))` for i=1..hidden_size
  - Summary statistic: "How much does the full hidden state encode this feature?"
  - Used for temporal trends (mi_trends.png)

- **Per-dimension MI**: `MI(h_i, feature)` for each i
  - Identifies individual neurons: "Which dimensions specialize for this feature?"
  - Used for specialization analysis (mi_per_dimension heatmap)
  - Only computed at final epoch (expensive)

### Feature categorization
**Classification features** (5):
- current_player (binary: 1 or 2)
- immediate_win_current (binary: 0 or 1)
- immediate_win_opponent (binary: 0 or 1)
- three_in_row_current (converted to binary: 0 vs ≥1)
- three_in_row_opponent (converted to binary: 0 vs ≥1)

**Regression features** (7):
- move_index (0..42)
- yellow_count (0..21)
- red_count (0..21)
- piece_diff (-21..+21)
- valid_moves (0..7)
- center_control_current (0..6)
- center_control_opponent (0..6)

## Knobs

Command-line options (via `scripts/compute_hidden_mutual_info.py`):
- `--analysis-dir`: Path to extract outputs [default: diagnostics/gru_observability]
- `--features`: List of feature names to analyze [default: all 12 features]
- `--max-samples`: Subsample hidden states per epoch [default: 4000]
  - Lower → faster but noisier MI estimates
  - Higher → slower but more accurate
- `--seed`: Random seed for reproducible subsampling [default: 0]
- `--output-dir`: Where to write plots/CSVs [default: visualizations/gru_observability]

Note: Respects hidden sample availability from the extract step. If you ran extract with `--sample-hidden 1500`, MI analysis can use at most 1500 samples per epoch (or fewer if `--max-samples` is lower).

## Performance notes

- MI computation scales as O(features × epochs × models)
- Per-dimension analysis adds O(hidden_size) overhead but only for final epoch
- Typical runtime for 9 models × 35 epochs × 12 features: ~2-3 minutes on CPU
- Bottleneck: sklearn's k-NN density estimation (parallelizes internally)
- Memory: Peak ~2GB for GRU128 with 4000 samples

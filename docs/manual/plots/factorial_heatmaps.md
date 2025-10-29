# Factorial Heatmaps

Overview
- Summarizes 9 key metrics across the full 3×3 architecture sweep to reveal main and interaction effects.
- Answers how CNN capacity and GRU capacity influence performance, stability, and representation quality.

How to Generate
- `./wt.sh factorial` (uses `scripts/visualize_unified.py`)
 
See also: [Methods Reference](../../reference/methods) for metric definitions and related libraries.

## Purpose

Visualize how 9 key metrics vary across a full factorial architecture sweep. Reveals main effects (CNN capacity, GRU capacity) and interaction effects in a single comprehensive figure.

## Architecture sweep

**Full factorial design**: 3×3 = 9 models

- **CNN channels**: 16, 64, 256 (4× and 16× increases)
- **GRU hidden size**: 8, 32, 128 (4× and 16× increases)
- **Kernel size**: Fixed at 3 (all models)

**Models**:
```
         GRU8    GRU32   GRU128
C16      k3_c16_gru8   k3_c16_gru32   k3_c16_gru128
C64      k3_c64_gru8   k3_c64_gru32   k3_c64_gru128
C256     k3_c256_gru8  k3_c256_gru32  k3_c256_gru128
```

## Output

### factorial_heatmaps.png
**Format**: 3×3 grid of heatmaps (9 subplots total)

**Layout**:
```
Weight Norm      Step Norm         Step Cosine
Total Variance   Top-1 Ratio       Min Val Loss
Final Val Loss   Train/Val Gap     Val Loss Increase
```

**Each heatmap**:
- **Rows**: CNN channels (16, 64, 256)
- **Columns**: GRU size (8, 32, 128)
- **Color**: Metric value (RdYlBu_r colormap)
- **Annotations**: Actual values overlaid on cells

## Metrics explained

### 1. Final Weight Norm
**What**: L2 norm of all weights at final epoch
**Formula**: `||w_final||_2`
**Interpretation**:
- Higher → Model learned more complex function
- Very high (>2000) → Potential overfitting
- Low (<500) → Underfit or simple task
**Expected pattern**: Increases with both CNN and GRU size

### 2. Mean Step Norm
**What**: Average L2 distance between consecutive checkpoints
**Formula**: `mean(||w_{t+1} - w_t||_2)`
**Interpretation**:
- Higher → More parameter churn during training
- Very high → Unstable training
- Low → Smooth convergence
**Expected pattern**: Higher for larger models (more parameters to optimize)

### 3. Final Step Cosine
**What**: Cosine similarity between final two checkpoints
**Formula**: `cos(w_99, w_100) = (w_99 · w_100) / (||w_99|| ||w_100||)`
**Interpretation**:
- Close to +1 → Training converged (moving in same direction)
- Close to 0 → Still exploring
- Negative → Oscillating (bad)
**Expected pattern**: High (>0.95) for all models if trained enough

### 4. Mean Total Variance
**What**: Average sum of squared singular values of GRU hidden representations
**Formula**: `mean(sum(S^2))` from SVD of centered hidden states
**Interpretation**:
- Higher → GRU using more representational capacity
- Very low → Representation collapse
- Increases with GRU size (more dimensions available)
**Expected pattern**: Strong dependence on GRU size, weak on CNN size

### 5. Mean Top-1 Ratio
**What**: Average fraction of representation variance captured by top singular vector
**Formula**: `mean(S[0]^2 / sum(S^2))`
**Interpretation**:
- Close to 1 → Severe rank-1 collapse (all hidden states on single line)
- Close to 1/gru_size → Healthy (uniform distribution)
- Acceptable: <0.3 for GRU8, <0.15 for GRU32, <0.10 for GRU128
**Expected pattern**: Decreases with GRU size

### 6. Min Validation Loss
**What**: Minimum validation loss achieved during training
**Formula**: `min(val_loss[epoch] for epoch in 0..100)`
**Interpretation**:
- Lower → Better peak performance
- **Main performance metric**
**Expected pattern**: Decreases with model capacity (up to a point)

### 7. Final Validation Loss
**What**: Validation loss at final epoch
**Formula**: `val_loss[100]`
**Interpretation**:
- Higher than min → Overfit after peak
- Equal to min → Still improving or converged
**Expected pattern**: Compare to min to detect overfitting

### 8. Final Train/Val Gap
**What**: Difference between train and validation loss at final epoch
**Formula**: `val_loss[100] - train_loss[100]`
**Interpretation**:
- Positive → Overfitting (memorizing training set)
- Near 0 → Generalizing well
- Large (>0.5) → Severe overfitting
**Expected pattern**: Larger for bigger models (more capacity to overfit)

### 9. Val Loss Increase
**What**: How much validation loss increased from minimum
**Formula**: `val_loss[100] - min(val_loss)`
**Interpretation**:
- 0 → No overfitting (min at end)
- Positive → Overfit after peak
- Large → Should have used early stopping
**Expected pattern**: Larger for bigger models

## Reading the heatmaps

### Main effects

**CNN capacity (vertical patterns)**:
- Look down each column: Does increasing channels (16→64→256) improve metrics?
- Example: If min val loss decreases down column → CNN capacity helps
- Example: If weight norm increases down column → CNNs getting more complex

**GRU capacity (horizontal patterns)**:
- Look across each row: Does increasing GRU size (8→32→128) improve metrics?
- Example: If total variance increases across row → GRU capacity utilized
- Example: If top-1 ratio decreases across row → avoiding collapse (good)

### Interaction effects

**Non-additive patterns** (look for diagonal, corners, centers):
- **Diagonal bright**: Small CNN + small GRU OR large CNN + large GRU → Balance matters
- **Corner bright**: Extreme combos (16×8 or 256×128) are best → Surprising!
- **Center bright**: Medium combo (64×32) is best → Sweet spot
- **Uniform**: No interaction → Main effects are independent

### Diagnostic patterns

**Pattern 1: GRU-dominated metric**
```
         GRU8    GRU32   GRU128
C16      10      30      90
C64      12      32      92
C256     11      31      91
```
- Strong horizontal gradient, weak vertical → GRU size dominates
- Example: Total variance (more GRU dims = more capacity)

**Pattern 2: CNN-dominated metric**
```
         GRU8    GRU32   GRU128
C16      5       6       7
C64      15      16      17
C256     45      46      47
```
- Strong vertical gradient, weak horizontal → CNN size dominates
- Example: Weight norm (CNN has more parameters than GRU)

**Pattern 3: Balanced scaling**
```
         GRU8    GRU32   GRU128
C16      10      20      30
C64      15      30      45
C256     20      40      60
```
- Both gradients strong → Both components contribute
- Example: Min val loss (both CNN and GRU improve performance)

**Pattern 4: Saturation**
```
         GRU8    GRU32   GRU128
C16      0.85    0.87    0.87
C64      0.86    0.90    0.90
C256     0.86    0.90    0.90
```
- Values plateau at GRU32+ and C64+ → Diminishing returns
- Interpretation: GRU32 + C64 is sufficient, larger is wasteful

**Pattern 5: Collapse in large models**
```
         GRU8    GRU32   GRU128
C16      0.05    0.03    0.02
C64      0.06    0.04    0.08  (!)
C256     0.07    0.05    0.15  (!!)
```
- Top-1 ratio increases for large models → Representation collapse
- Red flag: GRU128 + C256 is overfitting or unstable

## Example interpretation

**Hypothetical factorial_heatmaps.png analysis**:

### Min Validation Loss (subplot 6)
```
         GRU8    GRU32   GRU128
C16      0.82    0.71    0.68
C64      0.75    0.62    0.59
C256     0.73    0.61    0.58
```
**Observations**:
- Strong improvement: GRU8→GRU32 (−0.11 to −0.13)
- Weak improvement: GRU32→GRU128 (−0.03)
- CNN helps: C16→C64 (−0.07 to −0.09), but C64→C256 marginal (−0.02 to −0.03)
**Conclusion**: GRU32 + C64 is the sweet spot (~0.62 loss)

### Final Train/Val Gap (subplot 8)
```
         GRU8    GRU32   GRU128
C16      0.02    0.03    0.05
C64      0.03    0.05    0.12
C256     0.04    0.08    0.28
```
**Observations**:
- Large models (C256 + GRU128) have 0.28 gap → Severe overfitting
- Small models (C16 + GRU8) have 0.02 gap → Generalizing well
**Conclusion**: C256 + GRU128 needs regularization or early stopping

### Mean Top-1 Ratio (subplot 5)
```
         GRU8    GRU32   GRU128
C16      0.18    0.08    0.05
C64      0.19    0.09    0.04
C256     0.20    0.10    0.04
```
**Observations**:
- GRU8: 0.18-0.20 (acceptable for 8 dims, would be 0.125 if uniform)
- GRU32: 0.08-0.10 (healthy, ~3× uniform)
- GRU128: 0.04-0.05 (excellent, ~5× uniform)
**Conclusion**: All models have healthy representations (no collapse)

### Cross-metric decision

Combining Min Val Loss + Train/Val Gap:
- **Best performance**: C64 + GRU32 (loss=0.62)
- **Best generalization**: C16 + GRU8 (gap=0.02, loss=0.82)
- **Overfitting**: C256 + GRU128 (loss=0.58 but gap=0.28)
- **Recommended**: C64 + GRU32 (balanced performance + generalization)

## Data collection

**Prerequisites**:
1. Run `./wt.sh train-all` to train all 9 models
2. Run `./wt.sh metrics --board-source random --board-count 64` for each model
   - Generates `diagnostics/checkpoint_metrics/k3_c*_gru*_metrics.csv`
3. Ensure `training_history.json` exists in each checkpoint directory

**Command**:
```bash
./wt.sh factorial \
  --metrics-dir diagnostics/checkpoint_metrics \
  --checkpoint-dir checkpoints/save_every_3 \
  --output-dir visualizations/factorial
```

**Outputs**:
- `visualizations/factorial/factorial_heatmaps.png`
- Also generates: loss trajectory grids, representation grids, summary CSV

## Command options

```bash
./wt.sh factorial [options]

Options:
  --metrics-dir DIR        Directory with *_metrics.csv files [default: diagnostics/trajectory_analysis]
  --checkpoint-dir DIR     Base directory with model subdirs [default: checkpoints/save_every_3]
  --output-dir DIR         Where to save plots [default: visualizations]
```

**Note**: The script expects specific directory structure:
```
metrics_dir/
  k3_c16_gru8_metrics.csv
  k3_c16_gru32_metrics.csv
  ...
  k3_c256_gru128_metrics.csv

checkpoint_dir/
  k3_c16_gru8/training_history.json
  k3_c16_gru32/training_history.json
  ...
```

## Use cases

### Use case 1: Architecture selection
**Goal**: Choose best model for deployment

**Steps**:
1. Generate factorial heatmaps
2. Find Min Val Loss minimum (e.g., C64 + GRU32 = 0.62)
3. Check Train/Val Gap for that model (e.g., 0.05 - acceptable)
4. Check Top-1 Ratio (e.g., 0.09 - healthy)
5. Compare to neighbors: Is GRU32→GRU128 worth 4× capacity?
6. Decision: Use C64 + GRU32

### Use case 2: Diagnosing overfitting
**Goal**: Understand why large models underperform

**Steps**:
1. Compare Min Val Loss vs Final Val Loss heatmaps
2. Compute difference (Val Loss Increase) → Large for C256 models
3. Check Train/Val Gap → Also large for C256
4. Conclusion: Large models overfit after epoch ~60
5. Action: Use early stopping or add dropout

### Use case 3: Capacity analysis
**Goal**: Determine if more capacity helps

**Steps**:
1. Look at Total Variance heatmap
2. Compare GRU8 vs GRU32 vs GRU128 (increases)
3. Look at Top-1 Ratio (decreases with size)
4. Cross-reference with Min Val Loss
5. If variance ↑ but loss ↓ → Capacity is utilized productively
6. If variance ↑ but loss flat → Capacity wasted

### Use case 4: Identifying interactions
**Goal**: See if CNN and GRU effects are independent

**Steps**:
1. Pick a metric (e.g., Final Weight Norm)
2. Check if pattern is additive:
   - Additive: value[C64, GRU32] ≈ value[C16, GRU32] + (value[C64, GRU8] - value[C16, GRU8])
3. If not additive → Interaction exists
4. Example: If C256 + GRU128 has unusually high norm → Multiplicative interaction

## Cross-references

- **Weight metrics details**: [checkpoint_metrics_csv](manual/plots/checkpoint_metrics_csv)
- **Representation SVD**: [checkpoint_metrics_csv#4-representation-svd-analysis-optional](manual/plots/checkpoint_metrics_csv#4-representation-svd-analysis-optional)
- **Training command**: [train-all](manual/commands/train-all)
- **Metrics command**: [metrics](manual/commands/metrics)

# Per-Dimension Mutual Information Analysis

## Overview

The mutual information (MI) analysis has been extended to analyze **individual GRU hidden dimensions** rather than just the overall hidden state vector. This reveals **which specific neurons** encode which game features and **how** they encode them.

## What Gets Computed

### 1. Mean MI (Existing)
- **What**: Average MI across all hidden dimensions
- **Formula**: `mean(MI(h_i, feature))` for i=1..hidden_size
- **Output**: Single scalar per feature
- **Use**: Overall information encoding capacity

### 2. Per-Dimension MI (NEW)
- **What**: MI for each individual hidden dimension
- **Formula**: `MI(h_i, feature)` for each dimension i
- **Output**: Array of shape `(hidden_size,)` per feature
- **Use**: Identify specialized neurons

### 3. High-MI Dimension Values (NEW)
- **What**: Actual values in the most informative dimension
- **Formula**: For `best_dim = argmax(MI(h_i, feature))`, plot `h[best_dim]` vs `feature`
- **Output**: Scatter/distribution showing encoding relationship
- **Use**: Understand **how** the dimension encodes the feature

---

## New Visualizations

### `mi_per_dimension_<model>.png`

**Type**: Heatmap (features × dimensions)

**Layout**:
- Y-axis: 12 board features (immediate_win_current, three_in_row_opponent, etc.)
- X-axis: Hidden dimension indices (0..31 for GRU32, 0..63 for GRU64, etc.)
- Color: Mutual information magnitude (darker = higher MI)
- **★ Symbol**: Marks the dimension with highest MI for each feature

**Interpretation**:
1. **Specialization**: Bright spots show dimensions specialized for specific features
   - Example: "Dimension 17 has very high MI with `immediate_win_current`"

2. **Clustering**: Adjacent bright dimensions suggest feature co-encoding
   - Example: "Dimensions 23-27 all encode threat-related features"

3. **Redundancy**: Multiple bright spots per feature indicate distributed encoding
   - Example: "Both dim 5 and dim 78 encode `current_player`"

4. **Sparsity**: Mostly dark rows suggest features poorly encoded in hidden state
   - Example: "`center_control` has low MI across all dimensions"

**Expected Patterns**:
- **Trained models**: Clear specialization (sparse bright spots per feature)
- **Early epochs**: Diffuse/uniform MI (no specialization yet)
- **Large GRUs**: More dimensions with medium MI (distributed encoding)
- **Small GRUs**: Few dimensions with high MI (forced specialization)

---

### `mi_dimension_values_<model>.png`

**Type**: Grid of scatter/distribution plots (12 subplots, one per feature)

**Layout**:
- Each subplot: One feature
- X-axis: Feature values
- Y-axis: Hidden dimension values (the dimension with highest MI for that feature)
- Title: Feature name + best dimension index + MI score

**Plot Types**:
1. **Binary features** (e.g., `immediate_win_current`):
   - Violin plot or box plot
   - Shows distribution of dimension values for each class (0 vs 1)
   - Example: When immediate win=1, dimension 17 clusters around +2.5; when 0, clusters around -1.2

2. **Continuous features** (e.g., `move_index`, `piece_diff`):
   - Scatter plot
   - Shows relationship between feature value and dimension activation
   - Example: Dimension 42 linearly increases with `move_index` (game phase detector)

**Interpretation Examples**:

**Strong Encoding** (What you want to see):
```
Feature: immediate_win_current
Best dim: 17 (MI=0.842)
Plot: Two well-separated violin distributions
→ Dimension 17 is a reliable "winning threat detector"
```

**Linear Encoding**:
```
Feature: move_index
Best dim: 42 (MI=0.634)
Plot: Clear linear trend (correlation ≈0.8)
→ Dimension 42 acts as a "game phase counter"
```

**Weak Encoding**:
```
Feature: center_control_opponent
Best dim: 91 (MI=0.073)
Plot: Scattered blob with no clear pattern
→ This feature is not well-represented in hidden state
```

**Multi-Modal Encoding**:
```
Feature: three_in_row_current
Best dim: 23 (MI=0.521)
Plot: Multiple clusters in violin plot
→ Dimension encodes both presence AND number of threats
```

---

## Scientific Insights

### 1. Neuron Specialization
**Question**: Do GRU units specialize for specific game concepts?

**Answer from Plots**:
- **Yes**: Sparse heatmap with clear ★ symbols → dedicated feature detectors
- **No**: Uniform/diffuse heatmap → distributed encoding

**Connect Four Example**:
- Expected: Specialized neurons for immediate wins (critical tactical feature)
- Expected: Diffuse encoding for piece counts (less critical)

---

### 2. Encoding Mechanisms
**Question**: How does the GRU represent binary vs continuous features?

**Answer from Value Plots**:
- **Binary features**: Bimodal distributions (two peaks)
  - Clean separation → feature is well-learned
  - Overlap → ambiguous encoding

- **Continuous features**: Linear/monotonic trends
  - High correlation → simple encoding
  - Nonlinear → complex transformation

**Example**:
```python
# If dimension 17 encodes immediate_win_current as:
#   h[17] = +3.2 when win=1
#   h[17] = -2.1 when win=0
# → Simple threshold detector (like a ReLU on a linear feature)
```

---

### 3. Capacity Analysis
**Question**: Do larger GRUs waste capacity or use it effectively?

**Answer from Dimension Count**:
- Count dimensions with MI > threshold (e.g., 0.1)
- GRU32: 8 high-MI dimensions
- GRU64: 14 high-MI dimensions
- GRU128: 19 high-MI dimensions

**Interpretation**:
- **Linear scaling**: Larger models use extra capacity for redundancy/robustness
- **Sublinear scaling**: Diminishing returns (GRU32 might be sufficient)
- **Superlinear**: Larger models encode features current small models miss

---

### 4. Evolution During Training
**Future Extension** (would require computing per-dimension MI at multiple epochs):

Track which dimensions "specialize" first:
```
Epoch 10: All dimensions have uniform low MI (0.1-0.2)
Epoch 30: Dimension 17 MI spikes to 0.7 for immediate_win_current
Epoch 60: Dimensions 23-27 specialize for three_in_row features
Epoch 100: Stable specialization pattern
```

This reveals the **temporal dynamics of representation learning**.

---

## Usage Example

```bash
# Run MI analysis with per-dimension plots
python scripts/compute_hidden_mutual_info.py \
  --analysis-dir diagnostics/gru_observability \
  --output-dir visualizations/gru_observability \
  --features immediate_win_current three_in_row_current move_index
```

**Outputs**:
- `mi_results.csv` - Overall MI scores
- `mi_per_dimension_k3_c64_gru32.png` - Which dimensions encode what
- `mi_dimension_values_k3_c64_gru32.png` - How dimensions encode features
- (Repeated for each model architecture)

---

## Connection to Literature

### Karpathy et al. (2015): "Interpretable Units"
- Famous example: LSTM unit tracking "inside quotes"
- **Our version**: GRU dimension tracking "immediate win threat"
- **Advantage**: Systematic identification (not cherry-picking)

### Maheswaranathan et al. (2019): "Most units are not interpretable"
- Claimed individual neurons are often uninterpretable
- **Our check**: Count % of dimensions with MI > 0.1
  - If high → units ARE interpretable (disagrees with their finding)
  - If low → confirms distributed encoding

### Sussillo & Barak (2013): "Low-dimensional manifolds"
- Hidden states lie on manifolds despite high-dimensional space
- **Our version**: Few dimensions have high MI → effective dimensionality is low
- **Example**: GRU128 uses only ~20 dimensions meaningfully

---

## Expected Results for Connect Four

### High-MI Features (Tactical):
1. **immediate_win_current** (MI > 0.7)
   - Dimension shows clear bimodal separation
   - Critical for optimal play → must be well-encoded

2. **immediate_win_opponent** (MI > 0.6)
   - Similar encoding to current player's wins
   - May share dimension or use adjacent ones

3. **three_in_row_current** (MI > 0.5)
   - Shows multi-modal distribution (0 vs 1 vs 2+ threats)
   - Possibly multiple dimensions encode threat count

### Medium-MI Features (Strategic):
4. **center_control_current** (MI ≈ 0.3-0.5)
   - Linear trend with piece count
   - Important but not decisive

5. **move_index** (MI ≈ 0.4-0.6)
   - Strong linear encoding (game phase tracker)
   - Helps with opening vs endgame strategy

### Low-MI Features (Less Critical):
6. **piece_diff** (MI < 0.3)
   - In Connect Four, total pieces matter less than configuration
   - Diffuse encoding expected

---

## Diagnostic Use Cases

### Use Case 1: Debugging Poor Performance
**Scenario**: Model loses to random player

**Check**:
1. Look at `mi_per_dimension` heatmap
2. If `immediate_win_current` has low MI → **model isn't detecting wins!**
3. Likely issue: Training data doesn't emphasize end-game positions

**Fix**: Re-balance dataset or add loss term for tactical accuracy

---

### Use Case 2: Comparing Architectures
**Scenario**: GRU32 vs GRU64 performance similar

**Check**:
1. Compare # of high-MI dimensions
2. If GRU32 uses 18/32 dimensions, GRU64 uses 20/64
3. **Conclusion**: GRU32 is more efficient (uses capacity better)

**Decision**: Use GRU32 for deployment (smaller, faster, same quality)

---

### Use Case 3: Feature Engineering
**Scenario**: Should we add new board features?

**Check**:
1. Look at existing MI scores
2. If all features have MI > 0.3 → current features are sufficient
3. If some <0.1 → either useless OR not learned

**Experiment**: Remove low-MI features, retrain, check if performance drops

---

## Summary

**What changed**:
- Old: "The hidden state encodes `immediate_win_current` with MI=0.65"
- New: "Dimension 17 encodes `immediate_win_current` with MI=0.84; it clusters values at ±2.5 for win/no-win states"

**Why it matters**:
- Identifies **specialized neurons** (neuroscience-inspired interpretability)
- Shows **how** encoding works (threshold? linear? multi-modal?)
- Enables **capacity analysis** (how many dimensions are "active"?)
- Supports **debugging** (which features are poorly learned?)

**Recommended workflow**:
1. Run analysis on final epoch
2. Check per-dimension heatmap for specialization
3. Check value plots for encoding quality
4. Identify dimensions for further investigation (saliency, ablation, etc.)

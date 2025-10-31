# Mutual Information Analysis

**Idea in Brief**: Mutual information (MI) measures how much uncertainty about a
game feature disappears when you observe a hidden activation. Unlike linear
probes, MI sees any dependency—nonlinear, distributed, or otherwise—and thus
reveals representation structure that probes might miss.

**Quick start**:
- Run `./wt.sh observability mi` to execute
  `scripts/compute_hidden_mutual_info.py` after the standard analysis stage.
- Outputs cover both the final checkpoint and the best-validation checkpoint,
  with per-feature CSVs, heatmaps, and per-dimension activation plots stored
  under `visualizations/gru_observability/`.
- For a narrative walkthrough of the workflow and interpretation, see
  `guide_representation_diagnostics.md`.

## Conceptual Foundation

Mutual information quantifies statistical dependence between variables without assuming linear relationships. Applied to GRU hidden states, MI analysis reveals what information the network encodes and how it distributes that information across neurons. Unlike linear probes, which test only linear accessibility, MI detects any form of dependency—including nonlinear transformations and distributed codes.

For a hidden dimension h_i and target feature y, MI(h_i; y) measures how much knowing h_i reduces uncertainty about y. Measured in bits (log base 2) or nats (natural log), MI equals zero for independent variables and increases with stronger dependence. Computing MI at both aggregate and per-dimension levels distinguishes specialized neurons (single dimensions with high MI) from distributed encoding (information spread across many dimensions).

## Quantities Computed

### 1. Mean MI (aggregate)
- What: Average MI across hidden dimensions for a given feature.
- Definition: mean_i MI(h_i, y) where h_i is dimension i and y is the feature.
- Purpose: Summarize overall encoding capacity without assuming linearity.

### 2. Per‑dimension MI
- What: MI(h_i, y) for each hidden dimension i.
- Purpose: Identify specialized units and redundancy across units.

### 3. Value distributions for high‑MI units
- What: For best_dim = argmax_i MI(h_i, y), plot h_best_dim versus y.
- Purpose: Characterize encoding (bimodal thresholds, linear trends, multi‑modal patterns).

---

## Visualization Outputs

### Per-Dimension MI Heatmap

**Format**: Heatmap showing MI(h_i, y) for all hidden dimensions i and board features y.

**Axes**:
- Vertical: 12 board features (immediate_win_current, three_in_row_opponent, etc.)
- Horizontal: Hidden dimension indices (varies by GRU size: 32, 64, or 128)
- Color intensity: MI magnitude (darker indicates higher dependence)

**Interpretation patterns**:

**Specialization** appears as isolated high-MI cells—single dimensions strongly encoding specific features. Example: dimension 17 shows MI=0.84 with immediate_win_current while other dimensions show MI<0.2.

**Co-encoding** manifests as adjacent high-MI dimensions for related features. Example: dimensions 23-25 all show elevated MI with both current_player and immediate_win_current, suggesting a shared subspace for turn-dependent threat detection.

**Distributed encoding** appears as multiple scattered high-MI dimensions for one feature. Example: center_control shows moderate MI (0.3-0.5) across dimensions 8, 15, 29, and 41—information spread rather than concentrated.

**Weak representation** shows as uniformly low MI across all dimensions for a feature, indicating the network has not learned to encode that variable.

**Training dynamics**: Early checkpoints typically show diffuse, weak MI values. As training progresses, MI patterns sharpen—trained models exhibit clearer specialization. Architectural effects also emerge: larger GRUs tend toward distributed encoding while smaller GRUs show sharper specialization due to capacity constraints.

---

### Dimension Value Distributions

**Format**: Grid of plots showing how the highest-MI dimension for each feature relates to that feature's values.

**Layout**: One subplot per board feature. Each shows:
- Horizontal axis: Feature values (binary 0/1 or continuous range)
- Vertical axis: Activation of the dimension with highest MI for this feature
- Title: Feature name, dimension index, and MI score

**Plot types by feature**:

Binary features (immediate_win_current, current_player) use violin or box plots to show distribution separation between categories. Well-encoded binary features show distinct distributions with minimal overlap.

Continuous features (move_index, piece_count_diff) use scatter plots to reveal encoding structure. Strong linear trends indicate proportional encoding. Nonlinear patterns (clusters, thresholds, multi-modal distributions) indicate more complex transformations.

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
Neuron specialization
- Hypothesis: GRU units specialize for specific game concepts when capacity is sufficient and training progresses.
- Evidence: Sparse high‑MI patterns for critical features (e.g., immediate wins) versus diffuse MI for less decisive variables (e.g., piece counts).

---

### 2. Encoding Mechanisms
Encoding mechanisms
- Binary variables often appear as bimodal hidden activations (threshold‑like separation).
- Continuous variables may show linear/monotonic trends in high‑MI units; nonlinear patterns indicate more complex transformations.

---

### 3. Capacity Analysis
Capacity analysis
- Count dimensions with MI above a threshold to estimate effective capacity.
- Linear increase suggests useful redundancy; sublinear suggests diminishing returns; superlinear suggests qualitatively new encodings in larger models.

---

Training evolution
- Compute per‑dimension MI across checkpoints to track the onset and stabilization of specialization.
- Reveals temporal dynamics of representation learning and aligns with performance changes.

---

## Usage Example

```bash
# Run MI analysis with per-dimension plots
python scripts/compute_hidden_mutual_info.py \
  --analysis-dir diagnostics/gru_observability \
  --output-dir visualizations/gru_observability \
  --features immediate_win_current three_in_row_current move_index
```

Outputs:
- `mi_results.csv` – aggregated MI scores
- `mi_per_dimension_*.png` – dimension‑wise MI heatmaps
- `mi_dimension_values_*.png` – encoding patterns for highest‑MI units

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

**Analysis workflow**:
1. Examine per-dimension heatmap to identify specialized dimensions and encoding patterns
2. Review value distribution plots to understand encoding mechanisms (linear, threshold, multi-modal)
3. Cross-reference with probe accuracy to validate MI findings
4. For high-MI dimensions, consider ablation studies or gradient analysis to test causal importance

---

## Integration with Analysis Framework

**Theoretical background**: [Theoretical Foundations](theoretical_foundations.md#3-information-theory-for-interpretability) develops the mathematical foundation for MI estimation, including connections to entropy, conditional probability, and information-theoretic bounds.

**Comparative methods**: [Case Studies](case_studies.md) presents AlphaZero concept probing (case study #3) and Karpathy's interpretable neuron discovery (case study #6), showing how MI analysis complements and extends manual inspection approaches.

**Research frontiers**: [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md) identifies MI temporal tracking as a high‑impact addition, enabling analysis of how representations organize during training.

**Complete bibliography**: [References](references.md) section "Information Theory & Mutual Information" provides citations for MI estimation methods, applications to neural network interpretability, and theoretical foundations.

# Plot & Output Reference

Complete guide to interpreting every visualization and CSV output produced by the analysis tools.

---

## Quick Navigation by Analysis Type

### Weight-Space Analysis
- **[embeddings_weights](embeddings_weights.md)** - Weight trajectory embeddings (PCA, t-SNE, UMAP, PHATE)
- **[checkpoint_metrics_csv](checkpoint_metrics_csv.md)** - Weight norms, step statistics, SVD metrics
- **[trajectory_metric_space](trajectory_metric_space.md)** - Metric-based trajectory embeddings

### GRU Interpretability
- **[gru_observability](gru_observability.md)** - Gates, timescales, PHATE, probes
- **[gru_mutual_info](gru_mutual_info.md)** - Feature encoding analysis (12 features)
- **[fixed_points](fixed_points.md)** - Attractor finding and evolution

### Cross-Model Comparison
- **[cka](cka.md)** - Representational similarity heatmaps
- **[factorial_heatmaps](factorial_heatmaps.md)** - Factorial design analysis (9 metrics × 9 models)

### CNN Visualization
- **[activations](activations.md)** - Grad-CAM attention maps
- **[visualize_unified](visualize_unified.md)** - Unified visualization suite

---

## Plots by Command

| Command | Documentation | Key Outputs |
|---------|---------------|-------------|
| `./wt.sh embeddings` | [embeddings_weights](embeddings_weights.md) | CNN/GRU/all weight trajectories (4 methods) |
| `./wt.sh metrics` | [checkpoint_metrics_csv](checkpoint_metrics_csv.md) | `*_metrics.csv` with norms, cosines, SVD |
| `./wt.sh trajectory-embedding` | [trajectory_metric_space](trajectory_metric_space.md) | `trajectory_embedding_*.png` |
| `./wt.sh observability analyze` | [gru_observability](gru_observability.md) | Gate trajectories, timescales, PHATE, probes |
| `./wt.sh observability mi` | [gru_mutual_info](gru_mutual_info.md) | MI heatmaps, per-dimension analysis |
| `./wt.sh observability fixed` | [fixed_points](fixed_points.md) | Fixed-point NPZ files |
| `./wt.sh observability evolve` | [fixed_points](fixed_points.md) | Classification counts, drift plots |
| `./wt.sh cka` | [cka](cka.md) | CKA similarity heatmaps |
| `./wt.sh factorial` | [factorial_heatmaps](factorial_heatmaps.md) | Factorial analysis heatmaps (9 metrics) |
| `./wt.sh visualize --viz-type activations` | [activations](activations.md) | Grad-CAM heatmaps |
| `./wt.sh visualize` | [visualize_unified](visualize_unified.md) | Multi-type visualization suite |

---

## By Analysis Goal

### "I want to understand training dynamics"
1. **Start**: [checkpoint_metrics_csv](checkpoint_metrics_csv.md)
   - Weight norms over epochs
   - Step cosines (training stability)
   - Representation variance (capacity usage)

2. **Visualize**: [embeddings_weights](embeddings_weights.md)
   - PHATE trajectories showing parameter evolution
   - Compare CNN vs GRU learning rates

3. **Cross-compare**: [trajectory_metric_space](trajectory_metric_space.md)
   - See all 9 models in metric space
   - Identify architectural families

### "I want to interpret the GRU"
**Complete pipeline**: [GRU Interpretability Workflow](../workflows/gru_interpretability.md)

1. **Gates & memory**: [gru_observability](gru_observability.md)
   - Update/reset gate trajectories
   - Eigenvalue timescales (how long GRU remembers)
   - PHATE embeddings colored by features

2. **Feature encoding**: [gru_mutual_info](gru_mutual_info.md)
   - Which features are encoded? (MI heatmap)
   - Which neurons encode what? (per-dimension MI)
   - How well encoded? (violin plots)

3. **Dynamical systems**: [fixed_points](fixed_points.md)
   - How many attractors exist?
   - Are they stable?
   - How do they evolve during training?

### "I want to compare different architectures"
1. **Representational similarity**: [cka](cka.md)
   - Do GRU32 and GRU64 learn the same representations?
   - Hierarchical clustering shows architectural families

2. **Training dynamics**: [trajectory_metric_space](trajectory_metric_space.md)
   - Do models converge to similar metric profiles?
   - Which GRU sizes explore more?

### "I want to understand CNN decisions"
- **Grad-CAM**: [activations](activations.md)
  - Which board cells matter for policy?
  - Which cells matter for value?
  - Does CNN focus on threats/opportunities?

---

## Documentation Depth Guide

Each plot guide includes:

- Purpose: what question does this answer?
- Data pipeline: how is data collected?
- Algorithms: complete computational details
- Formulas: mathematical definitions
- Interpretation: how to read the plots
- Examples: concrete scenarios with expected patterns
- Diagnostics: troubleshooting and use cases

**Most comprehensive**:
- [gru_mutual_info](gru_mutual_info.md) (403 lines) - Board features, MI computation, neuron specialization
- [gru_observability](gru_observability.md) (213 lines) - GRU equations, eigenvalues, PHATE, probing
- [fixed_points](fixed_points.md) (162 lines) - Optimization algorithm, Jacobian, stability
- [checkpoint_metrics_csv](checkpoint_metrics_csv.md) (150 lines) - Weight metrics, SVD analysis

---

## Computational Methods Reference

Looking for specific algorithms?

| Method | Documentation | Location |
|--------|---------------|----------|
| **Board feature extraction** | [gru_mutual_info](gru_mutual_info.md#board-feature-extraction) | 12 features from Connect Four boards |
| **GRU gate computation** | [gru_observability](gru_observability.md#gate-activation-computation) | Manual GRU forward pass |
| **Eigenvalue timescales** | [gru_observability](gru_observability.md#timescale-computation) | τ = 1/\|log(\|λ\|)\| |
| **Reservoir sampling** | [gru_observability](gru_observability.md#gate-activation-computation) | Vitter's Algorithm R |
| **PHATE embedding** | [gru_observability](gru_observability.md#phate-embedding) | knn adaptation, preprocessing |
| **Logistic probing** | [gru_observability](gru_observability.md#logistic-regression-probing) | Train/test split, accuracy/F1 |
| **SVD for representations** | [checkpoint_metrics_csv](checkpoint_metrics_csv.md#4-representation-svd-analysis-optional) | Centering, variance ratios |
| **Fixed-point finding** | [fixed_points](fixed_points.md#fixed-point-finding-algorithm-discovery-stage) | Adam optimization, deduplication |
| **Jacobian stability** | [fixed_points](fixed_points.md#4-stability-classification) | Eigenvalues, spectral radius |
| **Linear CKA** | [cka](cka.md#2-linear-cka-formula) | Gram matrices, centering |
| **Grad-CAM** | [activations](activations.md#grad-cam-algorithm) | Gradient-weighted attention |
| **Feature standardization** | [trajectory_metric_space](trajectory_metric_space.md#3-standardization) | StandardScaler normalization |

---

## Cross-References

- **Commands that produce these**: [Commands Reference](../commands/)
- **End-to-end workflows**: [Workflows](../workflows/)
- **Main manual**: [User Manual Home](../)
- **Scientific background**: [Scientific Background](../../scientific/)

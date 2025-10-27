# Comparing Learning Trajectories Across Different Architectures
## Research-Backed Analysis Framework

**Problem**: You tried UMAP/t-SNE on weight trajectories but couldn't compare across settings (8,32,128 vs 16,64,256).
**Why it failed**: Different architectures have different parameter counts → incompatible vector spaces.

---

## What We Can Do (Without Re-Running Training)

### ✅ Option 1: Representation Similarity (CKA) **[RECOMMENDED]**

**What**: Compare learned representations (activations) instead of weights
**How**: Use Centered Kernel Alignment (CKA) - industry standard since 2019
**Data needed**: ✅ We already have! (board representations from checkpoints)

#### Why CKA?

From research (Kornblith et al., ICML 2019):
- "CKA can reliably identify correspondences between representations in networks trained from different initializations"
- "Unlike CCA, CKA works across different architectures"
- Used to compare: ResNet vs Inception, Wide vs Deep networks

#### Implementation Plan:

```python
# For each checkpoint epoch (3, 6, 9, ..., 100):
#   For each model (9 total):
#     1. Load checkpoint
#     2. Pass fixed test set through network
#     3. Extract GRU hidden states (128-dim for GRU128, 32-dim for GRU32, etc.)
#     4. Compute CKA between all model pairs

# Result: 9×9 similarity matrix per epoch
# Visualization: Heatmap animation showing how similarity evolves
```

**What this reveals**:
- Do c16_gru32 and c64_gru32 converge to similar representations?
- When does representation divergence happen?
- Which models learn similar solutions despite different architectures?

**Pros**:
- ✅ Works across different architectures
- ✅ We have the checkpoints
- ✅ Interpretable (0 = completely different, 1 = identical)
- ✅ Used in 100+ papers since 2019

**Cons**:
- Requires loading all checkpoints (~30 min computation)
- Needs careful choice of test set (use same boards for all models)

---

### ✅ Option 2: Metric-Space Trajectory Embedding **[EASIER]**

**What**: Embed trajectories based on METRICS we computed, not raw weights
**How**: UMAP/t-SNE on the feature vectors we already have!

#### We Already Have Rich Features:

Each checkpoint has:
- `weight_norm`, `step_norm`, `step_cosine`, `relative_step`
- `repr_total_variance`, `repr_top1_ratio`, `repr_top2_ratio`, ...
- Training/validation loss

**Key Insight**: These metrics are **architecture-agnostic**! A "step_norm" is comparable whether it's from GRU8 or GRU128.

#### Implementation:

```python
# Create feature matrix (306 checkpoints × ~15 features)
features = [
    'epoch',                   # Temporal context
    'weight_norm',            # Trajectory position
    'step_norm',              # Movement speed
    'step_cosine',            # Movement direction
    'relative_step',          # Scale-free speed
    'repr_total_variance',    # Representation quality
    'repr_top1_ratio',        # Collapse measure
    'val_loss',               # Performance
    'train_loss',
    'train_val_gap',          # Overfitting
    # ... normalized versions
]

# Apply UMAP with categorical coloring:
#   - Color by model (9 categories)
#   - Size by epoch (3 → 100)
#   - Shape by GRU size, transparency by channel count
```

**What this reveals**:
- Do GRU8 models cluster together?
- Does c64_gru32 occupy a unique region?
- Can we see the overfitting trajectory (GRU128 curves back?)
- Mode connectivity: smooth paths or jumps?

**Pros**:
- ✅ Uses data we already have (CSV files!)
- ✅ Easy to implement (~50 lines of code)
- ✅ Directly comparable across all architectures
- ✅ Can validate against known findings (GRU size >> channel count)

**Cons**:
- Doesn't capture full representational content
- UMAP hyperparameters matter (neighbors, min_dist)

---

### ⚠️ Option 3: Loss Landscape Connectivity **[EXPENSIVE]**

**What**: Analyze mode connectivity between checkpoints
**How**: Linear interpolation + loss evaluation

#### From Research (Draxler et al., 2018):

> "Optima of complex loss functions are connected by simple polygonal chains with only one bend"

We can check:
- Is c64_gru32 (best validation) connected to c64_gru128 via low-loss path?
- Do GRU8 checkpoints lie on a flat plateau?

#### Implementation:

```python
# For key checkpoint pairs (e.g., epoch 10 and 100):
#   For interpolation α ∈ [0, 1]:
#     θ_interp = α·θ_1 + (1-α)·θ_2
#     Compute val_loss(θ_interp)
#   Plot loss barrier
```

**Pros**:
- ✅ We have checkpoints
- ✅ Reveals optimization landscape geometry
- ✅ Can explain why GRU128 overfits (sharp vs flat minima)

**Cons**:
- ❌ Only works within same architecture (can't interpolate between different param counts)
- ❌ Computationally expensive (~100 loss evaluations per pair)
- ❌ Requires loading models and running forward passes

---

## About Gradients: Do We Need Them?

### Short Answer: **NO**, for your paper's goals

Here's why:

**What gradients tell us**:
1. **Gradient norms** → Training stability (exploding/vanishing)
2. **Gradient directions** → Update directions (we approximate this via `step_cosine`!)
3. **Hessian eigenvalues** → Loss curvature (sharp vs flat minima)

**What we can already infer without gradients**:

| Gradient Info | Our Approximation | Evidence |
|--------------|-------------------|----------|
| Gradient norm | `step_norm / learning_rate` | Step size ∝ gradient × LR |
| Gradient direction | `step_cosine` | Measures alignment of consecutive updates |
| Curvature | Val loss increase + weight norm flatness | Overfit = sharp minimum |
| Flow dynamics | Weight trajectory smoothness | Captures overall learning dynamics |

**From Research** (Garipov et al., 2018):
> "Mode connectivity analysis requires only checkpoints, not gradient history"

**When gradients ARE necessary**:
- Studying vanishing/exploding gradient problem specifically
- Analyzing gradient stochasticity (batch-to-batch variance)
- Computing exact Hessian eigenvalues for sharpness

**For your paper ("Connect-4 as Testbed")**:
- ❌ Not studying gradient pathologies
- ❌ Not studying SGD noise specifically
- ✅ Studying learning trajectories → checkpoint analysis suffices!
- ✅ Studying generalization → val loss + representation quality (we have both!)

---

## My Recommended Analysis Pipeline

### Phase 1: Quick Wins (What We Can Do This Week)

**1. Metric-Space UMAP** (2 hours)
```bash
# Script: scripts/visualize_trajectory_embedding.py
# Input:  diagnostics/trajectory_analysis/*.csv
# Output: visualizations/trajectory_umap.png
# Shows: All 9 models' trajectories in shared metric space
```

**2. Representation Similarity Matrix** (4 hours)
```bash
# Script: scripts/compute_representation_similarity.py
# Input:  checkpoints/save_every_3/*/checkpoint_*.pt
#         Fixed test set of 100 board positions
# Output: visualizations/cka_evolution.mp4 (animated heatmap)
# Shows: Which models converge to similar representations
```

**3. Per-Epoch Clustering** (1 hour)
```bash
# For epochs 10, 30, 60, 100:
#   Cluster models by (val_loss, repr_collapse, weight_growth)
#   Visualize: Does cluster membership change over training?
```

### Phase 2: Paper-Worthy Deep Dive (If Needed)

**4. Mode Connectivity** (1 day)
- Between best model (c64_gru32) and worst (c64_gru128)
- Shows: Is there a low-loss path connecting them?

**5. Representation Geometry** (2 days)
- PCA/UMAP on actual hidden states (not weights!)
- Compare: Does GRU8 occupy a lower-dimensional manifold?

---

## What NOT to Do

❌ **Don't**: Try to embed raw weight vectors across architectures
   - Different dimensions → incomparable
   - Weights are high-D and noisy

❌ **Don't**: Re-run training with gradient logging
   - Expensive (~1 week compute)
   - Marginal benefit for your research questions

❌ **Don't**: Use standard SVCCA/PWCCA for cross-architecture comparison
   - CKA is superior (proven in multiple papers)

✅ **Do**: Focus on representation similarity (CKA) + metric-space embedding
   - Uses existing checkpoints
   - Directly answers your questions
   - Standard methodology (cite Kornblith et al.)

---

## Concrete Next Steps

**I recommend we implement**:

1. **Metric-space UMAP** first (easiest, immediate results)
   - Creates compelling visualization for paper
   - Validates our factorial findings

2. **CKA analysis** second (more rigorous)
   - Produces quantitative similarity scores
   - Standard in field (easy to defend in reviews)

3. **Write it up** with proper framing:
   - "We analyze learning trajectories via checkpoint-based metrics and representation similarity (CKA)"
   - "Gradient-free analysis allows retrospective study of pre-trained models"
   - Cite: Kornblith et al. (2019), Draxler et al. (2018)

**Want me to implement the metric-space UMAP visualization?** It'll take ~30 min and give us immediate insights!

---

## References

**Representation Similarity**:
- Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited" (ICML) - CKA
- Raghu et al. (2021) "Do Vision Transformers See Like CNNs?" (NeurIPS) - CKA applications

**Mode Connectivity**:
- Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast Ensembling" (NeurIPS)
- Draxler et al. (2018) "Essentially No Barriers in Neural Network Energy Landscape" (ICML)

**Gradient Flow** (for context):
- Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets" (NeurIPS)
- Goodfellow et al. (2015) "Qualitatively characterizing neural network optimization problems" (ICLR)

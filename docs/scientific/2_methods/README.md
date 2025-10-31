# Interpretability Methods

This section provides detailed guides for interpretability techniques. Each method is documented with theory, implementation, use cases, and limitations.

---

## Method Selection Guide

| Question | Recommended Methods | Why |
|----------|-------------------|-----|
| Why did the model make this prediction? | [Grad-CAM](feature_visualization/gradient_based.md), [Integrated Gradients](feature_visualization/attribution_methods.md), [SHAP](feature_visualization/attribution_methods.md) | Provides input attribution |
| Is concept X encoded? | [Linear Probes](probing/linear_probes.md), [Mutual Information](probing/mutual_information.md) | Tests for specific knowledge |
| Does concept X causally matter? | [Causal Probing](probing/causal_probing.md), [Activation Patching](mechanistic_interpretability/activation_patching.md) | Validates causal importance |
| What patterns did it learn? | [UMAP](representation_analysis/dimensionality_reduction.md), [CKA](representation_analysis/similarity_metrics.md), Feature Visualization | Reveals learned structure |
| How do representations evolve? | [PHATE](dynamical_analysis/trajectory_analysis.md), [CKA over checkpoints](representation_analysis/similarity_metrics.md) | Tracks training dynamics |
| How does this RNN compute? | [Fixed Points](dynamical_analysis/fixed_points.md), [Attractor Analysis](dynamical_analysis/attractor_landscapes.md) | Reveals computational modes |
| What algorithm is implemented? | [Circuit Analysis](mechanistic_interpretability/circuits.md), [Activation Patching](mechanistic_interpretability/activation_patching.md) | Reverse-engineers computation |

---

## Method Categories

### [Feature Visualization](feature_visualization/)
Techniques for visualizing what features drive model decisions.

- [Gradient-Based Methods](feature_visualization/gradient_based.md): Saliency maps, Grad-CAM, SmoothGrad
- [Activation Maximization](feature_visualization/activation_maximization.md): DeepDream, feature visualization
- [Attribution Methods](feature_visualization/attribution_methods.md): Integrated Gradients, SHAP, LIME, Layer-wise Relevance Propagation

**When to use**: Explaining individual predictions, debugging misclassifications

---

### [Probing](probing/)
Testing what information is encoded in representations.

- [Linear Probes](probing/linear_probes.md): Simple classifiers on hidden states
- [Mutual Information](probing/mutual_information.md): Model-free dependence measurement
- [Causal Probing](probing/causal_probing.md): Intervention-based validation

**When to use**: Validating that expected concepts are learned, comparing representations

---

### [Representation Analysis](representation_analysis/)
Understanding structure and geometry of learned representations.

- [Dimensionality Reduction](representation_analysis/dimensionality_reduction.md): PCA, t-SNE, UMAP, PHATE
- [Similarity Metrics](representation_analysis/similarity_metrics.md): CKA, SVCCA, RSA
- [Geometry Analysis](representation_analysis/geometry_analysis.md): Manifold structure, curvature

**When to use**: Visualizing learned structure, comparing models, understanding representation space

---

### [Dynamical Analysis](dynamical_analysis/)
Methods for analyzing recurrent networks as dynamical systems.

- [Fixed Points](dynamical_analysis/fixed_points.md): Finding and characterizing stable states
- [Trajectory Analysis](dynamical_analysis/trajectory_analysis.md): Visualizing evolution in state/weight space
- [Attractor Landscapes](dynamical_analysis/attractor_landscapes.md): Understanding computational structure

**When to use**: Understanding RNNs, LSTMs, GRUs; analyzing training dynamics

---

### [Mechanistic Interpretability](mechanistic_interpretability/)
Reverse-engineering networks into interpretable algorithms and circuits.

- [Circuits](mechanistic_interpretability/circuits.md): Decomposing into functional components
- [Sparse Autoencoders](mechanistic_interpretability/sparse_autoencoders.md): Extracting monosemantic features
- [Activation Patching](mechanistic_interpretability/activation_patching.md): Causal interventions

**When to use**: Deep mechanistic understanding, validating hypotheses causally, safety-critical applications

---

## Method Comparison

### Computational Cost

| Method | Speed | Scalability | Notes |
|--------|-------|-------------|-------|
| Grad-CAM | Fast | High | Single backward pass |
| Integrated Gradients | Medium | Medium | Multiple forward passes |
| SHAP | Slow-Fast | Medium | Depends on variant (TreeExplainer vs KernelExplainer) |
| Linear Probes | Fast | High | Simple training |
| Mutual Information | Slow | Low | k-NN estimation expensive |
| t-SNE | Slow | Low | O(n²) complexity |
| UMAP | Fast | Medium | Much faster than t-SNE |
| PHATE | Medium | Medium | Comparable to UMAP |
| Fixed Points | Very Slow | Low | Optimization per context |
| CKA | Fast | High | Matrix operations |

---

### Assumptions and Limitations

| Method | Key Assumption | Main Limitation |
|--------|---------------|-----------------|
| Grad-CAM | Gradients reflect importance | Only applicable to differentiable models |
| Linear Probes | Linear accessibility sufficient | May miss non-linearly encoded info |
| Mutual Information | Sufficient samples available | Sample complexity, estimation bias |
| SHAP | Feature independence (for some variants) | Computationally expensive |
| t-SNE | Local structure matters most | Can distort global structure |
| Fixed Points | Dynamics can be linearized locally | Finding all fixed points is hard |
| Circuit Analysis | Circuits are modular | Labor-intensive manual analysis |

---

## Validation and Best Practices

### Essential Controls

For any interpretability method, include:

1. **Randomization tests**: Run on models with randomized weights or shuffled labels
2. **Baseline comparisons**: Compare to chance performance or simple baselines
3. **Multiple methods**: Validate findings with complementary techniques
4. **Statistical testing**: Quantify uncertainty, test significance
5. **Sanity checks**: Verify method behaves sensibly on known cases

### Reporting Standards

When presenting interpretability results:

- Report all hyperparameters
- Include confidence intervals or error bars
- Show representative examples AND aggregate statistics
- Document negative results and limitations
- Provide sufficient detail for reproduction

---

## Getting Started

### For Beginners
Start with:
1. [Grad-CAM](feature_visualization/gradient_based.md) for visual intuition
2. [Linear Probes](probing/linear_probes.md) for testing specific hypotheses
3. [UMAP](representation_analysis/dimensionality_reduction.md) for visualization

### For Intermediate Users
Explore:
1. [Integrated Gradients](feature_visualization/attribution_methods.md) for rigorous attribution
2. [Mutual Information](probing/mutual_information.md) for model-free analysis
3. [CKA](representation_analysis/similarity_metrics.md) for comparing models

### For Advanced Users
Investigate:
1. [Fixed Points](dynamical_analysis/fixed_points.md) for RNN dynamics
2. [Circuit Analysis](mechanistic_interpretability/circuits.md) for mechanistic understanding
3. [Causal Probing](probing/causal_probing.md) for rigorous validation

---

## Architecture-Specific Recommendations

### CNNs
Primary methods: Grad-CAM, feature visualization, layer-wise probes

### RNNs/LSTMs/GRUs
Primary methods: Fixed points, trajectory analysis, hidden state probes

### Transformers
Primary methods: Attention visualization, circuit analysis, activation patching

### General (any architecture)
Primary methods: Linear probes, SHAP, UMAP, CKA

---

## Further Resources

- [Tutorials](../6_tutorials/): Hands-on notebooks for each method
- [Case Studies](../5_case_studies/): Real-world applications
- [Tools](../4_tools_and_libraries/): Software implementations
- [References](../references/): Papers and citations

Return to [main handbook](../0_start_here/README.md)

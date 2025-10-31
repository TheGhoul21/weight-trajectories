# Interpretability for Researchers

This guide is for researchers developing new interpretability methods, studying representation learning, or exploring fundamental questions about neural network computation. We cover research frontiers, methodological rigor, and open problems.

---

## Contents

1. [Current Research Landscape](#current-research-landscape)
2. [Methodological Best Practices](#methodological-best-practices)
3. [Open Problems by Area](#open-problems-by-area)
4. [Experimental Design](#experimental-design)
5. [Theory Gaps](#theory-gaps)
6. [Recommended Reading Paths](#recommended-reading-paths)

---

## Current Research Landscape

### Major Research Directions (2024-2025)

#### 1. Mechanistic Interpretability

**Goal**: Reverse-engineer neural networks into interpretable algorithms/circuits.

**Key developments**:
- **Circuit discovery** (Anthropic, OpenAI): Decomposing transformer computations into functional components (induction heads, copying circuits, etc.)
- **Sparse autoencoders** (Anthropic): Extracting monosemantic features from superposition
- **Automated circuit discovery**: Scaling manual circuit analysis with search algorithms

**Frontier questions**:
- Can we fully reverse-engineer an LLM?
- How do circuits compose hierarchically?
- Can we transfer circuits between models?

**Key papers**:
- Elhage et al. (2021): Mathematical Framework for Transformer Circuits
- Olah et al. (2020): Zoom In - An Introduction to Circuits
- Cunningham et al. (2023): Sparse Autoencoders Find Highly Interpretable Features

**Relevant handbook sections**:
- [Mechanistic Interpretability](../2_methods/mechanistic_interpretability/)
- [Advanced Topics: Superposition](../7_advanced_topics/superposition.md)

---

#### 2. Scaling Interpretability

**Goal**: Develop methods that work for billion-parameter models.

**Key challenges**:
- Computational cost of analyzing activations
- Sampling strategies for massive datasets
- Automating interpretation (can't manually inspect everything)

**Approaches**:
- Automated feature labeling (using smaller models to interpret larger ones)
- Dimensionality reduction before detailed analysis
- Efficient attribution approximations

**Frontier questions**:
- Do interpretability methods scale differently than capabilities?
- Can we interpretinterpret emergent behaviors in large models?
- How to validate findings at scale?

**Key papers**:
- Bills et al. (2023): Language models can explain neurons in language models
- Kaplan et al. (2020): Scaling laws (capabilities context)

**Relevant handbook sections**:
- [Advanced Topics: Scaling Laws](../7_advanced_topics/scaling_laws.md)

---

#### 3. Representation Learning Dynamics

**Goal**: Understand how representations evolve during training.

**Key phenomena**:
- **Grokking**: Sudden generalization long after overfitting
- **Phase transitions**: Discrete jumps in capability
- **Progressive differentiation**: Gradual emergence of feature hierarchy
- **Lottery ticket hypothesis**: Existence of trainable subnetworks

**Methods**:
- Trajectory analysis (PHATE, CKA over checkpoints)
- Fixed-point tracking in RNNs
- Singular value decomposition of weight matrices
- Loss landscape visualization

**Frontier questions**:
- What determines phase transition timing?
- Can we predict when concepts will emerge?
- How do different layers co-evolve?

**Key papers**:
- Saxe et al. (2019): Mathematical theory of semantic development
- Nanda et al. (2024): Progress measures for grokking
- Fort & Jastrzębski (2019): Large scale structure of loss landscapes

**Relevant handbook sections**:
- [Dynamical Analysis Methods](../2_methods/dynamical_analysis/)
- [Advanced Topics: Emergence](../7_advanced_topics/emergence_and_phase_transitions.md)
- [Weight Trajectory Theory](../weight_embeddings_theory.md)

---

#### 4. Causal Understanding

**Goal**: Move beyond correlational to causal interpretability.

**Key developments**:
- **Activation patching**: Swap activations between runs to test causal importance
- **Causal abstraction framework**: Formal verification of interpretations
- **Causal mediation analysis**: Quantify indirect effects through representations

**Frontier questions**:
- When are correlational probes misleading?
- How to efficiently test causality at scale?
- Can we build causal models of entire networks?

**Key papers**:
- Geiger et al. (2021): Causal abstractions of neural networks
- Meng et al. (2022): Locating and editing factual associations (ROME)
- Pearl (2009): Causality (foundational)

**Relevant handbook sections**:
- [Probing: Causal Methods](../2_methods/probing/causal_probing.md)
- [Mechanistic Interpretability: Activation Patching](../2_methods/mechanistic_interpretability/activation_patching.md)

---

#### 5. Neuroscience-AI Alignment

**Goal**: Connect artificial and biological neural network computation.

**Key findings**:
- Convergent representations (CNNs and visual cortex)
- Shared dynamical motifs (RNNs and cortical circuits)
- Grid cells emerge in path integration RNNs
- Attractor dynamics in both systems

**Frontier questions**:
- Are computational primitives universal across substrate?
- What can neuroscience predict about optimal AI architectures?
- Can AI interpretability inform neuroscience experiments?

**Key papers**:
- Yamins & DiCarlo (2016): Using goal-driven deep learning to understand sensory cortex
- Cueva & Wei (2018): Emergence of grid-like representations
- Khona & Fiete (2022): Attractor and integrator networks in the brain

**Relevant handbook sections**:
- [Case Studies: Neuroscience](../5_case_studies/neuroscience/)
- [Theoretical Foundations](../theoretical_foundations.md) (§5)

---

### Cross-Cutting Themes

#### Superposition

**The problem**: Features are represented in superposition (multiple features per neuron), making interpretation hard.

**Why it happens**: More features to represent than dimensions available (compression).

**Current solutions**: Sparse autoencoders to decompose superposition.

**Open questions**: When does superposition vs dedicated dimensions occur? Can we control it?

---

#### Polysemanticity

**The problem**: Individual neurons respond to multiple unrelated concepts (not monosemantic).

**Relationship to superposition**: Polysemanticity is the observable consequence of superposition.

**Approaches**: Dictionary learning, sparse coding, independent component analysis.

---

## Methodological Best Practices

### Rigor Standards for Interpretability Research

#### 1. Controls and Baselines

**Always include**:
- **Randomization tests**: Shuffled labels, randomized models
- **Null models**: What would happen by chance?
- **Positive controls**: Known ground-truth cases

**Example**:
```python
# Probe analysis with proper controls
results = {
    'probe_accuracy': probe.score(X_test, y_test),
    'shuffled_labels': probe.score(X_test, np.random.permutation(y_test)),
    'random_features': probe_random_features(X_test, y_test),
    'chance_level': compute_chance_level(y_test)  # account for class imbalance
}

# Statistical test
from scipy.stats import binomtest
p_value = binomtest(
    correct_predictions,
    total_predictions,
    results['chance_level']
).pvalue
```

---

#### 2. Multiple Validation Methods

**Triangulation principle**: Don't rely on a single technique.

**Example validation chain**:
1. **Probe finds feature X is encoded** (correlational)
2. **MI confirms high mutual information** (model-free validation)
3. **Activation patching shows it's causal** (intervention)
4. **Ablation confirms necessity** (removal test)

---

#### 3. Statistical Rigor

**Requirements**:
- Report confidence intervals (bootstrap or analytic)
- Correct for multiple comparisons (Bonferroni, FDR)
- State assumptions explicitly
- Pre-register hypotheses when possible

**Example**:
```python
from statsmodels.stats.multitest import multipletests

# Testing 100 neurons for encoding of 10 features = 1000 tests
p_values = []  # collect from all tests

# Correct for multiple comparisons
rejected, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'  # Benjamini-Hochberg FDR control
)

print(f"Significant after FDR correction: {rejected.sum()}/{len(rejected)}")
```

---

#### 4. Reproducibility

**Best practices**:
- Fix random seeds
- Report all hyperparameters
- Archive checkpoints and activations
- Share code and data (when possible)
- Document preprocessing steps

**Checklist**:
- [ ] Random seeds documented
- [ ] Full hyperparameter specifications
- [ ] Data splits specified
- [ ] Software versions recorded
- [ ] Code repository public

---

### Common Pitfalls in Interpretability Research

#### Pitfall 1: Circular Reasoning

**Problem**: Using the model to interpret itself without external validation.

**Example**: Using model attention to explain model predictions (attention may not be faithful).

**Solution**: Validate with interventions, ground truth, or independent models.

---

#### Pitfall 2: Cherry-Picking

**Problem**: Showing only examples that support your hypothesis.

**Solution**: Systematic sampling, quantitative metrics, report negative results.

---

#### Pitfall 3: Overinterpreting Low-Quality Signals

**Problem**: Building elaborate theories on noisy, low-confidence findings.

**Solution**: Quantify uncertainty, replicate across seeds/models, use conservative thresholds.

---

#### Pitfall 4: Ignoring Task Context

**Problem**: Analyzing representations without considering task demands.

**Solution**: Compare representations trained on different tasks, ablate supervision.

---

## Open Problems by Area

### Fundamental Questions

#### Q1: What is the right formalism for interpretability?

**Current state**: Many ad-hoc methods, limited theoretical grounding.

**Approaches**:
- Information-theoretic frameworks (MI, information bottleneck)
- Causal inference frameworks (causal abstraction, SCMs)
- Dynamical systems theory (for RNNs)
- Algebraic frameworks (for transformers)

**Open questions**:
- Is there a unified mathematical theory?
- What are the fundamental limits of interpretability?
- Can we prove theorems about interpretability?

---

#### Q2: How do we validate interpretations?

**The problem**: Ground truth is often unavailable.

**Current approaches**:
- Synthetic datasets with known properties
- Neuroscience comparisons
- Causal interventions
- Predictive power (does interpretation enable control?)

**Open questions**:
- What makes an interpretation "correct"?
- Can automated validation replace human judgment?
- How do we measure interpretation quality?

---

### Architecture-Specific Questions

#### Transformers

**Q**: How do attention patterns relate to computation?
- Attention is not always faithful to information flow
- Multiple attention heads complicate interpretation
- Attention is just one mechanism (MLPs also matter)

**Q**: What algorithms are implemented in-context?
- How does in-context learning work mechanistically?
- Can we predict which algorithms will emerge?

**Q**: How does layer depth relate to abstraction?
- Early layers = syntax, late layers = semantics (but exceptions exist)
- How sharp are boundaries?

---

#### RNNs/LSTMs/GRUs

**Q**: How do attractors relate to computation?
- Finding all attractors is computationally hard
- Relationship between attractors and task solutions unclear

**Q**: How does memory work in gated architectures?
- When do networks use gates vs pure recurrence?
- Can we predict memory timescales from architecture?

---

#### CNNs

**Q**: What determines feature hierarchy?
- Why do some features emerge universally?
- How does architecture (depth, skip connections) affect features?

**Q**: Are adversarial examples fundamental or fixable?
- Do they reflect representational deficits or task artifacts?

---

### Scaling Questions

#### Q: Do methods scale to LLMs?

**Challenges**:
- 100B+ parameters makes exhaustive analysis impossible
- Sampling strategies matter enormously
- Interpretation drift during training

**Approaches**:
- Automated interpretation pipelines
- Focus on critical components
- Hierarchical analysis (zoom in/out)

---

#### Q: Do emergent behaviors have precursors?

**Context**: GPT-3 can perform tasks GPT-2 cannot. Why?

**Hypotheses**:
- Representational phase transitions
- Continuous sharpening (not discrete jump)
- More data reveals latent capabilities

**How to test**: Track representations across model scales.

---

### Safety and Alignment Questions

#### Q: Can we detect deceptive alignment?

**The problem**: Model behaves well in training but poorly in deployment.

**Current approaches**:
- Probe for internal goal representations
- Monitor representation drift
- Test consistency across contexts

**Challenges**: Adversarial optimization against detectors.

---

#### Q: Can interpretability improve robustness?

**Hypothesis**: Understanding failure modes enables fixes.

**Evidence**: Mixed - some successes (Grad-CAM-guided augmentation) but no general solution.

**Open questions**:
- Which interpretation methods best predict vulnerabilities?
- Can we automatically patch discovered issues?

---

## Experimental Design

### Designing a Strong Interpretability Study

#### 1. Formulate Precise Hypotheses

**Bad**: "We'll analyze what the model learned."

**Good**: "We hypothesize that layer 3 encodes syntactic features linearly while layer 6 encodes semantic features non-linearly. We will test this with probes and controlled interventions."

---

#### 2. Choose Appropriate Techniques

**Match method to question**:

| Question Type | Method | Why |
|--------------|--------|-----|
| "Is X encoded?" | Probe + MI | Correlational evidence |
| "Does X cause Y?" | Activation patching | Causal test |
| "Where is X encoded?" | Layer-wise probing | Localization |
| "How is X computed?" | Circuit analysis | Mechanistic detail |

---

#### 3. Design Control Conditions

**Essential controls**:
- **Shuffled labels**: Detects method artifacts
- **Random features**: Tests discriminability
- **Ablated supervision**: Tests task-dependence
- **Different architectures**: Tests generality

---

#### 4. Plan Statistical Analysis A Priori

**Pre-registration**: Specify hypotheses, methods, and analysis before seeing results.

**Power analysis**: Ensure sufficient sample size.

**Multiple comparisons**: Plan correction methods upfront.

---

#### 5. Build on Existing Work

**Literature review**: What has been tried? What worked/didn't?

**Replication**: Start by replicating key findings in your setting.

**Incremental progress**: Make specific contributions beyond prior work.

---

### Recommended Experimental Pipeline

```
1. SETUP PHASE
   ├─ Define research question precisely
   ├─ Survey related work
   ├─ Pre-register hypotheses
   └─ Prepare controlled datasets

2. PILOT PHASE
   ├─ Implement methods on toy problems
   ├─ Verify implementations with synthetic data
   ├─ Test on small models first
   └─ Refine based on pilot results

3. MAIN EXPERIMENT
   ├─ Run planned analyses
   ├─ Include all pre-registered controls
   ├─ Track all hyperparameters
   └─ Archive intermediate results

4. VALIDATION PHASE
   ├─ Cross-validate with multiple methods
   ├─ Test on held-out data/models
   ├─ Perform sensitivity analysis
   └─ Check for confounds

5. ANALYSIS PHASE
   ├─ Statistical tests with correction
   ├─ Visualizations with error bars
   ├─ Compare to baselines
   └─ Interpret cautiously

6. REPORTING PHASE
   ├─ Document negative results
   ├─ Report limitations explicitly
   ├─ Make code/data available
   └─ Solicit peer review
```

---

## Theory Gaps

### Formalizing Interpretability

**The challenge**: Most interpretability work is empirical without theoretical foundations.

**Desirable theoretical properties**:
- **Faithfulness**: Interpretation reflects true mechanism
- **Completeness**: Interpretation captures all relevant information
- **Soundness**: Valid interpretations only
- **Parsimony**: Simplest sufficient explanation

**Current frameworks** (partial):
- Information theory (MI, information bottleneck)
- Causal inference (interventions, counterfactuals)
- Dynamical systems (attractors, stability)

**Missing**:
- Unified framework encompassing all methods
- Formal guarantees on interpretation quality
- Complexity theory of interpretability (what's feasible?)

---

### Sample Complexity

**Question**: How much data is needed for reliable interpretability?

**Current state**: Mostly heuristic (e.g., "5000 samples for MI estimation").

**Needed**:
- Formal sample complexity bounds
- Dependence on model size, representation dimension
- Adaptive sampling strategies

---

### Faithfulness vs Plausibility

**The problem**: Interpretations can be plausible (convincing) but unfaithful (wrong).

**Example**: Attention weights are plausible explanations but may not reflect true information flow.

**Needed**: Formal definitions and tests for faithfulness.

---

## Recommended Reading Paths

### Path 1: Mechanistic Interpretability (Transformers)

**Foundation**:
1. Olah et al. (2020): Zoom In - Circuits
2. Elhage et al. (2021): Mathematical Framework for Transformer Circuits

**Core papers**:
3. Olsson et al. (2022): In-context Learning and Induction Heads
4. Wang et al. (2022): Interpretability in the Wild
5. Nanda et al. (2023): Progress Measures for Grokking

**Advanced**:
6. Cunningham et al. (2023): Sparse Autoencoders
7. Bills et al. (2023): Language Models Can Explain Neurons

---

### Path 2: Dynamical Systems (RNNs)

**Foundation**:
1. Strogatz (1994): Nonlinear Dynamics and Chaos (textbook)
2. Sussillo & Barak (2013): Opening the Black Box

**Core papers**:
3. Maheswaranathan et al. (2019): Reverse Engineering Sentiment Networks
4. Yang et al. (2019): Multi-task Representations
5. Huang et al. (2024): Dynamical Motifs

**Neuroscience connections**:
6. Khona & Fiete (2022): Attractor Networks in the Brain
7. Vyas et al. (2020): Computation Through Neural Population Dynamics

---

### Path 3: Representation Learning Dynamics

**Foundation**:
1. Saxe et al. (2014): Deep Linear Networks (exact solutions)
2. Saxe et al. (2019): Mathematical Theory of Semantic Development

**Core papers**:
3. Fort & Jastrzębski (2019): Loss Landscape Structure
4. Kornblith et al. (2019): Similarity of Neural Network Representations (CKA)
5. Nanda et al. (2024): Progress Measures for Grokking

**Advanced**:
6. Novak et al. (2020): Neural Tangent Kernel
7. Fort et al. (2021): Lottery Ticket Hypothesis at Scale

---

### Path 4: Causal Interpretability

**Foundation**:
1. Pearl (2009): Causality (textbook, chapters 1-3)
2. Geiger et al. (2021): Causal Abstractions

**Core papers**:
3. Meng et al. (2022): Locating and Editing Factual Associations
4. Belinkov (2022): Probing Classifiers (critique)
5. Elazar et al. (2021): Amnesic Probing

**Advanced**:
6. Goldowsky-Dill et al. (2023): Localizing Model Behavior
7. Hase et al. (2023): Does Localization Inform Editing?

---

### Path 5: Theory of Interpretability

**Information theory**:
1. Cover & Thomas (2006): Elements of Information Theory (textbook)
2. Tishby & Zaslavsky (2015): Deep Learning and Information Bottleneck

**Probing theory**:
3. Pimentel et al. (2020): Information-theoretic Probing
4. Hewitt & Liang (2019): Designing and Interpreting Probes

**Causal theory**:
5. Pearl & Mackenzie (2018): The Book of Why (accessible intro)
6. Schölkopf et al. (2021): Toward Causal Representation Learning

---

## Connecting to the Handbook

### For Empirical Work

Start here:
- [Methods](../2_methods/) - Technique deep-dives
- [Case Studies](../5_case_studies/) - Literature examples
- [Tutorials](../6_tutorials/) - Implementation guides

---

### For Theoretical Work

Start here:
- [Foundations](../1_foundations/) - Mathematical background
- [Advanced Topics](../7_advanced_topics/) - Cutting-edge theory
- [Theoretical Foundations](../theoretical_foundations.md) - Detailed derivations

---

### For Specific Architectures

Start here:
- [Architectures](../3_architectures/) - Architecture-specific guides
- Case studies by architecture (e.g., [Transformers](../5_case_studies/natural_language/))

---

## Contributing to Interpretability Research

### How to Make an Impact

**Underexplored areas**:
- RNN interpretability (overshadowed by transformers)
- Multimodal model interpretability
- Interpretability for embodied AI
- Automated interpretation at scale
- Theoretical foundations

**High-impact contributions**:
- Reproducing key findings across architectures
- Developing better benchmarks
- Creating open-source tools
- Finding failure modes of existing methods

---

### Publishing

**Key venues**:
- **ML conferences**: NeurIPS, ICML, ICLR (main venues)
- **Specialized**: EMNLP BlackboxNLP, ICLR DGMs+Apps
- **Journals**: Nature Communications, JMLR, Distill

**What reviewers look for**:
- Clear hypotheses and motivation
- Rigorous controls and baselines
- Multiple validation methods
- Honest discussion of limitations
- Open-source implementations

---

## Summary: Research Best Practices

1. **Formulate precise, testable hypotheses**
2. **Use multiple complementary methods**
3. **Include rigorous controls and baselines**
4. **Quantify uncertainty** (confidence intervals, p-values)
5. **Validate causally** (interventions, not just correlations)
6. **Report negative results** and limitations
7. **Make work reproducible** (code, data, seeds)
8. **Ground in existing theory** when possible
9. **Build incrementally** on prior work
10. **Stay humble** about interpretability claims

---

## Questions and Further Reading

This guide provides a starting point for interpretability research. For deeper dives:

- **[Advanced Topics](../7_advanced_topics/)**: Superposition, scaling, emergence, theory
- **[Methods](../2_methods/)**: Detailed technique documentation
- **[Case Studies](../5_case_studies/)**: Learn from literature examples
- **[References](../references/)**: Comprehensive bibliography and reading lists

---

**Ready to explore?** Return to the [main landing page](README.md) or dive into a specific topic.

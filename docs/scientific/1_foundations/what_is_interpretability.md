# What is Interpretability?

An overview of interpretability as a field: definitions, motivations, historical context, and fundamental concepts.

---

## Defining Interpretability

### The Core Question

**Interpretability**: The degree to which a human can understand the cause of a decision made by a model.

At its heart, interpretability seeks to answer: "Why did this model produce this output?"

### Multiple Perspectives

The field has no single universally accepted definition. Different communities emphasize different aspects:

**Machine learning perspective**: Understanding model mechanisms and decisions

**Human-computer interaction perspective**: Presenting model behavior to humans effectively

**Legal/ethical perspective**: Providing accountability and recourse

**Scientific perspective**: Using models to discover knowledge about data

**All share**: The goal of making opaque systems more transparent.

---

## Why Interpretability Matters

### 1. Trust and Verification

**Problem**: Models achieve high accuracy but we don't know how.

**Risk**: Hidden biases, spurious correlations, clever cheating

**Example**: Image classifier achieves 95% accuracy by detecting image compression artifacts instead of actual objects.

**Solution**: Interpretability reveals what the model actually learned.

**When critical**: Medical diagnosis, autonomous vehicles, financial decisions

### 2. Debugging and Improvement

**Problem**: Model fails on certain inputs; unclear why.

**Example**: Face detector fails on certain demographics.

**Interpretability helps**:
- Identify failure modes
- Understand what features are missing
- Guide data collection
- Inform architecture choices

**Practical value**: Better models faster.

### 3. Scientific Discovery

**Problem**: Complex data, unclear patterns.

**Opportunity**: Trained models might reveal hidden structure.

**Examples**:
- AlphaZero discovers novel chess strategies
- Image classifiers reveal hierarchical visual features
- RNNs rediscover neuroscience concepts (grid cells, attractors)

**Value**: Models as scientific instruments.

### 4. Fairness and Accountability

**Problem**: Models make decisions affecting people's lives.

**Legal/ethical requirements**:
- Right to explanation (GDPR)
- Demonstrate non-discrimination
- Enable appeals and recourse

**Interpretability provides**:
- Evidence of fair decision-making
- Ability to audit for bias
- Transparency for stakeholders

### 5. Safety and Robustness

**Problem**: Models vulnerable to adversarial examples, distribution shift.

**Interpretability helps**:
- Identify brittleness
- Understand failure modes
- Design more robust architectures
- Validate safety properties

**Critical for**: Deployment in high-stakes environments.

---

## Types of Interpretability

### Local vs Global

**Local interpretability**: Understanding a single prediction.

**Question**: "Why did this specific image get classified as 'cat'?"

**Methods**: Saliency maps, LIME, counterfactual explanations

**Use case**: Explaining individual decisions to users

---

**Global interpretability**: Understanding overall model behavior.

**Question**: "What patterns does the model use to classify cats in general?"

**Methods**: Feature visualization, probing, model distillation

**Use case**: Model validation, scientific understanding

---

**Trade-off**: Local explanations are easier to compute but don't reveal systemic issues. Global understanding is harder but more valuable for validation.

### Post-Hoc vs Intrinsic

**Post-hoc interpretability**: Analyzing a trained model after the fact.

**Characteristics**:
- Works with any model (black box)
- Flexibility in analysis methods
- No performance cost during training
- May not be fully faithful

**Examples**: Grad-CAM, probing, saliency maps

**Most interpretability research**: Post-hoc analysis

---

**Intrinsic interpretability**: Models designed to be interpretable from the start.

**Characteristics**:
- Interpretation is built into the model
- Often simpler models (decision trees, linear models)
- Guaranteed faithfulness
- May sacrifice some accuracy

**Examples**: Decision trees, linear regression, attention mechanisms (when designed for interpretability)

**Trade-off**: Intrinsically interpretable models may have limited expressiveness.

### Model-Specific vs Model-Agnostic

**Model-specific**: Techniques that exploit particular architecture properties.

**Examples**:
- Grad-CAM (requires convolutional structure)
- Attention visualization (requires attention mechanism)
- Fixed-point analysis (requires recurrent structure)

**Advantages**: Can leverage architectural insights for deeper understanding

**Disadvantages**: Don't generalize across architectures

---

**Model-agnostic**: Techniques that work for any model.

**Examples**:
- SHAP (treats model as black box)
- LIME (local linear approximation)
- Permutation importance

**Advantages**: Widely applicable, easy to compare models

**Disadvantages**: May miss architecture-specific insights

---

## Interpretability vs Related Concepts

### Interpretability vs Explainability

Often used interchangeably, but some distinguish:

**Interpretability**: Understanding the mechanism
- "How does it work?"
- Requires transparent model internals

**Explainability**: Providing human-understandable reasons
- "Why this output?"
- Can use post-hoc methods

**In practice**: The distinction is blurry and many use the terms synonymously.

### Interpretability vs Transparency

**Transparency**: Ability to inspect model components
- Code is available
- Weights are accessible
- Architecture is documented

**Interpretability**: Ability to understand what inspections reveal
- Understanding what weights mean
- Knowing why architecture matters
- Comprehending learned computations

**Relationship**: Transparency enables interpretability but doesn't guarantee it. A neural network can be fully transparent (all weights accessible) but still hard to interpret.

### Interpretability vs Causality

**Correlation-based interpretation**: Identifies associations between features and outputs
- "Feature X correlates with high predictions"
- Most standard interpretability methods

**Causal interpretation**: Identifies causal relationships
- "Feature X causes high predictions"
- Requires interventions

**Why it matters**: Correlational interpretations can be misleading. A feature might correlate with the output but not actually influence it.

**Trend**: Growing emphasis on causal interpretability (activation patching, causal mediation analysis)

---

## Fundamental Challenges

### The Faithfulness Problem

**Question**: Does an interpretation reflect how the model actually works?

**Example**: Attention weights may be plausible but not faithful (information flows through other mechanisms too)

**Challenge**: Hard to verify faithfulness without ground truth

**Approaches**:
- Synthetic data where truth is known
- Causal interventions
- Consistency across multiple methods

### The Complexity Problem

**Challenge**: Deep networks have millions of parameters and complex interactions.

**Question**: Can humans comprehend such complexity?

**Responses**:
1. **Abstraction**: Identify high-level patterns, ignore low-level details
2. **Simplification**: Focus on critical components
3. **Visualization**: Leverage human visual processing
4. **Automation**: Use AI to help interpret AI

**Fundamental limit**: Some models may be inherently too complex for complete human understanding.

### The Completeness-Simplicity Trade-off

**Completeness**: Explanation captures all relevant factors

**Simplicity**: Explanation is understandable to humans

**Tension**: Complete explanations of complex models are themselves complex.

**Example**: A complete explanation of GPT-4 would be extremely complicated; simple explanations omit important details.

**Resolution**: Match explanation granularity to use case. Medical doctor needs different explanation than ML researcher.

### The Rashomon Effect

**Observation**: Multiple different models can achieve similar performance on the same task.

**Implication**: Interpreting one model doesn't tell you about the general solution to the task—just one particular solution.

**Challenge**: Which model's interpretation is "correct"?

**Response**: Consider interpretations across model family, look for consistent patterns.

---

## Historical Context

### Pre-Deep Learning Era

**1990s-2000s**: Interpretability focused on intrinsically interpretable models
- Decision trees: human-readable rules
- Linear models: interpretable coefficients
- Rule-based systems: explicit logic

**Assumption**: Interpretability requires simple models

### The Deep Learning Revolution (2012+)

**ImageNet moment (2012)**: Deep networks dramatically outperform traditional methods

**Trade-off emerges**: Performance vs interpretability
- Deep networks: high accuracy, low interpretability
- Traditional methods: lower accuracy, higher interpretability

**Initial response**: "Black box" narrative—accept opacity for performance

### Rise of Interpretability Research (2015+)

**Catalysts**:
- High-stakes applications (medicine, autonomous vehicles)
- Regulatory pressures (GDPR right to explanation)
- Safety concerns (adversarial examples)
- Scientific interest (what did the model learn?)

**Key developments**:
- 2013: Zeiler & Fergus visualization techniques
- 2014: Simonyan et al. saliency maps
- 2016: LIME (Ribeiro et al.)
- 2017: Grad-CAM, Integrated Gradients
- 2019+: Mechanistic interpretability, circuits

**Shift**: From "interpretability vs accuracy" to "how to interpret powerful models"

### Current State (2024+)

**Mature techniques**: Grad-CAM, SHAP, probing all well-established

**Frontier areas**:
- Mechanistic interpretability (circuits, sparse autoencoders)
- Scaling interpretability to LLMs
- Causal interpretability
- Automated interpretation

**Open questions**: Can we fully understand foundation models? Are there fundamental limits?

---

## Schools of Thought

### 1. Feature Attribution

**Philosophy**: Understand predictions by attributing importance to input features.

**Key methods**: SHAP, Integrated Gradients, LIME

**Strengths**: Intuitive, actionable, well-grounded theoretically

**Limitations**: Local explanations, may not reveal global patterns

**Proponents**: Lundberg, Sundararajan, Ribeiro

### 2. Representation Analysis

**Philosophy**: Understand what models learn by analyzing internal representations.

**Key methods**: Probing, dimensionality reduction, similarity metrics

**Strengths**: Reveals learned abstractions, enables comparison

**Limitations**: Correlational (not causal), requires labeled concepts

**Proponents**: Belinkov, Rogers, Kornblith

### 3. Mechanistic Interpretability

**Philosophy**: Reverse-engineer models into interpretable algorithms.

**Key methods**: Circuit analysis, activation patching, sparse autoencoders

**Strengths**: Causal understanding, detailed mechanisms

**Limitations**: Labor-intensive, may not scale to largest models

**Proponents**: Anthropic, OpenAI interpretability teams, Neel Nanda

### 4. Neuroscience-Inspired

**Philosophy**: Understand artificial neural networks through analogy with biological brains.

**Key methods**: Fixed-point analysis, attractor dynamics, population coding

**Strengths**: Principled theory, connects AI to neuroscience

**Limitations**: Limited to certain architectures (RNNs especially)

**Proponents**: Sussillo, Ganguli, Fiete

### 5. Formal Verification

**Philosophy**: Mathematically prove properties of models.

**Key methods**: Abstract interpretation, SMT solvers, interval analysis

**Strengths**: Rigorous guarantees

**Limitations**: Computationally hard, limited to small models/properties

**Proponents**: Formal methods community

**Not mutually exclusive**: Many researchers draw from multiple schools.

---

## Philosophical Perspectives

### Reductionism vs Emergence

**Reductionist view**: Understand the whole by understanding parts
- Analyze individual neurons
- Decompose into circuits
- Bottom-up understanding

**Emergent view**: Properties arise from interactions, not reducible to parts
- Population-level dynamics
- Collective computation
- Top-down understanding

**Both necessary**: Low-level and high-level understanding complement each other.

### Interpretability as Science vs Engineering

**Science perspective**: Interpretability reveals truths about learning and computation
- What patterns exist in natural data?
- What algorithms does learning discover?
- How do neural networks generalize?

**Engineering perspective**: Interpretability is a tool for building better systems
- Debug failures
- Improve robustness
- Ensure safety

**Different goals**: Scientific interpretability seeks understanding; engineering interpretability seeks utility.

### Interpretation for Whom?

**Different stakeholders, different needs**:

**End users**: "Why did the system make this decision about me?"
- Need simple, actionable explanations
- Accuracy less important than trust

**Domain experts**: "Is the model using medically valid reasoning?"
- Need domain-appropriate concepts
- Want to validate against expertise

**ML practitioners**: "Why is performance poor on this subset?"
- Need detailed diagnostic information
- Want actionable debugging insights

**Researchers**: "What computation is implemented?"
- Need mechanistic understanding
- Want to test scientific hypotheses

**No single interpretation serves all needs**.

---

## Limits of Interpretability

### Fundamental Limits

**Computational complexity**: Some questions about models are provably hard (NP-complete or worse).

**Gödelian limits**: For sufficiently expressive models, complete formal verification is impossible.

**Quantum-like superposition**: When features are in superposition, asking "which feature?" may be wrong question.

### Practical Limits

**Human cognitive limits**: Humans can't comprehend millions of parameters in detail.

**Scale**: Interpretability methods that work for small models may not scale to GPT-4.

**Diversity**: Many different models solve the same task differently—which to interpret?

### Tension with Performance

**Hypothesis**: Most interpretable models sacrifice accuracy.

**Evidence**: Mixed
- Sometimes true (linear models vs deep networks)
- Sometimes false (attention mechanisms improve both)
- Depends on task and what "interpretable" means

**Open question**: Fundamental trade-off or engineering problem?

---

## Best Practices

### Matching Method to Question

Different questions need different techniques:

| Question | Appropriate Methods |
|----------|-------------------|
| Why this prediction? | Saliency, Grad-CAM, SHAP |
| What concepts learned? | Probing, clustering |
| How does it compute? | Circuit analysis, dynamics |
| Is it safe/fair? | Adversarial testing, bias audits |

### Triangulation

**Principle**: Use multiple methods to validate findings.

**Why**: Each method has limitations and biases
- Saliency maps can be noisy
- Probes might find spurious patterns
- Attention might not be faithful

**Convergence**: If multiple methods agree, higher confidence

**Divergence**: If methods disagree, investigate why

### Controls and Baselines

Always include:
- **Randomization tests**: Analyze random/untrained models
- **Shuffled controls**: Randomize labels/inputs
- **Null models**: What would happen by chance?

### Documentation

**Report**:
- Exact methods and hyperparameters
- Negative results and failures
- Limitations and caveats
- Computational costs

**Enable reproducibility**: Share code and data when possible

---

## Relationship to Other Fields

### Machine Learning Theory

**Connections**: Generalization, inductive bias, optimization

**Interpretability contributes**: Empirical insights into learning dynamics

**Theory contributes**: Formal frameworks for understanding models

### Cognitive Science

**Connections**: Human reasoning, concept learning, explanation

**Interpretability draws on**: Theories of human understanding

**Interpretability contributes**: Models as cognitive hypotheses

### Neuroscience

**Connections**: Neural coding, dynamics, computation

**Mutual inspiration**: Brain-inspired AI, AI-inspired neuroscience

**Shared methods**: Population analysis, dynamical systems, information theory

### Philosophy of Science

**Connections**: Explanation, understanding, reductionism

**Philosophical questions**:
- What constitutes an explanation?
- Can we understand what we cannot verify?
- Are models tools or theories?

### Law and Ethics

**Connections**: Accountability, fairness, transparency

**Legal drivers**: GDPR, fair lending laws, algorithmic accountability

**Ethical considerations**: Bias, privacy, autonomy

---

## The Road Ahead

### Open Challenges

**Scaling**: Interpretability for billion-parameter models

**Automation**: Can AI help interpret AI?

**Validation**: How to verify interpretations are correct?

**Integration**: Building interpretability into ML workflows

### Emerging Directions

**Causal interpretability**: Moving beyond correlation

**Learned representations**: Understanding emergence of concepts

**Interactive interpretation**: Human-in-the-loop analysis

**Interpretability for capabilities**: Understanding what models can/cannot do

### Vision for the Field

**Near-term (5 years)**:
- Standard practices for interpretability
- Tooling integrated into ML frameworks
- Regulatory requirements drive adoption

**Long-term (10+ years)**:
- Fundamental understanding of deep learning
- Provably interpretable powerful models
- Interpretability enables new capabilities

---

## Further Reading

**Surveys**:
- Lipton (2018): The Mythos of Model Interpretability
- Molnar (2020): Interpretable Machine Learning (online book)
- Räuker et al. (2023): Toward Transparent AI

**Perspectives**:
- Doshi-Velez & Kim (2017): Towards rigorous science of interpretability
- Rudin (2019): Stop explaining black box models
- Belinkov & Glass (2019): Analysis methods in neural NLP

**This handbook**:
- [Methods](../2_methods/) - Concrete techniques
- [Case Studies](../5_case_studies/) - Applications
- [For Researchers](../0_start_here/for_researchers.md) - Research landscape

Full bibliography: [References](../references/bibliography.md)

---

**Return to**: [Foundations](README.md) | [Main Handbook](../0_start_here/README.md)

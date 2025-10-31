# Interpretability Glossary

A comprehensive guide to terms used in neural network interpretability research. Terms are defined simply first, then explained in more detail with examples.

**Organization**: Alphabetical within thematic sections.

---

## Core Concepts

### Interpretability
**Simple**: Understanding what a neural network does and why.

**Detailed**: The degree to which a human can understand the cause of a decision or prediction made by a model. Encompasses both understanding individual predictions (local interpretability) and overall behavior (global interpretability).

**Example**: Being able to answer "Why did this image classifier think this was a cat?"

**Related**: Explainability, transparency

---

### Explainability
**Simple**: Providing human-understandable reasons for model outputs.

**Detailed**: The ability to present the internal workings or decisions of a model in terms understandable to humans. Often used interchangeably with interpretability, though some distinguish them: interpretability = understanding the mechanism, explainability = presenting it accessibly.

**Related**: Interpretability, post-hoc explanation

---

### Black Box
**Simple**: A model whose internal workings are not readily understood.

**Detailed**: A system where you can observe inputs and outputs but cannot see or understand the internal processing. Deep neural networks are often called black boxes because their millions of parameters make direct inspection infeasible.

**Example**: A large language model that produces answers without revealing its reasoning process.

**Related**: Interpretability, mechanistic understanding

---

### White Box
**Simple**: A model whose internal workings are transparent and understandable.

**Detailed**: A system where the internal mechanism is accessible and interpretable. Examples include decision trees, linear regression, and symbolic rule systems.

**Example**: A decision tree that explicitly shows the rules used for classification.

**Contrast with**: Black box

---

## Types of Interpretability

### Local Interpretability
**Simple**: Understanding a single prediction.

**Detailed**: Explaining why a model made a specific prediction for a particular input. Focuses on the model's behavior in a small region of input space around one example.

**Example**: "This image was classified as 'cat' because the model focused on the whiskers and pointed ears."

**Methods**: Grad-CAM, LIME, SHAP (local mode)

**Contrast with**: Global interpretability

---

### Global Interpretability
**Simple**: Understanding the model's overall behavior.

**Detailed**: Characterizing what the model has learned across all inputs. Reveals general patterns, feature hierarchies, and decision boundaries rather than individual predictions.

**Example**: "This classifier learned to recognize cats by detecting fur texture, facial features, and body shape."

**Methods**: Probing, feature visualization, representation analysis

**Contrast with**: Local interpretability

---

### Post-Hoc Interpretability
**Simple**: Explaining a model after it's trained.

**Detailed**: Applying interpretation techniques to an already-trained model without modifying its architecture or training. Most interpretability research is post-hoc.

**Example**: Using Grad-CAM on a pre-trained ResNet.

**Methods**: Most techniques in this handbook

**Contrast with**: Intrinsic interpretability

---

### Intrinsic Interpretability
**Simple**: Models designed to be interpretable from the start.

**Detailed**: Interpretability built into the model architecture or training process. The model is inherently understandable without requiring additional analysis tools.

**Example**: Decision trees, linear models with few features, attention mechanisms designed for interpretability.

**Contrast with**: Post-hoc interpretability

---

## Attribution and Saliency

### Attribution
**Simple**: Assigning credit to input features for a prediction.

**Detailed**: Computing how much each input feature contributed to a model's output. Usually represented as a score or heatmap indicating feature importance.

**Example**: Highlighting which pixels in an image most influenced the "dog" classification.

**Methods**: Integrated Gradients, SHAP, GradCAM

**Related**: Saliency, feature importance

---

### Saliency Map
**Simple**: A heatmap showing which input parts matter most.

**Detailed**: A visualization where each input location is colored by how much it affected the output. Brighter areas indicate higher importance. Computed from gradients of the output with respect to the input.

**Example**: A heatmap overlaid on an image showing which pixels the model "looked at."

**How it works**: `saliency = |∂output/∂input|`

**Methods**: Vanilla gradients, SmoothGrad

**Related**: Attribution, Grad-CAM

---

### Grad-CAM
**Simple**: Class Activation Mapping using gradients.

**Detailed**: A technique that produces heatmaps highlighting which regions of an image contributed most to a classification. Works by computing gradients of the class score with respect to the final convolutional layer, then weighting feature maps accordingly.

**Example**: Highlighting a cat's face in an image classified as "cat."

**Advantages**: Produces spatial heatmaps, class-discriminative, widely applicable

**Reference**: Selvaraju et al. (2017)

**Related**: CAM, Saliency maps

---

### Integrated Gradients (IG)
**Simple**: Rigorous gradient-based attribution method.

**Detailed**: Computes attribution by integrating gradients along the path from a baseline (e.g., black image) to the actual input. Satisfies desirable axioms like completeness (attributions sum to prediction difference from baseline).

**Example**: Showing which pixels differentiate a "dog" image from a blank baseline.

**Advantage**: Theoretically grounded, satisfies axioms

**Reference**: Sundararajan et al. (2017)

**Related**: Gradients, attribution

---

### SHAP (SHapley Additive exPlanations)
**Simple**: Game-theoretic feature importance scores.

**Detailed**: Uses Shapley values from cooperative game theory to assign each feature an importance score. Treats features as "players" and computes their marginal contribution to the prediction across all possible feature coalitions.

**Example**: "Feature A contributed +0.3 to the prediction, feature B contributed -0.1."

**Advantages**: Theoretically grounded, model-agnostic, local and global modes

**Disadvantages**: Can be computationally expensive

**Reference**: Lundberg & Lee (2017)

**Related**: Attribution, Shapley values

---

### LIME (Local Interpretable Model-agnostic Explanations)
**Simple**: Approximate model locally with a simple interpretable model.

**Detailed**: Explains a prediction by fitting a simple linear model (or decision tree) that approximates the black-box model's behavior in the neighborhood of the instance being explained.

**Example**: "Locally around this data point, the model behaves like: y ≈ 2*feature1 - 1*feature2."

**Advantages**: Model-agnostic, intuitive

**Disadvantages**: Local approximation may not reflect global behavior

**Reference**: Ribeiro et al. (2016)

**Related**: Post-hoc interpretability, attribution

---

## Probing and Representation Analysis

### Probe / Probing
**Simple**: Testing if a concept is encoded in hidden representations.

**Detailed**: Training a simple classifier (usually linear) on a model's internal activations to predict whether a specific concept or feature is present. Success indicates the concept is encoded and accessible.

**Example**: Train a logistic regression on BERT layer 5 activations to predict if a sentence is passive voice. High accuracy → layer 5 encodes voice.

**Advantages**: Simple, quantitative, tests specific hypotheses

**Limitations**: Success proves encoding but failure doesn't prove absence (may be non-linearly encoded)

**Related**: Linear probe, mutual information

---

### Linear Probe
**Simple**: A linear classifier trained on hidden representations.

**Detailed**: The simplest type of probe—just a linear transformation (e.g., logistic regression, linear SVM). Tests if a concept is linearly separable in the representation space.

**Example**: `y = sigmoid(W·h + b)` where h is hidden activations

**Why linear**: Tests if information is easily accessible (linearly decodable). Non-linear probes can extract more but are harder to interpret.

**Related**: Probing, linear separability

---

### Causal Probing
**Simple**: Testing if a representation causally matters, not just correlates.

**Detailed**: Goes beyond correlational probing by intervening on representations and measuring the effect. If changing a representation changes the output, it's causally relevant.

**Example**: Swap activations between two inputs; if output changes, those activations are causal.

**Methods**: Activation patching, causal mediation analysis

**Why it matters**: High probe accuracy might be spurious correlation

**Reference**: Geiger et al. (2021)

**Related**: Probing, activation patching

---

### Mutual Information (MI)
**Simple**: Measures how much two variables depend on each other.

**Detailed**: A measure from information theory quantifying the statistical dependence between two random variables. MI=0 means independence; higher values mean stronger dependence. Unlike correlation, MI captures non-linear dependencies.

**Formula**: `I(X;Y) = H(Y) - H(Y|X)` where H is entropy

**Example**: MI between hidden neuron activation and presence of a face in the image.

**Units**: bits (log base 2) or nats (natural log)

**Advantages**: Model-free, detects non-linear relationships

**Methods**: k-NN estimation (Kraskov et al.), MINE (neural estimation)

**Related**: Entropy, probing, information theory

---

### Representation
**Simple**: Internal activations or embeddings in a neural network.

**Detailed**: The pattern of activation values at a particular layer or time step in a network. Represents the network's internal "understanding" of the input at that processing stage.

**Example**: The 512-dimensional vector produced by a ResNet at layer 3 when processing an image.

**Types**: Input embeddings, hidden activations, final representations

**Related**: Embedding, latent space, hidden state

---

### Embedding
**Simple**: A dense vector representation of data.

**Detailed**: A continuous, typically low-dimensional vector representation learned by a model. Often used for discrete objects (words, users, items) to place them in a continuous space where similar items are close together.

**Example**: Word2Vec embeds words as 300-dimensional vectors where "king" and "queen" are nearby.

**Properties**: Dimensionality, distance metrics, structure

**Related**: Representation, latent space

---

### Latent Space
**Simple**: The internal representation space of a model.

**Detailed**: The multi-dimensional space where a model's representations live. The geometry of this space (distances, clusters, directions) reflects the model's learned structure.

**Example**: A 128-dimensional space where similar images have nearby representations.

**Analysis methods**: t-SNE, UMAP, PCA to visualize

**Related**: Representation, embedding, manifold

---

## Dimensionality Reduction and Visualization

### Dimensionality Reduction
**Simple**: Compressing high-dimensional data to 2D or 3D for visualization.

**Detailed**: Techniques for projecting high-dimensional data (e.g., 512-dimensional embeddings) into lower dimensions (typically 2 or 3) while preserving important structure. Enables human visualization of otherwise incomprehensible high-dimensional spaces.

**Why needed**: Can't visualize 512 dimensions; need to project to 2D/3D

**Methods**: PCA, t-SNE, UMAP, PHATE

**Trade-offs**: Different methods preserve different properties

**Related**: Visualization, manifold learning

---

### PCA (Principal Component Analysis)
**Simple**: Linear projection onto directions of maximum variance.

**Detailed**: Finds orthogonal axes (principal components) ordered by how much variance they explain. First component captures most variance, second captures most remaining variance orthogonal to first, etc.

**Advantages**: Fast, interpretable axes, exact solution

**Disadvantages**: Linear—misses non-linear structure

**Formula**: Eigendecomposition of covariance matrix

**Related**: SVD, eigenvalues, linear projection

---

### t-SNE (t-distributed Stochastic Neighbor Embedding)
**Simple**: Non-linear projection emphasizing local neighborhoods.

**Detailed**: Constructs a probability distribution over pairs of points in high and low dimensions, then minimizes the divergence between them. Preserves local structure (nearby points stay nearby) better than global structure.

**Advantages**: Beautiful visualizations, reveals clusters

**Disadvantages**: Slow, hyperparameter-sensitive, can distort global structure, non-deterministic

**When to use**: Visualizing clusters in high-dimensional data

**Reference**: van der Maaten & Hinton (2008)

**Related**: UMAP, dimensionality reduction

---

### UMAP (Uniform Manifold Approximation and Projection)
**Simple**: Fast non-linear projection preserving both local and global structure.

**Detailed**: Based on manifold learning and topological data analysis. Builds a high-dimensional graph, then finds a low-dimensional layout. Faster than t-SNE and better at preserving global structure.

**Advantages**: Fast, preserves global structure, fewer hyperparameters

**Disadvantages**: Still some parameter sensitivity

**When to use**: General-purpose replacement for t-SNE

**Reference**: McInnes et al. (2018)

**Related**: t-SNE, manifold learning

---

### PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)
**Simple**: Trajectory-preserving dimensionality reduction.

**Detailed**: Uses diffusion geometry to embed data while explicitly preserving trajectory structure and transition probabilities. Particularly good for temporal data like training checkpoints or sequential processes.

**Advantages**: Preserves trajectories, denoises, preserves both local and global structure

**When to use**: Training dynamics, time-series, developmental processes

**Reference**: Moon et al. (2019)

**Related**: UMAP, diffusion maps, training dynamics

---

### Manifold Hypothesis
**Simple**: High-dimensional data lies on a low-dimensional manifold.

**Detailed**: The hypothesis that natural high-dimensional data (images, text, etc.) actually occupies a much lower-dimensional structure (manifold) embedded in the high-dimensional space. Justifies dimensionality reduction.

**Example**: 1000x1000 images live in a billion-dimensional space, but natural images occupy a much lower-dimensional manifold.

**Implications**: Dimensionality reduction can work without losing too much information

**Related**: Manifold learning, intrinsic dimensionality

---

## Dynamical Systems

### Fixed Point
**Simple**: A state where the network doesn't change.

**Detailed**: For a recurrent network with update rule h_{t+1} = f(h_t, x), a fixed point h* satisfies h* = f(h*, x). If the network reaches this state and input stays constant, it stays there forever.

**Example**: In a flip-flop memory network, each bit configuration is a fixed point.

**Why it matters**: Fixed points represent stable computational modes

**Related**: Attractor, stability, RNN dynamics

---

### Attractor
**Simple**: A stable state that the network tends toward.

**Detailed**: A set of states (often a single fixed point) that trajectories converge to. Once near an attractor, the system stays there even with small perturbations. Different types include point attractors, line attractors, limit cycles.

**Example**: A GRU might have an "attack mode" attractor for offensive game states.

**Types**:
- **Point attractor**: Single fixed point
- **Line attractor**: Continuous line of equilibria
- **Limit cycle**: Periodic oscillation

**Related**: Fixed point, basin of attraction, dynamical systems

---

### Basin of Attraction
**Simple**: The set of states that lead to a specific attractor.

**Detailed**: All initial conditions that eventually converge to a particular attractor. The phase space is partitioned into basins, each leading to a different attractor.

**Example**: All game states where "attack" is the best strategy might lie in the attack attractor's basin.

**Related**: Attractor, separatrix, phase space

---

### Jacobian
**Simple**: Matrix of derivatives describing local dynamics.

**Detailed**: For a dynamical system h_{t+1} = f(h_t), the Jacobian J = ∂f/∂h evaluated at a point describes the local linear approximation of the dynamics. Its eigenvalues determine stability.

**Formula**: J_{ij} = ∂f_i/∂h_j

**Why it matters**: Eigenvalues tell you if a fixed point is stable (attractor) or unstable (saddle/repeller)

**Related**: Linearization, stability analysis, fixed points

---

### Stability
**Simple**: Whether perturbations grow or shrink.

**Detailed**: A fixed point is stable if small perturbations decay (system returns to fixed point). Determined by eigenvalues of the Jacobian: |λ| < 1 for all eigenvalues → stable.

**Types**:
- **Stable (attractor)**: all |λ| < 1
- **Unstable (repeller)**: any |λ| > 1
- **Saddle**: mixed (some |λ| < 1, others |λ| > 1)

**Related**: Eigenvalues, Jacobian, fixed points

---

### Trajectory
**Simple**: The path taken through state space over time.

**Detailed**: The sequence of hidden states h_0, h_1, h_2, ... as the network processes input. Visualizing trajectories reveals how the network transitions between computational modes.

**Example**: A path from "uncertain" to "confident attack" state as the game progresses.

**Visualization**: PHATE, PCA projections of state sequences

**Related**: Phase space, dynamical systems, hidden states

---

### Phase Space
**Simple**: The space of all possible states.

**Detailed**: The multi-dimensional space where each point represents a possible configuration of the system. For an n-dimensional RNN, phase space is n-dimensional (though trajectories may lie on a lower-dimensional manifold).

**Example**: For a 64-unit GRU, phase space is 64-dimensional.

**Visualization**: Project to 2D/3D with PCA or PHATE

**Related**: State space, trajectory, dynamical systems

---

## Mechanistic Interpretability

### Circuit
**Simple**: A minimal subnetwork implementing a specific behavior.

**Detailed**: In mechanistic interpretability, a circuit is a connected set of neurons, attention heads, and weights that together implement a particular algorithm or computation. Goal is to reverse-engineer networks into interpretable functional units.

**Example**: The "induction head" circuit in transformers that performs in-context copying.

**Methods**: Activation patching, causal tracing, manual analysis

**Reference**: Olah et al. (2020), Elhage et al. (2021)

**Related**: Mechanistic interpretability, feature decomposition

---

### Activation Patching
**Simple**: Swapping activations between runs to test causality.

**Detailed**: A causal intervention technique where you replace activations from one forward pass with activations from another, then measure the effect on the output. Identifies which activations are causally important.

**Example**: Swap layer 3 activations between "cat" and "dog" inputs. If output changes, layer 3 is causally relevant to the difference.

**Advantages**: Tests causality, not just correlation

**Methods**: Direct patching, path patching, denoising

**Related**: Causal probing, ablation, mechanistic interpretability

---

### Ablation
**Simple**: Removing a component and measuring the effect.

**Detailed**: Setting activations, weights, or entire components to zero (or random values) and observing the impact on model performance or behavior. If performance drops, the component was important.

**Example**: Set all layer 3 neurons to zero. If accuracy drops 20%, layer 3 is important.

**Advantages**: Simple, causal

**Disadvantages**: Can't distinguish between direct and indirect effects

**Related**: Activation patching, causal interpretability

---

### Superposition
**Simple**: Representing more features than dimensions available.

**Detailed**: The phenomenon where a network represents more concepts than it has neurons by using overlapping distributed codes. Multiple features are "superimposed" in the same set of neurons, recoverable through interference patterns.

**Example**: A 100-neuron layer representing 1000 features by having each neuron respond to multiple features in different combinations.

**Why it happens**: More features than dimensions → compression needed

**Implications**: Makes interpretation hard (polysemanticity)

**Reference**: Elhage et al. (2022), Anthropic

**Related**: Polysemanticity, sparse coding

---

### Polysemanticity
**Simple**: Neurons responding to multiple unrelated concepts.

**Detailed**: Individual neurons that activate for multiple semantically unrelated stimuli. The observable consequence of superposition—instead of one neuron = one concept, we see one neuron responding to many concepts.

**Example**: A neuron that responds to both "cat" images, the word "the", and left parentheses.

**Why it happens**: Superposition—efficient compression but less interpretable

**Solution**: Sparse autoencoders to decompose polysemantic neurons

**Related**: Superposition, monosemanticity, sparse coding

---

### Sparse Autoencoder (SAE)
**Simple**: Neural network that finds sparse feature representations.

**Detailed**: An autoencoder trained to reconstruct activations using a sparse bottleneck. Applied to network activations, SAEs can decompose polysemantic neurons into interpretable monosemantic features.

**Architecture**: encoder (activations → sparse features), decoder (sparse features → reconstructed activations)

**Goal**: Find over-complete basis where each feature is interpretable (monosemantic)

**Reference**: Cunningham et al. (2023)

**Related**: Superposition, polysemanticity, feature decomposition

---

### Monosemanticity
**Simple**: Neurons responding to a single coherent concept.

**Detailed**: The desirable property where each unit (neuron or feature) represents one well-defined concept. Opposite of polysemanticity. Makes interpretation straightforward: one unit = one meaning.

**Example**: A neuron that responds only to images of cats, nothing else.

**How to achieve**: Sparse autoencoders, specific training objectives

**Related**: Polysemanticity, interpretable features

---

## Information Theory

### Entropy
**Simple**: Measure of uncertainty or randomness.

**Detailed**: Quantifies the average amount of information (surprise) in a random variable. Higher entropy = more uncertain/random. Measured in bits (log base 2) or nats (natural log).

**Formula**: `H(X) = -Σ p(x) log p(x)`

**Example**: A fair coin has entropy 1 bit; a biased coin has less entropy.

**Intuition**: How many yes/no questions to determine the outcome?

**Related**: Mutual information, information theory

---

### Conditional Entropy
**Simple**: Remaining uncertainty about Y after observing X.

**Detailed**: The average entropy of Y for each value of X, weighted by probability. Measures how much uncertainty remains about Y after learning X.

**Formula**: `H(Y|X) = Σ p(x) H(Y|X=x)`

**Example**: Uncertainty about word identity after seeing its first letter.

**Related**: Mutual information (MI = H(Y) - H(Y|X))

---

### KL Divergence
**Simple**: Measure of difference between two probability distributions.

**Detailed**: Quantifies how much one probability distribution differs from another. Asymmetric: KL(P||Q) ≠ KL(Q||P). Used in VAEs, training objectives, and information theory.

**Formula**: `KL(P||Q) = Σ p(x) log[p(x)/q(x)]`

**Properties**: Non-negative, zero iff P = Q

**Interpretation**: Extra bits needed to encode P using code optimized for Q

**Related**: Cross-entropy, mutual information

---

### Information Bottleneck
**Simple**: Principle that networks compress input while preserving task-relevant information.

**Detailed**: A theory suggesting deep networks learn by compressing input information (reducing I(H;X)) while maintaining task information (keeping I(H;Y) high). Controversial but influential framework.

**Formula**: Minimize I(H;X) subject to constraint on I(H;Y)

**Debate**: Whether DNNs actually undergo compression phase

**Reference**: Tishby & Zaslavsky (2015)

**Related**: Mutual information, representation learning

---

## Model Behavior

### Adversarial Example
**Simple**: Input designed to fool a model.

**Detailed**: An input crafted by adding carefully chosen (usually imperceptible) perturbations to cause misclassification. Reveals brittleness in neural networks.

**Example**: Adding imperceptible noise to a "panda" image makes it classify as "gibbon."

**Why it matters**: Security, robustness, reveals what models really learn

**Methods**: FGSM, PGD, C&W attacks

**Related**: Robustness, failure modes

---

### Shortcut Learning
**Simple**: Learning spurious correlations instead of true patterns.

**Detailed**: When a model exploits dataset artifacts or superficial patterns instead of learning the underlying concept. Often invisible in training/test sets but fails on distribution shifts.

**Example**: Learning to recognize horses by detecting grass backgrounds instead of horse features.

**Detection**: Test on distribution shifts, analyze with saliency maps

**Related**: Dataset bias, spurious correlation

---

### Grokking
**Simple**: Sudden generalization long after overfitting.

**Detailed**: Phenomenon where a model continues training long after achieving zero training loss and eventually undergoes a sharp phase transition to perfect test accuracy. Challenges conventional understanding of overfitting.

**Example**: Modular arithmetic task: model memorizes for 10K steps, then suddenly generalizes perfectly.

**Why interesting**: Suggests there's structure beyond simply fitting training data

**Reference**: Power et al. (2022)

**Related**: Phase transitions, learning dynamics, double descent

---

### Double Descent
**Simple**: Test error decreases, then increases, then decreases again with model size.

**Detailed**: Counterintuitive phenomenon where test error follows a double-descent curve as model capacity increases: classical regime (U-shaped, risk of overfitting), interpolation threshold (worst performance), over-parameterized regime (lower error again).

**Why it matters**: Challenges classical bias-variance tradeoff

**Reference**: Belkin et al. (2019)

**Related**: Overfitting, model capacity, interpolation

---

## Similarity Metrics

### CKA (Centered Kernel Alignment)
**Simple**: Metric for comparing two sets of representations.

**Detailed**: Measures similarity between representations from two models (or layers) by comparing their kernel matrices. Invariant to orthogonal transformations and isotropic scaling. Values between 0 (completely different) and 1 (identical up to rotation/scaling).

**Formula**: `CKA(X,Y) = ||X^T Y||^2_F / (||X^T X||_F ||Y^T Y||_F)`

**Advantages**: Invariant to linear transformations, easy to compute

**Uses**: Comparing models, tracking training, finding equivalent layers

**Reference**: Kornblith et al. (2019)

**Related**: SVCCA, similarity metrics

---

### SVCCA (Singular Vector Canonical Correlation Analysis)
**Simple**: Technique for comparing neural network representations.

**Detailed**: Compares two sets of representations by first applying SVD to reduce dimensionality, then computing canonical correlation to measure similarity. Predecessor to CKA.

**Advantages**: Reduces noise via SVD

**Disadvantages**: CKA generally preferred now (simpler, more robust)

**Reference**: Raghu et al. (2017)

**Related**: CKA, representation similarity

---

## Training Dynamics

### Checkpoint
**Simple**: Saved model state at a specific training step.

**Detailed**: A snapshot of all model parameters saved during training. Used to resume training, track learning dynamics, or select the best model. Essential for interpretability studies of training dynamics.

**Typical saving**: Every N epochs or when validation improves

**Analysis uses**: PHATE trajectories, CKA similarity, probe accuracy over time

**Related**: Training dynamics, model evaluation

---

### Convergence
**Simple**: When training loss stops improving.

**Detailed**: The point at which gradient descent has reached a (local) optimum and further training produces negligible changes in loss. May not correspond to best test performance (early stopping).

**Measures**: Loss plateau, gradient magnitude, parameter change

**Related**: Early stopping, optimization

---

### Regime Change / Phase Transition
**Simple**: Qualitative shift in learning dynamics.

**Detailed**: Discrete transitions in representation structure or learning behavior during training. Can correspond to capability jumps, representational reorganization, or shifts in what features are learned.

**Example**: Sudden transition from memorization to rule-learning (grokking)

**Detection**: Abrupt changes in probes, representation similarity, loss curvature

**Related**: Grokking, training dynamics, critical learning periods

---

## Neuroscience Terms

### Population Code
**Simple**: Information represented by activity patterns across many neurons.

**Detailed**: Encoding where information is distributed across a population of neurons rather than individual neurons. The pattern of activity, not single-neuron responses, carries the information.

**Example**: Direction of movement encoded by pattern across motor cortex neurons.

**Contrast with**: Grandmother cell (single-neuron encoding)

**Related**: Distributed representation, neural coding

---

### Tuning Curve
**Simple**: Plot of neuron response vs stimulus property.

**Detailed**: Graph showing how a neuron's firing rate varies as a function of a stimulus parameter (e.g., orientation, position, frequency). Reveals what the neuron is selective for.

**Example**: Plot of visual neuron firing rate vs edge orientation (peaks at preferred angle).

**Computational analog**: Plot activation vs feature value for a network unit

**Related**: Feature selectivity, receptive field

---

### Receptive Field
**Simple**: The region of input that affects a neuron's response.

**Detailed**: In neuroscience: the spatial region of the visual field that modulates a neuron's activity. In CNNs: the region of the input image that influences a particular unit's activation.

**Example**: A V1 simple cell responding to edges at a specific location and orientation.

**In CNNs**: Grows with depth (deeper layers have larger receptive fields)

**Related**: Tuning curve, feature selectivity

---

## Miscellaneous

### Faithfulness
**Simple**: Whether an interpretation reflects the true mechanism.

**Detailed**: An interpretation is faithful if it accurately describes how the model actually makes decisions, not just plausible post-hoc rationalization. Key concern in interpretability research.

**Example**: Attention weights may be plausible but not faithful (other mechanisms matter too).

**Tests**: Causal interventions, consistency checks

**Related**: Validity, causal interpretability

---

### Sanity Check
**Simple**: Basic test that a method is working correctly.

**Detailed**: Experiments designed to verify an interpretability method produces sensible results. Common checks: randomization test (method should fail on random model), sensitivity (method should change when model changes).

**Examples**:
- Does saliency change when model is randomized?
- Does probe fail on shuffled labels?
- Does high-attribution region, when removed, affect prediction?

**Why important**: Catch method artifacts, build confidence

**Related**: Validation, controls

---

### Ground Truth
**Simple**: The correct answer known independently.

**Detailed**: Information known to be true from external sources, used to validate interpretability findings. Rare in real applications but valuable for controlled experiments.

**Examples**:
- Synthetic data where you know what concepts exist
- Neuroscience experiments with ground truth neural codes
- Controlled interventions with known effects

**Why valuable**: Direct validation of interpretations

**Related**: Validation, experimental design

---

## Thematic Cross-References

### For Understanding Single Predictions
See: **Attribution**, **Saliency Map**, **Grad-CAM**, **SHAP**, **LIME**, **Local Interpretability**

### For Understanding What's Learned
See: **Probing**, **Mutual Information**, **Global Interpretability**, **Representation**, **Feature Visualization**

### For Visualizing Structure
See: **PCA**, **t-SNE**, **UMAP**, **PHATE**, **Dimensionality Reduction**, **Manifold Hypothesis**

### For Understanding RNNs
See: **Fixed Point**, **Attractor**, **Trajectory**, **Jacobian**, **Stability**, **Basin of Attraction**

### For Mechanistic Understanding
See: **Circuit**, **Activation Patching**, **Sparse Autoencoder**, **Superposition**, **Mechanistic Interpretability**

### For Rigorous Analysis
See: **Mutual Information**, **CKA**, **Entropy**, **Causal Probing**, **Faithfulness**, **Sanity Check**

---

## Further Reading

For deeper understanding of these concepts:

- **[Foundations](../1_foundations/)**: Mathematical background
- **[Methods](../2_methods/)**: Technique deep-dives
- **[Tutorials](../6_tutorials/)**: Hands-on examples
- **[Glossary](glossary.md)**: This document

For specific use cases:
- **[For Beginners](for_beginners.md)**: Gentle introduction
- **[For Practitioners](for_practitioners.md)**: Applied recipes
- **[For Researchers](for_researchers.md)**: Research frontiers

**Return to**: [Main landing page](README.md)

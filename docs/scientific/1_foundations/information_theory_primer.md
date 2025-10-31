# Information Theory Primer

An accessible introduction to information-theoretic concepts used in interpretability research.

---

## Overview

Information theory provides a mathematical framework for quantifying information, uncertainty, and statistical dependence. For interpretability, it helps us ask: "How much information does this representation contain about that concept?"

**Key advantage**: Information-theoretic measures are model-free—they work regardless of whether the relationship is linear, non-linear, or highly complex.

---

## Core Concepts

### 1. Entropy: Measuring Uncertainty

**Intuition**: How surprising is a random variable? How many yes/no questions would you need to determine its value?

**Definition**: For a discrete random variable X with probability mass function p(x):

```
H(X) = -Σ p(x) log p(x)
```

**Units**: Bits (if log base 2) or nats (if natural log)

**Examples**:

**Fair coin**:
- p(heads) = p(tails) = 0.5
- H(X) = -0.5 log₂(0.5) - 0.5 log₂(0.5) = 1 bit
- Interpretation: Need 1 yes/no question to determine outcome

**Biased coin**:
- p(heads) = 0.9, p(tails) = 0.1
- H(X) ≈ 0.47 bits
- Interpretation: Less uncertain; usually heads

**Deterministic outcome**:
- p(heads) = 1.0
- H(X) = 0 bits
- Interpretation: No uncertainty; always heads

### Properties of Entropy

1. **Non-negative**: H(X) ≥ 0
2. **Maximum for uniform distribution**: H(X) ≤ log(|X|) where |X| is number of outcomes
3. **Zero iff deterministic**: H(X) = 0 ⟺ X has one outcome with probability 1

---

### 2. Conditional Entropy: Remaining Uncertainty

**Intuition**: How much uncertainty about Y remains after observing X?

**Definition**:
```
H(Y|X) = Σ p(x) H(Y|X=x)
       = -Σ p(x,y) log p(y|x)
```

**Example**: Predicting tomorrow's weather

Let X = today's weather, Y = tomorrow's weather

- H(Y) = entropy of tomorrow's weather with no information ≈ 1.5 bits (say)
- H(Y|X) = entropy after seeing today's weather ≈ 0.8 bits
- Interpretation: Knowing today reduces uncertainty about tomorrow

### Properties

1. **Chain rule**: H(X, Y) = H(X) + H(Y|X)
2. **Conditioning reduces entropy**: H(Y|X) ≤ H(Y) with equality iff X and Y are independent
3. **Non-negative**: H(Y|X) ≥ 0

---

### 3. Mutual Information: Shared Information

**Intuition**: How much information do X and Y share? How much does learning X tell you about Y?

**Definition**:
```
I(X; Y) = H(Y) - H(Y|X)
        = H(X) + H(Y) - H(X,Y)
```

**Alternative interpretation**: Reduction in uncertainty about Y from observing X.

**Visualization** (Venn diagram):
```
   ┌─────────────┐     ┌─────────────┐
   │    H(X)     │     │    H(Y)     │
   │             │     │             │
   │      ┌──────┴─────┴──────┐      │
   │      │   I(X;Y)           │      │
   │      │ (shared info)      │      │
   └──────┴────────────────────┴──────┘
```

**Examples**:

**Independent variables** (coin flips):
- I(coin1; coin2) = 0
- Learning coin1 tells you nothing about coin2

**Perfectly correlated**:
- Y = X (identical)
- I(X; Y) = H(X) = H(Y)
- Learning X completely determines Y

**Partially correlated** (weather example):
- I(today; tomorrow) = H(tomorrow) - H(tomorrow|today)
- = 1.5 - 0.8 = 0.7 bits
- Today's weather provides 0.7 bits of information about tomorrow

### Properties of Mutual Information

1. **Symmetric**: I(X; Y) = I(Y; X)
2. **Non-negative**: I(X; Y) ≥ 0
3. **Zero iff independent**: I(X; Y) = 0 ⟺ X and Y are independent
4. **Bounded**: I(X; Y) ≤ min(H(X), H(Y))

---

### 4. KL Divergence: Distribution Difference

**Intuition**: How different are two probability distributions?

**Definition**:
```
KL(P || Q) = Σ p(x) log[p(x)/q(x)]
```

**Interpretation**: Expected extra bits needed to encode data from P using a code optimized for Q.

**Properties**:

1. **Non-negative**: KL(P || Q) ≥ 0
2. **Zero iff identical**: KL(P || Q) = 0 ⟺ P = Q
3. **Asymmetric**: KL(P || Q) ≠ KL(Q || P) in general
4. **Not a metric**: Doesn't satisfy triangle inequality

**Example**:

Two models for coin fairness:
- P: true distribution, p(heads) = 0.6
- Q: assumed distribution, q(heads) = 0.5

```
KL(P || Q) = 0.6 log(0.6/0.5) + 0.4 log(0.4/0.5)
           ≈ 0.02 bits
```

Small divergence → Q is a reasonable approximation of P.

---

## Applications to Neural Network Interpretability

### 1. Representation Analysis

**Question**: How much information does hidden layer h encode about label y?

**Method**: Compute I(h; y)

**Interpretation**:
- I(h; y) = 0: Layer doesn't encode label information
- I(h; y) = H(y): Layer perfectly encodes label (can predict with certainty)
- Intermediate values: Partial encoding

**Practical use**:
- Compare layers: Which layer has most information about the task?
- Track training: Does I(h; y) increase during training?
- Compare architectures: Do wider networks encode more information?

### 2. Neuron Specialization

**Question**: Do individual neurons specialize for specific concepts?

**Method**: Compute I(h_i; y) for each neuron i

**Interpretation**:
- High I(h_i; y) for single neuron → specialized detector
- High I(h; y) but low I(h_i; y) for all i → distributed encoding

**Practical use**:
- Find interpretable neurons (high individual MI)
- Quantify polysemanticity (neurons with high MI to multiple concepts)

### 3. Information Bottleneck Theory

**Hypothesis**: Deep learning compresses input information while preserving task-relevant information.

**Formalization**: Networks minimize I(h; X) (compression) while maximizing I(h; Y) (relevance)

**Tracking**: Plot I(h; X) vs I(h; Y) over training epochs

**Interpretation** (controversial):
- Early training: Both increase (fitting phase)
- Late training: I(h; X) decreases, I(h; Y) plateaus (compression phase)

**Debate**: Whether compression phase actually occurs in practice.

---

## Estimation Methods

Information-theoretic quantities are defined for probability distributions, but in practice we only have samples. Estimation is non-trivial.

### For Discrete Variables

**Plug-in estimator**:
```python
def entropy_plugin(samples):
    """Estimate entropy from samples."""
    counts = np.bincount(samples)
    probs = counts / len(samples)
    probs = probs[probs > 0]  # remove zeros
    return -np.sum(probs * np.log2(probs))
```

**Issues**:
- Biased (underestimates true entropy)
- Requires enough samples per outcome
- Bias increases with number of outcomes

**Better estimators**: Miller-Madow correction, Bayesian estimators

### For Continuous Variables

**Differential entropy**:
```
h(X) = -∫ p(x) log p(x) dx
```

**Problem**: Requires estimating continuous density p(x)

**Practical approaches**:

**1. Binning** (discretize then compute):
```python
def entropy_binned(samples, n_bins=10):
    hist, _ = np.histogram(samples, bins=n_bins)
    probs = hist / len(samples)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))
```

Issues: Sensitive to bin size, loses precision

**2. k-NN estimator** (Kraskov et al., 2004):
- Uses distances to k-th nearest neighbor
- Nonparametric, no binning required
- Consistent (converges to true value as n → ∞)

```python
from sklearn.feature_selection import mutual_info_regression

# For continuous X and Y
mi = mutual_info_regression(X.reshape(-1, 1), Y, n_neighbors=3)
```

**3. Neural estimators** (MINE, InfoNCE):
- Train neural network to estimate MI
- Can handle high dimensions
- High variance, requires careful tuning

### Practical Recommendations

**For small datasets** (< 1000 samples):
- Discretize if possible
- Use k-NN with small k (k=3 to 5)
- Bootstrap for confidence intervals

**For medium datasets** (1000-10000):
- k-NN with k=5 to 10
- Consider neural estimators if high-dimensional

**For large datasets** (> 10000):
- Any method should work
- k-NN still robust choice
- Neural estimators feasible

**Always**:
- Compare multiple estimators
- Test on synthetic data with known ground truth
- Report confidence intervals

---

## Common Pitfalls

### 1. Small Sample Bias

**Problem**: MI estimates are biased upward for small samples.

**Manifestation**: High MI even for independent variables.

**Solution**:
- Permutation test: shuffle one variable, recompute MI. True MI should be >> shuffled MI.
- Increase sample size
- Use bias-corrected estimators

### 2. High Dimensionality

**Problem**: Curse of dimensionality—estimation becomes unreliable in high dimensions.

**Manifestation**: MI estimates vary wildly across random seeds.

**Solution**:
- Reduce dimensionality first (PCA, etc.)
- Use more sophisticated estimators (neural estimators)
- Aggregate over dimensions (average per-dimension MI)

### 3. Discretization Artifacts

**Problem**: Binning continuous variables creates artificial dependencies.

**Manifestation**: MI sensitive to number of bins.

**Solution**:
- Use k-NN estimators (no binning)
- If binning required, test multiple bin sizes
- Report sensitivity analysis

### 4. Confusing Correlation and MI

**Problem**: Correlation only captures linear dependence; MI captures any dependence.

**Example**:
- X uniform on [-1, 1]
- Y = X²
- Correlation(X, Y) = 0 (uncorrelated)
- I(X; Y) = H(X) > 0 (strong dependence!)

**Lesson**: MI detects non-linear relationships that correlation misses.

---

## Worked Example: Probing with MI

**Scenario**: We have a trained image classifier. Does layer 3 encode object presence?

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Extract activations from layer 3
# activations: (n_images, n_neurons)
# labels: (n_images,) binary [0=no object, 1=object present]

# Compute MI between each neuron and label
mi_scores = mutual_info_classif(activations, labels, random_state=42)

# mi_scores: (n_neurons,) one MI value per neuron

# Find top neurons
top_neurons = np.argsort(mi_scores)[-10:]  # top 10
print(f"Top neuron MI scores: {mi_scores[top_neurons]}")

# Compare to shuffled baseline
labels_shuffled = np.random.permutation(labels)
mi_shuffled = mutual_info_classif(activations, labels_shuffled, random_state=42)

print(f"True MI (mean): {mi_scores.mean():.3f}")
print(f"Shuffled MI (mean): {mi_shuffled.mean():.3f}")

# If true >> shuffled, layer 3 encodes object presence
```

**Interpretation**:
- High MI for specific neurons → those neurons specialize for object detection
- High mean MI → layer encodes object presence
- MI comparable to shuffled → layer doesn't encode this information

---

## Relationship to Other Concepts

**To entropy in thermodynamics**:
- Shannon borrowed the term from statistical mechanics
- Conceptual similarity: disorder, uncertainty
- Different contexts but related mathematics

**To compression**:
- Entropy H(X) = minimum bits needed to encode X
- Optimal compression achieves H(X) bits per symbol

**To channel capacity**:
- Maximum mutual information over input distributions
- Fundamental limit on communication

---

## Further Reading

**Introductory**:
- Cover & Thomas (2006): *Elements of Information Theory* (standard textbook)
- MacKay (2003): *Information Theory, Inference, and Learning Algorithms* (free online)

**For interpretability**:
- Tishby & Zaslavsky (2015): Deep learning and information bottleneck
- Gabrié (2018): Entropy and mutual information in models of deep learning
- Pimentel et al. (2020): Information-theoretic probing

**Estimation methods**:
- Kraskov et al. (2004): Estimating mutual information (k-NN method)
- Belghazi et al. (2018): MINE (neural estimator)

**In this handbook**:
- [Mutual Information for Probing](../2_methods/probing/mutual_information.md) - Practical application
- [Statistics for Interpretability](statistics_for_interpretability.md) - Complementary statistical concepts

Full bibliography: [References](../references/bibliography.md)

---

**Return to**: [Foundations](README.md) | [Main Handbook](../0_start_here/README.md)

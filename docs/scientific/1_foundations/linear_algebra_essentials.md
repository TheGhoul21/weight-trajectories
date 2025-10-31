# Linear Algebra Essentials for Interpretability

A practical introduction to the linear algebra concepts needed for neural network interpretability, with focus on intuition and application rather than mathematical formalism.

---

## Why Linear Algebra for Interpretability?

Neural networks are fundamentally linear algebra engines wrapped in nonlinearities. Understanding how information flows through networks requires understanding:

- **Matrix operations**: How layers transform data
- **Vector spaces**: The geometry of representations
- **Eigenanalysis**: Identifying dominant patterns and dynamics
- **Projections**: How information is preserved or discarded
- **Norms and distances**: Measuring similarity and change

This primer covers the essential concepts with interpretability applications.

---

## Vectors and Vector Spaces

### Vectors as Data Points

**Basic view**: A vector is a list of numbers.

```python
import numpy as np

# A 3-dimensional vector
v = np.array([1.5, -0.3, 2.1])
```

**Interpretability view**: A vector represents a state or representation.

Examples:
- Hidden state of RNN: `h_t ∈ ℝ^512`
- Word embedding: `e_word ∈ ℝ^300`
- Image representation: `z_image ∈ ℝ^2048`

### Vector Operations

**Addition**: Combining representations

```python
# Semantic composition in word embeddings
king = embeddings['king']
woman = embeddings['woman']
man = embeddings['man']

# Classic example: king - man + woman ≈ queen
queen_approx = king - man + woman
```

**Scalar multiplication**: Scaling activation strength

```python
# Amplifying a direction in representation space
direction = np.array([0.5, 0.8, -0.2])
stronger = 2.0 * direction  # Stronger effect
weaker = 0.5 * direction    # Weaker effect
```

**Dot product**: Measuring alignment

```python
def cosine_similarity(v1, v2):
    """How aligned are two vectors?"""
    dot_prod = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot_prod / norms

# Example: Compare hidden states
h1 = np.array([0.8, 0.6])
h2 = np.array([0.6, 0.8])
similarity = cosine_similarity(h1, h2)
# Returns ~0.96 (very similar)
```

**Interpretability application**: Dot products measure how much one representation "contains" another.

### Vector Norms

**L2 norm** (Euclidean length):

```python
def l2_norm(v):
    """Length of vector."""
    return np.sqrt(np.sum(v**2))

# Equivalent to:
l2_norm_builtin = np.linalg.norm(v)
```

**L1 norm** (Manhattan distance):

```python
def l1_norm(v):
    """Sum of absolute values."""
    return np.sum(np.abs(v))
```

**Infinity norm** (maximum absolute value):

```python
def linf_norm(v):
    """Largest component."""
    return np.max(np.abs(v))
```

**Why it matters**: Different norms capture different notions of "size" relevant for robustness, sparsity, and adversarial examples.

---

## Matrices and Transformations

### Matrices as Transformations

**Weight matrix**: Transforms input space to output space

```python
# Simple linear layer
W = np.array([
    [0.5, -0.3, 0.8],
    [0.2,  0.9, -0.4]
])  # 2 x 3 matrix

x = np.array([1.0, 0.5, -0.2])  # 3D input

y = W @ x  # 2D output
# Matrix multiplication: each row of W produces one output dimension
```

**Geometric interpretation**: Matrix multiplication stretches, rotates, and projects vectors.

### Key Matrix Operations

**Transpose**: Flip rows and columns

```python
W = np.array([[1, 2, 3],
              [4, 5, 6]])

W_T = W.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

**Interpretability use**: Analyzing gradient flow, computing activation patterns.

**Matrix multiplication**: Composing transformations

```python
# Two-layer network
W1 = np.random.randn(128, 512)  # First layer
W2 = np.random.randn(64, 128)   # Second layer

# Combined transformation
W_combined = W2 @ W1  # 64 x 512
# Equivalent to: W2(W1(x)) = (W2 @ W1)(x)
```

**Interpretability insight**: Deep networks compose many transformations. Understanding composition is key to understanding depth.

### Special Matrices

**Identity matrix**: Does nothing (preserves input)

```python
I = np.eye(3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

v = np.array([1, 2, 3])
I @ v == v  # True
```

**Diagonal matrix**: Scales each dimension independently

```python
D = np.diag([2.0, 0.5, 1.0])
# [[2.0, 0.0, 0.0],
#  [0.0, 0.5, 0.0],
#  [0.0, 0.0, 1.0]]

v = np.array([1, 2, 3])
D @ v  # [2.0, 1.0, 3.0] - each dimension scaled
```

**Interpretability use**: Gate mechanisms (LSTM, GRU) use element-wise scaling similar to diagonal matrices.

**Orthogonal matrix**: Preserves lengths and angles (rotation/reflection)

```python
def is_orthogonal(Q, tol=1e-6):
    """Check if Q is orthogonal: Q.T @ Q = I"""
    I_approx = Q.T @ Q
    I_exact = np.eye(Q.shape[0])
    return np.allclose(I_approx, I_exact, atol=tol)
```

**Why it matters**: Some architectures use orthogonal initialization to preserve gradient flow.

---

## Eigenvalues and Eigenvectors

### What Are They?

**Eigenvector**: A special direction that matrix only scales (doesn't rotate)

**Eigenvalue**: The scaling factor

**Formal definition**: If `A v = λ v`, then `v` is an eigenvector with eigenvalue `λ`.

### Computing Eigendecomposition

```python
# Symmetric matrix example (easier to interpret)
A = np.array([
    [3.0, 1.0],
    [1.0, 3.0]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
# [4. 2.]

print("Eigenvectors:")
print(eigenvectors)
# [[ 0.707  0.707]
#  [ 0.707 -0.707]]

# Verify: A @ v = λ * v
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print(np.allclose(A @ v1, lambda1 * v1))  # True
```

### Why Eigenanalysis Matters for Interpretability

**1. Dynamical systems analysis**

```python
def analyze_rnn_dynamics(W_rec):
    """Analyze RNN stability via eigenvalues.

    W_rec: Recurrent weight matrix
    """
    eigenvalues, eigenvectors = np.linalg.eig(W_rec)

    # Largest eigenvalue magnitude determines stability
    max_eigenvalue = np.max(np.abs(eigenvalues))

    if max_eigenvalue > 1.0:
        print("Unstable: activations will explode")
    elif max_eigenvalue < 1.0:
        print("Stable: converges to fixed point")
    else:
        print("Critical: dynamics on edge of stability")

    # Eigenvectors are the principal modes of dynamics
    return eigenvalues, eigenvectors
```

**2. Principal Component Analysis (PCA)**

```python
def pca_manual(X, n_components=2):
    """Manual PCA implementation using eigendecomposition.

    X: Data matrix (n_samples x n_features)
    Returns: Projected data (n_samples x n_components)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top components
    principal_components = eigenvectors[:, :n_components]
    X_projected = X_centered @ principal_components

    # Variance explained
    var_explained = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_projected, var_explained

# Example: Reduce 512D hidden states to 2D
hidden_states = np.random.randn(1000, 512)
projected, var_exp = pca_manual(hidden_states, n_components=2)
print(f"Variance explained: {var_exp}")
```

**3. Fixed-point stability analysis**

```python
def classify_fixed_point_stability(jacobian):
    """Determine fixed point type from Jacobian eigenvalues.

    jacobian: Local linearization around fixed point
    """
    eigenvalues = np.linalg.eigvals(jacobian)

    max_real = np.max(np.real(eigenvalues))

    if max_real < 0:
        stability = "stable (attractor)"
    elif max_real > 0:
        stability = "unstable (repeller)"
    else:
        stability = "marginal (saddle)"

    # Classify by imaginary components
    max_imag = np.max(np.abs(np.imag(eigenvalues)))

    if max_imag > 0.1:  # Threshold for considering complex
        dynamics = "oscillatory (spiral)"
    else:
        dynamics = "direct (node)"

    return stability, dynamics, eigenvalues
```

### Singular Value Decomposition (SVD)

**Most important factorization for interpretability.**

**Any matrix** can be decomposed as:
```
A = U Σ V^T
```

Where:
- `U`: Left singular vectors (output space directions)
- `Σ`: Singular values (scaling factors)
- `V`: Right singular vectors (input space directions)

```python
def analyze_weight_matrix(W):
    """Decompose weight matrix to understand transformation.

    W: Weight matrix (output_dim x input_dim)
    """
    U, s, Vt = np.linalg.svd(W, full_matrices=False)

    print(f"Matrix shape: {W.shape}")
    print(f"Rank (effective): {np.sum(s > 1e-6)}")
    print(f"Top singular values: {s[:5]}")

    # Singular values indicate importance of each component
    explained_var = s**2 / np.sum(s**2)
    print(f"Top component explains {explained_var[0]*100:.1f}% of transformation")

    return U, s, Vt

# Example: Analyze attention weight matrix
W_attn = np.random.randn(64, 64)
U, s, Vt = analyze_weight_matrix(W_attn)
```

**Interpretability applications**:
- **Low-rank structure**: If first few singular values are large, weight matrix has low effective dimensionality
- **Rank analysis**: Reveals redundancy in learned transformations
- **Probing**: SVD used to find interpretable subspaces

---

## Projections and Subspaces

### What is a Subspace?

**Intuition**: A lower-dimensional "slice" through a higher-dimensional space.

Examples:
- 2D plane through 3D space
- 1D line through 2D space
- 100D subspace of 512D hidden state space

**Why it matters**: Neural networks often use only a small subspace of their full capacity.

### Projection onto a Vector

```python
def project_onto_vector(x, v):
    """Project x onto direction v.

    Returns component of x in direction v.
    """
    v_normalized = v / np.linalg.norm(v)
    coefficient = np.dot(x, v_normalized)
    projection = coefficient * v_normalized
    return projection

# Example: Project hidden state onto "sentiment" direction
h = np.array([0.5, 0.8, -0.3, 0.6])
sentiment_dir = np.array([0.7, 0.7, 0.0, 0.0])  # Hypothetical direction

h_sentiment = project_onto_vector(h, sentiment_dir)
sentiment_strength = np.dot(h, sentiment_dir / np.linalg.norm(sentiment_dir))
```

**Interpretability use**: Measuring how much a representation encodes a concept.

### Projection onto a Subspace

```python
def project_onto_subspace(x, basis):
    """Project x onto subspace spanned by basis vectors.

    basis: Matrix where each column is a basis vector
    """
    # Orthonormalize basis (Gram-Schmidt or QR)
    Q, R = np.linalg.qr(basis)

    # Project: P = Q @ Q.T @ x
    projection = Q @ (Q.T @ x)

    return projection

# Example: Project onto 2D concept subspace
h = np.random.randn(128)
concept_basis = np.random.randn(128, 2)  # 2D subspace in 128D space

h_concept = project_onto_subspace(h, concept_basis)
# h_concept lives in the concept subspace
```

**Application**: Causal probing removes information by projecting onto orthogonal complement.

### Orthogonal Complement

```python
def project_away_from_subspace(x, basis):
    """Remove component in subspace (project onto orthogonal complement).

    Useful for ablation studies.
    """
    h_in_subspace = project_onto_subspace(x, basis)
    h_orthogonal = x - h_in_subspace
    return h_orthogonal

# Example: Remove gender information from embedding
embedding = np.random.randn(300)
gender_basis = np.random.randn(300, 1)  # 1D gender direction

embedding_debiased = project_away_from_subspace(embedding, gender_basis)
```

---

## Matrix Rank and Nullspace

### Rank

**Rank**: The dimension of the output space (number of linearly independent columns/rows).

```python
def effective_rank(A, threshold=1e-6):
    """Compute effective rank (number of non-negligible singular values)."""
    s = np.linalg.svd(A, compute_uv=False)
    return np.sum(s > threshold)

# Low-rank matrix: redundant transformation
W_lowrank = np.outer(np.random.randn(100), np.random.randn(50))
print(f"Rank: {effective_rank(W_lowrank)}")  # Will be 1

# Full-rank matrix: uses full capacity
W_fullrank = np.random.randn(100, 50)
print(f"Rank: {effective_rank(W_fullrank)}")  # Will be 50
```

**Interpretability insight**: Low-rank weight matrices indicate network uses simple transformations. High-rank indicates complex feature mixing.

### Nullspace (Kernel)

**Nullspace**: Set of inputs that get mapped to zero.

```python
def compute_nullspace(A, threshold=1e-6):
    """Find vectors that A maps to ~zero."""
    U, s, Vt = np.linalg.svd(A, full_matrices=True)

    # Nullspace is spanned by right singular vectors with zero singular values
    null_mask = s < threshold
    nullspace_basis = Vt[null_mask, :].T

    return nullspace_basis

# Example
A = np.array([
    [1, 2, 3],
    [2, 4, 6]
])  # Second row is 2x first row - rank 1

null_basis = compute_nullspace(A)
print(f"Nullspace dimension: {null_basis.shape[1]}")

# Verify: A @ v ≈ 0 for v in nullspace
v = null_basis[:, 0]
print(f"A @ v = {A @ v}")  # Should be near zero
```

**Why it matters**: Nullspace represents "invisible" directions that network can't distinguish. Important for understanding what network ignores.

---

## Distances and Similarities

### Common Distance Metrics

**Euclidean distance** (L2):

```python
def euclidean_distance(x, y):
    """Straight-line distance."""
    return np.linalg.norm(x - y)
```

**Cosine similarity** (angle between vectors):

```python
def cosine_similarity(x, y):
    """Similarity based on direction, ignoring magnitude."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_distance(x, y):
    """Distance version (0 = identical, 2 = opposite)."""
    return 1 - cosine_similarity(x, y)
```

**When to use which**:
- Euclidean: When magnitude matters (activation strength)
- Cosine: When only direction matters (semantic similarity)

### Matrix Distances

**Frobenius norm** (like L2 for matrices):

```python
def frobenius_distance(A, B):
    """Element-wise distance between matrices."""
    return np.linalg.norm(A - B, ord='fro')

# Example: Compare weight matrices at different training steps
W_init = np.random.randn(64, 128)
W_trained = W_init + np.random.randn(64, 128) * 0.1

distance = frobenius_distance(W_init, W_trained)
print(f"Training changed weights by {distance:.3f}")
```

**Centered Kernel Alignment (CKA)**: Sophisticated similarity for representations

```python
def linear_cka(X, Y):
    """CKA similarity between two representation sets.

    X, Y: (n_samples, n_features) matrices
    Returns: Similarity in [0, 1], 1 = identical up to linear transform
    """
    # Center representations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # Frobenius inner product
    numerator = np.linalg.norm(K_X @ K_Y, ord='fro')**2
    denominator = np.linalg.norm(K_X, ord='fro') * np.linalg.norm(K_Y, ord='fro')

    return numerator / (denominator**2)
```

**Usage**: Compare representations across layers, models, or training steps.

---

## Gradients and Jacobians

### Gradient Vectors

**Gradient**: Direction of steepest increase for a scalar function.

```python
def compute_gradient_numerical(f, x, epsilon=1e-5):
    """Numerical gradient using finite differences."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon

        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    return grad

# Example: Gradient of prediction w.r.t. input
def model_output(x):
    # Simple model: y = x^T W x
    W = np.array([[2, 1], [1, 3]])
    return x @ W @ x

x = np.array([1.0, 0.5])
grad = compute_gradient_numerical(model_output, x)
# grad tells us how to change x to increase output
```

**Interpretability use**: Saliency maps, gradient-based attribution.

### Jacobian Matrices

**Jacobian**: Gradient when output is a vector (derivative of vector function).

```python
def compute_jacobian_numerical(f, x, epsilon=1e-5):
    """Numerical Jacobian: derivative of vector function.

    f: Function from R^n -> R^m
    x: Input point (n,)
    Returns: Jacobian matrix (m, n)
    """
    f_x = f(x)
    m = len(f_x)
    n = len(x)

    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += epsilon
        f_plus = f(x_plus)

        J[:, j] = (f_plus - f_x) / epsilon

    return J

# Example: Jacobian of RNN cell
def rnn_cell(h):
    W = np.array([[0.5, -0.3], [0.8, 0.2]])
    b = np.array([0.1, -0.1])
    return np.tanh(W @ h + b)

h = np.array([0.5, 0.3])
J = compute_jacobian_numerical(rnn_cell, h)
print(f"Jacobian shape: {J.shape}")
# J tells us how small changes in h propagate through cell
```

**Interpretability applications**:
- **Fixed-point analysis**: Jacobian eigenvalues determine stability
- **Sensitivity analysis**: How robust is output to input changes?
- **Information flow**: Which hidden dimensions affect which outputs?

---

## Interpretability-Specific Operations

### Computing Concept Directions

```python
def find_concept_direction(positive_examples, negative_examples):
    """Find direction that separates positive from negative examples.

    Returns direction in representation space.
    """
    # Mean representations
    mean_pos = np.mean(positive_examples, axis=0)
    mean_neg = np.mean(negative_examples, axis=0)

    # Concept direction
    direction = mean_pos - mean_neg
    direction = direction / np.linalg.norm(direction)

    return direction

# Example: Find "sentiment" direction
positive_hidden_states = np.random.randn(100, 256)  # From positive reviews
negative_hidden_states = np.random.randn(100, 256)  # From negative reviews

sentiment_direction = find_concept_direction(positive_hidden_states,
                                             negative_hidden_states)

# Now can measure sentiment of any hidden state
def get_sentiment_score(h):
    return np.dot(h, sentiment_direction)
```

### Measuring Representational Geometry

```python
def compute_pairwise_distances(X, metric='euclidean'):
    """Compute all pairwise distances between representations.

    X: (n_samples, n_features)
    Returns: (n_samples, n_samples) distance matrix
    """
    n = X.shape[0]
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if metric == 'euclidean':
                dist = np.linalg.norm(X[i] - X[j])
            elif metric == 'cosine':
                dist = 1 - np.dot(X[i], X[j]) / (
                    np.linalg.norm(X[i]) * np.linalg.norm(X[j]))

            D[i, j] = dist
            D[j, i] = dist

    return D

# Analyze geometry of learned representations
representations = np.random.randn(50, 128)
distances = compute_pairwise_distances(representations)

# Are representations clustered or uniformly spread?
mean_dist = np.mean(distances[np.triu_indices(50, k=1)])
std_dist = np.std(distances[np.triu_indices(50, k=1)])
print(f"Mean distance: {mean_dist:.3f}, Std: {std_dist:.3f}")
```

### Orthogonal Basis for Probing

```python
def gram_schmidt(vectors):
    """Orthogonalize a set of vectors.

    Input: List of vectors or matrix with vectors as columns
    Output: Orthonormal basis
    """
    vectors = np.array(vectors)
    if vectors.ndim == 1:
        vectors = vectors.reshape(-1, 1)

    n_vectors = vectors.shape[1]
    basis = np.zeros_like(vectors, dtype=float)

    for i in range(n_vectors):
        # Start with current vector
        vec = vectors[:, i].astype(float)

        # Subtract projections onto previous basis vectors
        for j in range(i):
            vec -= np.dot(vec, basis[:, j]) * basis[:, j]

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            basis[:, i] = vec / norm
        else:
            # Vector was linearly dependent, leave as zero
            basis[:, i] = 0

    return basis

# Use for creating interpretable probe directions
raw_directions = np.random.randn(128, 5)  # 5 concept directions
orthogonal_directions = gram_schmidt(raw_directions)

# Now each direction is independent
print(np.allclose(orthogonal_directions.T @ orthogonal_directions, np.eye(5)))
# True
```

---

## Common Pitfalls and Best Practices

### Pitfall 1: Numerical Instability

**Problem**: Ill-conditioned matrices lead to unstable results.

```python
# Check condition number
def check_stability(A):
    """High condition number = numerical instability."""
    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"WARNING: Condition number {cond:.2e} is very high")
        print("Results may be numerically unstable")
    return cond
```

**Solution**: Regularization, use SVD instead of direct inversion.

### Pitfall 2: Not Centering Data

**Problem**: PCA and related methods require centered data.

```python
# Always center before eigenanalysis
X_centered = X - np.mean(X, axis=0)
```

### Pitfall 3: Confusing Row/Column Conventions

**Problem**: Different libraries use different conventions.

```python
# Sklearn: each row is a sample
# PyTorch: often each column is a sample or depends on batch_first

# Always check shapes
print(f"Data shape: {X.shape}")  # (n_samples, n_features) or (n_features, n_samples)?
```

### Pitfall 4: Ignoring Magnitude

**Problem**: Cosine similarity ignores whether vectors are small or large.

```python
# When magnitude matters, use Euclidean distance
# When only direction matters, use cosine similarity

# Often want both
def comprehensive_similarity(x, y):
    cos_sim = cosine_similarity(x, y)
    mag_ratio = np.linalg.norm(x) / np.linalg.norm(y)
    return cos_sim, mag_ratio
```

---

## Essential Linear Algebra Toolkit

```python
class InterpretabilityLinAlg:
    """Common linear algebra operations for interpretability."""

    @staticmethod
    def pca(X, n_components):
        """PCA dimensionality reduction."""
        X_centered = X - np.mean(X, axis=0)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        components = Vt[:n_components]
        projected = X_centered @ components.T
        var_explained = s[:n_components]**2 / np.sum(s**2)
        return projected, components, var_explained

    @staticmethod
    def concept_direction(pos_examples, neg_examples):
        """Find direction separating two classes."""
        direction = np.mean(pos_examples, axis=0) - np.mean(neg_examples, axis=0)
        return direction / np.linalg.norm(direction)

    @staticmethod
    def project_and_ablate(x, direction):
        """Remove component in direction from x."""
        direction = direction / np.linalg.norm(direction)
        component = np.dot(x, direction)
        x_ablated = x - component * direction
        return x_ablated, component

    @staticmethod
    def representational_similarity(X, Y, method='cka'):
        """Compare two sets of representations."""
        if method == 'cka':
            return linear_cka(X, Y)
        elif method == 'cosine':
            # Average cosine similarity between paired samples
            similarities = [cosine_similarity(X[i], Y[i])
                          for i in range(len(X))]
            return np.mean(similarities)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def effective_dimensionality(X, threshold=0.95):
        """Number of dimensions needed to explain threshold variance."""
        X_centered = X - np.mean(X, axis=0)
        s = np.linalg.svd(X_centered, compute_uv=False)
        var_explained_cumsum = np.cumsum(s**2) / np.sum(s**2)
        n_dims = np.argmax(var_explained_cumsum >= threshold) + 1
        return n_dims
```

---

## Connections to Interpretability Methods

### Linear Probes → Linear Algebra

Probes are just linear classifiers, which are:
```python
# Probe: f(h) = W @ h + b
# W is a weight matrix defining a separating hyperplane
```

See: [Linear Probes Guide](../2_methods/probing/linear_probes.md)

### Fixed Points → Eigenanalysis

Fixed point stability determined by Jacobian eigenvalues:
```python
# Stable if all |eigenvalues| < 1
# Unstable if any |eigenvalue| > 1
```

See: [Fixed Point Analysis](../2_methods/dynamical_analysis/fixed_points.md)

### Trajectory Analysis → SVD and Projections

Dimensionality reduction for visualizing trajectories:
```python
# PCA, t-SNE, UMAP all use linear algebra
# Project high-D trajectories to 2D/3D
```

See: [Trajectory Analysis](../2_methods/dynamical_analysis/trajectory_analysis.md)

### Representation Similarity → Matrix Norms

CKA, SVCCA use advanced matrix operations:
```python
# Compare learned representations across models
# Based on canonical correlation (eigenanalysis of covariance)
```

See: [Representation Analysis](../2_methods/representation_analysis/)

---

## Practice Exercises

### Exercise 1: Concept Directions

Given hidden states from positive and negative sentiment sentences, compute the sentiment direction and test it on new examples.

### Exercise 2: PCA from Scratch

Implement PCA using only basic NumPy (no sklearn), apply to high-dimensional representations.

### Exercise 3: Stability Analysis

Given an RNN weight matrix, determine if dynamics are stable by analyzing eigenvalues.

### Exercise 4: Probe Geometry

Train linear probes for multiple concepts. Are the learned weight vectors orthogonal? What does this tell you?

---

## Further Reading

**Textbooks**:
- Strang, *Introduction to Linear Algebra* - Intuitive and visual
- Trefethen & Bau, *Numerical Linear Algebra* - Practical computation

**For interpretability**:
- [What is Interpretability?](what_is_interpretability.md)
- [Information Theory Primer](information_theory_primer.md)
- [Dynamical Systems Primer](dynamical_systems_primer.md)

**Full bibliography**: [References](../references/bibliography.md)

---

**Return to**: [Foundations](README.md) | [Main Handbook](../0_start_here/README.md)

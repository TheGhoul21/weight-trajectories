# Reading Dimensionality Reduction Plots

This guide explains how to interpret dimensionality reduction visualizations used in weight trajectory analysis. Understanding what the axes, distances, and structures mean is crucial for drawing valid conclusions from these plots.

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
   - [The Curse of Dimensionality](#the-curse-of-dimensionality)
   - [The Manifold Hypothesis](#the-manifold-hypothesis)
   - [Distance Metrics](#distance-metrics)
   - [What Can We Preserve](#what-can-we-preserve)
2. [Overview: What Dimensionality Reduction Shows](#overview-what-dimensionality-reduction-shows)
3. [PCA: Principal Component Analysis](#pca-principal-component-analysis)
   - [PCA Mathematical Foundation](#pca-mathematical-foundation)
   - [PCA Algorithm](#pca-algorithm)
4. [PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding](#phate-potential-of-heat-diffusion-for-affinity-based-trajectory-embedding)
   - [PHATE Mathematical Foundation](#phate-mathematical-foundation)
   - [PHATE Algorithm](#phate-algorithm)
5. [t-SNE: t-Distributed Stochastic Neighbor Embedding](#t-sne-t-distributed-stochastic-neighbor-embedding)
   - [t-SNE Mathematical Foundation](#t-sne-mathematical-foundation)
   - [t-SNE Algorithm](#t-sne-algorithm)
6. [UMAP: Uniform Manifold Approximation and Projection](#umap-uniform-manifold-approximation-and-projection)
   - [UMAP Mathematical Foundation](#umap-mathematical-foundation)
   - [UMAP Algorithm](#umap-algorithm)
7. [Common Patterns and Their Meanings](#common-patterns-and-their-meanings)
8. [Method Comparison Table](#method-comparison-table)
9. [Best Practices for Interpretation](#best-practices-for-interpretation)

---

## Theoretical Background

Before diving into specific methods, it's crucial to understand why dimensionality reduction is necessary and what fundamental challenges it addresses.

### The Curse of Dimensionality

**Problem statement**: High-dimensional spaces behave counter-intuitively, making direct visualization and analysis difficult.

#### Key phenomena:

**1. Volume concentration**
- In high dimensions, most of the volume of a hypersphere lies near its surface
- Example: 1000-dimensional unit sphere has ~99.97% of its volume in the outer 1% shell
- Implication: All points are approximately equidistant from each other

**2. Distance concentration**
- As dimensions increase, the ratio of distances between nearest and farthest neighbors approaches 1
- Formula: `(d_max - d_min) / d_min → 0` as `d → ∞`
- Implication: Euclidean distance becomes less meaningful

**3. Empty space dominance**
- To cover 1% of a unit hypercube in `d` dimensions requires `0.01^(1/d)` sampling density
- For d=10: need to sample 63% of each dimension
- For d=100: need to sample 95.5% of each dimension
- Implication: Data becomes extremely sparse

#### Consequences for neural network weight analysis:

Consider a simple feedforward network with:
- Layer 1: 784 → 256 (200,704 weights)
- Layer 2: 256 → 128 (32,768 weights)
- Layer 3: 128 → 10 (1,280 weights)
- **Total: 234,752 parameters**

**Direct visualization is impossible**:
- Cannot plot 234,752-dimensional space
- Even understanding relationships between checkpoints is difficult
- Clustering, similarity measures, and trajectory analysis all suffer from curse of dimensionality

**Solution**: Dimensionality reduction finds low-dimensional representations that preserve essential structure.

---

### The Manifold Hypothesis

**Core idea**: High-dimensional data often lies on or near a low-dimensional manifold embedded in the high-dimensional space.

#### What is a manifold?

A **manifold** is a space that locally resembles Euclidean space but may have complex global topology.

**Examples**:
- **Circle (S¹)**: 1D manifold embedded in 2D plane
- **Sphere (S²)**: 2D manifold embedded in 3D space
- **Torus**: 2D manifold with genus 1 (one "hole")

**Key insight**: Although the manifold exists in high-D space, it has much lower **intrinsic dimensionality**.

#### Why neural network weights lie on manifolds:

**1. Functional constraints**
- Not all weight combinations produce useful functions
- Good solutions cluster near regions of parameter space that minimize loss
- Training trajectories follow continuous paths constrained by gradient descent

**2. Symmetries and invariances**
- Permutation symmetry: swapping neuron order doesn't change function
- Scaling symmetries: certain weight transformations preserve network output
- These create lower-dimensional structures in weight space

**3. Optimization paths**
- Gradient descent follows smooth trajectories
- Loss landscape creates "valleys" that constrain weight evolution
- Trajectories don't fill the entire weight space randomly

**Example**:
- Random 234,752-D points would be uniformly distributed in hypercube
- Trained weight checkpoints form continuous curves (1D manifolds)
- Even comparing multiple training runs might only explore a 3-5D subspace

#### Implications:

[YES] **Dimensionality reduction works** because:
- Effective dimensionality << nominal dimensionality
- Structure exists to be preserved

[NOTE] **But be careful**:
- Manifold might be curved (non-linear)
- Linear methods (PCA) may distort curved manifolds
- Non-linear methods (PHATE, UMAP) better capture complex topology

---

### Distance Metrics

Different dimensionality reduction methods preserve different notions of "distance." Understanding these is crucial for interpretation.

#### 1. Euclidean Distance

**Definition**: Straight-line distance in weight space
```
d_euclidean(x, y) = sqrt(Σᵢ (xᵢ - yᵢ)²)
```

**Properties**:
- [YES] Intuitive and mathematically simple
- [YES] Preserved by PCA
- [NO] Suffers from curse of dimensionality
- [NO] Doesn't respect manifold structure

**When it matters**: If data is truly linear (e.g., weights change via simple scaling), Euclidean distance is appropriate.

---

#### 2. Geodesic Distance

**Definition**: Shortest path along the manifold surface (not through high-D space)

**Example**:
- Two points on a sphere
- Euclidean distance: straight line through sphere interior
- Geodesic distance: arc along surface

**Properties**:
- [YES] Respects manifold structure
- [YES] More meaningful for curved data
- [NO] Computationally expensive to compute
- [NO] Requires knowing manifold structure

**Approximations**:
- **Isomap**: Approximates geodesic via graph shortest paths
- **PHATE**: Approximates via diffusion distances

**When it matters**: Training trajectories that curve through weight space (e.g., circling around a loss landscape minimum).

---

#### 3. Diffusion Distance

**Definition**: How easily a random walk can travel between two points on the data manifold

**Intuition**:
- If points are connected by many short paths through high-density regions → small diffusion distance
- If points separated by low-density regions → large diffusion distance

**Mathematical formulation**:
1. Build affinity matrix (similarity between nearby points)
2. Normalize to get transition probabilities (random walk)
3. Raise to power t (t-step random walk)
4. Measure: `D_t(x,y) = ||p_t(x,·) - p_t(y,·)||`

**Properties**:
- [YES] Robust to noise (diffusion smooths out local perturbations)
- [YES] Multi-scale: parameter `t` controls resolution
- [YES] Intrinsic to data geometry
- [NO] Less interpretable than Euclidean distance

**Used by**: PHATE (primary), diffusion maps

**When it matters**: Noisy trajectory data where you want to capture "flow" along manifold.

---

#### 4. Probabilistic Distances

**t-SNE approach**: Model high-D and low-D data as probability distributions, minimize divergence.

**High-dimensional probabilities**:
- For each point xᵢ, model similarity to neighbors as Gaussian
- `p_ij = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σ_k exp(-||xᵢ - x_k||² / 2σᵢ²)`
- σᵢ chosen based on perplexity (effective number of neighbors)

**Low-dimensional probabilities**:
- Use Student's t-distribution (heavier tails)
- `q_ij = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σ_kl (1 + ||y_k - y_l||²)⁻¹`

**Objective**: Minimize Kullback-Leibler divergence `KL(P||Q)`

**Properties**:
- [YES] Focuses on preserving local neighborhoods
- [NO] Global distances not preserved
- [NO] Optimization is non-convex, stochastic

**Used by**: t-SNE, Heavy-ball SNE

**When it matters**: Clustering, finding discrete groups regardless of global structure.

---

### What Can We Preserve

**Fundamental theorem (Johnson-Lindenstrauss)**: You can approximately preserve pairwise distances when projecting from high-D to low-D, but only if target dimension is large enough.

**For random projections**:
- To preserve distances within (1±ε): need `d ≥ O(log(n) / ε²)` dimensions
- Example: 1000 points, ε=0.1 → need ~7-8 dimensions
- But for 2D/3D visualization, this is too many!

#### Trade-offs in 2D/3D visualization:

| What to preserve | Best method | What's lost |
|------------------|-------------|-------------|
| **Global variance structure** | PCA | Non-linear relationships, manifold curvature |
| **Local neighborhoods** | t-SNE | Global structure, between-cluster distances |
| **Geodesic distances** | Isomap | Robustness to noise, short-circuit connections |
| **Diffusion geometry** | PHATE | Exact Euclidean distances, some details |
| **Balance (local + global)** | UMAP | Some precision in both local and global |

**Key insight**: **Perfect preservation is impossible in 2D/3D**. Every method makes a choice about what to prioritize.

#### What ALL methods preserve:

[YES] **Topology** (to some degree):
- If data has disconnected components, methods should show gaps
- If data is connected, methods should show connections

[YES] **Extreme dissimilarity**:
- Very different points generally remain far apart
- Very similar points generally remain close

[YES] **Qualitative structure**:
- Presence of clusters vs continuum
- Branching patterns
- Trajectory continuity (for trajectory-aware methods)

[NO] **What NO method preserves in 2D**:
- Exact distances (beyond nearest neighbors)
- All angles and orientations
- Absolute scales (units are arbitrary)

---

## Overview: What Dimensionality Reduction Shows

Dimensionality reduction takes high-dimensional data (e.g., thousands of network weights) and projects it into 2D or 3D for visualization. **The fundamental goal is to preserve meaningful relationships while discarding noise.**

### Universal Principles

**All methods create new axes** that don't directly correspond to original features:
- Original data: weights w₁, w₂, ..., wₙ
- Embedded data: coordinates x, y in a learned space

**What's preserved varies by method:**
- **PCA**: Global variance structure (linear relationships)
- **PHATE**: Trajectories and diffusion distances (manifold structure)
- **t-SNE**: Local neighborhoods (clusters)
- **UMAP**: Balance of local and global structure

**Critical insight**: The axes are **arbitrary orientations** in the embedded space. Rotation, reflection, or axis swapping doesn't change interpretation. What matters is:
- **Relative distances** between points
- **Neighborhood relationships**
- **Global topology** (how regions connect)

---

## PCA: Principal Component Analysis

### PCA Mathematical Foundation

PCA finds orthogonal directions of maximum variance in the data through eigendecomposition of the covariance matrix.

#### Core Mathematics

**Given**: Data matrix **X** ∈ ℝⁿˣᵈ (n samples, d dimensions)

**Goal**: Find projection **W** ∈ ℝᵈˣᵏ (k < d) that maximizes variance in projected space

**Steps**:

1. **Center the data**:
   ```
   X̄ = X - mean(X)
   ```

2. **Compute covariance matrix**:
   ```
   C = (1/n) X̄ᵀ X̄ ∈ ℝᵈˣᵈ
   ```

3. **Eigendecomposition**:
   ```
   C = V Λ Vᵀ
   ```
   - **V**: eigenvectors (principal components)
   - **Λ**: diagonal matrix of eigenvalues (variances)

4. **Project to k dimensions**:
   ```
   Y = X̄ W
   ```
   where W = first k columns of V (largest eigenvalues)

#### Why This Works

**Variance maximization**: The first principal component (PC1) is the direction that maximizes:
```
max_w ||Xw||² subject to ||w||=1
```

This is solved by the eigenvector with largest eigenvalue.

**Orthogonality**: Each subsequent PC is orthogonal to previous ones, capturing remaining variance.

**Optimal reconstruction**: PCA minimizes reconstruction error:
```
min_W ||X - X̂||² where X̂ = (XW)Wᵀ
```

#### Variance Explained

**Proportion of variance captured**:
```
Variance explained by PC_i = λᵢ / Σⱼ λⱼ
```

**Cumulative variance** (first k PCs):
```
Σᵢ₌₁ᵏ λᵢ / Σⱼ₌₁ᵈ λⱼ
```

**Example**: If λ₁=45, λ₂=23, total=100, then:
- PC1 explains 45% of variance
- PC1+PC2 explain 68% of variance

#### Loadings: Interpreting Components

**Loadings** = weights of original features in each PC

For PC1 = [w₁, w₂, ..., wₐ]:
- Large |wᵢ| → feature i contributes strongly to PC1
- Sign: positive wᵢ → feature increases with PC1

**Example** (simplified 3-weight network):
```
PC1 = [0.71, 0.10, 0.70]
→ Weights w₁ and w₃ dominate this direction
→ As training progresses along PC1, w₁ and w₃ change most
```

---

### PCA Algorithm

**Input**: Data matrix X (n × d)
**Output**: Embedded coordinates Y (n × k)

```
1. Center data:
   X̄ = X - mean(X, axis=0)

2. Option A: Eigendecomposition (small d)
   a. Compute covariance: C = X̄ᵀX̄ / n
   b. Eigendecomposition: V, Λ = eig(C)
   c. Sort by eigenvalues (descending)
   d. W = V[:, :k]  # first k eigenvectors

3. Option B: SVD (large d, more stable)
   a. SVD: U, S, Vᵀ = svd(X̄)
   b. W = V[:, :k]
   c. Note: singular values S = sqrt(n × eigenvalues)

4. Project:
   Y = X̄ @ W

5. (Optional) Compute variance explained:
   var_explained = S² / sum(S²)
```

**Computational complexity**:
- Covariance method: O(d²n + d³)
- SVD method: O(min(nd², n²d))
- For weight trajectories: typically d >> n, so SVD is preferred

---

### What the Axes Mean

**Principal Component 1 (PC1)** = Direction of maximum variance
- Points spread most along this axis
- Captures the "biggest difference" in the data
- Often correlates with training progress if weights change monotonically

**Principal Component 2 (PC2)** = Direction of second-most variance
- Orthogonal (perpendicular) to PC1
- Captures the "next biggest difference"
- Often reveals secondary effects (e.g., different optimization phases)

**Variance Explained**:
- PC1 might explain 45% of total variance
- PC2 might explain 23%
- Together: 68% (rest is in PC3, PC4, ..., PC_n)

### How to Read PCA Plots

#### Distances
[YES] **Euclidean distances are meaningful**
- Far apart = different weight configurations
- Close together = similar weights

[YES] **Preserved**: Global structure, linear relationships

[NO] **Not preserved**: Non-linear relationships (manifold curvature)

#### Trajectories
- **Straight lines**: Monotonic change in weight space
- **Curves**: Non-linear dynamics (but may be linearization artifact)
- **Direction**: Training typically progresses along PC1 (if variance dominates)

#### Example Interpretation

```
PC1 (45% variance explained)
      ↑
      |    • ← checkpoint 100
      |   /
      |  / ← smooth trajectory
      | /
      |• ← checkpoint 50
      |
Start •─────────→ PC2 (23% variance explained)
```

**Reading**: Training moves along PC1 (major direction of change) with slight PC2 variation (secondary effect, possibly learning rate oscillations or batch noise).

### When PCA is Best

- **Linear dynamics**: If weights change smoothly and linearly
- **Baseline comparison**: Always start with PCA (simplest method)
- **High-dimensional preprocessing**: Before applying PHATE or UMAP
- **Interpretability**: PC loadings show which original weights contribute most

### Limitations

- **Misses manifolds**: If data lies on a curved surface (e.g., sphere), PCA forces it flat
- **Variance ≠ importance**: High-variance noise might dominate low-variance signal
- **Example failure**: Circular trajectories become ellipses (topology lost)

---

## PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding

### PHATE Mathematical Foundation

PHATE uses diffusion geometry to embed data while preserving both local and global structure, with explicit design for continuous trajectories.

#### Core Concept: Diffusion Distances

**Intuition**: Measure similarity by simulating random walks on the data manifold.

**Mathematical framework**:

1. **Build affinity matrix** (similarity graph):
   ```
   A_ij = exp(-||xᵢ - xⱼ||² / ε)  for j in kNN(i)
   ```
   - Adaptive kernel bandwidth ε
   - Sparse: only k-nearest neighbors

2. **Normalize to get Markov transition matrix**:
   ```
   T_ij = A_ij / Σⱼ A_ij
   ```
   - Row-stochastic: each row sums to 1
   - T_ij = probability of random walk from i → j

3. **Diffuse by raising to power t**:
   ```
   T^t = T × T × ... × T  (t times)
   ```
   - t = diffusion time (number of steps)
   - Smooths noise, reveals global structure
   - As t↑: local noise averages out

4. **Compute potential distance**:
   ```
   V_ij = -log(T^t_ij)  (potential distance)
   D(i,j) = ||V_i - V_j||  (diffusion distance)
   ```
   - Log transform emphasizes long-range structure
   - Converts probabilities to "potential"

5. **Embed using MDS** (Multi-Dimensional Scaling):
   ```
   Minimize: Σᵢⱼ (D_ij - ||yᵢ - yⱼ||)²
   ```
   - Find low-D coordinates y that preserve distances D

#### Why Diffusion Works

**Multi-scale structure**:
- Small t: local neighborhoods (fine details)
- Large t: global structure (broad patterns)
- Auto-tuning finds optimal t via Von Neumann entropy

**Noise robustness**:
- Random walk averages over many paths
- Single noisy edges don't dominate
- Effective "denoising" through diffusion

**Trajectory continuity**:
- Temporal points are natural neighbors
- Diffusion preserves sequential order
- Better than distance-only methods for time series

#### T-PHATE: Temporal Extension

For autocorrelated time series (e.g., weight trajectories), T-PHATE adds temporal information:

**Feature augmentation**:
```
X̃ = [X, α × time_index]
```
- α controls temporal weight
- Ensures temporal neighbors stay close

**Delay embedding** (alternative):
```
X̃ᵢ = [Xᵢ, Xᵢ₋₁, ..., Xᵢ₋ₗ]
```
- Augments with lagged features
- Captures temporal dynamics directly

**Blended kernel**:
```
A_ij = exp(-||xᵢ - xⱼ||² / ε_spatial) × exp(-|tᵢ - tⱼ|² / ε_temporal)
```
- Combines spatial and temporal affinity
- Prevents "short-circuit" connections across time

---

### PHATE Algorithm

**Input**: Data X (n × d), parameters k, t, decay
**Output**: Embedding Y (n × 2)

```
1. Optional: PCA preprocessing
   if d > phate_n_pca:
       X = PCA(X, n_components=phate_n_pca)

2. Compute k-nearest neighbor graph:
   For each point i:
       Find k nearest neighbors by Euclidean distance
       Store indices and distances

3. Build adaptive affinity matrix:
   For each point i:
       Compute adaptive bandwidth: ε_i = distance to kth neighbor
       For each neighbor j:
           A_ij = exp(-||xᵢ - xⱼ||² / (ε_i × ε_j))

4. Symmetrize (optional):
   A = (A + Aᵀ) / 2

5. Normalize to Markov matrix:
   T_ij = A_ij / Σⱼ A_ij

6. Apply alpha decay (handling disconnected components):
   T = (1 - decay) × T + decay / n

7. Diffuse by powering:
   T_diff = T^t
   (Use eigendecomposition for efficiency: T = VΛVᵀ → T^t = VΛᵗVᵀ)

8. Compute potential distance:
   V = -log(T_diff + ε_small)  # ε_small avoids log(0)

9. Embed via MDS (metric multidimensional scaling):
   Option A: Classical MDS
       - Center distance matrix
       - Eigendecomposition
       - Take top k eigenvectors

   Option B: Force-directed layout (default)
       - Iterative optimization
       - Gradient descent on stress function
       - Better for complex geometries

10. Return Y (n × 2 coordinates)
```

**Computational complexity**:
- kNN search: O(n log n) with KD-tree or ball-tree
- Affinity matrix: O(nk)
- Diffusion: O(nk²) or O(n²) if dense
- MDS: O(n²) to O(n³) depending on method
- **Overall**: O(n²) to O(n³) — slower than PCA, faster than t-SNE

**Automatic parameter selection**:
- **k**: Adaptive based on n (default: 5-15)
- **t**: Von Neumann entropy maximization
- **decay**: Default 40 (handles disconnected regions)

---

### What the Axes Mean

**PHATE axes are diffusion coordinates**:
- Not directly interpretable like PCA components
- Represent positions in a space where **diffusion distances** are preserved
- Axes are arbitrary (can be rotated without changing meaning)

**Diffusion distance** = How easily you can "flow" between two points along the data manifold
- Small diffusion distance = points are connected by high-density paths
- Large diffusion distance = points separated by low-density regions or gaps

### How to Read PHATE Plots

#### Distances
[YES] **Preserved**: Trajectory continuity, manifold structure
- Points close in PHATE → close in original space along manifold
- Geodesic distances (along curved surfaces) preserved

[NO] **Not preserved**: Exact Euclidean distances
- Two points far apart in PHATE might be nearby in straight-line distance but separated by manifold structure

#### Trajectories
PHATE is **explicitly designed for trajectory data**:
- **Smooth curves**: Continuous evolution in weight space
- **Branching**: Bifurcations or phase transitions
- **Loops**: Cyclic dynamics (e.g., oscillating weights)
- **Velocity changes**: Closer points = slower change, sparser points = rapid change

#### Structure Interpretation

**Clusters**:
- Distinct regions in weight space
- Often correspond to different optimization phases
- Example: initialization → linear regime → feature learning → convergence

**Topology preservation**:
- If training revisits similar configurations, PHATE shows this as loops
- If training explores disconnected regions, PHATE shows gaps

#### Example Interpretation

```
      PHATE-Y
        ↑
        |     ╱──•
        |    │    (convergence plateau)
        |    │
        |     •
        |    ╱  (rapid transition)
        |   •
        |  ╱  (slow progress)
        | •
        •─────────→ PHATE-X
      (init)
```

**Reading**:
- Initial weights (bottom-left) evolve slowly (close points)
- Rapid transition (sparse points) suggests loss landscape change
- Convergence plateau (tight cluster) shows weights stabilizing

### When PHATE is Best

- **Trajectory analysis**: Weight evolution over training
- **Non-linear dynamics**: Capturing curved manifolds
- **Phase detection**: Identifying transitions between training regimes
- **Temporal continuity**: Preserving order of checkpoints

### Parameters and Their Effects

**`phate_knn` (k-nearest neighbors)**:
- **Small k** (5-10): Emphasizes local structure, more detailed trajectories
- **Large k** (20-50): Emphasizes global structure, smoother trajectories
- Auto-tuned based on sample count (default works well)

**`phate_t` (diffusion time)**:
- **Small t**: Local geometry (fine details)
- **Large t**: Global geometry (broad structure)
- Default `t='auto'` adaptively chooses

**`phate_decay` (kernel decay)**:
- Controls affinity sharpness
- Default 40 works for most cases

**`phate_n_pca` (PCA preprocessing)**:
- Reduces dimensionality before PHATE
- Speeds computation and removes noise
- Default 100 (or data dimensionality if smaller)

### Limitations

- **Computational cost**: Slower than PCA (but faster than t-SNE)
- **No explicit loadings**: Can't directly see which weights contribute (unlike PCA)
- **Stochastic**: Minor variations between runs (set random seed for reproducibility)

---

## t-SNE: t-Distributed Stochastic Neighbor Embedding

### t-SNE Mathematical Foundation

t-SNE minimizes the divergence between probability distributions representing pairwise similarities in high-dimensional and low-dimensional spaces.

#### Core Concept: Probabilistic Neighbors

**Key idea**: Instead of preserving distances directly, preserve the probability that points are neighbors.

**Mathematical framework**:

**1. High-dimensional probabilities (Gaussian kernel)**:

For each point xᵢ, define conditional probability that it picks xⱼ as neighbor:
```
p_j|i = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σ_k≠i exp(-||xᵢ - x_k||² / 2σᵢ²)
```

- σᵢ: bandwidth chosen to achieve desired perplexity
- Perplexity ≈ effective number of neighbors
- Different σᵢ for each point (adapts to density)

**Symmetrize** to get joint probabilities:
```
p_ij = (p_j|i + p_i|j) / 2n
```

**2. Perplexity**: Controls effective neighborhood size

```
Perplexity(Pᵢ) = 2^(H(Pᵢ))

where H(Pᵢ) = -Σⱼ p_j|i log₂(p_j|i)  (Shannon entropy)
```

- User specifies desired perplexity (e.g., 30)
- Binary search finds σᵢ to achieve target perplexity
- Typical range: 5-50

**3. Low-dimensional probabilities (Student's t-distribution)**:

For embedded points yᵢ, yⱼ in 2D:
```
q_ij = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σ_k≠l (1 + ||y_k - y_l||²)⁻¹
```

**Why t-distribution?**
- Heavy tails: allows moderate distances in high-D to become larger distances in low-D
- Alleviates "crowding problem": high-D neighbors can spread out in 2D
- Simpler than Gaussian: single parameter (df=1)

**4. Objective function (KL divergence)**:

Minimize:
```
KL(P||Q) = Σᵢⱼ p_ij log(p_ij / q_ij)
```

- Asymmetric: penalizes placing dissimilar points together more than similar points apart
- Focuses on preserving local structure

**5. Gradient descent optimization**:

Gradient with respect to embedding yᵢ:
```
∂KL/∂yᵢ = 4 Σⱼ (p_ij - q_ij)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹
```

Update rule (with momentum):
```
Y^(t) = Y^(t-1) + η∇KL + α(t)(Y^(t-1) - Y^(t-2))
```
- η: learning rate
- α(t): momentum (starts high, decreases)
- Early exaggeration: multiply pᵢⱼ by 4-12 initially to separate clusters

#### Why This Works

**Local structure preservation**:
- High pᵢⱼ (similar points) → optimization tries to make qᵢⱼ high
- Keeps neighborhoods intact

**Ignores global structure**:
- Distant points: pᵢⱼ ≈ 0, so their qᵢⱼ doesn't matter much
- All that matters: who your neighbors are, not absolute positions

**Crowding solution**:
- t-distribution's heavy tails allow dissimilar points to be far apart
- Without it: all points would cluster in center (crowding problem)

**Early exaggeration trick**:
- Multiply pᵢⱼ by 4-12 in first 250 iterations
- Forces tight clusters to form early
- Then refine with true probabilities

---

### t-SNE Algorithm

**Input**: Data X (n × d), perplexity, n_iter
**Output**: Embedding Y (n × 2)

```
1. Compute pairwise distances:
   D_ij = ||xᵢ - xⱼ||²  for all pairs

2. Compute high-dimensional probabilities:
   For each point i:
       Binary search to find σᵢ achieving target perplexity:
           Compute p_j|i with current σᵢ
           Compute perplexity
           Adjust σᵢ until perplexity matches target
   Symmetrize: p_ij = (p_j|i + p_i|j) / 2n

3. Initialize embedding:
   Y ~ N(0, 10⁻⁴ I)  # small random Gaussian

4. Optimization loop (typically 1000 iterations):

   For t = 1 to n_iter:

       a. Compute low-D probabilities:
          q_ij = (1 + ||yᵢ - yⱼ||²)⁻¹ / Z
          where Z = Σ_k≠l (1 + ||y_k - y_l||²)⁻¹

       b. Early exaggeration (first 250 iters):
          if t <= 250:
              p_ij_effective = 12 × p_ij
          else:
              p_ij_effective = p_ij

       c. Compute gradient:
          For each point i:
              ∇_i = 4 Σⱼ (p_ij - q_ij)(yᵢ - yⱼ)(1 + ||yᵢ - yⱼ||²)⁻¹

       d. Update with momentum:
          Y = Y - η∇ + α(t)(Y - Y_prev)
          Y_prev = Y

       e. Optional: Center Y (mean = 0)

5. Return final Y
```

**Computational complexity**:
- Naive: O(n²) per iteration × n_iter → O(n² × n_iter)
- Barnes-Hut approximation: O(n log n) per iteration
- **Typical**: 1000 iterations, so O(n² × 1000) or O(n log n × 1000)

**Approximations for speed**:
- **Barnes-Hut t-SNE**: Uses space-partitioning tree (quadtree/octree)
- Approximates far-field forces → O(n log n)
- Essential for n > 5000

**Stochasticity**:
- Random initialization → different runs give different layouts
- Optimization is non-convex → can get stuck in local minima
- Always use fixed random seed for reproducibility

---

### What the Axes Mean

**t-SNE axes have no inherent meaning**:
- Pure embedding coordinates
- Optimized to preserve local neighborhoods
- **Even more arbitrary than PHATE**: axes, scale, and distances are not comparable between runs

### How to Read t-SNE Plots

#### Distances
[YES] **Preserved**: Local neighborhoods (who your neighbors are)
- Points clustered together → genuinely similar in high-D space

[NO] **NOT preserved**: Global structure, distance magnitudes, cluster-cluster distances
- **Critical**: Distance between clusters is meaningless
- Empty space between clusters doesn't imply large separation in original space
- Cluster sizes don't reflect data density

#### Clusters
**Strong point of t-SNE**:
- **Tight clusters**: Data points very similar in original space
- **Separation**: Different clusters represent distinct regions
- Good for detecting discrete phases or modes

#### Example Interpretation

```
      t-SNE-Y
        ↑
        |   •••     •••
        |   •••     •••  ← two clusters
        |   •••     •••
        |
        |       •••
        |       •••  ← third cluster
        |       •••
        └─────────────→ t-SNE-X
```

**Reading**:
- Three distinct training phases or weight configurations
- Within each cluster: weights are similar
- **Cannot conclude**: How the clusters relate globally (order, distance, connections)

### When t-SNE is Best

- **Phase detection**: Finding discrete training stages
- **Outlier identification**: Detecting unusual checkpoints
- **Cluster discovery**: When you don't know structure a priori

### Parameters and Their Effects

**`perplexity`**:
- Roughly: "expected number of neighbors per point"
- **Small perplexity** (5-15): Many small clusters (emphasizes local structure)
- **Large perplexity** (30-50): Fewer, broader clusters (more global view)
- Rule of thumb: 5 < perplexity < n_samples/3
- Default auto-tuning adapts to sample count

### Limitations

- **Not for trajectories**: Temporal order is lost
- **No global distances**: Cannot compare cluster separations
- **Sensitive to parameters**: Different perplexities give very different plots
- **Slow**: Quadratic complexity in sample count
- **Not deterministic**: Different runs give different layouts (even with same random seed, minor variations occur)

### Common Misinterpretations

[NO] "Cluster A is far from cluster B, so they're very different"
- Cluster spacing is arbitrary

[NO] "The trajectory goes from cluster A → B → C"
- t-SNE doesn't preserve order; use PHATE for trajectories

[YES] "There are three distinct phases in training"
- Correct: clustering reveals discrete modes

---

## UMAP: Uniform Manifold Approximation and Projection

### UMAP Mathematical Foundation

UMAP constructs high-dimensional and low-dimensional fuzzy topological representations and minimizes the cross-entropy between them.

#### Core Concept: Fuzzy Simplicial Sets

**Key idea**: Model data as a fuzzy topological structure (weighted graph) where edges represent connectivity, then find low-D representation preserving this structure.

**Mathematical framework**:

**1. Theoretical foundation (Riemannian geometry)**:

Assumptions:
- Data uniformly distributed on Riemannian manifold
- Metric is locally constant
- Can approximate manifold structure from local distances

**Key result**: Under these assumptions, optimal representation uses exponentially decaying connectivity:
```
w_ij ∝ exp(-d(xᵢ, xⱼ))
```

where d is geodesic distance on manifold.

**2. Practical algorithm (fuzzy sets)**:

**Step 1**: Build k-nearest neighbor graph
- For each point xᵢ, find k nearest neighbors
- Adaptive: k controls local vs global focus

**Step 2**: Compute local distance scaling

For each point i:
```
ρᵢ = distance to nearest neighbor
σᵢ = chosen so that Σⱼ exp(-(d_ij - ρᵢ)/σᵢ) = log₂(k)
```

- ρᵢ: minimum distance (ensures all points connect to at least one neighbor)
- σᵢ: normalization factor (adaptive bandwidth)
- log₂(k): target "perplexity" (similar idea to t-SNE)

**Step 3**: Compute directed edge weights
```
w_j|i = exp(-(max(0, d_ij - ρᵢ))/σᵢ)
```

**Step 4**: Symmetrize (fuzzy set union)
```
w_ij = w_j|i + w_i|j - w_j|i × w_i|j
```

This is **probabilistic t-conorm** (fuzzy OR): gives high weight if either direction has high weight.

**3. Low-dimensional representation**:

**Similar construction** in low-D embedding space, but with simpler distance:
```
v_ij = 1 / (1 + a ||yᵢ - yⱼ||²ᵇ)
```

- a, b: hyperparameters controlling curve shape
- Default: a ≈ 1.93, b ≈ 0.79 (fitted to approximate exponential)
- `min_dist` parameter controls how tightly points cluster

**4. Objective function (cross-entropy)**:

Minimize:
```
CE = Σᵢⱼ w_ij log(w_ij / v_ij) + (1 - w_ij) log((1 - w_ij)/(1 - v_ij))
```

**Interpretation**:
- First term: attractive force (brings connected points together)
- Second term: repulsive force (pushes disconnected points apart)
- Balances local and global structure

**5. Optimization (stochastic gradient descent)**:

Gradient with respect to yᵢ:
```
∂CE/∂yᵢ = Σⱼ [ w_ij × (attractive gradient)
            + (1 - w_ij) × (repulsive gradient) ]
```

**Efficient sampling**:
- Don't compute all n² pairs
- Sample positive edges (high w_ij) from graph
- Sample negative edges (low w_ij) randomly
- Typical: 5-15 negative samples per positive edge

**Update rule**:
```
yᵢ = yᵢ - η × (∂CE/∂yᵢ)
```

with learning rate annealing over iterations.

#### Why This Works

**Balance of local and global**:
- k-NN graph: captures local neighborhoods
- Fuzzy set union: preserves connections at multiple scales
- Cross-entropy: attracts connected points, repels disconnected ones

**Better than t-SNE**:
- Fuzzy set union (vs simple average) better preserves topology
- Cross-entropy (vs KL divergence) symmetric, balances attraction/repulsion
- Negative sampling: efficiently handles disconnected points

**Computational efficiency**:
- k-NN graph: sparse (O(nk) edges vs O(n²))
- Negative sampling: O(n) per iteration, not O(n²)
- Result: much faster than t-SNE for large n

**Theoretical foundation**:
- Based on Riemannian geometry, not just heuristics
- Fuzzy topology provides mathematical rigor
- Parameters have geometric interpretations

#### Key parameters:

**n_neighbors**:
- Controls local vs global balance
- Small k → emphasizes local structure
- Large k → emphasizes global structure
- Directly affects σᵢ computation

**min_dist**:
- Controls tightness of clusters in embedding
- Small (0.01): tight clusters
- Large (0.5): spread out
- Affects a, b parameters in low-D similarity

---

### UMAP Algorithm

**Input**: Data X (n × d), n_neighbors, min_dist, n_epochs
**Output**: Embedding Y (n × 2)

```
1. Build k-nearest neighbor graph:
   For each point i:
       Find k nearest neighbors by Euclidean distance
       Store distances d_ij

2. Compute adaptive bandwidths:
   For each point i:
       ρᵢ = min distance to neighbors
       Binary search for σᵢ such that:
           Σⱼ exp(-(d_ij - ρᵢ)/σᵢ) = log₂(k)

3. Compute high-dimensional edge weights:
   For each edge (i, j):
       w_j|i = exp(-(max(0, d_ij - ρᵢ))/σᵢ)
   Symmetrize: w_ij = w_j|i + w_i|j - w_j|i × w_i|j

4. Compute low-D similarity parameters:
   Fit a, b to approximate exp(-x) with 1/(1 + ax^b)
   Default: a ≈ 1.93, b ≈ 0.79 (based on min_dist)

5. Initialize embedding:
   Option A: Spectral embedding (default)
       Compute Laplacian of graph
       Take eigenvectors corresponding to smallest eigenvalues
   Option B: Random initialization
       Y ~ N(0, 10 I)

6. Optimization (stochastic gradient descent):

   For epoch = 1 to n_epochs:

       For each edge (i,j) with w_ij > 0:

           a. Compute low-D similarity:
              v_ij = 1 / (1 + a ||yᵢ - yⱼ||²ᵇ)

           b. Attractive force (pull together):
              F_attr = -2ab ||yᵢ - yⱼ||^(2b-2) v_ij² (yᵢ - yⱼ)

           c. Sample negative edges (points that should be far):
              For n_neg samples:
                  Pick random point k
                  Repulsive force:
                      v_ik = 1 / (1 + a ||yᵢ - y_k||²ᵇ)
                      F_rep = 2ab ||yᵢ - y_k||^(2b-2) v_ik / (1-v_ik) (yᵢ - y_k)

           d. Update embedding:
              yᵢ = yᵢ + η_epoch × (w_ij × F_attr + F_rep)

       Anneal learning rate:
           η_epoch = η_initial × (1 - epoch/n_epochs)

7. Return final Y
```

**Computational complexity**:
- k-NN: O(n log n) with approximate NN algorithms
- Graph construction: O(nk)
- Optimization: O(nk × n_epochs) with negative sampling
- **Overall**: O(n log n + nk × n_epochs) ≈ O(n) for fixed k, n_epochs

**Typical settings**:
- n_neighbors: 15
- min_dist: 0.1
- n_epochs: 200-500
- negative_sample_rate: 5

**Speed tricks**:
- **Approximate NN**: Use NN-Descent or random projection trees
- **Negative sampling**: Don't compute all pairs
- **Early stopping**: Monitor convergence, stop if no improvement

---

### What the Axes Mean

**UMAP axes are manifold coordinates**:
- Similar to PHATE: represent positions in a learned geometric space
- Based on topological data analysis (Riemannian geometry)
- Axes are arbitrary but more stable than t-SNE

### How to Read UMAP Plots

#### Distances
[YES] **Preserved**: Local structure (neighborhoods) + global structure (topology)
- **Better than t-SNE**: Distances between clusters are somewhat meaningful
- **Better than PCA**: Captures non-linear manifolds

[NOTE] **Partially preserved**: Global distances are approximate
- Clusters far apart generally represent different regions
- But exact distances are not quantitatively precise

#### Balance of Local and Global
UMAP's key strength: **preserving both scales**
- Local neighborhoods (like t-SNE) + broad topology (like PHATE)
- Good for both clustering and trajectory analysis

#### Example Interpretation

```
      UMAP-Y
        ↑
        |    •──•──•  (branch A)
        |   ╱
        |  •
        |  │
        |  •─•─•─•  (branch B)
        |
        •─────────→ UMAP-X
    (shared init)
```

**Reading**:
- Training starts from common initialization (bottom)
- Bifurcation into two distinct trajectories (branches A & B)
- Could represent different hyperparameters or random seeds
- Distance between branches reflects genuine divergence (more reliable than t-SNE)

### When UMAP is Best

- **Trajectory + clustering**: Need both structure types
- **Multiple runs comparison**: Comparing different training configurations
- **Faster than t-SNE**: Scales better to large datasets
- **Reproducibility**: More consistent across runs than t-SNE

### Parameters and Their Effects

**`n_neighbors`**:
- Controls local vs global balance
- **Small n** (5-15): Emphasizes local structure (more clusters)
- **Large n** (30-50): Emphasizes global structure (broader view)
- Default auto-tuning: `min(15, max(2, n_samples-1))`

**`min_dist`**:
- Controls how tightly points cluster
- **Small min_dist** (0.01-0.1): Tight clusters
- **Large min_dist** (0.5-0.99): Looser, more spread out

### Limitations

- **Less trajectory-focused than PHATE**: Not explicitly designed for time series
- **Still stochastic**: Minor variations between runs
- **Hyperparameter sensitivity**: Results change with `n_neighbors` and `min_dist`

---

## Common Patterns and Their Meanings

### Pattern 1: Smooth Trajectories

**What it looks like**:
```
    •──•──•──•──•──•
  (start)        (end)
```

**Interpretation**:
- **Continuous optimization**: No abrupt changes
- **Stable dynamics**: Weights evolve smoothly
- **Good convergence**: Gradual approach to optimum

**Methods**: Best seen in PHATE, UMAP

---

### Pattern 2: Loops or Cycles

**What it looks like**:
```
      •──•
     ╱    ╲
    •      •
     ╲    ╱
      •──•
```

**Interpretation**:
- **Cyclic dynamics**: Weights return to similar configurations
- **Oscillations**: Learning rate too high, or inherent problem structure
- **Limit cycles**: Stable oscillatory regime
- **Note**: Verify not an artifact (check against loss curve)

**Methods**: Best seen in PHATE (preserves topology)

---

### Pattern 3: Sharp Turns or Transitions

**What it looks like**:
```
    •──•──•
            ╲
             •──•──•
```

**Interpretation**:
- **Phase transition**: Sudden change in learning dynamics
- **Loss landscape change**: Crossing from one basin to another
- **Learning rate schedule**: Step decay or warmup end
- **Critical period**: Feature learning onset

**Methods**: All methods can show this; PHATE best preserves transition sharpness

---

### Pattern 4: Plateaus or Clusters

**What it looks like**:
```
               •••
    •──•──•   •••
              •••
```

**Interpretation**:
- **Convergence**: Weights stabilizing in local/global optimum
- **Slow progress**: Small weight updates (look at loss curve)
- **Equilibrium**: Balance between gradient signal and noise

**Methods**: Best seen in t-SNE (clustering), PHATE (trajectory end)

---

### Pattern 5: Branching or Bifurcations

**What it looks like**:
```
        •──•──•  (branch A)
       ╱
    •─•
       ╲
        •──•──•  (branch B)
```

**Interpretation**:
- **Stochastic variation**: Different random seeds diverge
- **Symmetry breaking**: Model chooses one of multiple equivalent solutions
- **Critical point**: Unstable equilibrium with multiple exit paths
- **Hyperparameter effect**: Different learning rates or architectures

**Methods**: UMAP, PHATE (captures topology)

---

### Pattern 6: Outliers or Detached Points

**What it looks like**:
```
    •──•──•──•

           •  (outlier)
```

**Interpretation**:
- **Training instability**: Divergence event, NaN weights
- **Checkpoint error**: Corrupted save or incorrect loading
- **Rare configuration**: Unusual training regime (verify if artifact or genuine)

**Methods**: t-SNE, UMAP (emphasize clusters, outliers stand out)

---

### Pattern 7: Multiple Disconnected Components

**What it looks like**:
```
    •──•──•      •──•──•
   (group A)    (group B)
```

**Interpretation**:
- **Different initializations**: Random seeds or initialization schemes
- **Different architectures**: Comparing models with different structures
- **Hyperparameter regimes**: Distinct learning rate or batch size effects
- **Mode hopping**: Training switched between basins (rare)

**Methods**: UMAP, t-SNE, PHATE (all can show disconnected regions)

---

## Method Comparison Table

| Property                  | PCA | PHATE | t-SNE | UMAP |
|---------------------------|-----|-------|-------|------|
| **Preserves distances**   | [YES] (Euclidean) | [NOTE] (Diffusion) | [NO] | [NOTE] (Approximate) |
| **Preserves neighborhoods** | [NOTE] (Linear) | [YES] | [YES] | [YES] |
| **Preserves global structure** | [YES] | [YES] | [NO] | [NOTE] |
| **Good for trajectories** | [NOTE] | [YES][YES] | [NO] | [YES] |
| **Good for clustering**   | [NOTE] | [NOTE] | [YES][YES] | [YES] |
| **Interpretable axes**    | [YES] (Loadings) | [NO] | [NO] | [NO] |
| **Computational speed**   | [YES][YES] (Fast) | [NOTE] (Moderate) | [NO] (Slow) | [YES] (Fast) |
| **Deterministic**         | [YES] | [NOTE] (Mostly) | [NO] | [NOTE] (Mostly) |
| **Scales to large data**  | [YES] | [YES] | [NO] | [YES] |
| **Handles non-linear**    | [NO] | [YES] | [YES] | [YES] |

### Method Selection Decision Tree

```
Start: What do you want to understand?

├─ Baseline / quick check?
│  └─ → PCA
│
├─ Temporal dynamics / training trajectories?
│  ├─ Simple linear evolution?
│  │  └─ → PCA
│  └─ Complex non-linear evolution?
│     └─ → PHATE (or T-PHATE for temporal structure)
│
├─ Discrete phases / clusters?
│  ├─ Only clustering (order doesn't matter)?
│  │  └─ → t-SNE
│  └─ Clustering + how they relate?
│     └─ → UMAP
│
├─ Compare multiple runs / hyperparameters?
│  └─ → UMAP (or PHATE)
│
└─ Need interpretable axes (which weights change)?
   └─ → PCA
```

---

## Best Practices for Interpretation

### 1. Always Start with PCA

**Rationale**: Simplest method, fast, interpretable
- If PCA looks good, you might not need complex methods
- If PCA looks bad (e.g., curved trajectories forced into lines), try non-linear methods

### 2. Use Multiple Methods

**Rationale**: Each method highlights different aspects
- **PCA + PHATE**: Compare linear vs manifold interpretation
- **t-SNE + UMAP**: Validate clustering (if both show it, likely real)
- **Cross-reference**: Patterns seen in multiple methods are robust

### 3. Correlate with Other Metrics

**Never interpret embeddings in isolation**:
- **Loss curve**: Do sharp turns correspond to loss changes?
- **Accuracy**: Do clusters map to performance regimes?
- **Learning rate schedule**: Do transitions align with schedule changes?
- **Other visualizations**: Check CKA, mutual information, GRU observability

### 4. Check for Artifacts

**Common artifacts**:
- **Horseshoe effect** (PCA): Spurious curve from autocorrelation
- **Crowding problem** (t-SNE): High-D structure forced into low-D
- **Disconnected components**: May be noise or genuine signal (verify)

**Validation**:
- Try different hyperparameters (k, perplexity, n_neighbors)
- If structure disappears, it might be artifact
- If structure persists, likely genuine

### 5. Mind the Sample Size

**Too few samples** (< 20):
- All methods struggle
- Consider collecting more checkpoints

**Sweet spot** (50-500):
- All methods work well
- PHATE and UMAP shine

**Large samples** (> 1000):
- PCA and UMAP scale best
- PHATE and t-SNE slower (but see subsampling options)

### 6. Temporal Order Matters

**For trajectory analysis**:
- Connect points in time order (use line plots, not scatter)
- Add time colormaps (early = blue, late = red)
- Annotate key checkpoints (e.g., epoch 0, 50, 100)

**Example**:
```python
plt.plot(x, y, 'o-', alpha=0.6)  # Line connects trajectory
plt.scatter(x, y, c=time_steps, cmap='viridis')  # Color by time
```

### 7. Report Method and Parameters

**For reproducibility and interpretation**, always report:
- Method used (PCA, PHATE, t-SNE, UMAP)
- Key parameters (e.g., `phate_knn=10`, `perplexity=30`)
- Any preprocessing (e.g., "PCA to 100D before PHATE")
- Random seed (for stochastic methods)

### 8. Don't Over-interpret

**Be cautious about**:
- **Fine details**: Small wiggles might be noise
- **Exact positions**: Focus on relative structure, not absolute coordinates
- **Between-run comparisons**: Unless using same method + parameters, embeddings not directly comparable

**Safe interpretations**:
- Presence of clusters vs smooth continuum
- Trajectory smoothness vs abrupt transitions
- Relative ordering of phases (for PHATE/UMAP)
- Branching or convergence events

---

## Further Reading

- **PHATE**: Rübel et al. (2023), "Delayed-Embedding PHATE for Visualization and Computation on Autoregressive Trajectories"
- **UMAP**: McInnes et al. (2018), "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
- **t-SNE**: van der Maaten & Hinton (2008), "Visualizing Data using t-SNE"
- **PCA**: Jolliffe (2002), "Principal Component Analysis"
- **Methods reference**: See [`docs/reference/methods.md`](../../reference/methods.md)
- **Visualization commands**: See [`docs/manual/commands/visualize.md`](../commands/visualize.md)
- **Plot-specific guides**: See [`docs/manual/plots/`](../plots/README.md)

---

## Summary

**Key Takeaways**:

1. **Axes are not directly interpretable** (except PCA) — focus on relative distances and topology
2. **Each method preserves different properties** — choose based on your question
3. **PHATE for trajectories**, t-SNE for clusters, UMAP for both, PCA for baselines
4. **Always cross-reference with loss curves and other metrics**
5. **Use multiple methods** to validate findings
6. **Report methods and parameters** for reproducibility

**Golden Rule**: Dimensionality reduction reveals structure, but requires external validation. Never draw strong conclusions from embeddings alone — they are exploratory tools, not proof.

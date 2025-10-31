# Fixed-Point Analysis for RNNs

Finding and characterizing stable computational modes in recurrent neural networks through dynamical systems theory.

---

## Overview

**Fixed-point analysis** reveals the computational structure of trained RNNs by finding states where the dynamics stabilize.

**Key insight**: Recurrent networks perform computation by flowing through phase space and settling into stable modes (attractors). These modes correspond to discrete computations or memory states.

**Foundational work**: Sussillo & Barak (2013) demonstrated that fixed-point analysis reveals how RNNs solve tasks.

---

## Mathematical Foundation

### Definition

For an RNN with update rule:
```
h_{t+1} = f(h_t, x_t; θ)
```

A **fixed point** h* (for input x*) satisfies:
```
h* = f(h*, x*; θ)
```

If the network reaches h* and input stays at x*, it remains at h* forever.

### Why Fixed Points Matter

**Computational modes**: Each fixed point represents a stable state the network can maintain.

**Example - Memory**: In a 3-bit flip-flop task, 8 fixed points correspond to 8 memory configurations (000, 001, ..., 111)

**Example - Integration**: A sentiment analyzer may have a line of fixed points representing accumulated sentiment.

**Insight**: The structure of fixed points reveals what the network computes.

---

## Finding Fixed Points

###Step-by-step Algorithm

**Given**:
- Trained RNN
- Input context x* (e.g., a specific board state, sentence prefix)

**Goal**: Find all h* such that h* = f(h*, x*)

**Method**: Optimization-based search

```python
import torch
import numpy as np
from scipy.optimize import minimize

def find_fixed_points(rnn, input_context, n_inits=100, tol=1e-6):
    """
    Find fixed points of RNN for given input context.

    Args:
        rnn: trained RNN module (callable: h_next = rnn(x, h))
        input_context: fixed input x* (tensor)
        n_inits: number of random initializations
        tol: convergence tolerance

    Returns:
        fixed_points: list of found fixed points
        stabilities: list of stability classifications
    """
    fixed_points = []

    for i in range(n_inits):
        # Random initialization
        h0 = np.random.randn(rnn.hidden_size) * 0.5

        # Define objective: ||h - f(h, x*)||^2
        def objective(h):
            h_tensor = torch.tensor(h, dtype=torch.float32)
            with torch.no_grad():
                h_next = rnn(input_context, h_tensor)
            residual = h_next.numpy() - h
            return np.sum(residual**2)

        # Minimize
        result = minimize(
            objective,
            h0,
            method='L-BFGS-B',
            tol=tol,
            options={'maxiter': 1000}
        )

        # Check convergence
        if result.success and result.fun < tol:
            h_star = result.x

            # Check if this is a new fixed point (not a duplicate)
            is_new = True
            for existing_fp in fixed_points:
                if np.linalg.norm(h_star - existing_fp) < 0.01:
                    is_new = False
                    break

            if is_new:
                fixed_points.append(h_star)
                print(f"Found fixed point {len(fixed_points)}: residual = {result.fun:.2e}")

    return fixed_points
```

### Practical Considerations

**Number of initializations**:
- Simple tasks: 50-100
- Complex tasks: 500-1000
- Trade-off: More inits find more fixed points but take longer

**Tolerance**:
- Standard: 1e-6
- Tighter (1e-8): More precise but may fail to converge
- Looser (1e-4): Faster but less accurate

**Initialization strategy**:
- Random Gaussian: Simple, works well
- From trajectories: Sample from actual network states
- Grid: Systematic coverage (expensive)

---

## Stability Analysis

Once fixed points are found, classify their stability.

### Computing the Jacobian

The Jacobian J = ∂f/∂h at h* describes local linear dynamics.

```python
def compute_jacobian(rnn, h_star, input_context):
    """
    Compute Jacobian of RNN update at fixed point.

    Uses automatic differentiation.

    Args:
        rnn: RNN module
        h_star: fixed point (numpy array)
        input_context: input tensor

    Returns:
        jacobian: (hidden_size, hidden_size) numpy array
    """
    h_tensor = torch.tensor(h_star, dtype=torch.float32, requires_grad=True)

    # Forward pass
    h_next = rnn(input_context, h_tensor)

    # Compute Jacobian via autograd
    jacobian = torch.autograd.functional.jacobian(
        lambda h: rnn(input_context, h),
        h_tensor
    )

    return jacobian.detach().numpy()
```

### Eigenvalue Analysis

Stability is determined by eigenvalues of the Jacobian:

```python
def classify_stability(jacobian):
    """
    Classify fixed point stability from Jacobian eigenvalues.

    Returns:
        stability: 'attractor', 'saddle', or 'repeller'
        eigenvalues: complex eigenvalues
        timescales: characteristic timescales
    """
    eigenvalues = np.linalg.eigvals(jacobian)
    magnitudes = np.abs(eigenvalues)

    # Classify
    if np.all(magnitudes < 1):
        stability = 'attractor'
    elif np.all(magnitudes > 1):
        stability = 'repeller'
    else:
        stability = 'saddle'

    # Compute timescales
    # τ = -1 / ln|λ|, but avoid division by zero
    timescales = -1.0 / np.log(np.maximum(magnitudes, 1e-10))

    return stability, eigenvalues, timescales

# Usage
jacobian = compute_jacobian(rnn, h_star, input_context)
stability, eigs, timescales = classify_stability(jacobian)

print(f"Stability: {stability}")
print(f"Max timescale: {np.max(timescales):.2f} steps")
```

### Interpretation

**Stable attractor** (all |λ| < 1):
- Trajectories converge to this point
- Represents a stable computational mode
- Robust to small perturbations

**Saddle point** (mixed |λ|):
- Stable in some directions, unstable in others
- Often acts as transition state between attractors
- Can route trajectories

**Unstable repeller** (all |λ| > 1):
- Trajectories diverge
- Rarely relevant for trained networks

---

## Comprehensive Workflow

### End-to-End Pipeline

```python
def analyze_fixed_point_structure(rnn, input_contexts, n_inits=100):
    """
    Complete fixed-point analysis for multiple contexts.

    Args:
        rnn: trained RNN
        input_contexts: list of input tensors to analyze
        n_inits: number of random initializations per context

    Returns:
        results: dict with fixed points and analysis for each context
    """
    results = {}

    for i, context in enumerate(input_contexts):
        print(f"\n=== Context {i} ===")

        # Find fixed points
        fps = find_fixed_points(rnn, context, n_inits=n_inits)

        # Analyze each fixed point
        fp_data = []
        for fp in fps:
            jacobian = compute_jacobian(rnn, fp, context)
            stability, eigs, timescales = classify_stability(jacobian)

            fp_data.append({
                'state': fp,
                'stability': stability,
                'eigenvalues': eigs,
                'timescales': timescales,
                'jacobian': jacobian
            })

            print(f"  FP: stability={stability}, max_timescale={np.max(timescales):.2f}")

        results[i] = {
            'context': context,
            'fixed_points': fp_data,
            'n_attractors': sum(1 for fp in fp_data if fp['stability'] == 'attractor'),
            'n_saddles': sum(1 for fp in fp_data if fp['stability'] == 'saddle')
        }

    return results
```

### Visualization

Project high-dimensional fixed points to 2D:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_fixed_points(results):
    """Visualize fixed points in 2D PCA space."""
    # Collect all fixed points
    all_fps = []
    all_stabilities = []

    for context_results in results.values():
        for fp_data in context_results['fixed_points']:
            all_fps.append(fp_data['state'])
            all_stabilities.append(fp_data['stability'])

    all_fps = np.array(all_fps)

    # PCA projection
    pca = PCA(n_components=2)
    fps_2d = pca.fit_transform(all_fps)

    # Plot
    colors = {'attractor': 'blue', 'saddle': 'orange', 'repeller': 'red'}

    plt.figure(figsize=(10, 8))
    for stability in ['attractor', 'saddle', 'repeller']:
        mask = np.array(all_stabilities) == stability
        if mask.any():
            plt.scatter(fps_2d[mask, 0], fps_2d[mask, 1],
                       c=colors[stability], label=stability,
                       s=100, alpha=0.7, edgecolors='black')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Fixed Points in PCA Space')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
```

---

## Common Patterns

### Discrete Memory

**Pattern**: Small number of stable attractors

**Example**: 3-bit flip-flop → 8 attractors (one per configuration)

**Recognition**:
- Few attractors (< 20)
- Well-separated in phase space
- Correspond to discrete task states

### Continuous Integration

**Pattern**: Line of fixed points

**Example**: Sentiment integration → line from negative to positive

**Recognition**:
- Many fixed points forming continuous curve
- Eigenvalues near 1 along line (marginal stability)
- Eigenvalues << 1 perpendicular to line

### Decision Boundaries

**Pattern**: Saddle points between attractors

**Example**: Win/loss threshold in games

**Recognition**:
- Saddle points at boundary regions
- Attractors on either side
- Trajectories pass through or near saddles

---

## Trajectory Analysis

Complement fixed-point analysis with trajectory visualization:

```python
def simulate_trajectory(rnn, h0, input_sequence):
    """Simulate RNN trajectory through phase space."""
    trajectory = [h0]

    h = torch.tensor(h0, dtype=torch.float32)
    for x in input_sequence:
        with torch.no_grad():
            h = rnn(x, h)
        trajectory.append(h.numpy())

    return np.array(trajectory)

def visualize_trajectory_with_fps(trajectory, fixed_points, pca):
    """Show trajectory flowing through fixed-point landscape."""
    # Project trajectory and fixed points
    traj_2d = pca.transform(trajectory)
    fps_2d = pca.transform(fixed_points)

    plt.figure(figsize=(10, 8))

    # Plot fixed points
    plt.scatter(fps_2d[:, 0], fps_2d[:, 1],
               c='red', s=200, marker='*',
               label='Fixed Points', zorder=5)

    # Plot trajectory
    plt.plot(traj_2d[:, 0], traj_2d[:, 1],
            'b-', alpha=0.5, linewidth=2)
    plt.scatter(traj_2d[0, 0], traj_2d[0, 1],
               c='green', s=100, marker='o',
               label='Start', zorder=4)
    plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1],
               c='purple', s=100, marker='s',
               label='End', zorder=4)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Trajectory Through Fixed-Point Landscape')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
```

---

## Tracking Across Training

Analyze how fixed points evolve during training:

```python
def track_fixed_points_over_training(checkpoint_dir, input_context):
    """Track fixed-point evolution across epochs."""
    evolution = []

    for ckpt_path in sorted(glob.glob(f"{checkpoint_dir}/weights_epoch_*.pt")):
        epoch = int(ckpt_path.split('_')[-1].split('.')[0])

        # Load model
        rnn = load_model(ckpt_path)

        # Find fixed points
        fps = find_fixed_points(rnn, input_context, n_inits=50)

        evolution.append({
            'epoch': epoch,
            'n_fixed_points': len(fps),
            'fixed_points': fps
        })

        print(f"Epoch {epoch}: {len(fps)} fixed points found")

    return evolution

# Visualize evolution
def plot_fp_evolution(evolution):
    epochs = [e['epoch'] for e in evolution]
    n_fps = [e['n_fixed_points'] for e in evolution]

    plt.plot(epochs, n_fps, marker='o')
    plt.xlabel('Training Epoch')
    plt.ylabel('Number of Fixed Points')
    plt.title('Fixed-Point Emergence During Training')
    plt.grid(alpha=0.3)
    plt.show()
```

**Typical observations**:
- Early training: Few or no stable fixed points
- Middle training: Attractors begin to emerge
- Late training: Attractors sharpen, saddles appear

---

## Computational Considerations

### Scaling

**Problem**: Fixed-point finding is expensive

**Time complexity**:
- Per optimization: O(iterations × forward_pass)
- Total: O(n_inits × n_contexts × iterations)

**Practical limits**:
- Small RNNs (32-128 units): 100-1000 inits
- Large RNNs (>256 units): May need parallelization

### Parallelization

```python
from multiprocessing import Pool

def find_single_fp(args):
    """Worker function for parallel search."""
    rnn, input_context, seed = args
    np.random.seed(seed)
    return find_fixed_points(rnn, input_context, n_inits=1)

def parallel_fixed_point_search(rnn, input_context, n_inits=100, n_workers=8):
    """Parallelize fixed-point search."""
    args = [(rnn, input_context, i) for i in range(n_inits)]

    with Pool(n_workers) as pool:
        results = pool.map(find_single_fp, args)

    # Collect and de-duplicate
    all_fps = []
    for fps in results:
        all_fps.extend(fps)

    # De-duplicate
    unique_fps = deduplicate_fixed_points(all_fps, threshold=0.01)

    return unique_fps
```

### Caching

```python
def cache_fixed_points(fps, filepath):
    """Save fixed points to disk."""
    np.savez(filepath,
             fixed_points=[fp['state'] for fp in fps],
             stabilities=[fp['stability'] for fp in fps])

def load_cached_fixed_points(filepath):
    """Load cached fixed points."""
    data = np.load(filepath)
    return data['fixed_points'], data['stabilities']
```

---

## Limitations and Caveats

### When Fixed-Point Analysis Works Well

**Autonomous or slowly-changing input**:
- Board games (input changes discretely)
- Sentence processing (input changes gradually)

**Smooth dynamics**:
- Continuous activation functions (tanh, sigmoid)
- Well-conditioned Jacobians

**Moderate dimensionality**:
- Hidden size 32-512
- Higher dimensions harder to visualize/interpret

### When It May Not Work

**Rapidly varying input**:
- Video processing (input changes every frame)
- Real-time control (continuous input stream)

**Transient computation**:
- Some tasks solved during transients, not at attractors

**Chaotic dynamics**:
- If network exhibits chaos, fixed points less meaningful

**Very high dimensions**:
- Visualization/interpretation becomes difficult
- Computational cost increases

---

## Validation and Sanity Checks

### Check 1: Residual Magnitude

```python
def verify_fixed_point(rnn, h_star, input_context):
    """Verify h* is actually a fixed point."""
    h_tensor = torch.tensor(h_star, dtype=torch.float32)
    with torch.no_grad():
        h_next = rnn(input_context, h_tensor)

    residual = np.linalg.norm(h_next.numpy() - h_star)
    print(f"Residual: {residual:.2e}")

    return residual < 1e-5  # Should be very small
```

### Check 2: Trajectory Convergence

```python
def test_convergence(rnn, h_star, input_context, n_steps=100):
    """Test if nearby states converge to fixed point."""
    # Perturb fixed point
    h_perturbed = h_star + np.random.randn(len(h_star)) * 0.01

    # Simulate
    trajectory = simulate_trajectory(
        rnn,
        h_perturbed,
        [input_context] * n_steps
    )

    # Check convergence
    final_dist = np.linalg.norm(trajectory[-1] - h_star)
    print(f"Distance after {n_steps} steps: {final_dist:.2e}")

    return final_dist < 0.1  # Should converge if attractor
```

### Check 3: Consistency Across Optimizers

```python
# Try different optimization methods
for method in ['L-BFGS-B', 'trust-constr', 'SLSQP']:
    fps = find_fixed_points(rnn, input_context, method=method, n_inits=10)
    print(f"{method}: {len(fps)} fixed points")

# Should find similar numbers
```

---

## Integration with Other Methods

### With Probing

Fixed points reveal computational modes; probes test what they represent:

```python
# Find fixed points
fps = find_fixed_points(rnn, context, n_inits=100)

# Probe fixed points for concepts
from sklearn.cluster import KMeans

# Cluster fixed points
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(fps)

# Interpret clusters: what do they represent?
# Could use probes or visualize with PCA colored by cluster
```

### With Trajectory Visualization

Combine fixed points with PHATE embeddings:

```python
import phate

# Collect fixed points and trajectory states
all_states = np.vstack([fps, trajectory_states])
labels = ['FP'] * len(fps) + ['Traj'] * len(trajectory_states)

# Embed with PHATE
phate_op = phate.PHATE(n_components=2)
embedded = phate_op.fit_transform(all_states)

# Visualize
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels=='FP')
plt.title('Fixed Points and Trajectories in PHATE Space')
```

---

## Case Studies

**See also**:
- [Flip-Flop Attractors](../../5_case_studies/recurrent_networks/flip_flop_attractors.md) - Sussillo & Barak 2013
- [Sentiment Line Attractors](../../5_case_studies/natural_language/sentiment_line_attractors.md) - Maheswaranathan et al. 2019
- [Connect Four GRU](../../5_case_studies/board_games/connect_four_gru.md) - This project

---

## Further Reading

**Foundational papers**:
- Sussillo & Barak (2013): Opening the black box. *Neural Computation*
- Maheswaranathan et al. (2019): Reverse engineering sentiment RNNs. *NeurIPS*
- Golub & Sussillo (2018): FixedPointFinder software. *JOSS*

**Theory**:
- Strogatz (1994): Nonlinear Dynamics and Chaos (textbook)
- [Dynamical Systems Primer](../../1_foundations/dynamical_systems_primer.md)

**Related methods**:
- [Trajectory Analysis](trajectory_analysis.md) - PHATE embeddings
- [Attractor Landscapes](attractor_landscapes.md) - Theoretical framework

Full bibliography: [References](../../references/bibliography.md)

---

**Return to**: [Dynamical Analysis](README.md) | [Methods](../README.md)

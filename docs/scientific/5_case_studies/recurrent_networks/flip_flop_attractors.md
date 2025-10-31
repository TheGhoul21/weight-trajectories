# Case Study: 3-Bit Flip-Flop (Sussillo & Barak, 2013)

A foundational study demonstrating how fixed-point analysis reveals computational structure in trained RNNs.

---

## Task & Architecture

**Task**: Implement a 3-bit memory device that flips bits on command.
- **Inputs**: 3 flip commands (one per bit), given at arbitrary times
- **Outputs**: Current state of 3 bits (0 or 1)
- **Challenge**: Maintain memory between commands (temporal integration)

**Architecture**: Vanilla RNN with 24 hidden units, tanh activations.

---

## Analysis Methods

1. **Fixed-point finding**: Optimized to find `h*` where `h* = f(h*, x_context)`
2. **Stability analysis**: Computed Jacobians and eigenvalue spectra at each fixed point
3. **Visualization**: Projected 24D state space to 3D via PCA

---

## Key Findings

### Attractor Structure
- Network developed **8 stable fixed points**, one for each possible 3-bit configuration (000, 001, 010, ..., 111)
- Each attractor corresponds to a discrete memory state
- Flip commands cause **rapid transitions** between attractors

### Computation via Saddle Points
- **Saddle points** (unstable fixed points) lie between attractors
- Trajectories pass through saddle regions during bit flips
- Saddles act as "gates" controlling which attractor is reached

### Linearized Dynamics
- Near attractors: all eigenvalues |λ| < 0.5 (strong stability)
- Near saddles: 1-2 unstable eigenvalues |λ| > 1 (repelling directions)
- Stable manifolds of saddles form basin boundaries

### Visualization Result
3D projection shows 8 attractors at corners of a "cube" in state space—the network **explicitly represents the Boolean structure** of the task.

---

## Relevance to Other Work

### Analogies to Game-Playing RNNs
- Discrete strategic modes (attack/defend/explore) ↔ bit states (0/1)
- State changes ↔ flip commands
- GRU memory ↔ RNN sustained activity

### Differences from Continuous Tasks
- Discrete memory states vs continuous variables
- Small, well-defined state space vs large combinatorial spaces
- GRU gating may enable more flexible attractor structure

### Actionable Insights
- Expect small number of attractors (< 20) for discrete mode tasks
- Look for saddle points at decision boundaries
- Use dimensionality reduction (PCA, PHATE) to visualize attractor layout before computing fixed points

---

## Implementation Notes

### Fixed-Point Finding Algorithm

```python
# Pseudocode from Sussillo & Barak 2013
def find_fixed_points(rnn, input_context, n_inits=128):
    """
    Args:
        rnn: trained RNN module
        input_context: fixed input x* (e.g., a board state)
        n_inits: number of random initializations

    Returns:
        fixed_points: list of h* where h* = f(h*, x*)
        jacobians: list of dh_{t+1}/dh_t at each h*
        stability: classification (attractor/saddle/repeller)
    """
    fps = []
    for i in range(n_inits):
        h0 = random_init()  # sample random initial state

        # Minimize ||h - f(h, x*)||^2
        result = scipy.optimize.minimize(
            lambda h: np.sum((rnn(x*, h) - h)**2),
            h0,
            method='L-BFGS-B',
            tol=1e-6
        )

        if result.success and result.fun < 1e-6:
            h_star = result.x

            # Check if this is a new fixed point (not already found)
            if not any(np.allclose(h_star, fp) for fp in fps):
                fps.append(h_star)

    # Compute stability for each fixed point
    stability = []
    for h_star in fps:
        J = compute_jacobian(rnn, x*, h_star)
        eigs = np.linalg.eigvals(J)

        if all(np.abs(eigs) < 1):
            stability.append('attractor')
        elif all(np.abs(eigs) > 1):
            stability.append('repeller')
        else:
            stability.append('saddle')

    return fps, stability
```

### Key Parameters
- `n_inits=128`: Sussillo used 500-1000 for high-dimensional RNNs
- `tol=1e-6`: balance between precision and avoiding numerical issues
- Optimization method: L-BFGS-B works well; trust-region also used

### Practical Considerations
- Multiple random initializations needed to find all fixed points
- Clustering step required to remove duplicates
- Automatic differentiation (PyTorch/JAX) simplifies Jacobian computation

---

## Related Methods

**In this handbook**:
- [Fixed-Point Analysis](../../2_methods/dynamical_analysis/fixed_points.md) - Detailed methodology
- [Attractor Landscapes](../../2_methods/dynamical_analysis/attractor_landscapes.md) - Theoretical background
- [Dynamical Systems Primer](../../1_foundations/dynamical_systems_primer.md) - Mathematical foundations

**Tools**:
- FixedPointFinder (Golub & Sussillo, 2018): TensorFlow implementation
- Custom PyTorch implementations available

---

## References

**Primary paper**:
Sussillo, D., & Barak, O. (2013). Opening the black box: Low-dimensional dynamics in high-dimensional recurrent neural networks. *Neural Computation*, 25(3), 626-649. doi:10.1162/NECO_a_00409

**Related work**:
- Maheswaranathan et al. (2019): Extension to sentiment analysis
- Yang et al. (2019): Multi-task RNN dynamics
- Khona & Fiete (2022): Neuroscience perspective on attractors

Full bibliography: [References](../../references/bibliography.md)

---

**Return to**: [Case Studies](../README.md) | [Recurrent Networks](README.md)

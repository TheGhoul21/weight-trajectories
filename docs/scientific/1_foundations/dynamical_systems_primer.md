# Dynamical Systems Primer

An accessible introduction to dynamical systems concepts for understanding recurrent neural networks.

---

## Overview

A **dynamical system** describes how a state evolves over time according to fixed rules. Recurrent neural networks are dynamical systems: the hidden state at time t determines the state at t+1.

Understanding RNN computation through dynamical systems theory reveals:
- What stable computational modes exist (fixed points, attractors)
- How the network transitions between modes
- Why certain computations are robust or fragile

---

## What is a Dynamical System?

**General form**:
```
x(t+1) = f(x(t))
```

Where:
- `x(t)` is the **state** at time t (vector describing the system)
- `f` is the **update rule** (function determining evolution)
- System evolves by repeatedly applying f

**Example: Population growth**
```
population(t+1) = 1.1 × population(t)
```
Simple dynamics: exponential growth with factor 1.1

**Example: RNN**
```
h(t+1) = tanh(W_h · h(t) + W_x · x(t) + b)
```
Complex dynamics determined by learned weights W_h, W_x, b

---

## Phase Space and Trajectories

### Phase Space

The **phase space** is the space of all possible states.

**Example: Pendulum**
- State: (angle, angular velocity)
- Phase space: 2D (angle vs velocity)
- Each point represents a possible pendulum configuration

**Example: RNN with 64 units**
- State: h ∈ R^64
- Phase space: 64-dimensional
- Each point is a possible hidden state

### Trajectories

A **trajectory** is the path through phase space as the system evolves.

**Visualization** (2D example):
```
           State space (2D)

    h₂ ↑    ···○······
       │   ·   ○     ·
       │  ·    ○    ·
       │ ·     ○   ·
       │·      ○  ·
       ○       ○ ·
       │       ○·
       └───────○─────→ h₁
          trajectory
```

**For RNNs**: Trajectories show how hidden state evolves while processing a sequence.

---

## Fixed Points

A **fixed point** h* satisfies:
```
h* = f(h*)
```

If the system reaches h*, it stays there forever (assuming constant input).

### Finding Fixed Points

**Approach 1: Solve algebraically**

For simple systems, solve directly.

Example: `x(t+1) = 0.5x(t) + 3`
```
x* = 0.5x* + 3
0.5x* = 3
x* = 6
```

**Approach 2: Numerical optimization**

For complex systems (like RNNs), minimize:
```
||h - f(h)||²
```

Using gradient descent or other optimizers.

```python
from scipy.optimize import minimize

def find_fixed_point(f, h_init):
    """Find fixed point by minimizing ||h - f(h)||²"""
    result = minimize(
        lambda h: np.sum((h - f(h))**2),
        h_init,
        method='L-BFGS-B'
    )
    return result.x if result.success else None
```

### Multiple Fixed Points

Systems can have multiple fixed points.

**Example**: `x(t+1) = x(t)² - 1`

Solving `x* = x*² - 1`:
```
x*² - x* - 1 = 0
x* = (1 ± √5)/2

Two fixed points: x* ≈ 1.618 and x* ≈ -0.618
```

**RNNs**: Often have many fixed points, each representing a different computational mode.

---

## Stability Analysis

Not all fixed points are equal. Some attract nearby trajectories, others repel them.

### Linear Stability

Near a fixed point h*, linearize the dynamics:
```
h(t+1) - h* ≈ J · (h(t) - h*)
```

Where **J = ∂f/∂h** evaluated at h* is the **Jacobian matrix**.

### Eigenvalue Criterion

Stability is determined by eigenvalues λ of J:

**Stable (attractor)**: All |λ| < 1
- Small perturbations decay
- Trajectories converge to fixed point

**Unstable (repeller)**: Any |λ| > 1
- Small perturbations grow
- Trajectories diverge from fixed point

**Saddle**: Mixed eigenvalues (some |λ| < 1, others |λ| > 1)
- Stable in some directions, unstable in others
- Trajectories approach along stable manifold, repel along unstable manifold

### Visual Intuition

**1D Examples**:

**Stable fixed point** (|λ| < 1):
```
    f(x)
     ↑
     │   ╱
     │  ╱
     │ ╱
    ─┼╱────── x
     ╱│
   ╱  │

  Slope at crossing < 1 → stable
```

**Unstable fixed point** (|λ| > 1):
```
    f(x)
     ↑  ╱
     │ ╱
     │╱
    ─┼────── x
    ╱│
   ╱ │

  Slope at crossing > 1 → unstable
```

**2D Attractor**:
```
    h₂
     ↑
     │    →  →
     │  ↓     ↓
     │   ↓→  ←
     │    ●    ← trajectories spiral inward
     │   ← ↖
     └──────── h₁
```

---

## Attractors

An **attractor** is a set of states that trajectories converge to.

### Types of Attractors

**Point attractor**: Single fixed point
- Example: Pendulum with friction settles to rest
- RNN example: Discrete memory state

**Limit cycle**: Periodic oscillation
- Example: Heartbeat, planetary orbit
- RNN example: Rhythm generation (rare)

**Strange attractor**: Chaotic, fractal structure
- Example: Lorenz attractor (weather)
- RNN example: Pathological training (avoid!)

**Line attractor**: Continuous manifold
- Example: Integrator (memory of continuous variable)
- RNN example: Sentiment accumulation

**For interpretability**: We focus on point and line attractors.

### Basin of Attraction

The **basin** of an attractor is the set of initial conditions that converge to it.

**Visualization** (1D):
```
  Basins:    [---A---]  [---B---]
  Attractors:    ●          ●
             ←──  ──→  ←──  ──→
```

**Separatrix**: Boundary between basins (often contains saddle points).

**In RNNs**: Different input contexts may drive the system into different basins, corresponding to different computational modes.

---

## Jacobian and Local Dynamics

The **Jacobian matrix** J describes how small perturbations evolve.

### Computing the Jacobian

**Definition**:
```
J_ij = ∂f_i/∂h_j
```

**For RNNs**: Use automatic differentiation.

```python
import torch

def compute_jacobian(rnn, h, x):
    """Compute Jacobian of RNN update at state h with input x."""
    h_tensor = torch.tensor(h, requires_grad=True)
    x_tensor = torch.tensor(x)

    h_next = rnn(x_tensor, h_tensor)

    # Compute Jacobian via automatic differentiation
    jacobian = torch.autograd.functional.jacobian(
        lambda h: rnn(x_tensor, h),
        h_tensor
    )

    return jacobian.numpy()
```

### Eigenvalue Analysis

Eigenvalues of J reveal:

**Magnitude |λ|**: Speed of convergence/divergence
- |λ| << 1: Fast decay (strong stability)
- |λ| ≈ 1: Slow dynamics (marginally stable)
- |λ| >> 1: Fast divergence (strong instability)

**Complex eigenvalues**: Indicate oscillations/spirals
- λ = r e^(iθ)
- r = magnitude (as above)
- θ = rotation rate

**Time constant**: τ = -1/ln|λ|
- How many timesteps until perturbation decays by factor e

---

## RNNs as Dynamical Systems

### The RNN Update Rule

```
h(t+1) = σ(W_h h(t) + W_x x(t) + b)
```

Where σ is activation function (tanh, relu, etc.).

**With fixed input x***: This defines an autonomous dynamical system.

### Why This Perspective Matters

**Traditional view**: RNN processes sequences step-by-step.

**Dynamical view**: RNN trajectory flows through phase space, settling into computational modes (attractors).

**Benefits**:
- Explains robustness (attractors are robust to noise)
- Reveals computational structure (fixed points = discrete modes)
- Connects to neuroscience (brains also have attractor dynamics)

### Example: Flip-Flop Memory

**Task**: Remember 3 bits, flip each bit on command.

**Traditional explanation**: RNN learns to store bits in hidden state.

**Dynamical explanation**:
- Network has 8 stable fixed points (one per 3-bit configuration)
- Flip command causes transition between attractors
- Saddle points act as "gates" routing trajectories

**Insight**: The discrete memory structure is explicit in phase space geometry.

---

## Practical Analysis Workflow

### Step 1: Find Fixed Points

For several input contexts (board states, sentence prefixes, etc.):

```python
fixed_points = []
for context in contexts:
    # Multiple random initializations
    for _ in range(100):
        h0 = random_initial_state()
        h_star = find_fixed_point(
            lambda h: rnn(context, h),
            h0
        )
        if h_star is not None:
            fixed_points.append((context, h_star))
```

### Step 2: Classify Stability

For each fixed point:

```python
J = compute_jacobian(rnn, h_star, context)
eigenvalues = np.linalg.eigvals(J)

if all(abs(eigenvalues) < 1):
    stability = 'attractor'
elif all(abs(eigenvalues) > 1):
    stability = 'repeller'
else:
    stability = 'saddle'
```

### Step 3: Visualize

Project high-dimensional phase space to 2D/3D:

```python
from sklearn.decomposition import PCA

# Collect many states (fixed points + trajectories)
all_states = np.vstack([fps, trajectory_states])

# Project to 2D
pca = PCA(n_components=2)
states_2d = pca.fit_transform(all_states)

# Plot with color coding
plt.scatter(states_2d[:, 0], states_2d[:, 1],
            c=stability_labels, cmap='viridis')
```

### Step 4: Interpret

Ask:
- How many attractors? (Computational modes)
- Where are they? (What do they represent?)
- How do trajectories flow? (Computational transitions)
- Are attractors task-aligned? (Do they correspond to meaningful states?)

---

## Connections to Neuroscience

### Attractor Networks in the Brain

**Working memory** (prefrontal cortex):
- Sustained activity during delay periods
- Point attractors maintain discrete choices

**Eye position** (brainstem):
- Neurons encode current gaze angle
- Line attractor integrates velocity commands

**Head direction** (hippocampus):
- Neurons encode heading
- Ring attractor (periodic in angle)

**Motor planning** (motor cortex):
- Preparatory activity before movement
- Trajectories through phase space

### Computational Motifs

Both brains and RNNs use:
- **Point attractors** for discrete memory/decisions
- **Line attractors** for analog integration
- **Saddle points** for transitions/routing
- **Slow manifolds** for robust sequences

**Convergent evolution**: Similar computational problems → similar dynamical solutions.

---

## Limitations and Extensions

### What This Approach Doesn't Cover

**Non-autonomous dynamics**: When input varies rapidly, fixed-point analysis may not apply.

**Transient dynamics**: Some computations happen during transients, not at attractors.

**High-dimensional manifolds**: More complex than point/line attractors.

**Learning dynamics**: How attractors emerge during training (separate topic).

### Advanced Topics

**Slow manifolds**: Low-dimensional surfaces where dynamics are slow.

**Heteroclinic connections**: Trajectories between saddle points forming computational sequences.

**Bifurcations**: Qualitative changes in dynamics as parameters vary.

**Chaos**: Sensitive dependence on initial conditions (usually avoided in trained RNNs).

---

## Worked Example: Simple RNN

**System**:
```
x(t+1) = tanh(wx(t) + b)
```

1D RNN (scalar state) with weight w and bias b.

**Find fixed points**:
```
x* = tanh(wx* + b)
```

**Case 1**: w=0.5, b=1
```
x* = tanh(0.5x* + 1)
```

Numerical solution: x* ≈ 0.87

**Stability**:
```
J = d/dx tanh(0.5x + 1)|_{x=x*}
  = 0.5(1 - tanh²(0.5x* + 1))
  = 0.5(1 - 0.87²)
  ≈ 0.12
```

Since |J| = 0.12 < 1, this is a **stable attractor**.

**Case 2**: w=2, b=0
```
x* = tanh(2x*)
```

Three solutions: x* = 0, x* ≈ ±0.96

**Stability at x*=0**:
```
J = 2(1 - tanh²(0)) = 2
```

Since |J| = 2 > 1, x*=0 is **unstable**.

**Stability at x*=±0.96**:
```
J = 2(1 - 0.96²) ≈ 0.15
```

Since |J| < 1, both x*=±0.96 are **stable attractors**.

**Interpretation**: System has two stable states (bistability). Depending on initial condition, converges to either +0.96 or -0.96. This implements a 1-bit memory!

---

## Further Reading

**Textbooks**:
- Strogatz, S. (1994). *Nonlinear Dynamics and Chaos*. (Accessible classic)
- Wiggins, S. (2003). *Introduction to Applied Nonlinear Dynamical Systems*. (More rigorous)

**For RNNs**:
- Sussillo & Barak (2013). Opening the black box. *Neural Computation*. (Foundational paper)
- Maheswaranathan et al. (2019). Reverse engineering sentiment RNNs. *NeurIPS*. (Line attractors)

**Neuroscience perspective**:
- Khona & Fiete (2022). Attractor and integrator networks in the brain. *Nature Reviews Neuroscience*.
- Vyas et al. (2020). Computation through neural population dynamics. *Annual Review of Neuroscience*.

**In this handbook**:
- [Fixed-Point Analysis](../2_methods/dynamical_analysis/fixed_points.md) - Practical methods
- [Attractor Landscapes](../2_methods/dynamical_analysis/attractor_landscapes.md) - Advanced theory
- [RNN Case Studies](../5_case_studies/recurrent_networks/) - Real examples

Full bibliography: [References](../references/bibliography.md)

---

**Return to**: [Foundations](README.md) | [Main Handbook](../0_start_here/README.md)

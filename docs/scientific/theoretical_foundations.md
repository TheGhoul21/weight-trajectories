# Theoretical Foundations: Dynamical Systems View of RNNs

The theoretical foundations underlying the analyses in this repository ground the approach to GRU interpretability in dynamical systems theory, information theory, and computational neuroscience.

---

## Overview

Our approach treats trained GRUs as **dynamical systems** that implement computations through their temporal evolution in hidden state space. This perspective, pioneered by Sussillo & Barak (2013) and refined by decades of neuroscience research, enables principled reverse-engineering of what the network has learned.

**Core Insight**: Understanding how a recurrent network solves a task requires analyzing:
1. **Where** the dynamics go (attractors, fixed points, trajectories)
2. **How** they get there (flow fields, basins of attraction)
3. **What** information they encode along the way (probes, mutual information)
4. **When** these structures emerge during learning (training dynamics)

---

## 1. RNNs as Dynamical Systems

### 1.1 The Fundamental Equation

A GRU with frozen CNN features defines a **discrete-time dynamical system**:

```
h_{t+1} = f(h_t, x_t; θ)
```

where:
- `h_t ∈ ℝ^n` is the hidden state (phase space)
- `x_t ∈ ℝ^m` is the input (driving signal from CNN)
- `f` is the GRU update function (flow map)
- `θ` are trained parameters

For a fixed input context `x*` (e.g., a particular board position), we have an **autonomous system**:

```
h_{t+1} = f(h_t, x*; θ) =: F_{x*}(h_t)
```

This is analogous to continuous-time systems `dh/dt = f(h, x*)` studied in physics and neuroscience.

### 1.2 GRU Equations Unpacked

The GRU implements a gated version of the fundamental equation:

```
z_t = σ(W_z x_t + U_z h_{t-1} + b_z)          # update gate
r_t = σ(W_r x_t + U_r h_{t-1} + b_r)          # reset gate
h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)   # candidate state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t          # interpolation
```

**Dynamical interpretation**:
- **Update gate** `z_t`: controls integration timescale (z→0: slow dynamics, z→1: fast updating)
- **Reset gate** `r_t`: modulates recurrent feedback (r→0: history forgotten, r→1: history preserved)
- **Candidate state** `h̃_t`: proposes new state based on input and gated history
- **Final state** `h_t`: interpolates between holding previous state and accepting new candidate

This structure enables **adaptive timescales**: the network can learn to integrate over long horizons for some dimensions while rapidly updating others.

### 1.3 Continuous-Time Approximation

Following Laurent & von Brecht (2017), GRUs can be viewed as **numerical integrators** for an underlying ODE:

```
dh/dt ≈ z_t ⊙ (h̃_t - h_t)
```

This perspective:
- Connects to continuous-time RNN theory
- Explains why gate saturation matters (z→0 halts dynamics)
- Suggests analysis techniques from control theory

---

## 2. Fixed Points and Attractors

### 2.1 Fixed Points: Equilibria of Computation

A **fixed point** `h*` satisfies:

```
h* = f(h*, x*; θ)
```

Fixed points are the "resting states" of the dynamics—configurations where the network would remain indefinitely if inputs stayed constant.

**Why they matter**:
- **Stable fixed points** (attractors) represent **memory states** or **decisions**
- **Unstable fixed points** (saddles) act as **decision boundaries** or **transitions**
- The **set of fixed points** reveals the computational "modes" the network has learned

### 2.2 Stability Analysis via Linearization

Near a fixed point `h*`, the dynamics are approximately linear:

```
h_{t+1} - h* ≈ J(h*) · (h_t - h*)
```

where `J(h*) = ∂f/∂h|_{h=h*}` is the **Jacobian matrix** at the fixed point.

**Stability classification** (via eigenvalues λ_i of J):
- **Stable attractor**: all |λ_i| < 1 (trajectories converge to h*)
- **Unstable repeller**: any |λ_i| > 1 (trajectories diverge from h*)
- **Saddle point**: mixed eigenvalues (stable in some directions, unstable in others)

**Timescales** near the fixed point:
```
τ_i = 1 / |ln|λ_i||
```

Large τ → slow relaxation along that eigenvector direction.

### 2.3 Attractor Landscapes

The full **attractor landscape** describes:
- Number and location of stable fixed points
- **Basins of attraction**: which initial conditions lead to which attractors
- **Separatrices**: boundaries between basins (often pass through saddle points)

For Connect Four, we hypothesize:
- Different attractors for "offensive threat", "defensive response", "neutral exploration"
- Basin structure reflects learned strategy: game states cluster by attractor
- Training dynamics: attractors emerge, move, and sharpen as performance improves

### 2.4 Beyond Fixed Points: Limit Cycles and Chaos

While we focus on fixed points, RNNs can also exhibit:
- **Limit cycles**: periodic oscillations (rarely useful for board games)
- **Line attractors**: continuous manifolds of stable states (e.g., for analog working memory)
- **Chaotic attractors**: sensitive dependence on initial conditions (avoided in well-trained networks)

---

## 3. Information Theory for Interpretability

### 3.1 Mutual Information as a Probe-Free Measure

**Mutual information** `I(H; Y)` quantifies statistical dependence between hidden state `H` and target variable `Y`:

```
I(H; Y) = H(Y) - H(Y|H)
```

where `H(·)` is Shannon entropy.

**Interpretation**:
- `I(H; Y) = 0`: hidden state tells nothing about Y
- `I(H; Y) = H(Y)`: hidden state fully determines Y (perfect encoding)
- Measured in **bits** (log base 2) or **nats** (natural log)

**Advantages over linear probes**:
- Model-free: detects nonlinear relationships
- Principled: grounded in information theory
- Comparable: same units across features

### 3.2 Per-Dimension Mutual Information

For a single hidden dimension `h_i`:

```
I(h_i; Y) = ∫∫ p(h_i, y) log[p(h_i, y) / (p(h_i)p(y))] dh_i dy
```

**Practical estimation** (Kraskov et al., 2004):
- k-NN based: for each sample, find k nearest neighbors in joint space
- Nonparametric: no distributional assumptions
- Consistent: converges to true MI as sample size → ∞

**Use case: neuron specialization**:
- High `I(h_i; immediate_win)` for a single dimension → **specialized detector**
- High `I(H; immediate_win)` but low per-dimension → **distributed code**

### 3.3 Information Plane Analysis

Track `I(H; X)` (input information) vs `I(H; Y)` (task information) over training:

- **Fitting phase**: both increase (network learns to represent inputs and outputs)
- **Compression phase** (debated): `I(H; X)` may decrease while `I(H; Y)` plateaus (network discards irrelevant details)

For Connect Four:
- `I(H; board_features)` should increase early
- `I(H; win_probability)` should increase throughout
- `I(H; irrelevant_noise)` should remain low or decrease

### 3.4 Connection to Thermodynamics

Information theory originated in statistical mechanics. Analogies:
- **Entropy** `H(Y)` ↔ thermodynamic entropy (disorder)
- **Mutual information** ↔ reduction in uncertainty (information gain)
- **Free energy** principles in learning (variational inference)

Recent work (Maheswaranathan & Williams, 2024) formalizes neurons as optimizing local information-theoretic objectives, bridging AI and neuroscience.

---

## 4. Manifold Learning and Trajectory Embedding

### 4.1 The Manifold Hypothesis

**Hypothesis**: High-dimensional data (weights, hidden states) lie on or near a low-dimensional **manifold** embedded in the ambient space.

**Why this matters**:
- `n=64` dimensional hidden state may effectively use only `d≈10` dimensions
- Visualizing the manifold reveals structure invisible in raw coordinates
- Training trajectories follow smooth paths on the manifold (regime changes = manifold transitions)

### 4.2 PHATE: Diffusion Geometry for Trajectories

**PHATE** (Moon et al., 2019) constructs embeddings via:

1. **Affinity graph**: k-NN graph with Gaussian kernel
2. **Diffusion operator**: `P = D^{-1}A` (transition matrix)
3. **Powered diffusion**: `P^t` for diffusion time `t`
4. **Potential distance**: `d(i,j) = ||log P^t(i,:) - log P^t(j,:)||`
5. **MDS embedding**: project into 2D preserving potential distances

**Key properties**:
- Preserves **local structure** (like t-SNE)
- Preserves **global structure** (like PCA)
- Preserves **trajectory continuity** (unique to PHATE)
- Robust to noise (diffusion acts as denoising)

**Why PHATE for training dynamics**:
- Checkpoints form a temporal sequence → trajectory preservation crucial
- Few samples (20-50 epochs) → needs robust method
- Want to see both fine-grained changes and macro-level phases

### 4.3 T-PHATE for Time Series

**T-PHATE** (Rübel et al., 2023) extends PHATE for **autocorrelated time series**:

- Constructs affinities using **delayed embeddings**: `[h_t, h_{t-τ}, h_{t-2τ}, ...]`
- Exploits temporal structure to denoise and reveal latent manifold
- Originally developed for fMRI brain-state trajectories

**Application to hidden states**:
- Game sequences are autocorrelated (moves depend on history)
- T-PHATE could better separate strategic phases
- Not yet implemented but high-priority extension

### 4.4 Alternative Embeddings

**t-SNE** (van der Maaten & Hinton, 2008):
- Optimizes local neighborhood preservation
- Good for clustering, poor for trajectories
- Can distort global distances

**UMAP** (McInnes et al., 2018):
- Faster than t-SNE, better global structure
- Topology-preserving (Riemannian geometry)
- Less trajectory-preserving than PHATE

**PCA**:
- Linear projection onto principal components
- Fast, interpretable axes
- Misses nonlinear structure

**Recommendation**: Use PHATE as default, validate with PCA/UMAP for sensitivity.

---

## 5. Neuroscience Foundations: Computation via Attractors

### 5.1 Attractor Networks in the Brain

Decades of neuroscience research (Khona & Fiete, 2022; Wang, 2001) established that **attractor dynamics** underlie:

- **Working memory**: prefrontal cortex maintains task-relevant information via sustained activity
- **Decision-making**: evidence accumulation toward decision attractors
- **Spatial navigation**: grid cells and place cells encode position via ring/manifold attractors
- **Motor control**: preparation states as fixed points in motor cortex

**Key insight**: Brains solve tasks not by static representations, but by **trajectories through state space** guided by attractor landscapes.

### 5.2 Canonical Computations

Different attractor types implement different computations:

| Attractor Type | Computation | Neural Example | RNN Example |
|----------------|-------------|----------------|-------------|
| **Point attractor** | Discrete memory, decision | Working memory (PFC) | Sentiment classifier |
| **Line attractor** | Analog memory, integration | Eye position (brainstem) | Path integration |
| **Ring attractor** | Periodic variable | Head direction (hippocampus) | Angle tracking |
| **Slow manifold** | Continuous transformation | Motor preparation | Sequence generation |

For Connect Four, we expect **point attractors** corresponding to strategic modes (attack, defend, explore).

### 5.3 Why Study AI Through a Neuroscience Lens?

**Convergent evolution**:
- Brains and RNNs face similar computational problems (temporal integration, credit assignment, generalization)
- Both use recurrent connectivity
- Success of neuroscience-inspired algorithms (attention, working memory) suggests shared principles

**Practical benefits**:
- Neuroscience provides **interpretive frameworks** (attractor landscapes, representational geometry)
- Decades of analysis techniques (tuning curves, population dynamics, dimensionality reduction)
- Connects AI interpretability to biological intelligence

---

## 6. Learning Dynamics: How Attractors Emerge

### 6.1 Training as Trajectory Through Parameter Space

Training optimizes parameters `θ(t)` via gradient descent:

```
θ(t+1) = θ(t) - η ∇L(θ(t))
```

Each checkpoint defines a different dynamical system `f(·; θ(t))` with its own attractor landscape.

**Question**: How does the attractor landscape evolve during training?

### 6.2 Attractor Emergence

Empirical observations (Huang et al., 2024; Saxe et al., 2019):

1. **Early training**: Few or no stable attractors; dynamics are diffuse
2. **Middle training**: Attractors begin to emerge near task-relevant regions
3. **Late training**: Attractors sharpen and stabilize; basins expand
4. **Correlation**: Attractor emergence timing correlates with performance jumps

**Hypothesis for Connect Four**:
- Epochs 0-10: Random initial dynamics, no clear attractors
- Epochs 10-30: Attractor for "immediate win detection" emerges
- Epochs 30-50: Separate attractors for offensive/defensive strategies
- Epochs 50+: Fine-tuning of attractor locations and basin boundaries

### 6.3 Measuring Attractor Sharpening

**Quantitative metrics**:
- **Number of fixed points** per context (more structure = more computation)
- **Spectral radius** of Jacobian at fixed points (smaller = stronger attractor)
- **Basin volume** (larger = more robust to perturbations)
- **Fixed point stability over epochs** (displacement from previous checkpoint)

### 6.4 Connecting Dynamics to Performance

**Hypothesis**: Performance improvements correspond to attractor landscape changes:

- Validation loss decrease ↔ attractors move toward optimal positions
- Win rate increase ↔ basin boundaries align with game-theoretic strategy
- Plateaus in training ↔ local minima in attractor landscape configuration

**Test**: Correlate fixed-point movement with training metrics (see [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md)).

---

## 7. Connecting the Frameworks

### 7.1 The Full Picture

Our multi-level analysis combines:

1. **Weight space** (parameter landscape):
   - Eigenspectra of recurrent matrices → timescale capacity
   - Gate statistics → gating strategy
   - PHATE embeddings → training trajectory shape

2. **Hidden state space** (computation):
   - Fixed points and attractors → computational modes
   - Manifold geometry → representational structure
   - Trajectories → temporal processing

3. **Information encoding** (representation):
   - Mutual information → what is encoded
   - Linear probes → how it's encoded (linear vs distributed)
   - Per-dimension analysis → neuron specialization

4. **Learning dynamics** (development):
   - Checkpoint analysis → when features emerge
   - Attractor evolution → how computation crystallizes
   - MI over time → information organization process

### 7.2 Methodological Pipeline

```
Trained GRU Checkpoints
        ↓
┌───────┴───────┐
│ Weight Space  │ → Eigenvalues, gates, PHATE
└───────┬───────┘
        ↓
┌───────┴───────┐
│  Game Replay  │ → Hidden state samples
└───────┬───────┘
        ↓
    ┌───┴───┐
    │       │
┌───┴──┐ ┌──┴────┐
│ Geom │ │ Info  │
└───┬──┘ └──┬────┘
    │       │
PHATE   Probes
Fixed    MI
points   ↓
    │    Neuron
    │    Special.
    └──┬───┘
       ↓
Mechanistic
Understanding
```

### 7.3 Open Questions and Future Directions

**Theoretical**:
- Can we predict attractor landscapes from architecture and task structure?
- What governs the number and type of attractors that emerge?
- How does discrete-time stepping affect continuous attractor stability?

**Empirical**:
- Do game phase transitions correspond to attractor basin crossings?
- Can we intervene on attractors to modify strategy (neural network surgery)?
- How do attractor landscapes differ between GRU, LSTM, and Transformer?

**Practical**:
- Can attractor analysis guide architecture design?
- Does attractor-based interpretability transfer to language models?
- Can we use fixed-point structure for verification and safety?

---

## 8. Key Takeaways

1. **Dynamical systems theory provides a principled framework** for understanding what recurrent networks compute and how they learn.

2. **Fixed points and attractors are the "building blocks" of RNN computation**, analogous to computational primitives in traditional algorithms.

3. **Information theory quantifies what is represented**, complementing dynamical analysis of how computation unfolds.

4. **Manifold learning reveals structure** in both weight space (training trajectories) and hidden space (representational geometry).

5. **Neuroscience offers interpretive frameworks** validated by decades of studying biological neural networks.

6. **Training dynamics = evolution of attractor landscapes**, connecting learning to computational function.

7. **Many of these analyses are implemented**, with high-priority gaps in fixed-point finding and attractor evolution tracking (see [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md)).

---

## Further Reading

For readers new to these concepts:

- **Dynamical systems primer**: Strogatz, S. H. (1994). *Nonlinear Dynamics and Chaos*. Westview Press.
- **RNN dynamics**: Sussillo, D., & Barak, O. (2013). Opening the black box. *Neural Computation*, 25(3), 626-649.
- **Information theory**: Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
- **Computational neuroscience**: Dayan, P., & Abbott, L. F. (2001). *Theoretical Neuroscience*. MIT Press.
- **Attractor networks**: Khona, M., & Fiete, I. R. (2022). Attractor and integrator networks in the brain. *Nature Reviews Neuroscience*, 23, 744–766.

See [References](references.md) for full citations of all papers mentioned.

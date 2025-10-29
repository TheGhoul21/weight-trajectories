# Case Studies: Learning from RNN Interpretability Research

Overview
- Shows how dynamical systems, fixed points, and information‑theoretic tools revealed mechanisms in trained RNNs.
- Use this as a learning companion: skim the “Key Findings” for intuition, then see “Implementation Notes” for code ideas.

How to Use
- New to these methods? Cross‑check terms with [Theoretical Foundations](theoretical_foundations) and implementations with [Methods](../reference/methods).
- When code appears (e.g., probes, MI), we rely on standard libraries like scikit‑learn; see links in Methods.

---

## Overview

Each case study follows a common structure:
1. **Task & Architecture**: What problem was solved and with what network
2. **Analysis Methods**: Which techniques were applied
3. **Key Findings**: What was discovered about the computation
4. **Relevance to Our Work**: How this informs Connect Four analysis
5. **Implementation Notes**: Practical considerations for replication

---

## Case Study 1: 3-Bit Flip-Flop (Sussillo & Barak, 2013)

### Task & Architecture

**Task**: Implement a 3-bit memory device that flips bits on command.
- **Inputs**: 3 flip commands (one per bit), given at arbitrary times
- **Outputs**: Current state of 3 bits (0 or 1)
- **Challenge**: Maintain memory between commands (temporal integration)

**Architecture**: Vanilla RNN with 24 hidden units, tanh activations.

### Analysis Methods

1. **Fixed-point finding**: Optimized to find `h*` where `h* = f(h*, x_context)`
2. **Stability analysis**: Computed Jacobians and eigenvalue spectra at each fixed point
3. **Visualization**: Projected 24D state space to 3D via PCA

### Key Findings

**Attractor structure**:
- Network developed **8 stable fixed points**, one for each possible 3-bit configuration (000, 001, 010, ..., 111)
- Each attractor corresponds to a discrete memory state
- Flip commands cause **rapid transitions** between attractors

**Computation via saddle points**:
- **Saddle points** (unstable fixed points) lie between attractors
- Trajectories pass through saddle regions during bit flips
- Saddles act as "gates" controlling which attractor is reached

**Linearized dynamics**:
- Near attractors: all eigenvalues |λ| < 0.5 (strong stability)
- Near saddles: 1-2 unstable eigenvalues |λ| > 1 (repelling directions)
- Stable manifolds of saddles form basin boundaries

**Visualization result**: 3D projection shows 8 attractors at corners of a "cube" in state space—the network **explicitly represents the Boolean structure** of the task.

### Relevance to Our Work

**Analogies to Connect Four**:
- Discrete strategic modes (attack/defend/explore) ↔ bit states (0/1)
- Board state changes ↔ flip commands
- GRU memory ↔ RNN sustained activity

**Differences**:
- Connect Four has continuous board features (not discrete bits)
- Many more possible game states (potential for more attractors or continuous manifolds)
- GRU gating may enable more flexible attractor structure

**Actionable insights**:
- Expect small number of attractors (< 20) for strategy categorization
- Look for saddle points at decision boundaries (win/loss threshold)
- Use PHATE to visualize attractor layout before computing fixed points

### Implementation Notes
Libraries: scikit‑learn (`LogisticRegression`), NumPy, SciPy; see [Methods](../reference/methods).

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
    for h_star in fps:
        J = compute_jacobian(rnn, x*, h_star)
        eigs = np.linalg.eigvals(J)

        if all(np.abs(eigs) < 1):
            stability.append('attractor')
        elif all(np.abs(eigs) > 1):
            stability.append('repeller')
        else:
            stability.append('saddle')

    return fps, jacobians, stability
```

**Key parameters**:
- `n_inits=128`: Sussillo used 500-1000 for high-dimensional RNNs
- `tol=1e-6`: balance between precision and avoiding numerical issues
- Optimization method: L-BFGS-B works well; trust-region also used

---

## Case Study 2: Sentiment Analysis with Line Attractors (Maheswaranathan et al., 2019)

### Task & Architecture

**Task**: Classify movie review sentiment (positive/negative) from character sequences.
- **Input**: Characters encoded as one-hot vectors
- **Output**: Binary sentiment classification
- **Challenge**: Integrate sentiment cues over long sequences (hundreds of characters)

**Architecture**: 1-layer LSTM with 128 hidden units, trained on IMDB dataset.

### Analysis Methods

1. **Fixed-point finding** across different sentence prefixes
2. **PCA projection** of hidden states during sentence processing
3. **Slow-point analysis**: regions of phase space with slow dynamics (not true fixed points)
4. **Jacobian spectrum** analysis along trajectories

### Key Findings

**Line attractor structure**:
- Hidden states during sentence processing lie on an approximate **line attractor**
- Position along the line encodes accumulated sentiment (negative ← → positive)
- Line extends continuously through state space (not discrete like flip-flop)

**Dynamics on the line**:
- Positive words push state along line toward positive end
- Negative words push toward negative end
- Neutral words cause small perturbations but return to line
- Final position determines classification

**Why a line, not points?**:
- Task requires **integration** of evidence (not discrete memory)
- Line topology naturally implements accumulation
- Similar to neural integrators in neuroscience (eye position, evidence accumulation)

**Slow manifold = computational substrate**:
- Dynamics along the line are **slow** (eigenvalues near unit circle)
- Dynamics perpendicular to line are **fast** (eigenvalues well inside unit circle)
- This separation of timescales implements robust integration

### Relevance to Our Work

**When to expect line attractors in Connect Four**:
- If network integrates "threat level" as a continuous variable
- Positional evaluation (board score) accumulation
- Move quality assessment over game trajectory

**When to expect point attractors instead**:
- Discrete strategy selection (attack/defend as separate modes)
- Win/loss decision boundaries
- Opening vs midgame vs endgame phase detection

**Diagnostic**:
- If PHATE embeddings show **elongated structures**, investigate for line attractors
- If they show **distinct clusters**, look for point attractors
- Hybrid structures possible (line connecting discrete clusters)

### Implementation Notes
Libraries: scikit‑learn (PCA), NumPy; see [Methods](../reference/methods).

**Slow-point finding** (easier than fixed-point finding):

```python
def find_slow_points(rnn, sentence_trajectories, speed_threshold=0.01):
    """
    Find regions where ||h_{t+1} - h_t|| is small.
    These approximate fixed points without requiring exact convergence.
    """
    slow_points = []

    for trajectory in sentence_trajectories:
        speeds = [np.linalg.norm(trajectory[t+1] - trajectory[t])
                  for t in range(len(trajectory)-1)]

        # Find timesteps with slow dynamics
        slow_indices = np.where(np.array(speeds) < speed_threshold)[0]
        slow_points.extend([trajectory[i] for i in slow_indices])

    return np.array(slow_points)
```

**Advantage**: Doesn't require optimization, just threshold on speed.

**Disadvantage**: Less precise than true fixed-point finding; may miss structure.

---

## Case Study 3: AlphaZero Chess Concept Learning (McGrath et al., 2022)

### Task & Architecture

**Task**: Master chess through self-play reinforcement learning (no human games).
- **Input**: Board position (8×8×119 tensor encoding piece positions, castling rights, etc.)
- **Output**: Policy (move probabilities) and value (win probability)
- **Challenge**: Learn strategic concepts without explicit instruction

**Architecture**: ResNet CNN (20 blocks) + policy/value heads. No recurrent component, but included here for concept probing methods.

### Analysis Methods

1. **Linear probing**: Train linear classifiers on intermediate layer activations to predict:
   - Low-level features (piece counts, pawn structure)
   - High-level concepts (king safety, space advantage, initiative)
2. **Feature importance**: Regression weights to identify which concepts are encoded
3. **Temporal analysis**: Track when concepts emerge during training (self-play iterations)
4. **Layer-wise analysis**: Where in the network are concepts represented

### Key Findings

**Human concepts emerge without supervision**:
- AlphaZero learns 11/12 tested chess concepts (e.g., "king safety", "material advantage")
- Concepts are **linearly decodable** from hidden representations
- Emerges purely from win/loss signal and self-play

**Temporal emergence**:
- **Early concepts** (piece mobility, material): learned in first 10,000 games
- **Middle concepts** (pawn structure, king safety): 10,000-50,000 games
- **Late concepts** (initiative, space): 50,000+ games
- **Correlation**: concept emergence timing matches corresponding Elo rating jumps

**Spatial localization**:
- Earlier layers: low-level features (piece detection)
- Middle layers: tactical patterns (threats, pins, forks)
- Later layers: strategic concepts (positional evaluation)

**Comparison to human grandmasters**:
- AlphaZero's internal concepts align with human strategic thinking
- Some dimensions exceed human interpretability (novel concepts)

### Relevance to Our Work

**Direct applications to Connect Four GRU**:

1. **Probe for game concepts**:
   - Immediate threats (win-in-1, block-in-1)
   - Positional features (center control, two-in-a-row patterns)
   - Strategic concepts (tempo, initiative)

2. **Track emergence timing**:
   - Which concepts appear first? (expect immediate threat detection early)
   - Does emergence correlate with validation performance?
   - Compare across GRU sizes (do larger GRUs learn concepts earlier?)

3. **Layer-wise analysis** (if multi-layer GRU):
   - Do earlier layers encode board geometry?
   - Do later layers encode strategic evaluation?

**Our current implementation** already does some of this:
- `extract_gru_dynamics.py` computes probes for 5 board features
- Missing: temporal tracking of probe accuracy across epochs
- Gap: richer feature set (only 5 features now, expand to 15-20)

### Implementation Notes
Libraries: scikit‑learn (`LogisticRegression`), NumPy, Pandas; see [Methods](../reference/methods).

**Probing protocol** (from McGrath et al.):

```python
def train_probe(hidden_states, target_labels, train_frac=0.8):
    """
    Train linear probe to predict target concept from hidden states.

    Args:
        hidden_states: (n_samples, hidden_dim)
        target_labels: (n_samples,) binary or continuous
        train_frac: fraction of data for training

    Returns:
        probe_accuracy: test set performance
        feature_importance: which dimensions matter most
    """
    # Split data
    n_train = int(len(hidden_states) * train_frac)
    X_train, X_test = hidden_states[:n_train], hidden_states[n_train:]
    y_train, y_test = target_labels[:n_train], target_labels[n_train:]

    # Train logistic regression with L2 regularization
    probe = LogisticRegression(C=1.0, max_iter=1000)
    probe.fit(X_train, y_train)

    # Evaluate
    accuracy = probe.score(X_test, y_test)
    importance = np.abs(probe.coef_[0])  # magnitude of weights

    return accuracy, importance
```

**Best practices**:
- **Cross-validation**: 5-fold CV to ensure stability
- **Regularization**: tune C parameter (we use C=1.0 as default)
- **Baseline**: random classifier should get ~50% for binary; probe should beat this
- **Feature normalization**: standardize hidden states before probing

---

## Case Study 4: Grid Cell Emergence in Spatial RNNs (Cueva & Wei, 2018)

### Task & Architecture

**Task**: Path integration (track position by integrating velocity over time).
- **Input**: 2D velocity vector at each timestep
- **Output**: 2D position estimate
- **Challenge**: Maintain accurate position over long trajectories without drift

**Architecture**: Vanilla RNN and LSTM variants, 128-512 units, trained via supervised learning.

### Analysis Methods

1. **Tuning curves**: Plot firing rate of each neuron as a function of position
2. **Spatial autocorrelation**: 2D autocorrelation of tuning curves reveals periodic structure
3. **Fourier analysis**: Identify hexagonal grid patterns in autocorrelation
4. **Dimensionality reduction**: PCA and UMAP to visualize population code

### Key Findings

**Grid cell emergence**:
- ~10-20% of RNN units develop **hexagonal grid firing patterns**, resembling entorhinal cortex grid cells
- Multiple scales of grids emerge (different spatial frequencies)
- Grid structure enables efficient position encoding with periodic boundary conditions

**Why grids emerge**:
- **Optimal for path integration**: grids form basis functions for 2D torus topology
- **Energy efficiency**: sparse distributed code reduces representational cost
- **Error correction**: multiple grid scales enable hierarchical error correction

**Not just spatial tasks**:
- Follow-up work shows similar periodic structures in non-spatial tasks
- Suggests **periodic attractor manifolds** are a general computational motif

### Relevance to Our Work

**Connect Four differences**:
- Board game state space is **discrete** (not continuous like position)
- No obvious periodic structure in game states
- However, strategic "space" may have latent continuity

**Possible analogs**:
- Piece count gradients (continuous variable)
- Threat level assessment (continuous from safe to critical)
- Game phase (continuous from opening to endgame)

**Analysis to try**:
- **Tuning curves** for hidden units as a function of:
  - Piece count differential
  - Center control score
  - Move number (game phase)
- Look for **smooth, structured** responses (sign of continuous encoding)

### Implementation Notes

**Computing tuning curves**:

```python
def compute_tuning_curves(hidden_states, feature_values, n_bins=20):
    """
    Compute average activation of each hidden unit as function of feature.

    Args:
        hidden_states: (n_samples, hidden_dim)
        feature_values: (n_samples,) continuous feature (e.g., piece_diff)
        n_bins: number of bins to discretize feature

    Returns:
        tuning_curves: (hidden_dim, n_bins) average activation per bin
    """
    # Bin the feature values
    bins = np.linspace(feature_values.min(), feature_values.max(), n_bins+1)
    bin_indices = np.digitize(feature_values, bins) - 1

    # Compute mean activation in each bin
    tuning_curves = np.zeros((hidden_states.shape[1], n_bins))
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            tuning_curves[:, i] = hidden_states[mask].mean(axis=0)

    return tuning_curves

def find_tuned_units(tuning_curves, threshold=0.3):
    """
    Identify units with strong tuning (high variance across bins).
    """
    tuning_strength = tuning_curves.std(axis=1) / tuning_curves.mean(axis=1)
    tuned_units = np.where(tuning_strength > threshold)[0]
    return tuned_units
```

---

## Case Study 5: Multitask RNNs and Dynamical Motifs (Yang et al., 2019; Huang et al., 2024)

### Task & Architecture

**Tasks**: Train single RNN on 20 cognitive tasks simultaneously:
- Memory (delayed match-to-sample, working memory)
- Decision (perceptual discrimination, context-dependent choice)
- Motor (reaching, anti-reaching)

**Architecture**: GRU with 256 units, shared across all tasks; task identity provided as input.

### Analysis Methods

1. **Compositional PCA**: Separate task-specific vs shared dynamics
2. **Fixed-point finding** for each task
3. **Clustering** of fixed points to identify reusable motifs
4. **Dynamical motif detection**: line attractors, decision boundaries, rotations

### Key Findings

**Shared dynamical motifs**:
- Networks reuse **dynamical building blocks** across tasks:
  - **Line attractors**: for working memory and integration
  - **Decision boundaries**: for binary choices
  - **Rotational dynamics**: for timing and motor preparation
- Only ~5-7 motifs account for most task variance

**Compositionality**:
- Complex tasks combine multiple motifs
- Task identity input **reconfigures** connections between motifs
- Shared dynamics + flexible routing = efficient multi-tasking

**Comparison to neuroscience**:
- Similar motifs found in monkey prefrontal cortex during comparable tasks
- Suggests convergence on canonical computational solutions

### Relevance to Our Work

**Single-task Connect Four** (our current focus):
- Expect fewer motifs than multi-task RNNs (maybe 2-4)
- Motifs might include:
  - **Point attractors** for strategy modes
  - **Decision boundaries** at win/loss thresholds
  - **Integration dynamics** for threat accumulation

**Future extension to multi-game**:
- Train single GRU on Connect Four + Tic-Tac-Toe + Gomoku
- Look for **shared board game motifs** (e.g., threat detection across games)
- Compare motifs to game-specific vs universal strategy

### Implementation Notes

**Motif detection pipeline**:

1. Find fixed points for multiple contexts (board states)
2. Cluster fixed points by Jacobian eigenstructure
3. Identify clusters as motif types:
   - All eigenvalues < 0.5 → strong point attractor
   - One eigenvalue near 1, others < 0.5 → line attractor
   - Some eigenvalues > 1 → decision boundary/saddle

**Visualization**: Project all fixed points into shared 2D embedding (PHATE or PCA), color by motif type.

---

## Case Study 6: Language Model Neurons (Karpathy et al., 2015)

### Task & Architecture

**Task**: Character-level language modeling (predict next character).
- **Dataset**: Wikipedia, Linux source code, Shakespeare
- **Architecture**: Multi-layer LSTM, 512 units per layer

### Analysis Methods

1. **Manual inspection**: Examine activation patterns of individual neurons on test sentences
2. **Targeted search**: Look for neurons that track specific features (quotes, brackets, URLs)
3. **Ablation**: Silence neurons and observe output changes
4. **Visualization**: Highlight text with color-coded neuron activations

### Key Findings

**Interpretable neurons**:
- **Quote detector**: neuron activates inside quotes, deactivates outside
- **Line length tracker**: neuron counts characters since last newline
- **Code depth**: neuron tracks indentation level in code
- **Semantic units**: some neurons track high-level concepts (positive/negative sentiment)

**Not all neurons interpretable**:
- Many neurons have distributed, non-interpretable roles
- Interpretability varies by layer (earlier = more syntactic, later = more semantic)

**Causality**:
- Ablating interpretable neurons breaks corresponding behaviors
- Artificially activating neurons can inject behaviors (e.g., force closing quote)

### Relevance to Our Work

**MI-based discovery** (our approach) vs **manual search** (Karpathy):
- **Advantage of MI**: Systematic, no cherry-picking, quantitative
- **Advantage of manual**: Finds surprising interpretable units
- **Best practice**: Use MI to shortlist candidates, then manually inspect top units

**Connect Four interpretable units we might find**:
- **Immediate threat detector**: fires when win-in-1 available
- **Center column tracker**: monitors control of center (most strategic column)
- **Game phase counter**: correlates with move number
- **Opponent weakness detector**: identifies exploitable patterns

**Visualization parallel**:
- Karpathy colored text by activation → we color board squares by activation
- Use Grad-CAM or attention to visualize which board regions drive decisions

### Implementation Notes

**Candidate neuron inspection**:

```python
def inspect_top_neurons(hidden_states, feature, mi_scores, top_k=5):
    """
    Manually inspect the top-k neurons by MI score.

    Args:
        hidden_states: (n_samples, hidden_dim)
        feature: (n_samples,) target feature
        mi_scores: (hidden_dim,) MI score for each dimension
        top_k: number of top neurons to inspect

    Returns:
        plots showing activation vs feature value
    """
    top_indices = np.argsort(mi_scores)[-top_k:]

    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 3))
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        ax.scatter(feature, hidden_states[:, idx], alpha=0.3, s=1)
        ax.set_xlabel('Feature value')
        ax.set_ylabel(f'Neuron {idx} activation')
        ax.set_title(f'MI = {mi_scores[idx]:.3f}')

    plt.tight_layout()
    return fig
```

---

## Summary: Lessons for Connect Four GRU Analysis

### Methodological Principles

1. **Start with PHATE embeddings** to get intuition for structure (attractors? lines? clusters?)
2. **Use fixed-point finding** to formalize attractor landscape
3. **Compute mutual information** for systematic neuron specialization analysis
4. **Train probes** to validate MI findings and test linearity of encoding
5. **Track across epochs** to understand emergence dynamics

### Expected Findings (Hypotheses)

Based on these case studies, we predict:

**Attractor structure**:
- 3-5 **point attractors** for strategic modes (offensive, defensive, neutral)
- Possible **line attractor** for continuous threat evaluation
- **Saddle points** at decision thresholds (win/loss boundaries)

**Neuron specialization**:
- 10-20% of neurons highly specialized (high MI, interpretable)
- Top neurons detect: immediate threats, center control, game phase
- Larger GRUs → more distributed encoding

**Training dynamics**:
- **Epoch 0-10**: No clear attractors, diffuse dynamics
- **Epoch 10-30**: Threat detection attractor emerges
- **Epoch 30+**: Strategy attractors differentiate and stabilize

**Comparison to AlphaZero chess**:
- Similar concept emergence timeline (tactical before strategic)
- But Connect Four is simpler → concepts emerge faster

### Implementation Priority

Based on ROI from literature:

**P0 (Do first)**:
1. Fixed-point finding (Sussillo & Barak method)
2. MI per-dimension tracking across epochs
3. Expanded probe suite (15 features instead of 5)

**P1 (Do soon)**:
4. Tuning curve analysis (Cueva & Wei method)
5. Top neuron manual inspection (Karpathy method)
6. Attractor evolution visualization

**P2 (Nice to have)**:
7. Slow-point analysis (Maheswaranathan method)
8. Dynamical motif clustering (Yang/Huang method)
9. Ablation studies (causal validation)

---

## Further Reading

Each case study has an associated paper:

- **Sussillo & Barak (2013)**: Fixed-point method [Neural Computation]
- **Maheswaranathan et al. (2019)**: Line attractors [NeurIPS]
- **McGrath et al. (2022)**: AlphaZero probing [PNAS]
- **Cueva & Wei (2018)**: Grid cells [ICLR]
- **Yang et al. (2019)**: Multi-task motifs [Nature Neuroscience]
- **Huang et al. (2024)**: Motif reuse [Nature Neuroscience]
- **Karpathy et al. (2015)**: Interpretable neurons [arXiv]

See [References](scientific/references) for full citations and additional resources.

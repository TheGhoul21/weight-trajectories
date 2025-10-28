# GRU Observability: Gap Analysis

This document compares your current implementation against the research recommendations in *"Observability and Learning Dynamics of RNNs in Game Environments (Focus on GRUs)"*.

## Summary

**Current Implementation Status**: Strong foundation (70% coverage)
**Data Available**: Checkpoints every 3 epochs + training history
**Missing**: Primarily advanced dynamical systems analysis

---

## ✅ Currently Implemented

### 1. Weight Space Analysis
- **Eigenvalue analysis of W_hn** (candidate matrix) ✓
  - Scripts: `extract_gru_dynamics.py:422-451`
  - Computes eigenvalues, absolute values

- **Integration timescale computation** ✓
  - Formula: τ_i = 1/|ln|λ_i|| (line 432)
  - Median and max timescales tracked

- **Gate statistics over training** ✓
  - Update gate (z_t) and reset gate (r_t) mean/std
  - Per-unit and aggregate statistics
  - Visualization: `gate_mean_trajectories.png`

### 2. Representation Space Analysis
- **Hidden state sampling** ✓
  - Reservoir sampling (1500 samples per epoch)
  - Stored with board features for probing

- **Low-dimensional embeddings** ✓
  - PHATE embeddings at multiple epochs
  - Colored by interpretable features
  - Visual clustering analysis

### 3. Learning Dynamics
- **Gate evolution tracking** ✓
  - Mean/std trajectories across training
  - Shows when models "lock in" memories

- **Timescale evolution** ✓
  - Per-epoch eigenvalue summaries
  - Heatmap of final capacity

### 4. Interpretability Techniques
- **Probing classifiers** ✓
  - Logistic regression on hidden states
  - Binary features: current_player, immediate_win, three_in_row
  - Accuracy and F1 metrics tracked

- **Board feature extraction** ✓ (lines 245-277)
  - Current player, piece counts
  - Immediate win detection
  - Center control, three-in-row patterns
  - Move index (game phase proxy)

---

## 🔴 Critical Gaps (High Impact)

### 1. Fixed-Point Analysis ⭐⭐⭐
**What's Missing**: The cornerstone of modern RNN interpretability (Sussillo & Barak 2013)

**Recommended Addition**:
```python
def find_fixed_points(model, input_context, n_init=128):
    """
    Optimize to find fixed points h* where h* = GRU(x, h*)
    Returns: fixed points, jacobians, stability classification
    """
    # Sample random initial conditions
    # Run optimization to satisfy h_next ≈ h
    # Compute Jacobian at each fixed point
    # Classify: stable attractor, unstable saddle, etc.
```

**Why It Matters**:
- Reveals **what computations** the GRU performs (not just statistics)
- Identifies discrete behavioral modes (offensive/defensive in Connect Four)
- **Emergence during training**: Track when stable attractors form

**Data You Have**: ✅ Checkpoints every 3 epochs
**Implementation Effort**: High (2-3 days)
**Research Impact**: Very High

**References in PDF**:
- Page 2: "Fixed points (stable or unstable equilibrium states)"
- Page 3: "As learning progressed, new attractors (or stable states) can appear and sharpen"
- Page 3: "Stable fixed point attractor corresponding to balancing...emerged in hidden state space"

---

### 2. Attractor Landscape Evolution ⭐⭐⭐
**What's Missing**: Tracking how hidden state dynamics change during training

**Recommended Addition**:
```python
def analyze_attractor_evolution(checkpoints, test_games):
    """
    For each checkpoint epoch:
    1. Find fixed points
    2. Measure basin sizes
    3. Track fixed point movement
    4. Correlate with performance improvements
    """
    # Plot: fixed point locations in 2D PHATE space over epochs
    # Metric: distance moved by dominant attractor vs. win rate
```

**Why It Matters**:
- Answers "when does the GRU learn strategy X?"
- Validates that performance gains = clearer internal representations
- Connects hidden state geometry to gameplay milestones

**Data You Have**: ✅ Checkpoints every 3 epochs + loss history
**Implementation Effort**: Medium (builds on fixed-point finder)
**Research Impact**: Very High

**References in PDF**:
- Page 3: "Stable attractor moved closer to the ideal goal state over time, and this movement was tightly correlated with the increasing reward performance"
- Page 3: "Ring-shaped attractor to efficiently encode the pendulum's angular position"

---

### 3. Mutual Information Analysis ⭐⭐
**What's Missing**: Information-theoretic measure of learning progress

**Recommended Addition**:
```python
def compute_mutual_information_over_training(checkpoints, dataset):
    """
    For each epoch, compute MI(hidden_state, board_features)
    Track MI evolution for:
    - Task-relevant variables (should increase)
    - Irrelevant variables (should decrease)
    """
    # Use sklearn.feature_selection.mutual_info_regression
    # Plot MI curves for: win probability, threats, piece advantage
```

**Why It Matters**:
- Quantifies "how much game state is encoded" in hidden state
- Should correlate with probe accuracy
- Shows network learning to ignore distractions

**Data You Have**: ✅ Hidden samples with board features
**Implementation Effort**: Low (1 day)
**Research Impact**: Medium-High

**References in PDF**:
- Page 3: "Mutual information between the RNN's hidden state and the underlying belief state...steadily increased"
- Page 3: "Hidden state's mutual information with irrelevant variables decreased"

---

## 🟡 Moderate Gaps (Medium Impact)

### 4. Strategy Representation & Ranking ⭐⭐
**What's Missing**: Latent strategy embeddings (Lei et al. 2024 - STRIL)

**Your Use Case**: Analyzing AlphaZero self-play games
- Visualize strategy space: strong vs weak play styles
- Define indicators: randomness, exploited level
- Filter suboptimal strategies to improve learning

**Data You Have**: ✅ Game trajectories with outcomes
**Implementation Effort**: Medium
**Research Impact**: High (directly applicable to your Connect Four domain)

**References in PDF**:
- Page 6: "Learned latent space...cleanly separated dominant (strong) strategies on one side and dominated (weaker) strategies on the other"
- Page 6: "Randomness Indicator and Exploited Level derived from embeddings"

---

### 5. Expanded Probing Suite ⭐
**Current**: Binary classification only (5 features)

**Recommended Additions**:
```python
PROBE_FEATURES = {
    # Existing binary probes ✓
    "current_player": "binary",
    "immediate_win_current": "binary",

    # Add continuous regression probes
    "move_quality": "regression",  # If you have MCTS values
    "win_probability": "regression",
    "board_complexity": "regression",

    # Add multi-class probes
    "game_phase": "multiclass",  # opening/midgame/endgame
    "threat_level": "multiclass",  # none/low/medium/high
}
```

**Data You Have**: Partial - need to extract from AlphaZero value head
**Implementation Effort**: Low
**Research Impact**: Medium

---

### 6. Full Recurrent Matrix Eigenanalysis ⭐
**Current**: Only analyzing W_hn (candidate matrix)

**Recommended**: Analyze full linearized dynamics
```python
def compute_full_jacobian(gru, hidden_state, input_context):
    """
    Compute Jacobian of full GRU update:
    J = ∂h_{t+1}/∂h_t

    Includes effects of both gates at current operating point
    """
    # More accurate than just W_hn eigenvalues
    # Captures gate modulation
```

**Why It Matters**: W_hn alone misses gate modulation effects

**Implementation Effort**: Medium
**Research Impact**: Medium

---

## 🟢 Minor Enhancements

### 7. Eigenvalue Trajectory Visualization
**Current**: Final epoch heatmap only

**Add**: Plot eigenvalue movement in complex plane over training
```python
# Animate eigenvalues moving in complex plane
# Color by training epoch
# Show approach to unit circle (stability boundary)
```

### 8. Hidden State Trajectory Matching
**Current**: Static PHATE embeddings

**Add**: LSTMVis-style trajectory search
```python
def find_similar_trajectories(query_trajectory, all_trajectories):
    """Find game sequences with similar hidden state dynamics"""
    # Use DTW or cosine similarity in hidden space
    # Answer: "What other games looked like this to the GRU?"
```

### 9. Saliency Analysis
**Add**: Gradient-based attribution
```python
def compute_move_saliency(model, game_trajectory, decision_point):
    """Which past moves influenced this decision?"""
    # Backprop from current decision to past hidden states
    # Visualize attention-like weights over game history
```

### 10. Explicit Clustering + FSM Extraction
**Current**: Visual PHATE clustering

**Add**: Algorithmic clustering to discrete modes
```python
from sklearn.cluster import KMeans
def extract_behavioral_modes(hidden_states, n_modes=5):
    """
    Cluster hidden states → discrete strategy modes
    Output: Finite state machine representation
    """
```

---

## 📊 Implementation Priority Matrix

| Feature | Impact | Effort | Data Ready? | Priority |
|---------|--------|--------|-------------|----------|
| **Fixed-Point Finding** | Very High | High | ✅ | 🔥 **P0** |
| **Attractor Evolution** | Very High | Medium | ✅ | 🔥 **P0** |
| **Mutual Information** | High | Low | ✅ | ⭐ **P1** |
| **Strategy Embeddings** | High | Medium | ✅ | ⭐ **P1** |
| **Expanded Probing** | Medium | Low | Partial | **P2** |
| **Full Jacobian** | Medium | Medium | ✅ | **P2** |
| **Eigenvalue Trajectories** | Low | Low | ✅ | **P3** |
| **Saliency Maps** | Low | Medium | ✅ | **P3** |
| **Trajectory Matching** | Low | Medium | ✅ | **P3** |
| **FSM Extraction** | Low | Low | ✅ | **P3** |

---

## 🎯 Recommended Next Steps

### Phase 1: Core Dynamics (2-3 weeks)
1. **Implement fixed-point finder**
   - Start with single checkpoint validation
   - Visualize fixed points on PHATE embeddings

2. **Track attractor evolution**
   - Run fixed-point finder on all epochs
   - Plot movement vs. training loss/win rate

3. **Add mutual information tracking**
   - Quick win with existing hidden samples
   - Validates probe results

### Phase 2: Game-Specific Insights (1-2 weeks)
4. **Strategy representation analysis**
   - Embed game trajectories (not just states)
   - Separate AlphaZero skill levels

5. **Expanded probing**
   - Add regression probes if value head data available
   - Game phase classification

### Phase 3: Publication Polish (1 week)
6. **Eigenvalue evolution plots**
7. **Saliency visualizations** (if needed for paper)
8. **FSM extraction** (if you want verifiable policies)

---

## 📚 Code Stubs for Priority Items

### P0: Fixed-Point Finder

```python
#!/usr/bin/env python3
"""scripts/find_gru_fixed_points.py"""

import torch
from torch import nn
from scipy.optimize import minimize

def find_fixed_points(
    model: nn.Module,
    input_features: torch.Tensor,  # CNN features for a board state
    n_inits: int = 128,
    tolerance: float = 1e-6,
) -> dict:
    """
    Find fixed points h* where h* ≈ GRU(input_features, h*)

    Returns:
        fixed_points: List of fixed point vectors
        jacobians: Linearized dynamics at each fixed point
        stability: Classification (attractor/saddle/repeller)
    """
    gru = model.gru
    hidden_size = gru.hidden_size

    # Sample random initial conditions
    np.random.seed(0)
    h_inits = np.random.randn(n_inits, hidden_size) * 0.1

    fixed_points = []

    for h0 in h_inits:
        def objective(h):
            h_tensor = torch.tensor(h, dtype=torch.float32).unsqueeze(0)
            x_tensor = input_features.unsqueeze(0)
            with torch.no_grad():
                h_next, _ = gru(x_tensor, h_tensor.unsqueeze(0))
            residual = h_next.squeeze().numpy() - h
            return np.sum(residual ** 2)

        result = minimize(objective, h0, method='L-BFGS-B', tol=tolerance)

        if result.success and result.fun < tolerance:
            fixed_points.append(result.x)

    # Compute Jacobians and classify stability
    # ... (eigenvalue analysis of Jacobian at each fixed point)

    return {
        'fixed_points': np.array(fixed_points),
        # 'jacobians': ...,
        # 'stability': ...,
    }
```

### P0: Attractor Evolution Tracker

```python
#!/usr/bin/env python3
"""scripts/track_attractor_evolution.py"""

def track_fixed_point_evolution(checkpoint_dir, epochs, reference_boards):
    """
    For each epoch, find fixed points for reference board states.
    Track how fixed point locations move in hidden space.
    """
    results = []

    for epoch in epochs:
        model = load_checkpoint(checkpoint_dir, epoch)

        for board_idx, board in enumerate(reference_boards):
            features = model.resnet(board)
            fp_data = find_fixed_points(model, features)

            results.append({
                'epoch': epoch,
                'board': board_idx,
                'n_fixed_points': len(fp_data['fixed_points']),
                'fixed_points': fp_data['fixed_points'],
                # Could also store attractors in PHATE space
            })

    # Analyze movement
    # Plot: fixed point coordinates vs. epoch
    # Correlate with training metrics
```

### P1: Mutual Information Tracker

```python
#!/usr/bin/env python3
"""Add to analyze_gru_observability_results.py"""

from sklearn.feature_selection import mutual_info_regression

def compute_mi_evolution(analysis_dir, output_dir):
    """
    For each epoch, compute MI(hidden, feature) for all board features.
    """
    mi_results = []

    for model_dir in analysis_dir.iterdir():
        for epoch_file in (model_dir / "hidden_samples").glob("epoch_*.npz"):
            data = np.load(epoch_file)
            hidden = data['hidden']
            features = data['features']
            feature_names = data['feature_names']

            for i, fname in enumerate(feature_names):
                mi = mutual_info_regression(
                    hidden,
                    features[:, i],
                    random_state=0
                )[0]  # Returns array of MI per feature

                mi_results.append({
                    'model': model_dir.name,
                    'epoch': extract_epoch(epoch_file),
                    'feature': fname,
                    'mutual_information': mi,
                })

    df = pd.DataFrame(mi_results)
    # Plot MI evolution over training
    # Compare task-relevant vs irrelevant features
```

---

## 🔬 Expected Insights from Missing Analyses

### From Fixed-Point Analysis:
- **Q**: Does the GRU develop discrete "strategy modes"?
- **A**: Number and location of stable attractors
- **Example**: "2 stable attractors at epoch 100: one for aggressive play (center-focused), one for defensive"

### From Attractor Evolution:
- **Q**: When does the network "learn" Connect Four strategy?
- **A**: Epoch where stable attractor emerges correlating with win position
- **Example**: "Stable attractor for 'threat recognition' emerges at epoch 24, coinciding with 15% jump in win rate"

### From Mutual Information:
- **Q**: Is the GRU encoding game-relevant information?
- **A**: MI(hidden, immediate_win) should be high; MI(hidden, irrelevant_noise) should be low
- **Example**: "MI with immediate_win increases from 0.1 → 0.8 bits during training"

### From Strategy Embeddings:
- **Q**: Can we visualize strong vs weak gameplay in latent space?
- **A**: 2D projection separating AlphaZero iteration 10 vs iteration 100
- **Example**: "Early iterations cluster in 'random play' region; later iterations form tight cluster in 'strategic play' region"

---

## 📖 Key References by Priority

### P0 - Fixed Points & Attractors
- **Sussillo & Barak (2013)**: "Opening the black box" - fixed point finding method
- **Huang et al. (2024)**: Fixed point emergence during RL training
- **Maheswaranathan et al. (2019)**: Line attractors in sentiment RNNs

### P1 - Learning Dynamics
- **Lambrechts et al. (2022)**: Mutual information evolution in POMDPs
- **Lei et al. (2024)**: STRIL - strategy representation for Connect Four

### P2 - Advanced Interpretability
- **Carr et al. (2021)**: FSM extraction from RNN policies
- **Strobelt et al. (2018)**: LSTMVis trajectory matching

---

## ✅ Validation Checklist

Before considering the analysis "complete," you should be able to answer:

- [ ] **Fixed Points**: How many stable attractors does the GRU have at final epoch?
- [ ] **Evolution**: At what epoch does the primary attractor stabilize?
- [ ] **Information**: Does MI(hidden, immediate_win) increase monotonically?
- [ ] **Correlation**: Do attractor movements correlate with loss decrease?
- [ ] **Strategy**: Can we separate strong vs weak strategies in latent space?
- [ ] **Probing**: Can a linear probe decode win probability from hidden state?

---

## 💡 Bottom Line

**You have**: Excellent infrastructure for weight-space and basic representation analysis.

**You're missing**: The dynamical systems core that makes modern RNN interpretability powerful.

**Biggest ROI**: Implement fixed-point finding + attractor evolution tracking. These are the analyses that will:
1. Provide novel scientific insights
2. Distinguish your work from "just running probes"
3. Connect internal GRU dynamics to Connect Four gameplay

**Time Investment**: ~3-4 weeks for P0+P1 priorities
**Research Impact**: Transforms analysis from "descriptive" to "mechanistic understanding"

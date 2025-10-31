# Case Study: Sentiment Analysis with Line Attractors (Maheswaranathan et al., 2019)

Demonstrates how line attractors emerge for continuous integration tasks in RNNs.

---

## Task & Architecture

**Task**: Classify movie review sentiment (positive/negative) from character sequences.
- **Input**: Characters encoded as one-hot vectors
- **Output**: Binary sentiment classification
- **Challenge**: Integrate sentiment cues over long sequences (hundreds of characters)

**Architecture**: 1-layer LSTM with 128 hidden units, trained on IMDB dataset.

---

## Analysis Methods

1. **Fixed-point finding** across different sentence prefixes
2. **PCA projection** of hidden states during sentence processing
3. **Slow-point analysis**: regions of phase space with slow dynamics (not true fixed points)
4. **Jacobian spectrum** analysis along trajectories

---

## Key Findings

### Line Attractor Structure
- Hidden states during sentence processing lie on an approximate **line attractor**
- Position along the line encodes accumulated sentiment (negative ← → positive)
- Line extends continuously through state space (not discrete like flip-flop)

### Dynamics on the Line
- Positive words push state along line toward positive end
- Negative words push toward negative end
- Neutral words cause small perturbations but return to line
- Final position determines classification

### Why a Line, Not Points?
- Task requires **integration** of evidence (not discrete memory)
- Line topology naturally implements accumulation
- Similar to neural integrators in neuroscience (eye position, evidence accumulation)

### Slow Manifold = Computational Substrate
- Dynamics along the line are **slow** (eigenvalues near unit circle)
- Dynamics perpendicular to line are **fast** (eigenvalues well inside unit circle)
- This separation of timescales implements robust integration

---

## Relevance to Other Work

### When to Expect Line Attractors
For tasks requiring continuous integration:
- Threat level accumulation
- Positional evaluation (board score)
- Move quality assessment over game trajectory
- Evidence accumulation in decision-making

### When to Expect Point Attractors Instead
For tasks with discrete modes:
- Discrete strategy selection (attack/defend as separate modes)
- Win/loss decision boundaries
- Phase detection (opening vs midgame vs endgame)

### Diagnostic Heuristic
- If dimensionality reduction (PHATE, UMAP) shows **elongated structures** → investigate for line attractors
- If it shows **distinct clusters** → look for point attractors
- Hybrid structures possible (line connecting discrete clusters)

---

## Implementation Notes

### Slow-Point Finding (Easier than Fixed-Point Finding)

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

### Advantages and Disadvantages

**Advantage of slow-point finding**:
- Doesn't require optimization, just threshold on speed
- Fast and easy to implement
- Works for approximate attractors (not just exact fixed points)

**Disadvantage**:
- Less precise than true fixed-point finding
- May miss structure if threshold is poorly chosen
- Conflates true fixed points with transiently slow regions

---

## Related Methods

**In this handbook**:
- [Fixed-Point Analysis](../../2_methods/dynamical_analysis/fixed_points.md)
- [Trajectory Analysis](../../2_methods/dynamical_analysis/trajectory_analysis.md)
- [Attractor Landscapes](../../2_methods/dynamical_analysis/attractor_landscapes.md)

**Alternative techniques**:
- Principal curves (fitting 1D manifolds)
- Trajectory clustering
- Dimensionality reduction followed by curve fitting

---

## Neuroscience Connections

### Neural Integrators
Line attractors are well-studied in neuroscience:

**Eye position integration** (brainstem):
- Neurons maintain eye position through persistent activity
- Forms a line attractor in neural state space

**Evidence accumulation** (parietal cortex):
- Neurons accumulate sensory evidence toward a decision
- Ramp-like activity reflects integration along a line

**Head direction** (hippocampus):
- Ring attractor (topologically related to line attractor with wrapped ends)

### Computational Principle
Line attractors enable **analog working memory**:
- Store continuous variables (not just discrete states)
- Robust to noise (attractor property)
- Biologically plausible (observed in real brains)

---

## Practical Applications

### For Sentiment Analysis
- Understanding line structure can guide:
  - Feature extraction (position along line)
  - Debugging (are neutral words handled correctly?)
  - Architecture design (encourage line formation)

### For Other Sequence Tasks
Tasks that might benefit from line attractors:
- Cumulative reward estimation (RL)
- Position tracking
- Temperature/resource monitoring
- Any task requiring robust integration of incremental evidence

---

## References

**Primary paper**:
Maheswaranathan, N., Williams, A. H., Golub, M. D., Ganguli, S., & Sussillo, D. (2019). Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics. *NeurIPS*.

**Related work**:
- Seung (1996): Theory of neural integrators
- Mante et al. (2013): Context-dependent integration in PFC
- Cueva & Wei (2018): Emergent grid cells (related manifold structure)

Full bibliography: [References](../../references/bibliography.md)

---

**Return to**: [Case Studies](../README.md) | [Natural Language](README.md)

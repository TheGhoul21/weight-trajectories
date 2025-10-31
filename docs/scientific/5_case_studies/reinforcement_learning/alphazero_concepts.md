# Case Study: AlphaZero Chess Concept Learning (McGrath et al., 2022)

Demonstrates how human-interpretable concepts emerge from self-play reinforcement learning without explicit supervision.

---

## Task & Architecture

**Task**: Master chess through self-play reinforcement learning (no human games).
- **Input**: Board position (8×8×119 tensor encoding piece positions, castling rights, etc.)
- **Output**: Policy (move probabilities) and value (win probability)
- **Challenge**: Learn strategic concepts without explicit instruction

**Architecture**: ResNet CNN (20 blocks) + policy/value heads. No recurrent component, but included here for concept probing methods.

---

## Analysis Methods

1. **Linear probing**: Train linear classifiers on intermediate layer activations to predict:
   - Low-level features (piece counts, pawn structure)
   - High-level concepts (king safety, space advantage, initiative)
2. **Feature importance**: Regression weights to identify which concepts are encoded
3. **Temporal analysis**: Track when concepts emerge during training (self-play iterations)
4. **Layer-wise analysis**: Where in the network are concepts represented

---

## Key Findings

### Human Concepts Emerge Without Supervision
- AlphaZero learns 11/12 tested chess concepts (e.g., "king safety", "material advantage")
- Concepts are **linearly decodable** from hidden representations
- Emerges purely from win/loss signal and self-play

### Temporal Emergence
Concepts emerge in stages:

**Early concepts** (first 10,000 games):
- Piece mobility
- Material advantage

**Middle concepts** (10,000-50,000 games):
- Pawn structure
- King safety

**Late concepts** (50,000+ games):
- Initiative
- Space advantage

**Critical observation**: Concept emergence timing matches corresponding Elo rating jumps.

### Spatial Localization
Layer-wise analysis reveals hierarchy:

**Earlier layers**: Low-level features
- Piece detection
- Local patterns (pins, forks)

**Middle layers**: Tactical patterns
- Threats
- Piece coordination

**Later layers**: Strategic concepts
- Positional evaluation
- Long-term planning

### Comparison to Human Grandmasters
- AlphaZero's internal concepts align with human strategic thinking
- Some dimensions exceed human interpretability (novel concepts)
- Suggests convergent evolution on canonical chess knowledge

---

## Relevance to Other Work

### Direct Applications to Game-Playing Networks

For any game-playing agent (Connect Four, Go, etc.):

**1. Probe for game concepts**:
- Immediate threats (win-in-1, block-in-1)
- Positional features (center control, pattern formations)
- Strategic concepts (tempo, initiative, pressure)

**2. Track emergence timing**:
- Which concepts appear first? (expect immediate threat detection early)
- Does emergence correlate with validation performance?
- Compare across model sizes (do larger models learn concepts earlier?)

**3. Layer-wise analysis**:
- Do earlier layers encode board geometry?
- Do later layers encode strategic evaluation?
- Where is the transition from tactics to strategy?

### Implications for Interpretability

**Encouraging findings**:
- Complex concepts emerge without supervision
- Linear probes work (concepts are linearly accessible)
- Temporal tracking reveals learning dynamics

**Open questions**:
- Why these particular concepts?
- Are they necessary or incidental?
- Can we steer concept learning?

---

## Implementation Notes

### Probing Protocol (from McGrath et al.)

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

### Best Practices

**Cross-validation**: Use 5-fold CV to ensure stability

**Regularization**: Tune C parameter (default C=1.0 often works)

**Baseline**: Random classifier should get ~50% for binary; probe should beat this significantly

**Feature normalization**: Standardize hidden states before probing

**Controls**: Always include shuffled-label baseline

### Concept Label Generation

For board games, concept labels must be generated:

```python
# Example: "king safety" in chess
def compute_king_safety(board):
    """
    Compute king safety metric from board state.

    Returns value in [0, 1] where 1 = very safe, 0 = under attack
    """
    king_pos = find_king(board)

    # Count defenders vs attackers near king
    defenders = count_friendly_pieces_near(king_pos, board, radius=2)
    attackers = count_enemy_pieces_attacking(king_pos, board)

    # Pawn shield
    pawn_shield = count_pawn_shield(king_pos, board)

    # Escape squares
    escape_squares = count_safe_escape_squares(king_pos, board)

    # Combine into safety metric
    safety = weighted_sum(defenders, pawn_shield, escape_squares) - attackers
    return normalize(safety)
```

For game-specific concepts, domain expertise helps design label functions.

---

## Extensions and Open Questions

### Extending to Other Domains

**Board games** (Connect Four, Go, Tic-Tac-Toe):
- Similar probing protocol
- Simpler concept space than chess
- Faster concept emergence expected

**Video games** (Atari, Starcraft):
- Higher-dimensional input
- More complex concepts
- Temporal concepts (timing, rhythm)

**Robotics**:
- Physical concepts (balance, stability)
- Goal-oriented concepts (progress, efficiency)

### Open Research Questions

**Causality**: Are learned concepts causal or epiphenomenal?
- Test with activation patching
- Ablate concept-encoding dimensions

**Transferability**: Can concepts transfer across games?
- Train multi-game agents
- Test concept probe transfer

**Novel concepts**: What concepts does AI learn that humans don't have names for?
- Unsupervised concept discovery
- Cluster analysis of representations

---

## Related Methods

**In this handbook**:
- [Linear Probes](../../2_methods/probing/linear_probes.md) - Detailed methodology
- [Mutual Information](../../2_methods/probing/mutual_information.md) - Complementary approach
- [Representation Analysis](../../2_methods/representation_analysis/) - Understanding learned structure

**Related case studies**:
- [Sentiment Line Attractors](../natural_language/sentiment_line_attractors.md) - RNN probing
- [Interpretable Neurons](../recurrent_networks/interpretable_neurons.md) - Karpathy's manual inspection
- Connect Four GRU (this project) - Similar game-playing context

---

## References

**Primary paper**:
McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Wattenberg, M., Hassabis, D., Kim, B., Paquet, U., & Kramnik, V. (2022). Acquisition of chess knowledge in AlphaZero. *Proceedings of the National Academy of Sciences (PNAS)*, 119(47), e2206625119. doi:10.1073/pnas.2206625119

**AlphaZero architecture**:
Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.

**Follow-up work**:
Tian, C., Xu, K., & Levine, S. (2024). Bridging the human–AI knowledge gap through concept discovery and transfer in AlphaZero. *PNAS*, 122(3), e2406675122. (Extracting novel concepts and transferring to grandmasters)

Full bibliography: [References](../../references/bibliography.md)

---

**Return to**: [Case Studies](../README.md) | [Reinforcement Learning](README.md)

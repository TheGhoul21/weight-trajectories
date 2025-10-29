# Scientific Background

## Overview

Understanding recurrent neural networks requires moving beyond training metrics to analyze the computational mechanisms learned during training. The GRU in Connect Four learns to integrate temporal information, maintain memory of board states, and develop internal representations of game strategy. Our analysis framework draws on three complementary perspectives:

**Dynamical Systems Theory** treats the GRU as a system evolving through state space, where computation emerges from attractor landscapes and fixed points. This perspective, pioneered by Sussillo & Barak (2013), reveals how networks implement algorithms through their temporal dynamics.

**Information Theory** quantifies what information the network encodes and where it is stored. Mutual information analysis identifies which hidden units represent specific game features, while probing classifiers test the accessibility of this information.

**Computational Neuroscience** provides interpretive frameworks validated by decades of studying biological neural networks. Concepts like attractor networks, working memory, and representational geometry apply equally to artificial and biological systems.

## Analysis Framework

Our approach examines GRU learning across four levels:

**Weight Space Analysis** characterizes the parameter landscape. Eigenvalue spectra of recurrent matrices reveal memory timescales—how long the network can integrate information. Gate statistics show when the network decides to update or reset its state. PHATE embeddings visualize how parameters evolve during training, revealing phases and regime transitions.

**Representation Dynamics** examines the hidden state space where computation occurs. Fixed points are equilibrium states where the network would remain indefinitely under constant input. Attractors are stable fixed points that pull nearby trajectories toward them, implementing discrete memory states or decisions. The geometry of these attractors reflects the computational structure the network has learned.

**Feature Encoding** determines what the network has learned to represent. Linear probes test whether game variables (current player, threats, win conditions) can be read out from hidden states. Mutual information quantifies how much information each hidden dimension carries about each game feature, identifying specialized neurons versus distributed codes.

**Learning Dynamics** tracks how these structures emerge during training. Early training typically shows diffuse, unstructured dynamics. As training progresses, attractors crystallize, gates learn when to update or hold state, and representations organize around task-relevant features. Performance improvements correlate with sharpening of attractor landscapes and increasing mutual information with strategic variables.

## Implementation Status

The repository implements comprehensive weight-space analysis (eigenvalues, gates, PHATE), representation analysis (hidden state sampling, embeddings), and interpretability tools (linear probes, mutual information). The primary gaps are advanced dynamical systems analysis—specifically fixed-point finding and attractor evolution tracking—which would connect internal structure directly to learning progress and gameplay strategy.

---

## Document Map

### Core Documentation

- **[Theoretical Foundations](theoretical_foundations.md)** - Dynamical systems theory, information theory, and computational neuroscience foundations
- **[GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md)** - Implementation status, research gaps, and opportunities for dynamical systems analysis
- **[Case Studies](case_studies.md)** - Detailed examples from RNN interpretability literature with implementation approaches

### Analysis Methods

- **[Mutual Information Analysis](mutual_information_theory.md)** - Per-dimension MI computation, neuron specialization, and encoding patterns
- **[Weight Trajectory Analysis with PHATE](weight_embeddings_theory.md)** - Manifold learning for parameter evolution and training dynamics
- **[References](references.md)** - Comprehensive bibliography organized by research area

### Practical Workflows

- **End‑to‑end pipeline** to reproduce figures → User Manual workflow: [GRU Interpretability Pipeline](../manual/workflows/gru_interpretability.md)

---

## Quick Start: Where to Begin

**New to dynamical systems & RNN interpretability?**
→ Start with [Theoretical Foundations](theoretical_foundations.md) for conceptual grounding

**Want to see concrete examples?**
→ Jump to [Case Studies](case_studies.md) to see how these methods revealed mechanisms in sentiment RNNs, AlphaZero, and more

**Ready to implement missing analyses?**
→ Check [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md) for implementation priorities and code stubs

**Need specific technical details?**
→ Consult [Mutual Information Analysis](mutual_information_theory.md) or [Weight Trajectory Analysis with PHATE](weight_embeddings_theory.md)

**Looking for papers to cite?**
→ Browse [References](references.md) organized by topic with annotations

---

## Key Concepts Across Documents

These central ideas appear throughout the documentation:

1. **Fixed points and attractors** - Equilibrium states revealing computational structure
   - Theory: [Theoretical Foundations §2](theoretical_foundations.md#2-fixed-points-and-attractors)
   - Examples: [Case Studies #1, #2](case_studies.md)
   - Implementation: [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md)

2. **Mutual information for representation analysis** - Model-free quantification of encoded information
   - Theory: [Theoretical Foundations §3](theoretical_foundations.md#3-information-theory-for-interpretability)
   - Methods: [Mutual Information Analysis](mutual_information_theory.md)
   - Examples: [Case Studies #3, #6](case_studies.md)

3. **Training dynamics and attractor emergence** - Evolution of computational structure during learning
   - Theory: [Theoretical Foundations §6](theoretical_foundations.md#6-learning-dynamics-how-attractors-emerge)
   - Implementation: [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md)
   - Examples: [Case Studies #3, #5](case_studies.md)

4. **Manifold learning and trajectory visualization** - Revealing structure in high-dimensional parameter spaces
   - Theory: [Theoretical Foundations §4](theoretical_foundations.md#4-manifold-learning-and-trajectory-embedding)
   - Methods: [Weight Trajectory Analysis with PHATE](weight_embeddings_theory.md)
   - Applications: Throughout [Case Studies](case_studies.md)

---

## References

This overview synthesizes results from:
- **Dynamical systems**: Sussillo & Barak (2013), Maheswaranathan et al. (2019), Huang et al. (2024)
- **Information theory**: Kraskov et al. (2004), Maheswaranathan & Williams (2024)
- **Manifold learning**: Moon et al. (2019) - PHATE, Rübel et al. (2023) - T-PHATE
- **Game-playing**: Silver et al. (2018) - AlphaZero, McGrath et al. (2022), Tian et al. (2024)
- **Neuroscience**: Khona & Fiete (2022), Vyas et al. (2020), Wang (2001)

See [complete bibliography](references.md) for full citations with DOIs and annotations.

# Scientific Background

Research context, literature reviews, and theoretical foundations for weight trajectory analysis and GRU interpretability.

---

## Contents

### [GRU Observability Literature Review](./gru_observability_literature.md)
Comprehensive comparison with state-of-the-art RNN interpretability research.

**What's covered**:
- Weight space analysis (eigenvalues, timescales, gate dynamics)
- Representation space (fixed points, attractors, manifolds)
- Learning dynamics evolution during training
- Interpretability techniques (probing, visualization, FSM extraction)
- Game-specific studies (Connect Four and beyond)

**Key insights**:
- ✅ What we implement (70% coverage): gates, eigenvalues, probes, MI, PHATE embeddings
- 🔴 Critical gaps: fixed-point analysis, attractor evolution, mutual information evolution
- 📊 Priority matrix: effort vs impact for missing features

**Use this for**:
- Understanding the research landscape
- Justifying analysis choices in papers
- Identifying future work directions

---

### [Mutual Information Theory](./mutual_information_theory.md)
Deep dive into per-dimension MI analysis and neuron specialization.

**What's covered**:
- Per-dimension vs mean MI computation
- Neuron specialization patterns (sparse vs distributed encoding)
- Encoding mechanisms (binary threshold detectors, linear counters, multi-modal)
- Expected results for Connect Four features
- Diagnostic use cases (debugging, architecture selection, feature engineering)

**Key insights**:
- Individual neurons can specialize for specific game concepts
- High-MI dimensions reveal interpretable "feature detectors"
- Encoding quality visible in value distributions (violin plots, scatter patterns)

**Use this for**:
- Understanding what MI plots mean
- Interpreting neuron-level representations
- Connecting to neuroscience interpretability literature (Karpathy, Maheswaranathan)

---

### [Weight Trajectory Embeddings](./weight_embeddings_theory.md)
Technical explanation of PHATE embeddings for training dynamics.

**What's covered**:
- Why PHATE over PCA/t-SNE for sparse checkpoint sampling
- Data flow: checkpoint ingestion → weight extraction → embedding → plotting
- knn parameter adaptation for small datasets
- Interpretation of weight vs representation trajectories
- Highlighting minimal-loss and final-epoch checkpoints

**Key insights**:
- PHATE preserves progressive structure in high-dimensional weight changes
- Smooth trajectories reveal continuous learning (vs abrupt jumps = regime shifts)
- Weight trajectories show parameter space exploration
- Representation trajectories show how board encodings evolve

**Use this for**:
- Understanding trajectory visualization plots
- Troubleshooting PHATE parameters
- Interpreting weight-space vs representation-space dynamics

---

## How to Use These Docs

### For Paper Writing
1. Read [GRU Observability Literature Review](./gru_observability_literature.md) first
   - Use for "Related Work" section
   - Cite key papers (Sussillo & Barak, Maheswaranathan, Lei et al.)
2. Reference [MI Theory](./mutual_information_theory.md) for "Methods" section
   - Explain per-dimension analysis
   - Justify neuron specialization claims
3. Use [Weight Embeddings](./weight_embeddings_theory.md) for trajectory plots
   - Explain PHATE choice in methods
   - Interpret learning phases from trajectories

### For Understanding Analysis Results
1. **MI plots confusing?** → [MI Theory](./mutual_information_theory.md)
2. **Trajectory plots unexpected?** → [Weight Embeddings](./weight_embeddings_theory.md)
3. **Want to add new analysis?** → [Literature Review](./gru_observability_literature.md) (see gap analysis)

### For Extending the Project
Check [Literature Review](./gru_observability_literature.md) priority matrix:
- **P0**: Fixed-point finding, attractor evolution (very high impact)
- **P1**: Mutual information evolution, strategy embeddings (high impact)
- **P2**: Full Jacobian analysis, expanded probing (medium impact)

---

## Cross-References

- **User Manual**: [`../manual/`](../manual/) - How to run analyses
- **Plot Guides**: [`../manual/plots/`](../manual/plots/) - How to read outputs
- **Workflows**: [`../manual/workflows/gru_interpretability.md`](../manual/workflows/gru_interpretability.md) - End-to-end pipelines

---

## Key Papers Referenced

### RNN Interpretability
- **Sussillo & Barak (2013)**: "Opening the black box" - Fixed-point finding methodology
- **Maheswaranathan et al. (2019)**: Line attractors in sentiment RNNs
- **Jordan et al. (2019)**: GRUs as continuous-time dynamical systems
- **Lambrechts et al. (2022)**: Mutual information in POMDPs

### Game-Playing RNNs
- **Lei et al. (2024)**: STRIL - Strategy representation for Connect Four
- **Ni et al. (2022)**: Recurrent RL baselines for POMDPs

### Interpretability Techniques
- **Karpathy et al. (2015)**: Visualizing and understanding RNNs
- **Strobelt et al. (2018)**: LSTMVis - trajectory visualization
- **Carr et al. (2021)**: Extracting verifiable FSMs from RNN policies

Full citations available in each document.

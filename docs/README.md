# Weight Trajectories Documentation

Documentation for the weight-trajectories project: analyzing neural network training dynamics and GRU interpretability in Connect Four agents.

---

## 🚀 Quick Start

**New users**: Start with the [User Manual](./manual/README.md) for complete command reference and workflows.

**Running GRU analysis**: See the [GRU Interpretability Workflow](./manual/workflows/gru_interpretability.md)

---

## 📚 Documentation Structure

### [User Manual](./manual/) (`./manual/`)
Complete operational guide for running experiments and generating visualizations.

- **Commands**: Detailed reference for all `./wt.sh` commands
  - [observability](./manual/commands/observability.md) - GRU dynamics and interpretability
  - [train](./manual/commands/train.md), [metrics](./manual/commands/metrics.md), [visualize](./manual/commands/visualize.md), etc.
- **Plots**: How to read every CSV and figure generated
  - [GRU mutual information](./manual/plots/gru_mutual_info.md)
  - [GRU observability + probes](./manual/plots/gru_observability.md)
  - [Fixed points + evolution](./manual/plots/fixed_points.md)
- **Workflows**: End-to-end analysis pipelines
  - [GRU interpretability pipeline](./manual/workflows/gru_interpretability.md)

### [Scientific Background](./scientific/) (`./scientific/`)
Research context, literature reviews, and theoretical foundations.

- **[GRU Observability Literature Review](./scientific/gru_observability_literature.md)**
  - Comparison with Sussillo & Barak (2013), Maheswaranathan et al. (2019), Lei et al. (2024)
  - Gap analysis: what we implement vs state-of-the-art
  - Priority recommendations for future work

- **[Mutual Information Theory](./scientific/mutual_information_theory.md)**
  - Per-dimension MI analysis explained
  - Neuron specialization vs distributed encoding
  - Connection to neuroscience interpretability literature

- **[Weight Trajectory Embeddings](./scientific/weight_embeddings_theory.md)**
  - Why PHATE for training dynamics
  - Interpretation guide for trajectory plots
  - Data flow and algorithmic details

### [Technical Reference](./reference/) (`./reference/`)
Architecture details and internal documentation.

- **[Model Architecture Diagrams](./reference/architecture_diagrams.md)**
  - How to generate paper-ready ResNet+GRU diagrams
  - PlotNeuralNet and Netron setup

- **[Changelog: Manual Updates](./reference/changelog/)**
  - Internal documentation of major feature additions

---

## 📖 Common Tasks

### I want to...

**Run the GRU interpretability pipeline**
→ [GRU Interpretability Workflow](./manual/workflows/gru_interpretability.md)

**Understand what a plot means**
→ [Plot Explainers](./manual/plots/) (find your plot by filename)

**Learn about the science behind the analysis**
→ [Scientific Background](./scientific/)

**Generate architecture diagrams for a paper**
→ [Architecture Diagrams Guide](./reference/architecture_diagrams.md)

**See what features were recently added**
→ [Manual Update Changelog](./reference/changelog/)

**Understand how mutual information analysis works**
→ [MI Theory](./scientific/mutual_information_theory.md) + [MI Plot Guide](./manual/plots/gru_mutual_info.md)

**Debug a failed analysis**
→ [GRU Workflow Troubleshooting](./manual/workflows/gru_interpretability.md#troubleshooting)

**Compare our GRU analysis with research papers**
→ [Literature Review & Gap Analysis](./scientific/gru_observability_literature.md)

---

## 🎓 For Researchers

If you're writing a paper or want deep scientific context:

1. **Start**: [GRU Observability Literature Review](./scientific/gru_observability_literature.md)
   - See what techniques are implemented (fixed points, MI, probes, etc.)
   - Understand priority gaps (what's missing from state-of-the-art)

2. **Theory**: [Mutual Information Theory](./scientific/mutual_information_theory.md)
   - Neuron specialization patterns
   - Encoding mechanisms (binary vs continuous features)
   - Expected results for Connect Four domain

3. **Practice**: [GRU Interpretability Workflow](./manual/workflows/gru_interpretability.md)
   - Run the full analysis pipeline
   - Generate figures for paper
   - Identify interpretable neurons

4. **Interpretation**: [Plot Explainers](./manual/plots/)
   - `gru_mutual_info.md` - Comprehensive guide with examples
   - `gru_observability.md` - Gates, timescales, PHATE embeddings
   - `fixed_points.md` - Attractor dynamics and evolution

---

## 🛠️ For Engineers

If you're debugging models or optimizing architectures:

1. **Quick Reference**: [Command Index](./manual/README.md)
   - All `./wt.sh` commands with options

2. **Diagnostics**: [GRU Workflow](./manual/workflows/gru_interpretability.md#diagnostic-scenarios)
   - Scenario 1: Model plays poorly despite low loss
   - Scenario 2: Large model no better than small model
   - Scenario 3: Preparing paper figures

3. **Plot Interpretation**: [Plot Explainers](./manual/plots/)
   - See "Diagnostic use cases" sections
   - Architecture selection, neuron ablation, feature engineering

---

## 📂 Directory Structure

```
docs/
├── README.md (this file)
│
├── manual/                          # User-facing operational manual
│   ├── README.md                    # Manual index
│   ├── commands/                    # Command reference
│   │   ├── observability.md
│   │   ├── train.md
│   │   └── ...
│   ├── plots/                       # Plot interpretation guides
│   │   ├── gru_mutual_info.md
│   │   ├── gru_observability.md
│   │   └── ...
│   └── workflows/                   # End-to-end pipelines
│       └── gru_interpretability.md
│
├── scientific/                      # Research background & theory
│   ├── gru_observability_literature.md
│   ├── mutual_information_theory.md
│   └── weight_embeddings_theory.md
│
└── reference/                       # Technical specs & internal docs
    ├── architecture_diagrams.md
    └── changelog/
        └── 2025-10-28_mi_dimension_analysis.md
```

---

## 🔗 External Resources

- **Project Repository**: [GitHub](https://github.com/yourusername/weight-trajectories) (if applicable)
- **Related Papers**:
  - Sussillo & Barak (2013): "Opening the black box" - [arXiv](https://arxiv.org/abs/1211.4722)
  - Maheswaranathan et al. (2019): "Line attractors in RNNs" - [NeurIPS](https://papers.nips.cc/paper/9419-reverse-engineering-recurrent-networks-for-sentiment-classification-reveals-line-attractor-dynamics)
  - Lei et al. (2024): "STRIL for Connect Four" - [arXiv](https://arxiv.org/html/2409.19363v2)

---

## 📝 Contributing to Documentation

When adding new features:

1. **User-facing**: Update `manual/commands/` and `manual/plots/`
2. **Scientific context**: Add to `scientific/` if novel technique
3. **Changelog**: Document major changes in `reference/changelog/`
4. **Cross-link**: Add entries to this README under "Common Tasks"

See [`reference/changelog/2025-10-28_mi_dimension_analysis.md`](./reference/changelog/2025-10-28_mi_dimension_analysis.md) for an example.

---

## ⚡ Version

**Last Updated**: 2025-10-28
**Major Features**:
- Complete GRU observability pipeline (gates, eigenvalues, probes, MI)
- Per-dimension mutual information analysis (neuron specialization)
- Fixed-point finding and attractor evolution
- Weight trajectory embeddings (PHATE)

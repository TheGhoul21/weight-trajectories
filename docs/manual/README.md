# wt.sh User Manual

Complete practical handbook for the `wt` tool — the single entrypoint for dataset generation, training, diagnostics, visualizations, and analysis.

**Audience**: Researchers and engineers running experiments
**Scope**: Every command, option, output, and plot interpretation

---

## Navigation

**Looking for**:
- **Scientific background?** → [Scientific Background](../scientific/) (theory, literature reviews)
- **Architecture diagrams?** → [Architecture Diagrams](../reference/architecture_diagrams)
- **Main docs index?** → [Documentation Home](../)

---

## Quick Start
- Read first: [Getting Started](manual/getting_started) – three common paths
- Print available commands: `./wt.sh help`
- See Python interpreter: `./wt.sh python-path`
- Complete workflow: [GRU Interpretability Pipeline](manual/workflows/gru_interpretability)

## Contents

### Commands
How to run, options, defaults, inputs/outputs
- [Commands Overview](manual/commands/) - All commands indexed
- [dataset](manual/commands/dataset) - Generate training data
- [train](manual/commands/train) and [train-all](manual/commands/train-all) - Model training
- [model](manual/commands/model) - Model inspection
- [metrics](manual/commands/metrics) - Weight/representation metrics
- [visualize](manual/commands/visualize) - Trajectory visualization
- [analyze](manual/commands/analyze) - Interactive wizard
- [embeddings](manual/commands/embeddings) - Weight embeddings
- [cka](manual/commands/cka) - Representational similarity
- [trajectory-embedding](manual/commands/trajectory-embedding) - Metric space
- [observability](manual/commands/observability) - GRU analysis suite
- [report](manual/commands/report) - Generate reports
- [onnx](manual/commands/onnx) - Model export
- [python-path](manual/commands/python-path) - Print interpreter

### Plots and CSVs
What you're looking at and how to read it
- [Plots Overview](manual/plots/) - All outputs indexed
- [Checkpoint Metrics CSV](manual/plots/checkpoint_metrics_csv) - Weight statistics
- [Unified Visualization Suite](manual/plots/visualize_unified) - Multi-type plots
- [Metric-Space Trajectories](manual/plots/trajectory_metric_space) - Embedding in metric space
- [Weight/Representation Embeddings](manual/plots/embeddings_weights) - PHATE, PCA, t-SNE, UMAP
- [CKA Similarity](manual/plots/cka) - Cross-model comparison
- [GRU Observability](manual/plots/gru_observability) - Gates, timescales, probes
- [GRU Mutual Information](manual/plots/gru_mutual_info) - Feature encoding analysis
- [GRU Fixed Points](manual/plots/fixed_points) - Attractors and evolution
- [Factorial Analysis](manual/plots/factorial_heatmaps) - Architecture sweep heatmaps
- [CNN Activation Maps](manual/plots/activations) - Grad-CAM visualization

### Workflows
End-to-end analysis guides
- [Workflows Overview](manual/workflows/) - All pipelines indexed
- [GRU Interpretability Pipeline](manual/workflows/gru_interpretability) - Complete 5-stage guide

Conventions
- Checkpoints live under `checkpoints/` (often `checkpoints/save_every_3/<model>/weights_epoch_XXXX.pt`).
- Metrics and analysis go to `diagnostics/...`.
- Visual assets go to `visualizations/...`.
- All commands accept extra script-specific flags after `--`.
- Override Python with env var: `PYTHON_BIN=/path/to/python ./wt.sh <cmd>`.

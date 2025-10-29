# wt.sh User Manual

Complete practical handbook for the `wt` tool — the single entrypoint for dataset generation, training, diagnostics, visualizations, and analysis.

**Audience**: Researchers and engineers running experiments
**Scope**: Every command, option, output, and plot interpretation

---

## Navigation

**Looking for**:
- **Scientific background?** → [Scientific Background](../scientific/) (theory, literature reviews)
- **Architecture diagrams?** → [Architecture Diagrams](../reference/architecture_diagrams.md)
- **Main docs index?** → [Documentation Home](../)

---

## Quick Start
- Read first: [Getting Started](getting_started.md) – three common paths
- Print available commands: `./wt.sh help`
- See Python interpreter: `./wt.sh python-path`
- Complete workflow: [GRU Interpretability Pipeline](workflows/gru_interpretability.md)

## Contents

### Commands
How to run, options, defaults, inputs/outputs
- [Commands Overview](manual/commands/) - All commands indexed
- [dataset](commands/dataset.md) - Generate training data
- [train](commands/train.md) and [train-all](commands/train-all.md) - Model training
- [model](commands/model.md) - Model inspection
- [metrics](commands/metrics.md) - Weight/representation metrics
- [visualize](commands/visualize.md) - Trajectory visualization
- [analyze](commands/analyze.md) - Interactive wizard
- [embeddings](commands/embeddings.md) - Weight embeddings
- [cka](commands/cka.md) - Representational similarity
- [trajectory-embedding](commands/trajectory-embedding.md) - Metric space
- [observability](commands/observability.md) - GRU analysis suite
- [report](commands/report.md) - Generate reports
- [onnx](commands/onnx.md) - Model export
- [python-path](commands/python-path.md) - Print interpreter

### Plots and CSVs
What you're looking at and how to read it
- [Plots Overview](manual/plots/) - All outputs indexed
- [Checkpoint Metrics CSV](plots/checkpoint_metrics_csv.md) - Weight statistics
- [Unified Visualization Suite](plots/visualize_unified.md) - Multi-type plots
- [Metric-Space Trajectories](plots/trajectory_metric_space.md) - Embedding in metric space
- [Weight/Representation Embeddings](plots/embeddings_weights.md) - PHATE, PCA, t-SNE, UMAP
- [CKA Similarity](plots/cka.md) - Cross-model comparison
- [GRU Observability](plots/gru_observability.md) - Gates, timescales, probes
- [GRU Mutual Information](plots/gru_mutual_info.md) - Feature encoding analysis
- [GRU Fixed Points](plots/fixed_points.md) - Attractors and evolution
- [Factorial Analysis](plots/factorial_heatmaps.md) - Architecture sweep heatmaps
- [CNN Activation Maps](plots/activations.md) - Grad-CAM visualization

Note: The Unified Visualization Suite includes temporal trajectory plots across both training and game time, with T‑PHATE support (delay embeddings and temporal kernel blending) for clearer learning dynamics.

### Workflows
End-to-end analysis guides
- [Workflows Overview](manual/workflows/) - All pipelines indexed
- [GRU Interpretability Pipeline](workflows/gru_interpretability.md) - Complete 5-stage guide

Conventions
- Checkpoints live under `checkpoints/` (often `checkpoints/save_every_3/<model>/weights_epoch_XXXX.pt`).
- Metrics and analysis go to `diagnostics/...`.
- Visual assets go to `visualizations/...`.
- All commands accept extra script-specific flags after `--`.
- Override Python with env var: `PYTHON_BIN=/path/to/python ./wt.sh <cmd>`.

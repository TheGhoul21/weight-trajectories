# wt.sh User Manual

Complete practical handbook for the `wt` tool — the single entrypoint for dataset generation, training, diagnostics, visualizations, and analysis.

**Audience**: Researchers and engineers running experiments
**Scope**: Every command, option, output, and plot interpretation

---

## Navigation

**Looking for**:
- **Scientific background?** → [`../scientific/`](../scientific/) (theory, literature reviews)
- **Architecture diagrams?** → [`../reference/architecture_diagrams.md`](../reference/architecture_diagrams.md)
- **Main docs index?** → [`../README.md`](../README.md)

---

## Quick Start
- Print available commands: `./wt.sh help`
- See Python interpreter: `./wt.sh python-path`
- **Complete workflow**: [GRU Interpretability Pipeline](./workflows/gru_interpretability.md)

Contents
- Commands (how to run, options, defaults, inputs/outputs)
  - [dataset](./commands/dataset.md)
  - [train](./commands/train.md) and [train-all](./commands/train-all.md)
  - [model](./commands/model.md)
  - [metrics](./commands/metrics.md)
  - [visualize](./commands/visualize.md)
  - [analyze (wizard)](./commands/analyze.md)
  - [embeddings](./commands/embeddings.md)
  - [cka](./commands/cka.md)
  - [trajectory-embedding](./commands/trajectory-embedding.md)
  - [observability](./commands/observability.md)
  - [report](./commands/report.md)
  - [onnx](./commands/onnx.md)
  - [python-path](./commands/python-path.md)
- Plots and CSVs (what you're looking at and how to read it)
  - [Checkpoint metrics CSV](./plots/checkpoint_metrics_csv.md)
  - [Unified visualization suite](./plots/visualize_unified.md)
  - [Metric-space trajectory embeddings](./plots/trajectory_metric_space.md)
  - [Weight/representation embeddings](./plots/embeddings_weights.md)
  - [CKA similarity](./plots/cka.md)
  - [GRU observability + probes](./plots/gru_observability.md)
  - [GRU mutual information](./plots/gru_mutual_info.md)
  - [GRU fixed points + evolution](./plots/fixed_points.md)
  - [CNN activation maps](./plots/activations.md)
- Workflows (end-to-end analysis guides)
  - [GRU interpretability pipeline](./workflows/gru_interpretability.md)

Conventions
- Checkpoints live under `checkpoints/` (often `checkpoints/save_every_3/<model>/weights_epoch_XXXX.pt`).
- Metrics and analysis go to `diagnostics/...`.
- Visual assets go to `visualizations/...`.
- All commands accept extra script-specific flags after `--`.
- Override Python with env var: `PYTHON_BIN=/path/to/python ./wt.sh <cmd>`.

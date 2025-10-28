# wt.sh User Guide

This is your complete, practical handbook for the wt tool — the single entrypoint that orchestrates dataset generation, training, diagnostics, visualizations, reports, and exports for the weight-trajectories project.

- Audience: researchers/engineers running experiments and visualizations.
- Scope: every command and option, what it reads/writes, and a plot/CSV explainer for all artifacts.

Quick start:
- Print available commands: `./wt.sh help`
- See Python used: `./wt.sh python-path`

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
- Plots and CSVs (what you’re looking at and how to read it)
  - [Checkpoint metrics CSV](./plots/checkpoint_metrics_csv.md)
  - [Unified visualization suite](./plots/visualize_unified.md)
  - [Metric-space trajectory embeddings](./plots/trajectory_metric_space.md)
  - [Weight/representation embeddings](./plots/embeddings_weights.md)
  - [CKA similarity](./plots/cka.md)
  - [GRU observability + probes](./plots/gru_observability.md)
  - [GRU mutual information](./plots/gru_mutual_info.md)
  - [GRU fixed points + evolution](./plots/fixed_points.md)
  - [CNN activation maps](./plots/activations.md)

Conventions
- Checkpoints live under `checkpoints/` (often `checkpoints/save_every_3/<model>/weights_epoch_XXXX.pt`).
- Metrics and analysis go to `diagnostics/...`.
- Visual assets go to `visualizations/...`.
- All commands accept extra script-specific flags after `--`.
- Override Python with env var: `PYTHON_BIN=/path/to/python ./wt.sh <cmd>`.

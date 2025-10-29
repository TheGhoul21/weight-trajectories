# wt analyze (Wizard)

Overview
- Runs the end‑to‑end analysis pipeline: metrics → visuals → report, across all saved runs.
- Good starting point for users who want a complete, reproducible pass.

How to Run
- `./wt.sh analyze` (invokes `scripts/analyze_trajectories_wizard.sh`)

Pipeline steps
1) Metrics: compute_checkpoint_metrics across all runs in `checkpoints/save_every_3`
2) Advanced derived metrics: inline Python computes trajectory_summary.csv
3) Visuals: generates 5 unified plots and a PHATE metric-space embedding
4) Report: writes ANALYSIS_REPORT.md summarizing results

Defaults
- Checkpoint base: `checkpoints/save_every_3`
- Output (metrics): `diagnostics/trajectory_analysis`
- Visualizations: `visualizations`
- Python: `.venv/bin/python3` unless overridden by PYTHON_BIN

Reads
- All checkpoint subdirs under the base
- training_history.json per model (if present)

Writes
- diagnostics/trajectory_analysis/*.csv (per-run metrics + trajectory_summary.csv)
- visualizations/*.png and visualizations/metric_embeddings_phate/*.png
- diagnostics/trajectory_analysis/ANALYSIS_REPORT.md

Artifacts are explained under:
- [Checkpoint metrics CSV](manual/plots/checkpoint_metrics_csv)
- [Unified visualization suite](manual/plots/visualize_unified)
- [Metric-space trajectory embeddings](manual/plots/trajectory_metric_space)
- [Report generation](manual/plots/visualize_unified)

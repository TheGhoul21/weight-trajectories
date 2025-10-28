# wt report

Generate the unified markdown analysis report summarizing metrics, histories, and key figures.

Backed by: `scripts/generate_report.py`

Options
- --metrics-dir [default diagnostics/trajectory_analysis]
- --checkpoint-dir [default checkpoints/save_every_3]
- --viz-dir [default visualizations]
- --output [default diagnostics/trajectory_analysis/ANALYSIS_REPORT.md]

Reads
- Metrics CSVs: `{metrics-dir}/k3_c{channels}_gru{hidden}_metrics.csv`
- Training histories: `{checkpoint-dir}/k3_c{channels}_gru{hidden}/training_history.json`
- Visualizations directory is referenced in the report text.

Writes
- Markdown report at `--output` with:
  - Executive summary and key findings
  - Generalization performance tables (min val loss, epoch of min)
  - Overfitting dynamics (final gap, degree)
  - Weight trajectory dynamics (growth, correlation)
  - Representation analysis (top-1 ratio, total variance)
  - Factorial design effects (variance decomposition)
  - Practical recommendations
  - Links/mentions to figures under `--viz-dir`

Notes
- The report computes summary stats across the 3×3 grid of models (channels ∈ {16,64,256}, GRU ∈ {8,32,128}).
- Any missing CSV or history per model is silently skipped.

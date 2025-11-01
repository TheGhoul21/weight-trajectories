# wt factorial

Generate factorial heatmaps and companion 3×3 summaries across the full architecture sweep (channels × GRU size).

Overview
- Command: `./wt.sh factorial`
- Script: `scripts/visualize_unified.py`
- Inputs: per-model metrics CSVs and training histories
- Outputs: 5 figures summarizing metrics, trajectories, and capacity effects

## Usage

```bash
./wt.sh factorial \
  --metrics-dir diagnostics/trajectory_analysis \
  --checkpoint-dir checkpoints/save_every_1 \
  --output-dir visualizations/factorial
```

Options
- `--metrics-dir DIR`  Directory containing per-model `k3_c*_gru*_metrics.csv`
  - Default: `diagnostics/trajectory_analysis`
- `--checkpoint-dir DIR`  Base directory with model subfolders containing `training_history.json`
  - Default: `checkpoints/save_every_1`
- `--output-dir DIR`  Destination for figures
  - Default: `visualizations`

Directory layout expectations
```
metrics_dir/
  k3_c16_gru8_metrics.csv
  k3_c16_gru32_metrics.csv
  ...
  k3_c256_gru128_metrics.csv

checkpoint_dir/
  k3_c16_gru8/training_history.json
  k3_c16_gru32/training_history.json
  ...
  k3_c256_gru128/training_history.json
```

## Figures written
- `factorial_heatmaps.png`  3×3 heatmaps for 9 key metrics:
  - Weight Norm (final), Step Norm (mean), Step Cosine (final)
  - Total Variance (mean), Top‑1 Ratio (mean)
  - Min Val Loss, Final Val Loss, Train/Val Gap (final), Val Loss Increase
- `loss_trajectory_grid.png`  3×3 grid overlaying weight norm vs train/val loss (with best‑epoch marker)
- `representation_grid.png`  3×3 SV top‑k variance ratio trends (shared legend)
- `gru_sweep_comparison.png`  Main effect of GRU size per channel (3×3)
- `channel_sweep_comparison.png`  Main effect of channel count per GRU size (3×3)

All figures are saved under `--output-dir`.

## Notes
- Train all 9 architectures first (e.g., `./wt.sh train-all`).
- Compute metrics before running this command (e.g., `./wt.sh metrics --output-dir diagnostics/trajectory_analysis`).
- Heatmaps use `RdYlBu_r` and annotate values for quick reading.

## See also
- Plots explanation: [Factorial Analysis](/manual/plots/factorial_heatmaps)
- Metrics CSV format: [Checkpoint Metrics CSV](/manual/plots/checkpoint_metrics_csv)
- Unified plots overview: [Unified Visualizations](/manual/plots/visualize_unified)

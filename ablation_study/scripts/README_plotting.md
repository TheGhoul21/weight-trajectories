Usage

- Create venv and install deps (using uv):
  - `uv venv .venv`
  - `source .venv/bin/activate`
  - `uv pip install pandas matplotlib seaborn`

- Generate plots from `all.json`:
  - `python scripts/plot_ablations.py --input all.json --outdir outputs/plots --mode final --overfit-threshold 1.15 --csv`
  - Optional: use best validation epoch instead of final: `--mode minval`
  - Limit learning-curve grids to the best `K` runs: add `--top-k K` (omit or set `0` to include all runs)

Outputs

- `outputs/plots/scatter_train_vs_val(final).png`: Train vs Val loss; red outlined points exceed overfit threshold (val/train ≥ threshold). Diagonal y=x is drawn.
- `outputs/plots/heatmap_gap_k3_{True|False}(final).png`: (val - train) heatmap across `c` x `gru` per `k3` setting.
- `outputs/plots/heatmap_ratio_k3_{True|False}(final).png`: (val/train) heatmap across `c` x `gru`.
- `outputs/plots/summary_final.csv`: Tabular summary used to plot.
- `outputs/plots/learning_curves_all_pXX<(mode)>.png`: Paginated grids of learning curves (one image per chunk of runs). When `--top-k` is supplied, filenames use `top{K}_pXX`.

Notes

- Overfit threshold is configurable via `--overfit-threshold` (default 1.15).
- The script parses `k3`, `c`, and `gru` from checkpoint directory names like `k3_c16_gru32_*`.

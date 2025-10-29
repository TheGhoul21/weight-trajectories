# wt metrics

Overview
- Computes per‑checkpoint diagnostics (weight norms, step statistics, optional representation SVD) used across visualizations and embeddings.

How to Run
- `./wt.sh metrics` (invokes `python scripts/compute_checkpoint_metrics.py`)

Maps to: `python scripts/compute_checkpoint_metrics.py`

Reads
- One or more checkpoint directories: `.../k{kernel}_c{channels}_gru{hidden}/weights_epoch_*.pt`
- Optional dataset if `--board-source=dataset`

Writes
- Per-run CSV under `diagnostics/checkpoint_metrics/`: `{run}_metrics.csv`

Options and defaults
- --checkpoint-dirs [one or more, REQUIRED]
- --component [cnn|gru|all, default gru] which weights are flattened
- --epoch-min [int] include epochs ≥ value
- --epoch-max [int] include epochs ≤ value
- --epoch-step [int, default 1] stride over checkpoints
- --board-source [none|random|dataset, default none] if not none, compute hidden-state variance stats
- --board-count [int, default 16]
- --board-dataset [path] required when board-source=dataset
- --board-seed [int, default 37]
- --top-singular-values [int, default 4]
- --output-dir [path, default diagnostics/checkpoint_metrics]

CSV columns
- epoch
- weight_norm
- step_norm, step_cosine, relative_step (empty for first row)
- repr_total_variance (optional)
- repr_top{1..K}_ratio (optional)

See also: Plot/CSV reference → [Checkpoint metrics CSV](manual/plots/checkpoint_metrics_csv)

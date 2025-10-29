# wt observability

Overview
- Runs the GRU observability and dynamics pipeline: extract data, analyze gates/embeddings/probes, compute MI, and (optionally) find/evolve fixed points.

How to Run
- `./wt.sh observability extract|analyze|mi|fixed|evolve [options]`

Subcommands
- extract → scripts/extract_gru_dynamics.py
- analyze (alias: summarize) → scripts/analyze_gru_observability_results.py + compute_hidden_mutual_info.py
- fixed → scripts/find_gru_fixed_points.py
- evolve (alias: evolution) → scripts/analyze_fixed_point_evolution.py

extract (hidden dynamics, gates, eigen-timescales)
- Options
  - --checkpoint-dir [default checkpoints/save_every_3]
  - --dataset [default data/connect4_sequential_10k_games.pt]
  - --max-games [int, default 256]
  - --sample-hidden [int, default 1500]
  - --device [cpu|cuda, default cpu]
  - --output-dir [default diagnostics/gru_observability]
  - --seed [int, default 0]
  - --verbose [flag]
- Reads: sequential dataset; per-model checkpoints
- Writes per model under `diagnostics/gru_observability/<model>/`:
  - metrics.csv (per-epoch gate stats and timescales)
  - unit_gate_stats.csv (per-unit gate mean/std)
  - hidden_samples/epoch_XXX.npz (hidden/update/reset arrays, features)
  - epoch_XXX_eigenvalues.npz (per-unit eigenvalue spectra)

analyze (plots + linear probes)
- Options
  - --analysis-dir [default diagnostics/gru_observability]
  - --output-dir [default visualizations/gru_observability]
  - --embedding-epochs [list, default 3 30 60 100]
  - --embedding-feature [default move_index]
  - --probe-epochs [list, default 30 60 100]
  - --probe-features [list, default current_player immediate_win_current immediate_win_opponent]
  - --probe-components [list, default gru] (`gru`, `cnn`, or both)
  - --max-hidden-samples [int, default 2000]
  - --seed [int, default 0]
  - --palette [str, default Set2]
  - --skip-embedding [flag]
  - --skip-probing [flag]
- Reads: outputs from extract
- Writes (under `visualizations/gru_observability/`):
  - gate_mean_trajectories.png
  - timescale_heatmap.png
  - phate_epoch_XXX_<feature>.png (unless skipped)
  - probe_results.csv and probe_accuracy.png (per requested component; CNN probes write under `cnn/`)

mutual information (run within analyze)
- `scripts/compute_hidden_mutual_info.py` is executed automatically after analyze completes
- Uses same --analysis-dir and --output-dir as analyze
- Options
  - --features [list, default: all 12 board features]
  - --max-samples [int, default 4000] — subsample hidden states per epoch for MI estimation
  - --seed [int, default 0]
  - --output-dir [default visualizations/gru_observability]
- Writes (to `visualizations/gru_observability/`)
  - **Overview plots** (all models, all epochs):
    - mi_results.csv — long-form table: model/epoch/feature/mi/type
    - mi_heatmap_final.png — cross-model comparison at final epoch
    - mi_trends.png — MI evolution over training per feature
    - mi_metadata.json — run parameters
  - **Per-dimension analysis** (per model, final epoch only):
    - mi_per_dimension_<model>.png — heatmap of MI per hidden dimension; ★ marks best dimension
    - mi_dimension_values_<model>.png — violin/scatter plots showing HOW best dimension encodes each feature

fixed (find fixed points / stability per epoch)
- Options
  - --checkpoint-dir [default checkpoints/save_every_3]
  - --dataset [default data/connect4_sequential_10k_games.pt]
  - --max-contexts [int, default 12]
  - --restarts [int, default 8]
  - --max-iter [int, default 400]
  - --tolerance [float, default 1e-5]
  - --epoch-min / --epoch-max / --epoch-step [default 1]
  - --device [default cpu]
  - --output-dir [default diagnostics/gru_fixed_points]
  - --seed [int, default 0]
- Writes
  - diagnostics/gru_fixed_points/<model>/epochs/epoch_XXX_fixed_points.npz (hidden/residual/spectral/classification/context/eigvals)
  - diagnostics/gru_fixed_points/<model>/fixed_points_summary.csv
  - diagnostics/gru_fixed_points/contexts_metadata.json (+ contexts.pt cache)

evolve (visualize fixed-point evolution)
- Options
  - --fixed-dir [default diagnostics/gru_fixed_points]
  - --output-dir [default visualizations/gru_fixed_points]
- Writes per model
  - <model>_classification_counts.png
  - <model>_spectral_radius.png
  - <model>_attractor_drift.png

Explainers
- See [GRU Observability](manual/plots/gru_observability)
- See [GRU Mutual Information](manual/plots/gru_mutual_info)
- See [GRU Fixed Points and Evolution](manual/plots/fixed_points)
- See [Methods Reference](../../reference/methods) for library links (scikit‑learn MI, PHATE, UMAP, Grad‑CAM).

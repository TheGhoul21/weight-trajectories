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
  - --embedding-epochs [list, default 1 3 5 … 99]
  - --embedding-feature [default move_index]
  - --embedding-animate [flag] render an animated 3×3 grid over epochs (saved as mp4 or gif)
  - --embedding-mode [{separate, joint}, default separate]
      • separate: fit PHATE independently per epoch (fast; orientation stabilized via feature-correlated axis/sign)
      • joint: pool a capped number of samples per epoch and fit PHATE once per model for stable axes across epochs
  - --embedding-joint-samples [int, default 300] per-epoch cap used when --embedding-mode=joint
  - --embedding-fps [int, default 4] animation framerate
  - --embedding-format [{auto, mp4, gif}, default auto]
  - --embedding-dpi [int, default 150] animation DPI
  - --embedding-point-size [float, default 12.0] marker size for PHATE scatter points
  - --embedding-alpha [float, default 0.8] transparency for PHATE points
  - --embedding-dedup [{auto, off}, default auto] deduplicate identical hidden states before PHATE to avoid zero-distance artifacts (turn off to visualize all samples; may be numerically fragile)
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
  - phate_animation_<feature>.mp4 (or .gif) when `--embedding-animate` is set; grid sorted by (kernel, channels, GRU)
  - probe_results.csv and probe_accuracy.png (per requested component; CNN probes write under `cnn/`)
  - probe_confusion_matrices_epoch_XXX.png, probe_feature_summary_epoch_XXX.png, probe_score_density_epoch_XXX.png (final probed epoch)
  - probe_confusion_matrices_best.png, probe_feature_summary_best.png, probe_score_density_best.png (aggregated across models at their own best validation-loss epoch)

mutual information (run within analyze)
- `scripts/compute_hidden_mutual_info.py` is executed automatically after analyze completes
- Uses same --analysis-dir and --output-dir as analyze
- Options
  - --features [list, default: all 12 board features]
  - --max-samples [int, default 4000] — subsample hidden states per epoch for MI estimation
  - --seed [int, default 0]
  - --output-dir [default visualizations/gru_observability]
  - --force [flag] recompute MI even if `mi_results.csv` is present
- Writes (to `visualizations/gru_observability/`)
  - **Overview plots** (all models, all epochs):
    - mi_results.csv — long-form table: model/epoch/feature/mi/type
    - mi_heatmap_final.png — cross-model comparison at final epoch
    - mi_heatmap_best.png — cross-model comparison at each model’s best validation-loss epoch
    - mi_trends.png — MI evolution over training per feature
    - mi_metadata.json — run parameters
  - **Per-dimension analysis** (per model, final and best epochs):
    - mi_per_dimension_<model>.png — per-dimension MI heatmap at final epoch; ★ marks best dimension
    - mi_dimension_values_<model>.png — violin/scatter plots (final)
    - best_epoch/mi_per_dimension_<model>.png and best_epoch/mi_dimension_values_<model>.png — best-epoch variants

Caching and force
- When `mi_results.csv` exists, plots are regenerated from that cache without recomputing MI.
- Use `--force` to ignore the cache and recompute MI for all requested models/epochs.

Notes on epoch selection
Animation format and ffmpeg
- By default, the analyzer prefers MP4 when ffmpeg is available; otherwise it automatically falls back to GIF (PillowWriter).
- If you use uv/pip and want MP4 without a system ffmpeg install, you can add a bundled binary via:
  - `uv pip install imageio-ffmpeg`
  The script will auto-detect it and configure Matplotlib accordingly.
- You can also force a format via `--embedding-format mp4` or `--embedding-format gif`.

- Best epoch is derived from `checkpoints/<model>/training_history.json` using `epochs_saved` to index into `val_loss` and select the minimum; final epoch is the maximum saved epoch.

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
  - <model>_spectral_radius_enhanced.png (mean±SEM ribbon + faint per-context lines)
  - <model>_attractor_drift_enhanced.png (mean±SEM ribbon + faint per-context lines)

Explainers
- See [GRU Observability](../plots/gru_observability.md)
- See [GRU Mutual Information](../plots/gru_mutual_info.md)
- See [GRU Fixed Points and Evolution](../plots/fixed_points.md)
- See [Methods Reference](../../reference/methods.md) for library links (scikit‑learn MI, PHATE, UMAP, Grad‑CAM).

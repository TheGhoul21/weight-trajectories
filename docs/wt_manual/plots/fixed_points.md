# GRU fixed points + evolution

Produced by:
- Discovery: `./wt.sh observability fixed` → `scripts/find_gru_fixed_points.py`
- Evolution: `./wt.sh observability evolve` → `scripts/analyze_fixed_point_evolution.py`

Inputs
- diagnostics/gru_fixed_points/<model>/epochs/epoch_XXX_fixed_points.npz
  - Arrays: hidden, residual, spectral_radius, classification, context_index, eigvals_real/imag
- diagnostics/gru_fixed_points/<model>/fixed_points_summary.csv
- diagnostics/gru_fixed_points/contexts_metadata.json (and contexts.pt)

Outputs (evolution)
- <model>_classification_counts.png: For each epoch, count of {stable, marginal, unstable} fixed points
- <model>_spectral_radius.png: Spectral radius over epochs for stable fixed points; hue=context
- <model>_attractor_drift.png: L2 drift of centroids of stable fixed points per context across epochs

Interpretation
- Classification: stable (spectral radius < 1), unstable (> 1), marginal (~1)
- Spectral radius trends indicate stability changes during learning
- Attractor drift shows how stable equilibria move as weights evolve

Knobs (discovery)
- --max-contexts, --restarts, --max-iter, --tolerance
- --epoch-min/--epoch-max/--epoch-step to sub-sample checkpoints
- --device, --seed

Notes
- Fixed points are computed in GRU hidden space given a static CNN feature vector for a board context
- Deduplication removes nearby duplicates; residual threshold enforces quality

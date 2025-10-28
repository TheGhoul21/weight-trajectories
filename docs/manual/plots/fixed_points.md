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
- `<model>_classification_counts.png` – bars per epoch counting {stable, marginal, unstable} fixed points for each board context.
- `<model>_spectral_radius.png` – spectral radius over epochs for stable points; hue = context id.
- `<model>_attractor_drift.png` – L2 drift of stable-centroid locations per context between consecutive epochs.

What “contexts” mean
- During discovery we freeze a particular board position (taken from the sequential dataset) and feed its CNN feature vector into the GRU. Each sampled position is a **context**.
- Context indices are stable across runs: e.g. context 0 might be an empty board, context 4 a mid-game with a tactical threat, etc. See `contexts_metadata.json` for per-context features (piece counts, immediate wins, …).
- Comparing contexts shows how the GRU reacts to different strategic situations—opening, forcing win, defensive posture, etc.—without recomputing the entire dataset.

How to read the plots
- **Classification counts**
  - Rising stable bars ⇒ the network is carving out more attractors for that context (richer internal model).
  - Persistent unstable bars ⇒ the hidden dynamics are still bifurcating; often happens early in training or when tolerance is loose.
- **Spectral radius**
  - Values ≪ 1 mean the attractor is strongly stable (fast convergence).
  - Values creeping toward 1 indicate a “soft” attractor—the GRU retains information longer but is more sensitive to perturbations.
- **Attractor drift**
  - Drift ≈ 0 ⇒ the attractor has settled; no major internal reorganisation.
  - Spikes highlight epochs where the network reshapes its strategy for that context (often correlates with validation improvements or sudden overfitting).

Practical tips
- Use `--epoch-step` to sample fewer checkpoints when iterating takes too long (e.g. analyse every 3rd epoch first).
- Loosen `--tolerance` (e.g. `1e-2`) for quick exploratory runs, then tighten it for publication-quality numbers once you know interesting regions.
- Combine these plots with PHATE embeddings: contexts whose attractors move a lot usually correspond to regions where the hidden manifold reshapes.

Knobs (discovery)
- --max-contexts, --restarts, --max-iter, --tolerance
- --epoch-min/--epoch-max/--epoch-step to sub-sample checkpoints
- --device, --seed

Notes
- Fixed points are computed in GRU hidden space given a static CNN feature vector for a board context
- Deduplication removes nearby duplicates; residual threshold enforces quality

# Unified Visualization Suite

Overview
- Generates the standard set of plots (CNN/GRU weight trajectories, representations, summaries, activations, ablations) with consistent styling and annotations.
- Intended for quick end‑to‑end reporting and reproducible figures.

How to Generate
- `./wt.sh visualize --config <json>` (invokes `python -m src.visualize_trajectories` with presets)
- Direct CLI: `python -m src.visualize_trajectories` with `--viz-type` of `all|cnn|gru|boards|summary|joint|ablation-cnn|ablation-gru|activations`.

Key figures (by viz-type)
- cnn_trajectory.png: PHATE embedding of CNN weight snapshots; points colored by epoch with start/end markers and min/final val-loss markers if history is present
- gru_trajectory.png: PHATE embedding of GRU weight snapshots; same annotations
- board_representations.png: PHATE embedding of GRU hidden vectors for random boards at latest checkpoint; color encodes board/sample index
- summary.png: 2×2 panel with CNN and GRU trajectories plus loss curves (train/val and value/policy)
- cnn_gru_joint.png (joint): Shared PHATE embedding overlaying CNN and GRU trajectories for the same run; min/final loss markers drawn as filled (CNN) vs outline (GRU)
- ablation_{cnn,gru}_trajectories.png (+ optional .gif): Multi-run overlay comparing trajectories across runs for the selected component; labels annotate epochs periodically
- activations/activation_###.png: Grad-CAM heatmaps showing which board cells drive CNN activations for policy/value; labeled with focused move (and its probability) and predicted value

Axes/encodings
- PHATE 1/2 are non-linear embedding axes; only relative positions and paths matter
- Colorbar encodes epoch progression
- Lines show the chronological path; green circle = start; orange diamond = min val loss; red star = final val loss

Common options
- --checkpoint-dir or --ablation-dirs (2+ for ablations)
- --epoch-min/--epoch-max/--epoch-step: subsample checkpoints
- --phate-n-pca: cap dimensionality before PHATE for very high-D weights (auto when large)
- --phate-knn/--phate-t/--phate-decay: PHATE controls
- --ablation-center [none|anchor|normalize]: post alignment for multi-run overlays
- --activation-target [policy|value], --activation-move, --activation-max-examples: Grad-CAM specifics

Reading the plots
What to look for
- **CNN / GRU trajectories**
  - Colour progression (early dark → late bright) tracks checkpoint order.
  - Smooth arcs = steady optimisation; zig-zags/bends = LR schedule changes, regularisation kicks, or overfitting.
  - Compare the total path length between CNN and GRU panels: more movement means more parameter churn in that block.
- **Summary panel**
  - Correlate loss curves with trajectory marks (green circle = start, orange diamond = min val loss, red star = final). If the trajectory keeps moving far after min val loss, the model may be drifting into worse generalisation.
- **Joint plot**
  - Overlay reveals whether CNN/GRU evolve in sync (paths overlapping) or in phases (one moves while the other stays still).
- **Ablation trajectories**
  - Use identical colouring/annotation to compare multiple runs; if paths collapse onto each other the ablations are redundant.
- **Board representations**
  - Clustering by sample index indicates the GRU groups similar positions; diffuse clouds hint at entangled representations.
- **Activation maps**
  - As with the dedicated activations doc: bright cells = influential board regions; ensure they align with tactical expectations.

Tips
- Use `--epoch-step` to reduce PHATE compute for long runs.
- `--ablation-center normalize` recentres trajectories, making cross-run comparisons easier when absolute positions differ.
- When Grad-CAM is noisy, set `--activation-max-examples` to a small value and verify a few boards manually.

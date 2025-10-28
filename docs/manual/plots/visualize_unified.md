# Unified visualization suite

Produced by: `./wt.sh visualize --config <json>` → runs `python -m src.visualize_trajectories` with presets and writes a markdown summary.
See also: direct CLI `python -m src.visualize_trajectories` (viz-type: all|cnn|gru|boards|summary|joint|ablation-cnn|ablation-gru|activations).

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
- Smooth, monotone paths often align with steady training; sharp bends suggest phase shifts
- Broad CNN movement with compact GRU can indicate feature extractor churn with stable memory; the reverse suggests recurrent dynamics shifting
- Joint plots help compare CNN vs GRU temporal alignment
- Activation maps: brighter cells signal spatial regions influencing the prediction focus (policy logit or value)

# CKA similarity

Produced by: `./wt.sh cka [--representation gru|cnn]`
Backed by: `scripts/compute_cka_similarity.py`

What it does
- For chosen epochs, loads all 9 models, extracts either GRU hidden vectors or CNN feature maps on a fixed set of test boards, and computes linear CKA pairwise across models. Emits per-epoch heatmaps, clustered heatmaps, evolution curves, and CSV matrices. Optional animated heatmap across epochs.

Outputs (under `visualizations/<representation>/`)
- cka_<rep>_similarity_epoch_<E>.png: 9×9 heatmap (0..1), annotated per-cell
- cka_<rep>_clustered_epoch_<E>.png: Hierarchically clustered heatmap with dendrogram (requires SciPy)
- cka_<rep>_evolution.png: Line plots of selected model pairs across epochs
- cka_<rep>_heatmap_animation.{gif|mp4}: Optional animated heatmap (if --animate)
- cka_<rep>_similarity_epoch_<E>.csv: Numeric matrix for the heatmap

Axes/encodings
- Heatmaps: rows/cols are model ids (k3_c{channels}_gru{hidden}); color encodes CKA similarity
- Evolution: x-axis epoch; y-axis CKA similarity (0..1) for a set of illustrative pairs

Key options
- --epochs E1 E2 ... or --epoch-step to auto-generate [3,3+step,...,99,100]
- --representation [gru|cnn]
- --num-boards [default 64], --seed
- --device [cpu|cuda]
- --animate, --animate-fps, --animate-format [gif|mp4]

Reading the plots
- **Heatmaps**
  - Values in [0, 1]; 1 = identical representations on the sampled boards, 0 = orthogonal.
  - Block structure reveals factor effects: e.g. a GRU-size block with high CKA means different channel counts behave similarly when GRU size matches.
  - Off-diagonal lows highlight architectural combos that learn genuinely different features.
- **Clustered view**
  - Dendrogram groups models with similar representations; use it to summarise “families” of solutions.
  - Large branch distance between two leaves = representations diverge strongly.
- **Evolution plot**
  - Track specific pairs (e.g., best baseline vs ablation). Rising curves indicate convergence during training; falling curves show divergence.
- **Animation**
  - Helps spot phase changes: sudden brighten/dim blocks often align with architectural turning points (e.g., post-min-loss fine-tuning).

Tips
- Compare GRU vs CNN CKA: GRU similarity reflects memory dynamics; CNN similarity focuses on spatial feature extractors.
- Use --num-boards to balance compute vs stability—more boards smooths the estimate.
- Keep epochs aligned with checkpoints from `save_every_3`; for a quick scan, use `--epoch-step 9` to capture 3–30–60–90–100.

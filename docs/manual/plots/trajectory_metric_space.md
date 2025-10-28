# Metric-space trajectory embeddings

Produced by: `./wt.sh trajectory-embedding [--method umap|tsne|phate]`
Backed by: `scripts/visualize_trajectory_embedding.py`

What it does
- Loads per-run metrics CSVs and training histories, builds an architecture-agnostic feature vector per checkpoint, standardizes features, then embeds all checkpoints from all 9 models into a shared 2D space.

Inputs
- diagnostics/trajectory_analysis/k3_c{channels}_gru{hidden}_metrics.csv
- checkpoints/save_every_3/k3_c{channels}_gru{hidden}/training_history.json

Outputs
- trajectory_embedding_all.png: All models overlaid; color is epoch, marker encodes GRU size (o,s,^), edge color encodes channels
- trajectory_embedding_by_gru.png: 1×3 facet by GRU size; lines/points per channel; colorbar = epoch
- trajectory_embedding_3d.png: Optional 3D embedding when available (PCA fallback)

Axes/encodings
- UMAP/t-SNE/PHATE component 1/2
- Point size scales with epoch to emphasize later training
- Model labels annotate start/end; facets show within-GRU channel effects

Features used (subset if missing)
- epoch, weight_norm, step_norm, step_cosine, relative_step
- repr_total_variance, repr_top1..4_ratio
- train_loss, val_loss, train_val_gap

Reading the plots
Interpreting the panels
- **All models overlay**
  - Colour = epoch (darker→lighter). Use it to spot which architectures lag or race ahead in feature space.
  - Marker shapes separate GRU sizes; edge colour encodes CNN channels. Look for grouped paths to see main effects.
- **Faceted by GRU**
  - Within each subplot, directly compare CNN capacities while holding GRU fixed. If paths align, CNN width has little effect on these metrics; divergence means the CNN is driving behaviour.
- **3D rendering**
  - Adds depth for non-linear methods; rotate in an external viewer if you need to inspect crossings.

Reading patterns
- **Compact clusters** → models share similar training dynamics; good when you expect invariance.
- **Outliers** → architecture-specific behaviour (e.g. GRU128 taking a long exploratory loop, signalling overfitting risk).
- **Loops/backtracking** → validation dips or LR schedule kicks; cross-reference with loss curves.
- **Straight-shot trajectories** → smooth convergence; often the best generalisers.

Tips
- You can limit epochs with `--epoch-min/--epoch-max/--epoch-step` to focus on salient phases (e.g. early vs late training).
- Try alternative metrics (add/remove columns in `create_feature_matrix`) when experimenting; the script auto-drops missing ones.
- Combine with `visualizations/gru_observability/` plots to explain why a particular trajectory segment behaves strangely.

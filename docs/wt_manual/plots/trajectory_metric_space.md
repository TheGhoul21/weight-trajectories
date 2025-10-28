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
- Clusters indicate similar training dynamics across models
- Trajectories that drift far from others may reflect capacity-induced overfitting regimes
- Within a GRU panel, diverging channel trajectories quantify CNN contribution holding GRU fixed

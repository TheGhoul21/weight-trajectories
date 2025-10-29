# wt trajectory-embedding

Embed per-checkpoint trajectories into a common metric space (UMAP/t-SNE/PHATE) using architecture-agnostic features.

Maps to: `scripts/visualize_trajectory_embedding.py`

Options and defaults
- --metrics-dir [path, default diagnostics/trajectory_analysis]
- --checkpoint-dir [path, default checkpoints/save_every_3]
- --output-dir [path, default visualizations]
- --method [umap|tsne|phate, default umap]
- --n-neighbors [int, default 15]
- --min-dist [float, default 0.1]

Reads
- Per-run metrics CSVs and training history to build a feature matrix.

Writes (under `<output-dir>`)
- `trajectory_embedding_all.png` color by epoch, per-model tracks
- `trajectory_embedding_by_gru.png` faceted by GRU size
- `trajectory_embedding_3d.png` optional 3D projection

Explainers: [Metric-space trajectory embeddings](manual/plots/trajectory_metric_space)

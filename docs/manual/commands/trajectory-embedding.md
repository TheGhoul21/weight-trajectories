# wt trajectory-embedding

Embed per-checkpoint trajectories into a common metric space (UMAP/t-SNE/PHATE/T‑PHATE) using architecture-agnostic features.

Maps to: `scripts/visualize_trajectory_embedding.py`

Options and defaults
- --metrics-dir [path, default diagnostics/trajectory_analysis]
- --checkpoint-dir [path, default checkpoints/save_every_3]
- --output-dir [path, default visualizations]
- --method [umap|tsne|phate|tphate, default umap]
- --n-neighbors [int, default 15]
- --min-dist [float, default 0.1]
- --time-alpha [float, default 3.0] (only for `--method tphate`) — scales the epoch feature to bias temporal continuity
- --dims [2|3, default 2] — compute 3D embeddings when set to 3
- --animate-3d — save a rotating 3D GIF (camera orbit)
- --anim-frames [int, default 180] — number of frames for the GIF
- --anim-seconds [float, default 12.0] — duration of the GIF in seconds
- --plotly-html [path] — write an interactive 3D HTML (drag to orbit/zoom)

Reads
- Per-run metrics CSVs and training history to build a feature matrix.

Writes (under `<output-dir>`)
- `trajectory_embedding_all.png` color by epoch, per-model tracks
- `trajectory_embedding_by_gru.png` faceted by GRU size
- `trajectory_embedding_3d.png` static 3D projection (when `--dims 3` or any 3D option is set)
- `trajectory_embedding_3d_rotate.gif` rotating 3D animation (when `--animate-3d`)
- `trajectory_embedding_3d.html` interactive 3D Plotly (when `--plotly-html PATH`)

Explainers: [Metric-space trajectory embeddings](../plots/trajectory_metric_space.md)

Notes
- T‑PHATE in this command is the metric-space variant: it emphasizes chronological smoothness by scaling the `epoch` feature prior to PHATE.
- For raw weight/representation T‑PHATE with delay embeddings and temporal kernel blending, see the unified visualizer: `./wt.sh visualize` with `--viz-type [all|cnn|gru|joint|ablation-*|temporal]` and the `--t-phate*` flags.

### Examples

2D (default UMAP):
```bash
./wt.sh trajectory-embedding --metrics-dir diagnostics/trajectory_analysis \
	--checkpoint-dir checkpoints/save_every_1 --output-dir visualizations/metric_space
```

PHATE 3D + rotating GIF + interactive HTML:
```bash
./wt.sh trajectory-embedding \
	--metrics-dir diagnostics/trajectory_analysis \
	--checkpoint-dir checkpoints/save_every_1 \
	--output-dir visualizations/metric_space \
	--method phate --n-neighbors 10 \
	--dims 3 --animate-3d --anim-frames 180 --anim-seconds 12 \
	--plotly-html visualizations/metric_space/trajectory_embedding_3d.html
```

Shortcuts (aliases in `wt.sh`):
```bash
./wt.sh trajectory-embedding-3d --metrics-dir diagnostics/trajectory_analysis --output-dir visualizations/metric_space
./wt.sh trajectory-embedding-interactive --metrics-dir diagnostics/trajectory_analysis --output-dir visualizations/metric_space
```

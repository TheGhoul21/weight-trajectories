# Weight and representation embeddings

Produced by: `./wt.sh embeddings` → `scripts/analyze_weight_embeddings.py`

Outputs per run (under `visualizations/simple_embeddings/<run>/`)
- cnn_{pca,tsne,umap,phate}.png: 2D embeddings of CNN weight snapshots
- gru_{pca,tsne,umap,phate}.png: 2D embeddings of GRU weight snapshots
- repr_{pca,tsne,umap,phate}.png: 2D embeddings of GRU hidden vectors (enable with --board-representations)
- Optional CSVs with coordinates if `--export-csv` is set
- Optional per-run animations `<component>_<method>_anim.gif` with checkpoint reveal

Comparisons (under `visualizations/simple_embeddings/comparisons/`)
- <component>_<method>_comparison.png: Overlays embeddings across all selected runs; optional GIF

Axes/encodings
- Component 1/2 of the chosen embedding method
- Colorbar encodes epoch for weight trajectories; start (green circle), end (red star)
- Representation embeddings color by sample index

Key options
- --checkpoint-dirs ... (1+; multiple enables comparisons)
- --component [cnn|gru|all]
- --methods [pca tsne umap phate]
- --epoch-min/--epoch-max/--epoch-step
- --annotate to label epochs periodically
- --board-representations with --board-source [random|dataset], --board-count, --board-dataset
- --compare to produce overlays; --animate/--animate-fps for GIFs

Reading the plots
- PCA often captures coarse drift; t-SNE/UMAP/PHATE emphasize local vs global structure
- Long, smooth arcs with increasing epoch indicators show progressive learning; tight loops or reversals can indicate instability
- Representation plots reveal clustering of board positions; separability hints at feature encoding strength

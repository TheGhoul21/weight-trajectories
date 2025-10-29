# Getting Started

This manual doubles as a book: you can learn the core ideas while running the code. Start with one of three common paths and follow the links for details.

## 1) Train a Model

- Goal: produce checkpoints for analysis.
- Run: `./wt.sh dataset flat --test-run` then `./wt.sh train --data data/connect4_test.pt --epochs 30`
- Next: inspect outputs with `./wt.sh metrics` and `./wt.sh visualize`.
- Learn more:
  - Commands: [dataset](../manual/commands/dataset), [train](../manual/commands/train)
  - Plots: [Checkpoint Metrics CSV](../manual/plots/checkpoint_metrics_csv), [Unified Visualization Suite](../manual/plots/visualize_unified)

## 2) Analyze GRU Dynamics

- Goal: understand memory, representations, and mechanisms.
- Run: `./wt.sh observability extract` then `./wt.sh observability analyze`
- Optional: `./wt.sh observability mi`, `./wt.sh observability fixed`, `./wt.sh observability evolve`
- Learn more:
  - Plots: [GRU Observability](../manual/plots/gru_observability), [GRU Mutual Information](../manual/plots/gru_mutual_info), [Fixed Points](../manual/plots/fixed_points)
  - Scientific: [GRU Observability (Intuition)](../scientific/gru_observability_literature), [Theoretical Foundations](../scientific/theoretical_foundations)

## 3) Compare Architectures

- Goal: see how models differ across capacities.
- Run: `./wt.sh train-all`, then `./wt.sh trajectory-embedding`, and `./wt.sh cka --representation gru`
- Learn more:
  - Plots: [Metric-Space Trajectories](../manual/plots/trajectory_metric_space), [CKA Similarity](../manual/plots/cka)
  - Theory: [Weight Trajectory Theory](../scientific/weight_embeddings_theory)

## Libraries and Methods

- Methods reference: [Methods](../reference/methods) (CKA, MI, PHATE/UMAP/PCA, Grad‑CAM; with links to scikit‑learn, umap-learn, phate, PyTorch).
- Examples:
  - Mutual information via scikit‑learn: `sklearn.feature_selection.mutual_info_classif` / `mutual_info_regression`
  - Probes via scikit‑learn: `sklearn.linear_model.LogisticRegression` (+ `sklearn.preprocessing.StandardScaler`)
  - Embeddings: `umap.UMAP`, `phate.PHATE`, `sklearn.manifold.TSNE`, `sklearn.decomposition.PCA`

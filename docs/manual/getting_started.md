# Getting Started

This manual doubles as a book: you can learn the core ideas while running the code. Start with one of three common paths and follow the links for details.

## 1) Train a Model

- Goal: produce checkpoints for analysis.
- Run: `./wt.sh dataset flat --test-run` then `./wt.sh train --data data/connect4_test.pt --epochs 30`
- Next: inspect outputs with `./wt.sh metrics` and `./wt.sh visualize`.
- Learn more:
  - Commands: [dataset](commands/dataset.md), [train](commands/train.md)
  - Plots: [Checkpoint Metrics CSV](plots/checkpoint_metrics_csv.md), [Unified Visualization Suite](plots/visualize_unified.md)

## 2) Analyze GRU Dynamics

- Goal: understand memory, representations, and mechanisms.
- Run: `./wt.sh observability extract` then `./wt.sh observability analyze`
- Optional: `./wt.sh observability mi`, `./wt.sh observability fixed`, `./wt.sh observability evolve`
- Learn more:
  - Plots: [GRU Observability](plots/gru_observability.md), [GRU Mutual Information](plots/gru_mutual_info.md), [Fixed Points](plots/fixed_points.md)
  - Scientific: [GRU Observability (Intuition)](../scientific/gru_observability_literature.md), [Theoretical Foundations](../scientific/theoretical_foundations.md)

## 3) Compare Architectures

- Goal: see how models differ across capacities.
- Run: `./wt.sh train-all`, then `./wt.sh trajectory-embedding`, and `./wt.sh cka --representation gru`
- Learn more:
  - Plots: [Metric-Space Trajectories](plots/trajectory_metric_space.md), [CKA Similarity](plots/cka.md)
  - Theory: [Weight Trajectory Theory](../scientific/weight_embeddings_theory.md)

## Libraries and Methods

- Methods reference: [Methods](../reference/methods.md) (CKA, MI, PHATE/UMAP/PCA, Grad‑CAM; with links to scikit‑learn, umap-learn, phate, PyTorch).
- Examples:
  - Mutual information via scikit‑learn: `sklearn.feature_selection.mutual_info_classif` / `mutual_info_regression`
  - Probes via scikit‑learn: `sklearn.linear_model.LogisticRegression` (+ `sklearn.preprocessing.StandardScaler`)
  - Embeddings: `umap.UMAP`, `phate.PHATE`, `sklearn.manifold.TSNE`, `sklearn.decomposition.PCA`

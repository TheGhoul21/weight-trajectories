# Methods Reference

Concise reference for algorithms and measures used across the manual, with links to libraries and primary sources. Use this page to avoid repetition and for quick lookups while reading plot pages or running commands.

## Dimensionality Reduction

- PCA: Linear projection maximizing variance. Library: scikit‑learn (`sklearn.decomposition.PCA`). Docs: https://scikit-learn.org/
- t‑SNE: Stochastic neighbor embedding emphasizing local neighborhoods. Library: scikit‑learn (`sklearn.manifold.TSNE`). Paper: van der Maaten & Hinton (2008).
- UMAP: Uniform Manifold Approximation and Projection; topology‑preserving with good global structure. Library: `umap-learn` (`umap.UMAP`). Docs: https://umap-learn.readthedocs.io/
- PHATE: Diffusion‑based embedding preserving trajectories. Library: `phate` (`phate.PHATE`). Paper: Moon et al. (2019). Docs: https://phate.readthedocs.io/
- T‑PHATE (temporal PHATE): Extends PHATE for autocorrelated time series via delay embeddings and temporal affinity shaping. Paper: Rübel et al. (2023).
  - Delay embedding: concatenate `[x_t, x_{t-τ}, x_{t-2τ}, …]` to encode short‑term history.
  - Temporal kernel blending: blend Euclidean feature distance with a scaled time distance inside sequences; cross‑sequence pairs penalized.
  - Provided here in two forms:
    - Feature augmentation: `--t-phate --t-phate-alpha A [--t-phate-delay τ --t-phate-lags L]` (simple, robust)
    - Precomputed blended distances: `--t-phate-kernel --t-phate-kernel-alpha A --t-phate-kernel-tau τ` (more faithful)
  - Metric‑space variant: `scripts/visualize_trajectory_embedding.py --method tphate --time-alpha A` (scales epoch feature).

## Similarity and Alignment

- Linear CKA: Centered Kernel Alignment using linear kernels on representations. Implementation: custom (NumPy/SciPy), alternately `cka` community libs. Reference: Kornblith et al. (2019). See Plot → CKA Similarity for formula.

## Information Measures

- Mutual Information (classification): scikit‑learn `sklearn.feature_selection.mutual_info_classif`. Docs: https://scikit-learn.org/ Note: nonparametric estimator robust for mixed distributions.
- Mutual Information (regression): scikit‑learn `sklearn.feature_selection.mutual_info_regression`.
- Alternative estimators: KSG (Kraskov et al., 2004) via `NPEET` or `dit` (not used here by default).

## Probing

- Logistic Regression (binary probes): scikit‑learn `sklearn.linear_model.LogisticRegression`, solver `lbfgs`, `max_iter` typically ≥ 5000. Standardize inputs with `sklearn.preprocessing.StandardScaler`.

## Visualization

- Grad‑CAM: Gradient‑weighted class activation mapping. Implemented in‑repo (PyTorch hooks), or via `pytorch-grad-cam`. Paper: Selvaraju et al. (2017).
- Matplotlib/Seaborn: plotting; `matplotlib`, `seaborn`.

## Sampling and Data Handling

- Reservoir Sampling: Vitter’s Algorithm R for uniform sampling from streams (used for hidden states). Implemented in‑repo.
- Standardization: `sklearn.preprocessing.StandardScaler` for zero‑mean/unit‑variance features before linear models and embeddings.
- Pandas: CSV I/O and data wrangling (`pandas`).

## Numerical and Scientific Libraries

- PyTorch: model definition, checkpoints, tensor ops (`torch`).
- NumPy: numerical arrays (`numpy`).
- SciPy: linear algebra and utilities (`scipy`).

## Cross‑References

- Plot docs: [CKA Similarity](../manual/plots/cka.md), [Metric‑Space Trajectory Embeddings](../manual/plots/trajectory_metric_space.md), [GRU Observability](../manual/plots/gru_observability.md), [Activations](../manual/plots/activations.md).
- Scientific background: [Theoretical Foundations](../scientific/theoretical_foundations.md), [GRU Observability](../scientific/gru_observability_literature.md), [Weight Trajectory Theory](../scientific/weight_embeddings_theory.md).

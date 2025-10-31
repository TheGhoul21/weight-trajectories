# Dimensionality Reduction for Representation Analysis

A comprehensive guide to reducing high-dimensional neural network representations to interpretable low-dimensional visualizations and embeddings.

---

## Overview

**Problem**: Neural networks use high-dimensional hidden states (128D, 512D, 2048D+) that humans cannot directly visualize or understand.

**Solution**: Dimensionality reduction projects representations into 2D or 3D while preserving important structure.

**Applications**:
- Visualize learned representations
- Identify clusters and trajectories
- Discover emergent structure
- Compare across models or layers

This guide covers both linear (PCA, CCA) and nonlinear (t-SNE, UMAP, PHATE) methods with interpretability-focused examples.

---

## Why Dimensionality Reduction Matters

### Visualization

**Human visual system**: Excellent at pattern recognition in 2D/3D, useless in 512D

**Example**: Hidden states during sequence processing - can't plot 512D trajectory, but can plot 2D projection

### Structure Discovery

**Clustering**: Do representations naturally group?

**Trajectories**: How do representations evolve over time or depth?

**Geometry**: Are concepts organized linearly, hierarchically, or in complex manifolds?

### Hypothesis Testing

**Before**: "I think the model learns a hierarchy"

**After dimensionality reduction**: "Plot shows clear hierarchical organization (or doesn't)"

---

## Linear Methods

### Principal Component Analysis (PCA)

**Idea**: Find directions of maximum variance

**Math**: Eigendecomposition of covariance matrix

**Properties**:
- Linear projection
- Preserves global distances
- Fast (O(n²d) for n samples, d dimensions)
- Interpretable components

#### Implementation

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def apply_pca(representations, n_components=2):
    """Apply PCA to high-dimensional representations.

    Args:
        representations: (n_samples, n_features) array
        n_components: Number of principal components

    Returns:
        projected: (n_samples, n_components) low-D data
        pca_model: Fitted PCA object
        variance_explained: Fraction of variance per component
    """
    # Fit PCA
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(representations)

    # Variance explained
    var_explained = pca.explained_variance_ratio_

    print(f"Variance explained by {n_components} components: {var_explained.sum()*100:.1f}%")
    print(f"Per component: {var_explained}")

    return projected, pca, var_explained

# Example: Reduce 512D hidden states to 2D
hidden_states = np.random.randn(1000, 512)  # Replace with actual data
projected_2d, pca_model, var_exp = apply_pca(hidden_states, n_components=2)

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(projected_2d[:, 0], projected_2d[:, 1], alpha=0.5)
plt.xlabel(f'PC1 ({var_exp[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({var_exp[1]*100:.1f}% variance)')
plt.title('PCA Projection of Hidden States')
plt.show()
```

#### Choosing Number of Components

**Scree plot**: Variance explained vs number of components

```python
def plot_scree(representations, max_components=50):
    """Visualize variance explained by components."""
    pca = PCA(n_components=min(max_components, representations.shape[1]))
    pca.fit(representations)

    var_exp = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(var_exp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Individual variance
    ax1.plot(range(1, len(var_exp) + 1), var_exp, 'o-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained')
    ax1.set_title('Scree Plot')
    ax1.grid(True)

    # Cumulative variance
    ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Find elbow
    n_for_95 = np.argmax(cumsum_var >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_for_95}")

    return var_exp, cumsum_var

# Usage
plot_scree(hidden_states, max_components=50)
```

#### Interpreting Principal Components

**Components as directions in representation space**

```python
def interpret_principal_components(pca_model, feature_names=None, n_top_features=10):
    """Identify which original features contribute to each PC.

    Useful when features have semantic meaning (e.g., specific neurons).
    """
    components = pca_model.components_  # (n_components, n_features)

    for i, component in enumerate(components):
        print(f"\nPrincipal Component {i+1}:")

        # Top positive contributors
        top_pos_idx = np.argsort(component)[-n_top_features:][::-1]
        print("Top positive contributors:")
        for idx in top_pos_idx:
            feature = feature_names[idx] if feature_names else f"Feature {idx}"
            print(f"  {feature}: {component[idx]:.3f}")

        # Top negative contributors
        top_neg_idx = np.argsort(component)[:n_top_features]
        print("Top negative contributors:")
        for idx in top_neg_idx:
            feature = feature_names[idx] if feature_names else f"Feature {idx}"
            print(f"  {feature}: {component[idx]:.3f}")

# Example: If you have neuron labels
# feature_names = ['neuron_0', 'neuron_1', ..., 'neuron_511']
# interpret_principal_components(pca_model, feature_names)
```

#### Visualizing Trajectories with PCA

**Sequence processing**: How representations evolve

```python
def visualize_trajectory_pca(sequence_representations, labels=None):
    """Visualize trajectory through representation space.

    Args:
        sequence_representations: (n_timesteps, n_features) or
                                  list of (n_features,) arrays
        labels: Optional timestep labels
    """
    # Stack if list
    if isinstance(sequence_representations, list):
        sequence_representations = np.array(sequence_representations)

    # Apply PCA
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(sequence_representations)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'o-', alpha=0.6, markersize=4)

    # Mark start and end
    plt.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1],
               s=200, c='green', marker='*', label='Start', zorder=5)
    plt.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1],
               s=200, c='red', marker='*', label='End', zorder=5)

    # Annotate timesteps if labels provided
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (trajectory_2d[i, 0], trajectory_2d[i, 1]),
                        fontsize=8, alpha=0.7)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Trajectory in Representation Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: RNN processing sequence
# hidden_states_over_time = [h_t for t in range(sequence_length)]
# visualize_trajectory_pca(hidden_states_over_time, labels=range(len(hidden_states_over_time)))
```

### Canonical Correlation Analysis (CCA)

**Use case**: Compare representations from two different sources (e.g., two layers, two models)

**Idea**: Find correlated dimensions between two representation spaces

```python
from sklearn.cross_decomposition import CCA

def compare_representations_cca(repr_A, repr_B, n_components=10):
    """Find correlated dimensions between two representation sets.

    Args:
        repr_A: (n_samples, n_features_A)
        repr_B: (n_samples, n_features_B)
        n_components: Number of canonical dimensions

    Returns:
        correlations: Canonical correlations
        A_canonical, B_canonical: Projected representations
    """
    # Fit CCA
    cca = CCA(n_components=n_components)
    A_canonical, B_canonical = cca.fit_transform(repr_A, repr_B)

    # Compute correlations per dimension
    correlations = [np.corrcoef(A_canonical[:, i], B_canonical[:, i])[0, 1]
                   for i in range(n_components)]

    print("Canonical correlations:")
    for i, corr in enumerate(correlations):
        print(f"  Dimension {i+1}: {corr:.3f}")

    # Plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(correlations) + 1), correlations)
    plt.xlabel('Canonical Dimension')
    plt.ylabel('Correlation')
    plt.title('Canonical Correlations')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(A_canonical[:, 0], B_canonical[:, 0], alpha=0.5)
    plt.xlabel('Representation A (Canonical Dim 1)')
    plt.ylabel('Representation B (Canonical Dim 1)')
    plt.title(f'Top Canonical Dimension (r={correlations[0]:.3f})')
    plt.plot([-3, 3], [-3, 3], 'r--', alpha=0.5)  # Diagonal reference
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return correlations, A_canonical, B_canonical

# Example: Compare two layers
# layer3_repr = extract_layer_representations(model, data, layer=3)
# layer5_repr = extract_layer_representations(model, data, layer=5)
# corrs, layer3_can, layer5_can = compare_representations_cca(layer3_repr, layer5_repr)
```

---

## Nonlinear Methods

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Idea**: Preserve local neighborhoods (nearby points stay nearby)

**Strengths**:
- Reveals clusters beautifully
- Handles complex nonlinear structure

**Limitations**:
- Stochastic (different runs give different results)
- Doesn't preserve global distances
- Slow for large datasets (O(n²) to O(n log n))
- Hyperparameters matter a lot

#### Implementation

```python
from sklearn.manifold import TSNE

def apply_tsne(representations, perplexity=30, n_iter=1000, random_state=42):
    """Apply t-SNE for visualization.

    Args:
        representations: (n_samples, n_features)
        perplexity: Balance local vs global structure (5-50 typical)
        n_iter: Number of optimization iterations
        random_state: For reproducibility

    Returns:
        embedding: (n_samples, 2) low-D coordinates
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )

    embedding = tsne.fit_transform(representations)

    print(f"Final KL divergence: {tsne.kl_divergence_:.3f}")

    return embedding

# Example with labeled data
def visualize_tsne(representations, labels=None, title="t-SNE Projection"):
    """Create t-SNE visualization with optional labels."""
    embedding = apply_tsne(representations)

    plt.figure(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       label=str(label), alpha=0.6, s=30)
        plt.legend()
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=30)

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return embedding

# Usage
# representations = extract_representations(model, data)
# labels = get_labels(data)  # e.g., class labels, clusters, etc.
# embedding = visualize_tsne(representations, labels)
```

#### Hyperparameter Tuning

**Perplexity**: Most important parameter

```python
def compare_perplexities(representations, perplexities=[5, 30, 50, 100], labels=None):
    """Visualize effect of perplexity parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, perp in enumerate(perplexities):
        print(f"Running t-SNE with perplexity={perp}...")
        embedding = apply_tsne(representations, perplexity=perp)

        ax = axes[i]
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          label=str(label), alpha=0.6, s=20)
            ax.legend(fontsize=8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20)

        ax.set_title(f'Perplexity = {perp}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.show()

# Guideline:
# - Small perplexity (5-10): Emphasizes local structure, more clusters
# - Medium perplexity (30-50): Balanced (default is usually good)
# - Large perplexity (50-100): Emphasizes global structure
```

#### Common Pitfalls with t-SNE

```python
def tsne_best_practices_demo():
    """Demonstrate t-SNE best practices."""

    print("t-SNE Best Practices:")
    print("=" * 50)

    print("\n1. DISTANCES:")
    print("   ✓ Cluster separation is meaningful")
    print("   ✗ Distances within/between clusters are NOT meaningful")
    print("   → Don't measure distances in t-SNE space!")

    print("\n2. CLUSTER SIZE:")
    print("   ✗ Cluster sizes don't reflect true proportions")
    print("   → t-SNE expands dense clusters and compresses sparse regions")

    print("\n3. MULTIPLE RUNS:")
    print("   ✓ Run with different random seeds")
    print("   → Check if structure is consistent")

    print("\n4. PREPROCESSING:")
    print("   ✓ Standardize features (zero mean, unit variance)")
    print("   ✓ Consider PCA first (50-100 dims) for very high-D data")

    print("\n5. PERPLEXITY:")
    print("   ✓ Try multiple values")
    print("   → Perplexity ~ number of close neighbors to preserve")
    print("   → Should be less than n_samples")

    print("\n6. INTERPRETATION:")
    print("   ✓ Look for clusters and separation")
    print("   ✗ Don't interpret axis directions")
    print("   ✗ Don't measure distances")

# Call to print guidelines
tsne_best_practices_demo()
```

### UMAP (Uniform Manifold Approximation and Projection)

**Advantages over t-SNE**:
- Preserves more global structure
- Faster (can handle larger datasets)
- Deterministic (same result each run with same seed)
- Can embed new points without refitting

**Use when**: Larger datasets, need consistency, want global structure

#### Implementation

```python
# Requires: pip install umap-learn
import umap

def apply_umap(representations, n_neighbors=15, min_dist=0.1, random_state=42):
    """Apply UMAP for dimensionality reduction.

    Args:
        representations: (n_samples, n_features)
        n_neighbors: Balance local vs global (5-50, default 15)
        min_dist: Minimum distance between points (0.0-1.0, default 0.1)
        random_state: For reproducibility

    Returns:
        embedding: (n_samples, 2) low-D coordinates
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )

    embedding = reducer.fit_transform(representations)

    return embedding, reducer

def visualize_umap(representations, labels=None, n_neighbors=15, min_dist=0.1):
    """Create UMAP visualization."""
    embedding, reducer = apply_umap(representations, n_neighbors, min_dist)

    plt.figure(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       label=str(label), alpha=0.6, s=30)
        plt.legend()
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=30)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})')
    plt.tight_layout()
    plt.show()

    return embedding, reducer

# Usage
# embedding, reducer = visualize_umap(representations, labels)
```

#### Embedding New Data

```python
def embed_new_data_umap(reducer, new_representations):
    """Project new data into existing UMAP space.

    Args:
        reducer: Fitted UMAP object
        new_representations: (n_new_samples, n_features)

    Returns:
        new_embedding: (n_new_samples, 2)
    """
    new_embedding = reducer.transform(new_representations)
    return new_embedding

# Example: Train on one dataset, project test set
# embedding_train, reducer = apply_umap(train_representations)
# embedding_test = embed_new_data_umap(reducer, test_representations)
```

#### Comparing UMAP Parameters

```python
def umap_parameter_sweep(representations, labels=None):
    """Visualize effect of UMAP hyperparameters."""
    n_neighbors_list = [5, 15, 50]
    min_dist_list = [0.0, 0.1, 0.5]

    fig, axes = plt.subplots(len(n_neighbors_list), len(min_dist_list),
                            figsize=(15, 15))

    for i, n_neighbors in enumerate(n_neighbors_list):
        for j, min_dist in enumerate(min_dist_list):
            print(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")

            embedding, _ = apply_umap(representations, n_neighbors, min_dist)

            ax = axes[i, j]
            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    ax.scatter(embedding[mask, 0], embedding[mask, 1],
                             alpha=0.6, s=10)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=10)

            ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# Usage
# umap_parameter_sweep(representations, labels)
```

### PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)

**Specialty**: Trajectory and time-series data

**Advantages**:
- Preserves both local and global structure
- Excellent for trajectories
- Denoisesdata
- Handles branching structures

**Use when**: Analyzing temporal dynamics, developmental trajectories, state transitions

#### Implementation

```python
# Requires: pip install phate
import phate

def apply_phate(representations, knn=5, decay=40, t='auto'):
    """Apply PHATE for trajectory visualization.

    Args:
        representations: (n_samples, n_features)
        knn: Number of nearest neighbors
        decay: Decay parameter for adaptive kernel
        t: Diffusion time (higher = more global)

    Returns:
        embedding: (n_samples, 2) low-D coordinates
    """
    phate_op = phate.PHATE(
        n_components=2,
        knn=knn,
        decay=decay,
        t=t,
        verbose=True
    )

    embedding = phate_op.fit_transform(representations)

    return embedding, phate_op

def visualize_trajectory_phate(trajectory_representations, timesteps=None):
    """Visualize trajectory in PHATE space.

    Args:
        trajectory_representations: (n_timesteps, n_features)
        timesteps: Optional time labels for coloring
    """
    embedding, _ = apply_phate(trajectory_representations)

    plt.figure(figsize=(10, 8))

    if timesteps is None:
        timesteps = np.arange(len(embedding))

    # Color by time
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                         c=timesteps, cmap='viridis',
                         s=50, alpha=0.6)
    plt.colorbar(scatter, label='Time')

    # Draw trajectory
    plt.plot(embedding[:, 0], embedding[:, 1], 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    plt.scatter(embedding[0, 0], embedding[0, 1],
               s=200, c='green', marker='*', label='Start', zorder=5, edgecolor='black')
    plt.scatter(embedding[-1, 0], embedding[-1, 1],
               s=200, c='red', marker='*', label='End', zorder=5, edgecolor='black')

    plt.xlabel('PHATE 1')
    plt.ylabel('PHATE 2')
    plt.title('Trajectory in PHATE Space')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return embedding

# Usage for RNN hidden state trajectories
# hidden_states = [model.forward(x_t)[1] for x_t in sequence]  # Extract hidden states
# embedding = visualize_trajectory_phate(np.array(hidden_states))
```

---

## Method Comparison

### When to Use Which Method

```python
def method_selection_guide():
    """Guide for choosing dimensionality reduction method."""

    guide = {
        "PCA": {
            "Best for": ["Global structure", "Quick exploration", "Preprocessing"],
            "Strengths": ["Fast", "Deterministic", "Interpretable components"],
            "Limitations": ["Only linear", "May miss clusters"],
            "Use when": "Need fast, interpretable, global view"
        },

        "t-SNE": {
            "Best for": ["Cluster visualization", "Local structure", "Publication figures"],
            "Strengths": ["Beautiful clusters", "Reveals nonlinear structure"],
            "Limitations": ["Stochastic", "Slow", "Doesn't preserve distances"],
            "Use when": "Want to discover/visualize clusters, n < 10000"
        },

        "UMAP": {
            "Best for": ["Large datasets", "General-purpose nonlinear reduction"],
            "Strengths": ["Fast", "Preserves global structure", "Can embed new data"],
            "Limitations": ["Less established than t-SNE", "Sensitive to parameters"],
            "Use when": "Need fast nonlinear method, n > 10000, or need to embed new data"
        },

        "PHATE": {
            "Best for": ["Trajectories", "Time series", "Developmental data"],
            "Strengths": ["Excellent for dynamics", "Denoises", "Handles branching"],
            "Limitations": ["Newer method", "Less well-known"],
            "Use when": "Analyzing temporal or sequential data"
        },

        "CCA": {
            "Best for": ["Comparing two representation spaces"],
            "Strengths": ["Finds shared structure", "Quantifies similarity"],
            "Limitations": ["Requires paired data", "Linear"],
            "Use when": "Comparing layers, models, or modalities"
        }
    }

    for method, info in guide.items():
        print(f"\n{'='*60}")
        print(f"{method}")
        print(f"{'='*60}")
        for key, value in info.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")

    return guide

# Print guide
method_selection_guide()
```

### Side-by-Side Comparison

```python
def compare_all_methods(representations, labels=None, sample_size=1000):
    """Apply and visualize all major methods side-by-side.

    Args:
        representations: (n_samples, n_features)
        labels: Optional (n_samples,) labels for coloring
        sample_size: Subsample for speed (t-SNE is slow)
    """
    # Subsample if needed
    if len(representations) > sample_size:
        indices = np.random.choice(len(representations), sample_size, replace=False)
        repr_subset = representations[indices]
        labels_subset = labels[indices] if labels is not None else None
    else:
        repr_subset = representations
        labels_subset = labels

    # Apply methods
    print("Computing PCA...")
    pca_embedding, _, _ = apply_pca(repr_subset, n_components=2)

    print("Computing t-SNE...")
    tsne_embedding = apply_tsne(repr_subset, perplexity=30)

    print("Computing UMAP...")
    umap_embedding, _ = apply_umap(repr_subset)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    embeddings = [pca_embedding, tsne_embedding, umap_embedding]
    titles = ['PCA', 't-SNE', 'UMAP']

    for ax, embedding, title in zip(axes, embeddings, titles):
        if labels_subset is not None:
            unique_labels = np.unique(labels_subset)
            for label in unique_labels:
                mask = labels_subset == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          label=str(label), alpha=0.6, s=20)
            ax.legend(fontsize=8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f'{title} 1')
        ax.set_ylabel(f'{title} 2')

    plt.tight_layout()
    plt.show()

    return {
        'pca': pca_embedding,
        'tsne': tsne_embedding,
        'umap': umap_embedding
    }

# Usage
# embeddings = compare_all_methods(representations, labels)
```

---

## Interpretability Applications

### Application 1: Layer-wise Representation Evolution

**Question**: How do representations change across network depth?

```python
def analyze_layer_representations(model, data, layer_indices):
    """Visualize how representations evolve through layers.

    Args:
        model: Neural network
        data: Input data
        layer_indices: Which layers to extract

    Returns:
        Dictionary of embeddings per layer
    """
    from collections import defaultdict

    # Extract representations from each layer
    layer_representations = defaultdict(list)

    for layer_idx in layer_indices:
        repr_layer = extract_layer_representations(model, data, layer_idx)
        layer_representations[layer_idx] = repr_layer

    # Apply UMAP to each
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(5*len(layer_indices), 5))
    if len(layer_indices) == 1:
        axes = [axes]

    embeddings = {}

    for i, layer_idx in enumerate(layer_indices):
        repr_layer = layer_representations[layer_idx]
        embedding, _ = apply_umap(repr_layer, n_neighbors=15)

        embeddings[layer_idx] = embedding

        # Visualize (color by label if available)
        labels = get_labels(data)
        unique_labels = np.unique(labels)

        ax = axes[i]
        for label in unique_labels:
            mask = labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      label=str(label), alpha=0.6, s=20)

        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(fontsize=8)

    plt.suptitle('Representation Evolution Across Layers', fontsize=16)
    plt.tight_layout()
    plt.show()

    return embeddings

# Example
# layer_embeddings = analyze_layer_representations(
#     model, data, layer_indices=[0, 3, 6, 9, 12]
# )
```

### Application 2: Concept Discovery

**Question**: Do representations naturally cluster by semantic concept?

```python
def discover_clusters(representations, n_clusters_range=range(2, 11)):
    """Find natural clusters in representations.

    Uses dimensionality reduction + clustering.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Reduce dimensions first (PCA for speed)
    pca = PCA(n_components=50)
    repr_reduced = pca.fit_transform(representations)

    # Try different numbers of clusters
    silhouette_scores = []
    inertias = []

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(repr_reduced)

        score = silhouette_score(repr_reduced, cluster_labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)

    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(n_clusters_range, silhouette_scores, 'o-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Analysis')
    ax1.grid(True)

    ax2.plot(n_clusters_range, inertias, 'o-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Elbow Method')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Best number of clusters (highest silhouette)
    best_n = list(n_clusters_range)[np.argmax(silhouette_scores)]
    print(f"Best number of clusters: {best_n}")

    # Final clustering with best n
    kmeans_final = KMeans(n_clusters=best_n, random_state=42)
    final_labels = kmeans_final.fit_predict(repr_reduced)

    # Visualize with UMAP
    embedding, _ = apply_umap(repr_reduced)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                         c=final_labels, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(f'Discovered Clusters (n={best_n})')
    plt.tight_layout()
    plt.show()

    return final_labels, best_n

# Usage
# cluster_labels, n_clusters = discover_clusters(representations)
```

### Application 3: Training Dynamics

**Question**: How do representations evolve during training?

```python
def visualize_training_dynamics(checkpoints_representations, checkpoint_steps):
    """Visualize how representations change during training.

    Args:
        checkpoints_representations: List of (n_samples, n_features) arrays
        checkpoint_steps: Training steps for each checkpoint
    """
    # Stack all checkpoints
    all_repr = np.vstack(checkpoints_representations)
    n_per_checkpoint = len(checkpoints_representations[0])

    # Apply UMAP to all jointly (ensures aligned embedding space)
    embedding, _ = apply_umap(all_repr)

    # Split back into checkpoints
    checkpoint_embeddings = []
    for i in range(len(checkpoints_representations)):
        start_idx = i * n_per_checkpoint
        end_idx = (i + 1) * n_per_checkpoint
        checkpoint_embeddings.append(embedding[start_idx:end_idx])

    # Visualize evolution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (emb, step) in enumerate(zip(checkpoint_embeddings, checkpoint_steps)):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.scatter(emb[:, 0], emb[:, 1], alpha=0.5, s=10)
        ax.set_title(f'Step {step}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    plt.suptitle('Representation Evolution During Training', fontsize=16)
    plt.tight_layout()
    plt.show()

    return checkpoint_embeddings

# Usage
# Collect checkpoints during training
# checkpoints = []
# for step in [0, 100, 500, 1000, 5000, 10000]:
#     repr = extract_representations(model_at_step[step], data)
#     checkpoints.append(repr)
# visualize_training_dynamics(checkpoints, [0, 100, 500, 1000, 5000, 10000])
```

---

## Best Practices

### Preprocessing

```python
def preprocess_for_dim_reduction(representations, method='pca'):
    """Preprocess representations before dimensionality reduction.

    Common preprocessing steps:
    1. Remove NaNs and Infs
    2. Standardize (zero mean, unit variance)
    3. Optional: PCA pre-reduction for very high-D data
    """
    # 1. Clean data
    representations = representations[~np.isnan(representations).any(axis=1)]
    representations = representations[~np.isinf(representations).any(axis=1)]

    # 2. Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    representations_scaled = scaler.fit_transform(representations)

    # 3. Optional PCA pre-reduction
    if method in ['tsne', 'umap'] and representations_scaled.shape[1] > 100:
        print(f"Pre-reducing from {representations_scaled.shape[1]}D to 50D with PCA...")
        pca = PCA(n_components=50)
        representations_scaled = pca.fit_transform(representations_scaled)

    return representations_scaled

# Always preprocess
# representations_clean = preprocess_for_dim_reduction(representations)
# embedding = apply_tsne(representations_clean)
```

### Validation

```python
def validate_embedding_quality(original_data, embedding, k=10):
    """Check if embedding preserves neighborhood structure.

    Computes k-NN overlap between original and embedded space.
    """
    from sklearn.neighbors import NearestNeighbors

    # k-NN in original space
    nn_original = NearestNeighbors(n_neighbors=k+1)
    nn_original.fit(original_data)
    _, indices_original = nn_original.kneighbors(original_data)

    # k-NN in embedded space
    nn_embedded = NearestNeighbors(n_neighbors=k+1)
    nn_embedded.fit(embedding)
    _, indices_embedded = nn_embedded.kneighbors(embedding)

    # Compute overlap
    overlaps = []
    for i in range(len(original_data)):
        neighbors_orig = set(indices_original[i, 1:])  # Exclude self
        neighbors_emb = set(indices_embedded[i, 1:])
        overlap = len(neighbors_orig & neighbors_emb) / k
        overlaps.append(overlap)

    mean_overlap = np.mean(overlaps)
    print(f"Mean k-NN overlap (k={k}): {mean_overlap:.3f}")
    print("Higher = better preservation of local structure")

    return mean_overlap

# Usage
# overlap = validate_embedding_quality(representations, embedding, k=10)
```

---

## Further Reading

**Comprehensive resources**:
- [Linear Algebra Essentials](../1_foundations/linear_algebra_essentials.md) - PCA, SVD, projections
- [Trajectory Analysis](../dynamical_analysis/trajectory_analysis.md) - Using embeddings for dynamics
- [Similarity Metrics](similarity_metrics.md) - Quantifying representation similarity

**Papers**:
- van der Maaten & Hinton (2008): "Visualizing Data using t-SNE"
- McInnes et al. (2018): "UMAP: Uniform Manifold Approximation and Projection"
- Moon et al. (2019): "Visualizing structure and transitions in high-dimensional biological data"

**Full bibliography**: [References](../../references/bibliography.md)

---

**Return to**: [Methods](../README.md) | [Main Handbook](../../0_start_here/README.md)

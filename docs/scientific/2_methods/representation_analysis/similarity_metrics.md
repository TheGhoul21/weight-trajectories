# Representation Similarity Metrics

A comprehensive guide to measuring and comparing neural network representations across layers, models, training steps, and architectures.

---

## Overview

**Problem**: How do we quantify whether two sets of representations are similar?

**Applications**:
- Compare representations across layers
- Track training dynamics
- Compare different models
- Identify when representations converge
- Measure architectural differences

**Challenge**: Representations can be similar in meaningful ways even if raw vectors differ (e.g., rotated, scaled, or using different numbers of dimensions).

This guide covers metrics from simple distance-based measures to sophisticated invariant comparisons.

---

## Why Representation Similarity Matters

### Model Comparison

**Question**: Do transformer and CNN learn similar representations?

**Method**: Extract representations from both, measure similarity

**Insight**: High similarity suggests convergence on similar features despite different architectures

### Training Dynamics

**Question**: When do representations stabilize during training?

**Method**: Compare representations at different checkpoints

**Insight**: Similarity plateaus indicate convergence

### Layer Analysis

**Question**: How much do representations change between layers?

**Method**: Compare adjacent layers

**Insight**: Large changes indicate significant transformation; small changes suggest redundancy

### Transfer Learning

**Question**: Which pre-trained model is closest to target task?

**Method**: Measure similarity between pre-trained and fine-tuned representations

**Insight**: Higher similarity suggests better initialization

---

## Distance-Based Metrics

### Euclidean Distance

**Simplest metric**: L2 distance between representation vectors

```python
import numpy as np

def euclidean_distance(repr1, repr2):
    """Compute Euclidean distance between single representations.

    Args:
        repr1, repr2: (n_features,) arrays

    Returns:
        distance: Scalar distance
    """
    return np.linalg.norm(repr1 - repr2)

def average_euclidean_distance(reprs_A, reprs_B):
    """Average distance between paired representations.

    Args:
        reprs_A, reprs_B: (n_samples, n_features) arrays

    Returns:
        mean_distance: Average pairwise distance
    """
    assert reprs_A.shape == reprs_B.shape, "Representations must have same shape"

    distances = np.linalg.norm(reprs_A - reprs_B, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    print(f"Mean Euclidean distance: {mean_dist:.3f} ± {std_dist:.3f}")

    return mean_dist

# Example
# layer1_repr = extract_representations(model, data, layer=1)
# layer2_repr = extract_representations(model, data, layer=2)
# dist = average_euclidean_distance(layer1_repr, layer2_repr)
```

**Limitations**:
- Sensitive to scale (large activations → large distances)
- Not invariant to rotation
- Requires same dimensionality

### Cosine Similarity

**Angle between vectors**: Invariant to magnitude

```python
def cosine_similarity_single(repr1, repr2):
    """Cosine similarity between two representations.

    Returns value in [-1, 1]:
    - 1: Identical direction
    - 0: Orthogonal
    - -1: Opposite direction
    """
    dot_product = np.dot(repr1, repr2)
    norms = np.linalg.norm(repr1) * np.linalg.norm(repr2)

    if norms == 0:
        return 0

    return dot_product / norms

def average_cosine_similarity(reprs_A, reprs_B):
    """Average cosine similarity for paired representations.

    Args:
        reprs_A, reprs_B: (n_samples, n_features) arrays

    Returns:
        mean_similarity: Average cosine similarity
    """
    assert reprs_A.shape == reprs_B.shape

    # Normalize rows
    reprs_A_norm = reprs_A / (np.linalg.norm(reprs_A, axis=1, keepdims=True) + 1e-8)
    reprs_B_norm = reprs_B / (np.linalg.norm(reprs_B, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity per sample
    similarities = np.sum(reprs_A_norm * reprs_B_norm, axis=1)

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    print(f"Mean cosine similarity: {mean_sim:.3f} ± {std_sim:.3f}")

    return mean_sim

# Example
# similarity = average_cosine_similarity(layer1_repr, layer2_repr)
```

**Advantages**:
- Invariant to scaling
- Normalized to [-1, 1]

**Limitations**:
- Still not invariant to rotation
- Requires same dimensionality

---

## Alignment-Based Metrics

### Procrustes Distance

**Idea**: Optimally align representations with rotation/reflection, then measure residual

```python
from scipy.linalg import orthogonal_procrustes

def procrustes_similarity(reprs_A, reprs_B):
    """Compute similarity after optimal alignment.

    Finds rotation matrix R that minimizes ||A - BR||.

    Args:
        reprs_A, reprs_B: (n_samples, n_features) arrays

    Returns:
        similarity: 1 - (aligned_distance / scale_factor)
        R: Optimal rotation matrix
    """
    assert reprs_A.shape == reprs_B.shape

    # Center representations
    reprs_A_centered = reprs_A - np.mean(reprs_A, axis=0)
    reprs_B_centered = reprs_B - np.mean(reprs_B, axis=0)

    # Normalize by Frobenius norm
    norm_A = np.linalg.norm(reprs_A_centered, ord='fro')
    norm_B = np.linalg.norm(reprs_B_centered, ord='fro')

    reprs_A_scaled = reprs_A_centered / norm_A
    reprs_B_scaled = reprs_B_centered / norm_B

    # Find optimal rotation
    R, scale = orthogonal_procrustes(reprs_A_scaled, reprs_B_scaled)

    # Apply rotation
    reprs_A_aligned = reprs_A_scaled @ R

    # Compute residual
    residual = np.linalg.norm(reprs_A_aligned - reprs_B_scaled, ord='fro')

    # Similarity (1 = identical after rotation)
    similarity = 1 - (residual ** 2) / 2

    print(f"Procrustes similarity: {similarity:.3f}")

    return similarity, R

# Example
# similarity, rotation_matrix = procrustes_similarity(layer1_repr, layer2_repr)
```

**Advantages**:
- Invariant to rotation and reflection
- Handles aligned representations

**Limitations**:
- Still requires same dimensionality
- Sensitive to outliers

---

## Canonical Correlation Analysis (CCA)

### Linear CCA

**Idea**: Find maximally correlated linear projections of two representation spaces

**Use when**: Comparing representations of different dimensionality

```python
from sklearn.cross_decomposition import CCA

def linear_cca_similarity(reprs_A, reprs_B, n_components=None):
    """Measure similarity via canonical correlation.

    Args:
        reprs_A: (n_samples, n_features_A)
        reprs_B: (n_samples, n_features_B)
        n_components: Number of canonical dimensions (default: min dimension)

    Returns:
        mean_correlation: Average canonical correlation
        correlations: Individual canonical correlations
    """
    if n_components is None:
        n_components = min(reprs_A.shape[1], reprs_B.shape[1], reprs_A.shape[0] - 1)

    # Fit CCA
    cca = CCA(n_components=n_components)
    reprs_A_c, reprs_B_c = cca.fit_transform(reprs_A, reprs_B)

    # Compute correlations per dimension
    correlations = []
    for i in range(n_components):
        corr = np.corrcoef(reprs_A_c[:, i], reprs_B_c[:, i])[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    mean_corr = np.mean(correlations)

    print(f"Mean CCA correlation: {mean_corr:.3f}")
    print(f"Individual correlations: {correlations[:5]}")  # Show first 5

    return mean_corr, correlations

# Example
# mean_corr, corrs = linear_cca_similarity(layer1_repr, layer5_repr, n_components=10)
```

### Projection-Weighted CCA (PWCCA)

**Improvement**: Weight canonical correlations by how much variance they explain

```python
def pwcca_similarity(reprs_A, reprs_B, n_components=None):
    """Projection-weighted CCA similarity.

    Weights correlations by variance explained in original space.

    Returns:
        pwcca_score: Weighted mean correlation
    """
    if n_components is None:
        n_components = min(reprs_A.shape[1], reprs_B.shape[1], reprs_A.shape[0] - 1)

    # Fit CCA
    cca = CCA(n_components=n_components)
    reprs_A_c, reprs_B_c = cca.fit_transform(reprs_A, reprs_B)

    # Correlations
    correlations = []
    for i in range(n_components):
        corr = np.corrcoef(reprs_A_c[:, i], reprs_B_c[:, i])[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Compute weights: variance explained in A by each canonical variable
    # Project canonical variables back to A space
    reprs_A_centered = reprs_A - np.mean(reprs_A, axis=0)
    total_var_A = np.sum(np.var(reprs_A_centered, axis=0))

    weights = []
    for i in range(n_components):
        # Variance in canonical variable i
        var_i = np.var(reprs_A_c[:, i])
        weight = var_i / total_var_A
        weights.append(weight)

    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Normalize

    # Weighted correlation
    pwcca_score = np.sum(weights * correlations)

    print(f"PWCCA similarity: {pwcca_score:.3f}")

    return pwcca_score

# Example
# pwcca_score = pwcca_similarity(layer1_repr, layer5_repr)
```

---

## Centered Kernel Alignment (CKA)

**Most popular modern metric**

**Advantages**:
- Invariant to orthogonal transformations and isotropic scaling
- Works with different dimensionalities
- Theoretically grounded
- Widely used in recent literature

### Linear CKA

```python
def linear_cka(reprs_A, reprs_B):
    """Compute linear CKA similarity.

    CKA measures similarity of Gram matrices (inner product matrices).

    Args:
        reprs_A: (n_samples, n_features_A)
        reprs_B: (n_samples, n_features_B)

    Returns:
        cka_score: Similarity in [0, 1], 1 = identical (up to orthogonal transform)
    """
    # Center representations
    reprs_A = reprs_A - np.mean(reprs_A, axis=0, keepdims=True)
    reprs_B = reprs_B - np.mean(reprs_B, axis=0, keepdims=True)

    # Gram matrices
    gram_A = reprs_A @ reprs_A.T  # (n_samples, n_samples)
    gram_B = reprs_B @ reprs_B.T

    # Frobenius inner product of Gram matrices
    numerator = np.linalg.norm(gram_A @ gram_B, ord='fro') ** 2

    # Normalization
    denominator = np.linalg.norm(gram_A, ord='fro') * np.linalg.norm(gram_B, ord='fro')

    cka = numerator / (denominator ** 2)

    print(f"Linear CKA: {cka:.3f}")

    return cka

# Example
# cka_score = linear_cka(layer1_repr, layer3_repr)
```

### RBF CKA (Nonlinear)

**For nonlinear similarity**: Use RBF kernel instead of linear

```python
def rbf_kernel(X, Y=None, gamma=None):
    """Compute RBF (Gaussian) kernel matrix.

    K(x, y) = exp(-gamma * ||x - y||²)
    """
    if Y is None:
        Y = X

    # Compute pairwise squared distances
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    distances_sq = X_sq + Y_sq.T - 2 * X @ Y.T

    # Gamma parameter (default: 1 / median distance)
    if gamma is None:
        median_dist = np.median(distances_sq[distances_sq > 0])
        gamma = 1.0 / median_dist

    # RBF kernel
    K = np.exp(-gamma * distances_sq)

    return K

def rbf_cka(reprs_A, reprs_B, gamma=None):
    """Compute RBF CKA similarity.

    Captures nonlinear relationships.

    Args:
        reprs_A, reprs_B: (n_samples, n_features) arrays
        gamma: RBF bandwidth parameter (default: auto)

    Returns:
        cka_score: Nonlinear similarity
    """
    # Compute RBF kernel matrices
    K_A = rbf_kernel(reprs_A, gamma=gamma)
    K_B = rbf_kernel(reprs_B, gamma=gamma)

    # Center kernels
    n = K_A.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix

    K_A_centered = H @ K_A @ H
    K_B_centered = H @ K_B @ H

    # CKA computation
    numerator = np.linalg.norm(K_A_centered @ K_B_centered, ord='fro') ** 2
    denominator = np.linalg.norm(K_A_centered, ord='fro') * np.linalg.norm(K_B_centered, ord='fro')

    cka = numerator / (denominator ** 2)

    print(f"RBF CKA: {cka:.3f}")

    return cka

# Example
# rbf_cka_score = rbf_cka(layer1_repr, layer3_repr)
```

### CKA Across All Layer Pairs

```python
def compute_cka_matrix(model, data, layer_indices):
    """Compute CKA similarity for all pairs of layers.

    Args:
        model: Neural network
        data: Input data
        layer_indices: List of layer indices to compare

    Returns:
        cka_matrix: (n_layers, n_layers) similarity matrix
    """
    n_layers = len(layer_indices)
    cka_matrix = np.zeros((n_layers, n_layers))

    # Extract representations from all layers
    layer_reprs = []
    for layer_idx in layer_indices:
        repr_layer = extract_layer_representations(model, data, layer_idx)
        layer_reprs.append(repr_layer)

    # Compute pairwise CKA
    for i in range(n_layers):
        for j in range(n_layers):
            if i == j:
                cka_matrix[i, j] = 1.0  # Perfect self-similarity
            elif i < j:
                cka = linear_cka(layer_reprs[i], layer_reprs[j])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka  # Symmetric

    # Visualize
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 7))
    plt.imshow(cka_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='CKA Similarity')
    plt.xticks(range(n_layers), layer_indices)
    plt.yticks(range(n_layers), layer_indices)
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.title('CKA Similarity Matrix')

    # Add values
    for i in range(n_layers):
        for j in range(n_layers):
            plt.text(j, i, f'{cka_matrix[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cka_matrix[i, j] < 0.5 else 'black',
                    fontsize=8)

    plt.tight_layout()
    plt.show()

    return cka_matrix

# Example
# cka_matrix = compute_cka_matrix(model, data, layer_indices=[0, 2, 4, 6, 8, 10])
```

---

## Mutual Information-Based Metrics

### Discrete Mutual Information

**For discrete representations**: Direct MI calculation

```python
from sklearn.metrics import mutual_info_score

def discrete_mutual_information(reprs_A_discrete, reprs_B_discrete):
    """Compute MI between discrete representations.

    Args:
        reprs_A_discrete: (n_samples,) integer cluster labels
        reprs_B_discrete: (n_samples,) integer cluster labels

    Returns:
        mi: Mutual information in bits
    """
    mi = mutual_info_score(reprs_A_discrete, reprs_B_discrete)

    print(f"Mutual Information: {mi:.3f} bits")

    return mi

# Example: After clustering representations
# from sklearn.cluster import KMeans
# clusters_A = KMeans(n_clusters=10).fit_predict(layer1_repr)
# clusters_B = KMeans(n_clusters=10).fit_predict(layer3_repr)
# mi = discrete_mutual_information(clusters_A, clusters_B)
```

### Continuous MI Estimation

**For continuous representations**: Estimate MI using k-NN methods

```python
from sklearn.feature_selection import mutual_info_regression

def continuous_mutual_information(reprs_A, reprs_B, n_neighbors=3):
    """Estimate MI for continuous representations.

    Uses k-nearest neighbor MI estimator.

    Args:
        reprs_A: (n_samples, n_features_A)
        reprs_B: (n_samples, n_features_B)

    Returns:
        mi_scores: MI for each dimension of B given A
    """
    # Compute MI of each dimension of B with all of A
    mi_scores = mutual_info_regression(reprs_A, reprs_B.T, n_neighbors=n_neighbors)

    mean_mi = np.mean(mi_scores)
    print(f"Mean MI: {mean_mi:.3f}")

    return mean_mi, mi_scores

# Note: This is asymmetric (MI of B given A)
# For symmetric measure, average both directions
```

---

## Task-Based Similarity

### Representational Similarity Analysis (RSA)

**Idea**: Compare how representations dissociate between stimuli

```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

def representational_similarity_analysis(reprs_A, reprs_B, metric='correlation'):
    """Compare representational dissimilarity matrices (RDMs).

    Args:
        reprs_A, reprs_B: (n_samples, n_features) arrays
        metric: Distance metric for RDM ('euclidean', 'correlation', 'cosine')

    Returns:
        rsa_similarity: Spearman correlation between RDMs
    """
    # Compute RDMs (pairwise distances)
    rdm_A = squareform(pdist(reprs_A, metric=metric))
    rdm_B = squareform(pdist(reprs_B, metric=metric))

    # Flatten upper triangles
    triu_indices = np.triu_indices(rdm_A.shape[0], k=1)
    rdm_A_flat = rdm_A[triu_indices]
    rdm_B_flat = rdm_B[triu_indices]

    # Spearman correlation
    rsa_corr, p_value = spearmanr(rdm_A_flat, rdm_B_flat)

    print(f"RSA similarity (Spearman ρ): {rsa_corr:.3f}")
    print(f"p-value: {p_value:.4f}")

    return rsa_corr, rdm_A, rdm_B

# Example
# rsa_corr, rdm_layer1, rdm_layer3 = representational_similarity_analysis(
#     layer1_repr, layer3_repr, metric='correlation'
# )

def visualize_rdms(rdm_A, rdm_B, labels=None):
    """Visualize two RDMs side-by-side."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot RDM A
    im1 = ax1.imshow(rdm_A, cmap='viridis')
    ax1.set_title('RDM A')
    plt.colorbar(im1, ax=ax1)

    # Plot RDM B
    im2 = ax2.imshow(rdm_B, cmap='viridis')
    ax2.set_title('RDM B')
    plt.colorbar(im2, ax=ax2)

    if labels is not None:
        ax1.set_xticks(range(len(labels)))
        ax1.set_yticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=90)
        ax1.set_yticklabels(labels)

        ax2.set_xticks(range(len(labels)))
        ax2.set_yticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=90)
        ax2.set_yticklabels(labels)

    plt.tight_layout()
    plt.show()

# visualize_rdms(rdm_layer1, rdm_layer3, labels=stimulus_names)
```

---

## Model Comparison Framework

### Complete Similarity Analysis

```python
def comprehensive_similarity_analysis(reprs_A, reprs_B, labels_A=None, labels_B=None):
    """Run all similarity metrics and compare.

    Args:
        reprs_A, reprs_B: Representation matrices
        labels_A, labels_B: Optional descriptive labels

    Returns:
        results: Dictionary of all similarity scores
    """
    results = {}

    print("="*60)
    print("COMPREHENSIVE SIMILARITY ANALYSIS")
    print("="*60)

    # 1. Basic distance metrics (if same shape)
    if reprs_A.shape == reprs_B.shape:
        print("\n1. DISTANCE-BASED METRICS")
        results['euclidean'] = average_euclidean_distance(reprs_A, reprs_B)
        results['cosine'] = average_cosine_similarity(reprs_A, reprs_B)
        results['procrustes'], _ = procrustes_similarity(reprs_A, reprs_B)

    # 2. CCA-based metrics
    print("\n2. CCA-BASED METRICS")
    results['linear_cca'], _ = linear_cca_similarity(reprs_A, reprs_B)
    results['pwcca'] = pwcca_similarity(reprs_A, reprs_B)

    # 3. CKA metrics
    print("\n3. CKA METRICS")
    results['linear_cka'] = linear_cka(reprs_A, reprs_B)
    results['rbf_cka'] = rbf_cka(reprs_A, reprs_B)

    # 4. RSA
    print("\n4. REPRESENTATIONAL SIMILARITY ANALYSIS")
    results['rsa'], _, _ = representational_similarity_analysis(reprs_A, reprs_B)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.3f}")

    return results

# Usage
# results = comprehensive_similarity_analysis(layer1_repr, layer5_repr)
```

---

## Interpretability Applications

### Application 1: Training Dynamics

**Track representation stability during training**

```python
def track_training_similarity(model_checkpoints, data, layer_idx):
    """Measure how representations change during training.

    Args:
        model_checkpoints: List of model states at different steps
        data: Evaluation data
        layer_idx: Which layer to analyze

    Returns:
        similarities: CKA similarity to final checkpoint
    """
    # Extract representations from all checkpoints
    checkpoint_reprs = []
    for checkpoint in model_checkpoints:
        reprs = extract_layer_representations(checkpoint, data, layer_idx)
        checkpoint_reprs.append(reprs)

    # Compare each to final checkpoint
    final_repr = checkpoint_reprs[-1]
    similarities = []

    for i, repr_i in enumerate(checkpoint_reprs):
        if i == len(checkpoint_reprs) - 1:
            sim = 1.0  # Self-similarity
        else:
            sim = linear_cka(repr_i, final_repr)
        similarities.append(sim)

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(similarities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Checkpoint')
    plt.ylabel('CKA Similarity to Final Checkpoint')
    plt.title(f'Representation Convergence (Layer {layer_idx})')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()

    return similarities

# Usage
# checkpoints = [model_at_epoch_0, model_at_epoch_5, ..., model_final]
# similarities = track_training_similarity(checkpoints, data, layer_idx=5)
```

### Application 2: Architecture Comparison

**Compare representations across architectures**

```python
def compare_architectures(models_dict, data, layer_mapping):
    """Compare representations across different architectures.

    Args:
        models_dict: {'model_name': model_object}
        data: Common evaluation data
        layer_mapping: {'model_name': layer_idx} for comparison

    Returns:
        similarity_matrix: Pairwise CKA between all models
    """
    model_names = list(models_dict.keys())
    n_models = len(model_names)

    # Extract representations
    reprs = {}
    for name in model_names:
        model = models_dict[name]
        layer_idx = layer_mapping[name]
        reprs[name] = extract_layer_representations(model, data, layer_idx)

    # Compute pairwise CKA
    similarity_matrix = np.zeros((n_models, n_models))

    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            elif i < j:
                sim = linear_cka(reprs[name_i], reprs[name_j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

    # Visualize
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 7))
    plt.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='CKA Similarity')
    plt.xticks(range(n_models), model_names, rotation=45, ha='right')
    plt.yticks(range(n_models), model_names)
    plt.title('Cross-Architecture Representation Similarity')

    # Add values
    for i in range(n_models):
        for j in range(n_models):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if similarity_matrix[i, j] < 0.5 else 'black')

    plt.tight_layout()
    plt.show()

    return similarity_matrix

# Example
# models = {
#     'ResNet50': resnet_model,
#     'ViT-B': vit_model,
#     'ConvNeXt': convnext_model
# }
# layer_map = {'ResNet50': 30, 'ViT-B': 10, 'ConvNeXt': 20}
# sim_matrix = compare_architectures(models, data, layer_map)
```

### Application 3: Identify Redundant Layers

**Find layers with highly similar representations**

```python
def find_redundant_layers(model, data, layer_indices, threshold=0.95):
    """Identify layers with very similar representations.

    High similarity suggests potential for pruning.

    Args:
        model: Neural network
        data: Evaluation data
        layer_indices: Layers to analyze
        threshold: CKA similarity threshold for "redundant"

    Returns:
        redundant_pairs: List of (layer_i, layer_j) pairs
    """
    # Compute CKA matrix
    cka_matrix = compute_cka_matrix(model, data, layer_indices)

    # Find high-similarity pairs (excluding diagonal)
    redundant_pairs = []

    for i in range(len(layer_indices)):
        for j in range(i+1, len(layer_indices)):
            if cka_matrix[i, j] >= threshold:
                redundant_pairs.append((layer_indices[i], layer_indices[j]))
                print(f"Layers {layer_indices[i]} and {layer_indices[j]}: "
                      f"CKA = {cka_matrix[i, j]:.3f} (redundant)")

    if not redundant_pairs:
        print(f"No redundant layer pairs found (threshold = {threshold})")

    return redundant_pairs

# Usage
# redundant = find_redundant_layers(model, data, layer_indices=range(0, 12), threshold=0.95)
```

---

## Best Practices

### Choosing the Right Metric

```python
def metric_selection_guide():
    """Guide for selecting similarity metric."""

    guide = {
        "Euclidean/Cosine": {
            "Use when": "Same dimensionality, quick comparison",
            "Pros": "Fast, simple, intuitive",
            "Cons": "Not invariant to rotation, requires same dimensions",
            "Best for": "Preliminary analysis, same architecture comparisons"
        },

        "Procrustes": {
            "Use when": "Same dimensionality, rotation-invariant comparison",
            "Pros": "Invariant to orthogonal transform",
            "Cons": "Still requires same dimensions",
            "Best for": "Comparing trained vs randomly initialized"
        },

        "CCA/PWCCA": {
            "Use when": "Different dimensionalities, want correlated subspaces",
            "Pros": "Handles different dimensions, finds shared structure",
            "Cons": "Linear only, can miss global structure",
            "Best for": "Comparing different architectures, modalities"
        },

        "Linear CKA": {
            "Use when": "General-purpose comparison, different dimensions OK",
            "Pros": "Invariant to isotropic scaling and rotation, well-studied",
            "Cons": "Linear only",
            "Best for": "Default choice for most comparisons"
        },

        "RBF CKA": {
            "Use when": "Need to capture nonlinear relationships",
            "Pros": "Captures nonlinear similarity",
            "Cons": "Hyperparameter sensitive (gamma)",
            "Best for": "When linear CKA gives low scores but you suspect similarity"
        },

        "RSA": {
            "Use when": "Comparing stimulus representations",
            "Pros": "Theory from neuroscience, interpretable",
            "Cons": "Requires structured stimulus set",
            "Best for": "Neuroscience comparisons, interpretable domains"
        }
    }

    for metric, info in guide.items():
        print(f"\n{'='*60}")
        print(f"{metric}")
        print(f"{'='*60}")
        for key, value in info.items():
            print(f"{key}: {value}")

    return guide

# Print guide
metric_selection_guide()
```

### Statistical Significance

```python
def test_similarity_significance(reprs_A, reprs_B, n_permutations=1000):
    """Test if similarity is statistically significant.

    Null hypothesis: Representations are independent (permutation test).

    Args:
        reprs_A, reprs_B: Representation matrices
        n_permutations: Number of permutation samples

    Returns:
        p_value: Probability under null hypothesis
    """
    # Observed similarity
    observed_cka = linear_cka(reprs_A, reprs_B)

    # Permutation distribution
    null_distribution = []

    for _ in range(n_permutations):
        # Shuffle samples in B (breaks pairing)
        perm_indices = np.random.permutation(len(reprs_B))
        reprs_B_perm = reprs_B[perm_indices]

        # Compute CKA
        cka_perm = linear_cka(reprs_A, reprs_B_perm)
        null_distribution.append(cka_perm)

    null_distribution = np.array(null_distribution)

    # p-value: fraction of permutations ≥ observed
    p_value = np.mean(null_distribution >= observed_cka)

    print(f"Observed CKA: {observed_cka:.3f}")
    print(f"Null mean: {np.mean(null_distribution):.3f}")
    print(f"p-value: {p_value:.4f}")

    # Visualize
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.hist(null_distribution, bins=50, alpha=0.7, label='Null distribution')
    plt.axvline(observed_cka, color='red', linestyle='--', linewidth=2,
                label=f'Observed (p={p_value:.3f})')
    plt.xlabel('CKA Similarity')
    plt.ylabel('Frequency')
    plt.title('Permutation Test for Representation Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return p_value, null_distribution

# Usage
# p_value, null_dist = test_similarity_significance(layer1_repr, layer5_repr, n_permutations=1000)
```

---

## Further Reading

**Foundational papers**:
- Kornblith et al. (2019): "Similarity of Neural Network Representations Revisited"
- Morcos et al. (2018): "Insights on representational similarity in neural networks with canonical correlation"
- Raghu et al. (2017): "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability"

**Related guides**:
- [Dimensionality Reduction](dimensionality_reduction.md) - Visualizing representations
- [Linear Algebra Essentials](../../1_foundations/linear_algebra_essentials.md) - Mathematical foundations
- [Statistics for Interpretability](../../1_foundations/statistics_for_interpretability.md) - Significance testing

**Full bibliography**: [References](../../references/bibliography.md)

---

**Return to**: [Methods](../README.md) | [Main Handbook](../../0_start_here/README.md)

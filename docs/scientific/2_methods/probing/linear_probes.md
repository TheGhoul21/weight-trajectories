# Linear Probes

Testing what information is encoded in neural network representations using simple linear classifiers.

---

## Overview

**Linear probing** tests whether a concept is encoded in a model's internal representations by training a linear classifier (probe) to predict the concept from hidden activations.

**Key idea**: If a linear classifier can predict concept C from layer L's activations, then C is linearly encoded in layer L.

**Why linear**: Tests if information is easily accessible. Non-linear probes can extract more information but are harder to interpret and may find spurious patterns.

---

## The Probing Protocol

### Basic Setup

**Given**:
- Trained model with hidden representations h
- Concept labels y (e.g., "contains a face", "positive sentiment", "win-in-1 available")
- Training samples {(h_i, y_i)}

**Goal**: Train linear classifier f(h) = w·h + b to predict y from h

**Success**: High accuracy → concept is (linearly) encoded

**Failure**: Low accuracy → concept is not linearly accessible (but might be non-linearly encoded)

### Standard Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train_linear_probe(hidden_states, labels, test_size=0.2, C=1.0):
    """
    Train linear probe to predict concept from hidden states.

    Args:
        hidden_states: (n_samples, hidden_dim) numpy array
        labels: (n_samples,) binary or multiclass labels
        test_size: fraction for test set
        C: inverse regularization strength (higher = less regularization)

    Returns:
        accuracy: test set accuracy
        probe: trained classifier
        weights: probe weights (interpretable!)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, labels,
        test_size=test_size,
        stratify=labels,  # maintain class balance
        random_state=42
    )

    # Train logistic regression probe
    probe = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42
    )
    probe.fit(X_train, y_train)

    # Evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Extract interpretable weights
    weights = probe.coef_[0] if probe.coef_.shape[0] == 1 else probe.coef_

    return accuracy, probe, weights
```

---

## Essential Controls

Probes can achieve high accuracy for wrong reasons. Always include controls.

### 1. Shuffled Label Baseline

**Purpose**: Ensure accuracy is not due to class imbalance or method artifacts

```python
def shuffled_baseline(hidden_states, labels, n_shuffles=10):
    """Compute baseline accuracy with shuffled labels."""
    accuracies = []

    for _ in range(n_shuffles):
        labels_shuffled = np.random.permutation(labels)
        acc, _, _ = train_linear_probe(hidden_states, labels_shuffled)
        accuracies.append(acc)

    return np.mean(accuracies), np.std(accuracies)

# Usage
true_acc, _, _ = train_linear_probe(hidden_states, labels)
baseline_mean, baseline_std = shuffled_baseline(hidden_states, labels)

print(f"True accuracy: {true_acc:.3f}")
print(f"Baseline: {baseline_mean:.3f} ± {baseline_std:.3f}")

# True accuracy should be >> baseline
```

### 2. Random Feature Control

**Purpose**: Ensure accuracy is not due to the probe itself being too powerful

```python
def random_feature_control(hidden_states, labels):
    """Test probe on random features (same dimensionality)."""
    random_features = np.random.randn(*hidden_states.shape)
    acc, _, _ = train_linear_probe(random_features, labels)
    return acc

# Should be close to chance
random_acc = random_feature_control(hidden_states, labels)
```

### 3. Stratified Splits

**Purpose**: Avoid train/test leakage, especially with temporal data

```python
# For sequential data (e.g., training epochs), use temporal split
n_train = int(0.8 * len(hidden_states))
X_train, X_test = hidden_states[:n_train], hidden_states[n_train:]
y_train, y_test = labels[:n_train], labels[n_train:]

# For non-sequential, use stratified split (shown above)
```

### 4. Cross-Validation

**Purpose**: Ensure results are stable across different train/test splits

```python
from sklearn.model_selection import cross_val_score

probe = LogisticRegression(C=1.0, max_iter=1000)
scores = cross_val_score(probe, hidden_states, labels, cv=5)

print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Hyperparameter Considerations

### Regularization Strength (C)

**Purpose**: Control overfitting

**C parameter in sklearn**: Inverse regularization (higher C = less regularization)

**Guidance**:
- **C=1.0**: Good default, moderate regularization
- **C=0.1**: Strong regularization, use if overfitting suspected
- **C=10**: Weak regularization, use with large datasets

**Tuning**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1.0, 10, 100]}
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
print(f"Best C: {best_C}")
```

### Solver Choice

**Options**:
- `'lbfgs'`: Default, good for small to medium datasets
- `'saga'`: Faster for large datasets, supports L1 regularization
- `'liblinear'`: Good for small datasets, supports L1 and L2

**Recommendation**: Use default unless you have specific needs

---

## Interpreting Results

### Accuracy Interpretation

**High accuracy (>80%)**:
- Concept is strongly encoded
- Linearly accessible
- Layer is relevant for this concept

**Medium accuracy (60-80%)**:
- Concept partially encoded
- May be distributed across dimensions
- May require non-linear combination

**Low accuracy (≈ baseline)**:
- Concept not encoded (or not in this layer)
- May be encoded non-linearly
- May emerge in different layers

### Weight Analysis

Probe weights reveal which dimensions matter:

```python
def analyze_probe_weights(weights, hidden_dim, top_k=10):
    """Identify most important dimensions."""
    importance = np.abs(weights)
    top_dims = np.argsort(importance)[-top_k:][::-1]

    print(f"Top {top_k} dimensions:")
    for i, dim in enumerate(top_dims):
        print(f"  {i+1}. Dimension {dim}: weight = {weights[dim]:.3f}")

    return top_dims

# Usage
_, _, weights = train_linear_probe(hidden_states, labels)
top_dims = analyze_probe_weights(weights, hidden_states.shape[1])
```

**Interpretation**:
- **Sparse weights**: Few dimensions encode concept (specialized neurons)
- **Dense weights**: Many dimensions contribute (distributed encoding)
- **Large magnitude**: Strong contribution to prediction

---

## Layer-wise Analysis

Test which layer encodes concepts:

```python
def layer_wise_probing(model, dataloader, concept_labels, layer_names):
    """
    Probe all layers to find where concept is encoded.

    Args:
        model: neural network
        dataloader: data to extract activations from
        concept_labels: (n_samples,) target concept
        layer_names: list of layer names to probe

    Returns:
        results: dict mapping layer_name -> accuracy
    """
    results = {}

    for layer_name in layer_names:
        # Extract activations from this layer
        activations = extract_activations(model, dataloader, layer_name)

        # Train probe
        acc, _, _ = train_linear_probe(activations, concept_labels)
        results[layer_name] = acc

        print(f"{layer_name}: {acc:.3f}")

    return results

# Usage
results = layer_wise_probing(
    model, val_loader,
    concept_labels,
    layer_names=['layer1', 'layer2', 'layer3', 'fc']
)

# Visualize
import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values())
plt.ylabel('Probe Accuracy')
plt.title('Concept Encoding Across Layers')
plt.xticks(rotation=45)
plt.show()
```

**Expected patterns**:
- **Low-level concepts** (edges, colors): Early layers
- **Mid-level concepts** (textures, parts): Middle layers
- **High-level concepts** (objects, semantics): Late layers

---

## Temporal Analysis

Track concept emergence during training:

```python
def probe_across_checkpoints(checkpoint_dir, dataloader, concept_labels):
    """Track probe accuracy across training."""
    accuracies = []
    epochs = []

    for ckpt_path in sorted(glob.glob(f"{checkpoint_dir}/weights_epoch_*.pt")):
        # Extract epoch number
        epoch = int(ckpt_path.split('_')[-1].split('.')[0])

        # Load model at this checkpoint
        model = load_model(ckpt_path)

        # Extract activations
        hidden_states = extract_activations(model, dataloader, 'layer3')

        # Train probe
        acc, _, _ = train_linear_probe(hidden_states, concept_labels)

        epochs.append(epoch)
        accuracies.append(acc)

        print(f"Epoch {epoch}: {acc:.3f}")

    return epochs, accuracies

# Visualize emergence
epochs, accs = probe_across_checkpoints(ckpt_dir, val_loader, labels)
plt.plot(epochs, accs, marker='o')
plt.xlabel('Training Epoch')
plt.ylabel('Probe Accuracy')
plt.title('Concept Emergence During Training')
plt.show()
```

**Insights**:
- When do concepts emerge?
- Do they emerge gradually or suddenly?
- Which concepts emerge first?
- Correlate with validation performance

---

## Advanced Techniques

### Selective Probing

Test if specific dimensions are necessary:

```python
def ablation_probing(hidden_states, labels, dim_to_ablate):
    """Train probe with specific dimension removed."""
    # Create copy with dimension zeroed out
    h_ablated = hidden_states.copy()
    h_ablated[:, dim_to_ablate] = 0

    # Train probe
    acc_ablated, _, _ = train_linear_probe(h_ablated, labels)

    # Compare to full probe
    acc_full, _, _ = train_linear_probe(hidden_states, labels)

    drop = acc_full - acc_ablated
    print(f"Ablating dim {dim_to_ablate}: {acc_full:.3f} -> {acc_ablated:.3f} (drop: {drop:.3f})")

    return drop

# Test importance of top dimensions
for dim in top_dims:
    ablation_probing(hidden_states, labels, dim)
```

### Control Task Probing

Ensure probe isn't learning from confounds:

```python
# Example: Face detection probe might learn from image background
# Control: Probe should also predict face with background randomized

def control_task_probe(hidden_states, primary_labels, control_labels):
    """
    Test if probe learns from confounds.

    primary_labels: target concept (e.g., face present)
    control_labels: confound (e.g., indoor vs outdoor)
    """
    acc_primary, _, _ = train_linear_probe(hidden_states, primary_labels)
    acc_control, _, _ = train_linear_probe(hidden_states, control_labels)

    print(f"Primary task: {acc_primary:.3f}")
    print(f"Control task: {acc_control:.3f}")

    # If control accuracy is high, probe might be using confound
    if acc_control > 0.7:
        print("Warning: Probe may be learning from confound!")
```

### Minimum Description Length (MDL) Probing

Control for probe complexity:

```python
from sklearn.linear_model import LogisticRegressionCV

def mdl_probe(hidden_states, labels):
    """Use cross-validation to select regularization automatically."""
    probe = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1, 10, 100],
        cv=5,
        max_iter=1000
    )
    probe.fit(hidden_states, labels)

    print(f"Best C: {probe.C_[0]:.3f}")
    print(f"CV score: {probe.scores_[1].mean():.3f}")

    return probe
```

---

## Common Pitfalls

### Pitfall 1: Ignoring Class Imbalance

**Problem**: 90% of samples are negative; probe gets 90% accuracy by always predicting negative.

**Solution**:
```python
from sklearn.metrics import balanced_accuracy_score, f1_score

# Use balanced metrics
y_pred = probe.predict(X_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Balanced accuracy: {balanced_acc:.3f}")
print(f"F1 score: {f1:.3f}")
```

### Pitfall 2: Information Leakage

**Problem**: Test samples correlated with train samples (e.g., frames from same video).

**Solution**: Ensure independent test set
```python
# Group by source (video, document, etc.) before splitting
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=video_ids))
```

### Pitfall 3: Overinterpreting Probe Weights

**Problem**: Assuming large weight = important neuron.

**Issue**: Weights depend on scale of activations.

**Solution**: Normalize activations before probing
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Now weights are comparable
```

### Pitfall 4: Multiple Comparisons

**Problem**: Testing 100 concepts, expect 5 significant by chance at p=0.05.

**Solution**: Correct for multiple comparisons
```python
from statsmodels.stats.multitest import multipletests

# Collect p-values from all probes
p_values = [...]  # from statistical tests

# Correct using FDR
rejected, p_corrected, _, _ = multipletests(
    p_values,
    alpha=0.05,
    method='fdr_bh'
)

print(f"Significant after correction: {rejected.sum()}/{len(rejected)}")
```

---

## Relationship to Mutual Information

**Comparison**:

| Aspect | Linear Probe | Mutual Information |
|--------|-------------|-------------------|
| Assumption | Linear accessibility | Any dependency |
| Computation | Fast | Slower (k-NN estimation) |
| Interpretation | Clear (weights) | Opaque (single number) |
| False negatives | May miss non-linear encoding | Detects any dependence |

**When to use which**:
- **Probe first**: Fast, interpretable, actionable
- **MI for validation**: Model-free confirmation
- **Both together**: Triangulate findings

**Example workflow**:
```python
# 1. Quick probe
acc_probe, _, weights = train_linear_probe(hidden_states, labels)

# 2. If high probe accuracy, concept is linearly encoded
if acc_probe > 0.7:
    print("Concept is linearly accessible")
    analyze_probe_weights(weights, hidden_states.shape[1])

# 3. If low probe accuracy, check MI
else:
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(hidden_states, labels)

    if mi.mean() > 0.1:
        print("Concept is encoded but not linearly accessible")
    else:
        print("Concept is not encoded")
```

---

## Case Study: BERT Probing

**Question**: Does BERT encode syntactic information?

**Method**: Probe BERT layers for part-of-speech tags

```python
# Simplified example
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extract BERT activations
def get_bert_activations(sentences, layer_idx=-1):
    activations = []

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors='pt')
        outputs = model(**inputs, output_hidden_states=True)

        # Get activations from specified layer
        layer_acts = outputs.hidden_states[layer_idx]
        activations.append(layer_acts.mean(dim=1).detach().numpy())

    return np.vstack(activations)

# Probe for POS tags
sentences = [...]  # your sentences
pos_tags = [...]   # your POS labels (e.g., NOUN, VERB, ADJ)

activations = get_bert_activations(sentences, layer_idx=6)
acc, _, _ = train_linear_probe(activations, pos_tags)

print(f"Layer 6 POS probe accuracy: {acc:.3f}")
```

**Findings** (from literature):
- **Layer 1-4**: Low-level syntax (POS tags)
- **Layer 5-8**: Syntactic structure (constituency parsing)
- **Layer 9-12**: Semantics (word sense, coreference)

---

## Practical Recommendations

### Sample Size

**Minimum**: 500 samples (more if high-dimensional hidden states)

**Recommended**: 5,000+ samples for stable estimates

**Rule of thumb**: 10x hidden dimension

### Feature Engineering

**Normalization**: Standardize hidden states
```python
from sklearn.preprocessing import StandardScaler
hidden_states = StandardScaler().fit_transform(hidden_states)
```

**Dimensionality reduction** (if hidden_dim >> n_samples):
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=min(128, n_samples // 10))
hidden_states_reduced = pca.fit_transform(hidden_states)
```

### Reporting

**Essential information**:
- Probe accuracy with confidence intervals
- Baseline comparisons (shuffled, random features)
- Number of training samples
- Regularization strength
- Layer(s) analyzed

**Example**:
```
Layer 3 probe accuracy: 0.847 ± 0.023 (5-fold CV)
Shuffled baseline: 0.512 ± 0.031
Training samples: 5,000
Regularization: C=1.0
```

---

## Further Reading

**Key papers**:
- Alain & Bengio (2017): Understanding intermediate layers (early probing paper)
- Belinkov & Glass (2019): Analysis methods for neural NLP
- Hewitt & Liang (2019): Designing and interpreting probes
- Pimentel et al. (2020): Information-theoretic probing
- Elazar et al. (2021): Amnesic probing (causal variant)

**Related methods**:
- [Mutual Information](mutual_information.md) - Complementary approach
- [Causal Probing](causal_probing.md) - Testing causal importance
- [Representation Similarity](../representation_analysis/similarity_metrics.md) - Comparing representations

**Case studies**:
- [AlphaZero Concept Learning](../../5_case_studies/reinforcement_learning/alphazero_concepts.md)
- BERT linguistic knowledge (Tenney et al., 2019)

Full bibliography: [References](../../references/bibliography.md)

---

**Return to**: [Probing Methods](README.md) | [Methods](../README.md)

# Interpretability for Practitioners

This guide is for engineers and data scientists who want to understand and debug their trained models. Skip the theory - let's get to practical techniques and code.

---

## Quick Start: "I Have a Model, What Now?"

Use this decision tree to find the right technique for your goal:

```
Do you have a trained model you want to understand?
│
├─ YES: What type of question do you have?
│  │
│  ├─ "Why did it make THIS specific prediction?"
│  │  ├─ Image model → Grad-CAM (5 min setup)
│  │  ├─ Text model → Attention visualization or Integrated Gradients
│  │  └─ Tabular model → SHAP values
│  │
│  ├─ "What patterns did it learn OVERALL?"
│  │  ├─ Extract activations on validation set
│  │  ├─ Apply UMAP or t-SNE → visualize clusters
│  │  └─ Train probes for concepts of interest
│  │
│  ├─ "Does it know about concept X?"
│  │  ├─ Linear probe → test if concept is linearly encoded
│  │  ├─ Mutual information → quantify statistical dependence
│  │  └─ Causal probe → test if concept causally matters
│  │
│  ├─ "How does training change representations?"
│  │  ├─ Save checkpoints during training
│  │  ├─ Extract representations at each checkpoint
│  │  └─ Visualize with PHATE or track with CKA similarity
│  │
│  └─ "Is component/feature X important?"
│     ├─ Ablation → remove it, measure performance drop
│     ├─ Activation patching → swap activations, see effect
│     └─ Feature attribution → SHAP or Integrated Gradients
│
└─ NO: What stage are you at?
   │
   ├─ Planning training? → Design interpretability checks upfront
   ├─ Debugging poor performance? → Start with loss curves + grad-CAM
   └─ Preparing for deployment? → Validate with probes + failure mode analysis
```

---

## Common Tasks with Code Snippets

### Task 1: "Explain this prediction" (Image Classification)

**Goal**: Visualize which parts of an image drove the prediction.

**Best tool**: Grad-CAM (Gradient-weighted Class Activation Mapping)

**Quick code** (using Captum):

```python
import torch
from captum.attr import LayerGradCam
from torchvision import transforms
from PIL import Image

# Load your model
model = YourModel()
model.eval()

# Load and preprocess image
img = Image.open('image.jpg')
transform = transforms.Compose([...])  # your preprocessing
input_tensor = transform(img).unsqueeze(0)

# Initialize Grad-CAM (targeting last conv layer)
grad_cam = LayerGradCam(model, model.layer4)  # adjust layer name

# Compute attribution
target = 243  # ImageNet class for "dog"
attribution = grad_cam.attribute(input_tensor, target=target)

# Visualize
from captum.attr import visualization as viz
viz.visualize_image_attr(
    attribution[0].cpu().permute(1,2,0).detach().numpy(),
    original_image=img,
    method='blended_heat_map',
    sign='positive',
    show_colorbar=True
)
```

**Time**: 10-15 minutes

**When it's useful**: Debugging misclassifications, validating the model looks at reasonable features.

**Pitfalls**: Grad-CAM can be noisy; average over multiple similar inputs for robust patterns.

---

### Task 2: "Does my model understand concept X?" (Probing)

**Goal**: Test if a specific concept (e.g., "contains a face", "is positive sentiment") is encoded.

**Best tool**: Linear probe with baseline comparison

**Quick code**:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Extract hidden activations
def extract_activations(model, dataloader, layer_name='layer3'):
    """Extract activations from a specific layer."""
    activations = []
    labels = []

    def hook(module, input, output):
        activations.append(output.cpu().detach())

    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook)

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            model(batch_x)
            labels.extend(batch_y.numpy())

    handle.remove()
    return torch.cat(activations).numpy(), np.array(labels)

# Step 2: Get activations and concept labels
X, y_concept = extract_activations(model, val_loader)

# X: (n_samples, hidden_dim)
# y_concept: (n_samples,) binary labels for concept (1=present, 0=absent)

# Step 3: Train probe
X_train, X_test, y_train, y_test = train_test_split(X, y_concept, test_size=0.2)

probe = LogisticRegression(max_iter=1000, C=1.0)
probe.fit(X_train, y_train)

# Step 4: Evaluate
accuracy = probe.score(X_test, y_test)
print(f"Probe accuracy: {accuracy:.3f}")

# Step 5: Compare to baseline (shuffled labels)
y_shuffled = np.random.permutation(y_train)
baseline_probe = LogisticRegression(max_iter=1000, C=1.0)
baseline_probe.fit(X_train, y_shuffled)
baseline_acc = baseline_probe.score(X_test, y_test)
print(f"Baseline (shuffled) accuracy: {baseline_acc:.3f}")

# Interpretation:
# - If accuracy >> baseline: concept is encoded
# - If accuracy ≈ baseline: concept not accessible (or not learned)
# - If accuracy < baseline: something is wrong with setup
```

**Time**: 30 minutes (including data preparation)

**When it's useful**: Validating that expected concepts are learned, finding what layer encodes what.

**Pitfalls**: Always compare to shuffled baseline! High accuracy might just reflect class imbalance.

---

### Task 3: "Visualize representation structure" (t-SNE/UMAP)

**Goal**: See how representations cluster, identify patterns.

**Best tool**: UMAP (faster than t-SNE, better global structure)

**Quick code**:

```python
import umap
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Extract activations (as above)
X, y = extract_activations(model, val_loader, layer_name='final_layer')
# X: (n_samples, hidden_dim)
# y: class labels for coloring

# Step 2: Run UMAP
reducer = umap.UMAP(
    n_neighbors=15,  # local neighborhood size
    min_dist=0.1,    # minimum distance between points
    n_components=2,  # output dimensions
    metric='cosine',  # distance metric (cosine good for NN activations)
    random_state=42
)

embedding = reducer.fit_transform(X)

# Step 3: Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y,
    cmap='tab10',
    alpha=0.6,
    s=10
)
plt.colorbar(scatter, label='Class')
plt.title('UMAP Projection of Hidden Representations')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.tight_layout()
plt.savefig('umap_embedding.png', dpi=150)
plt.show()
```

**Time**: 15-20 minutes

**When it's useful**: Getting intuition for what the model learned, identifying confusable classes, detecting representation collapse.

**Pitfalls**: UMAP hyperparameters matter; try a few settings. Don't over-interpret fine details.

---

### Task 4: "Compare two models" (CKA Similarity)

**Goal**: Quantify how similar representations are between two models or training runs.

**Best tool**: Centered Kernel Alignment (CKA)

**Quick code**:

```python
import torch
import numpy as np

def centering(K):
    """Center kernel matrix."""
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)

def linear_CKA(X, Y):
    """
    Compute linear CKA between two sets of representations.
    X, Y: (n_samples, n_features)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T

    # CKA
    nom = np.linalg.norm(K_X @ K_Y, ord='fro')**2
    denom = np.linalg.norm(K_X, ord='fro') * np.linalg.norm(K_Y, ord='fro')

    return nom / denom

# Usage:
# Extract activations from both models at the same layer
X_model1, _ = extract_activations(model1, dataloader, layer='layer3')
X_model2, _ = extract_activations(model2, dataloader, layer='layer3')

cka_score = linear_CKA(X_model1, X_model2)
print(f"CKA similarity: {cka_score:.3f}")

# Interpretation:
# CKA ≈ 1.0: very similar representations
# CKA ≈ 0.5: moderately similar
# CKA ≈ 0.0: completely different
```

**Time**: 20 minutes

**When it's useful**: Comparing architectures, tracking representation changes during training, identifying which layers are affected by a change.

**Pitfalls**: CKA is invariant to orthogonal transformations - two representations can be "similar" even if features are rotated.

---

### Task 5: "Find important features" (SHAP for Tabular Data)

**Goal**: Determine which input features matter most for predictions.

**Best tool**: SHAP (SHapley Additive exPlanations)

**Quick code**:

```python
import shap
import pandas as pd

# Load your model and data
# model = trained sklearn/xgboost/neural net model
# X_train, X_test = feature matrices

# Choose explainer based on model type
# For tree models:
explainer = shap.TreeExplainer(model)
# For neural networks:
# explainer = shap.DeepExplainer(model, X_train[:100])
# For any model:
# explainer = shap.KernelExplainer(model.predict, X_train[:100])

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Visualization 1: Summary plot (feature importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Visualization 2: Single prediction explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test[0],
    feature_names=feature_names
)

# Visualization 3: Dependence plot (how feature X affects output)
shap.dependence_plot("feature_name", shap_values, X_test)
```

**Time**: 10-30 minutes (depends on model size and data)

**When it's useful**: Understanding global feature importance, explaining individual predictions, detecting biases.

**Pitfalls**: SHAP can be slow for large datasets; sample data if needed. Kernel SHAP is model-agnostic but slowest.

---

## Tool Selection Guide

Choose the right tool for your situation:

| Your Situation | Recommended Tool | Why | Setup Time |
|----------------|------------------|-----|------------|
| **Image classification, need quick explanations** | Grad-CAM (Captum) | Visualizes spatial importance, fast | 10 min |
| **Text model, explain predictions** | Attention viz or Integrated Gradients | Shows which tokens matter | 15 min |
| **Tabular data, any model** | SHAP | Model-agnostic, rigorous theory | 20 min |
| **Testing if concept is learned** | Linear probe | Simple, interpretable, has baselines | 30 min |
| **Visualizing learned structure** | UMAP | Fast, good global structure | 15 min |
| **Comparing models/checkpoints** | CKA | Quantitative similarity metric | 20 min |
| **Understanding RNN dynamics** | Fixed-point analysis | Reveals computational structure | 2 hours |
| **Deep mechanistic understanding** | Circuit analysis (TransformerLens) | Reverse-engineers algorithms | Days |

**General advice**: Start simple (Grad-CAM, SHAP), go complex as needed.

---

## Integration into ML Pipelines

### During Training

**Checkpoint interpretability checks**:

```python
# In your training loop
if epoch % 5 == 0:
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

    # Run quick interpretability checks
    run_probe_suite(model, val_loader, epoch)
    visualize_embeddings(model, val_loader, epoch)

    # Log to TensorBoard or W&B
    wandb.log({
        'probe_accuracy': probe_acc,
        'embedding_plot': fig
    })
```

**Benefits**: Catch representation collapse early, track concept emergence, correlate with performance.

---

### Before Deployment

**Validation checklist**:

- [ ] **Probe for expected concepts**: Does it encode what it should?
- [ ] **Test failure modes**: Run adversarial examples, edge cases
- [ ] **Compare to baselines**: Ensure it's not learning shortcuts
- [ ] **Visualize decision boundaries**: Check for unexpected patterns
- [ ] **Attribution sanity checks**: Do explanations make sense?

**Example script**:

```python
def pre_deployment_checks(model, val_loader, test_suite):
    """Run interpretability validation before deployment."""

    print("Running pre-deployment interpretability checks...")

    # 1. Probe suite
    print("\n1. Checking concept encoding...")
    probe_results = {}
    for concept in ['safety_critical_feature', 'fairness_attribute', ...]:
        acc = train_and_test_probe(model, val_loader, concept)
        probe_results[concept] = acc
        print(f"  {concept}: {acc:.2%}")

    # 2. Attribution sanity checks
    print("\n2. Running attribution sanity checks...")
    for test_case in test_suite:
        attribution = compute_attribution(model, test_case)
        assert sanity_check_passes(attribution), f"Failed on {test_case}"

    # 3. Adversarial robustness
    print("\n3. Testing adversarial robustness...")
    adv_accuracy = test_adversarial(model, val_loader)
    print(f"  Adversarial accuracy: {adv_accuracy:.2%}")

    # 4. Generate report
    report = generate_interpretability_report(probe_results, adv_accuracy)
    report.save('deployment_validation_report.pdf')

    print("\nAll checks passed! ✓")
```

---

## Debugging Guide

### Problem: "My model has high train accuracy but low test accuracy"

**Interpretability diagnosis**:

1. **Compare train vs test representations**:
   ```python
   # Extract embeddings
   X_train, y_train = extract_activations(model, train_loader)
   X_test, y_test = extract_activations(model, test_loader)

   # Visualize jointly
   X_combined = np.vstack([X_train, X_test])
   labels = ['train']*len(X_train) + ['test']*len(X_test)

   umap_viz(X_combined, labels)  # Look for separation
   ```

   **If train and test form separate clusters**: Overfitting, poor generalization

   **If they overlap well**: Problem might not be in representations

2. **Probe generalization**:
   Train probes on train set, test on test set. Do probe accuracies also drop?

---

### Problem: "Model works on average but fails on specific subgroup"

**Interpretability diagnosis**:

1. **Stratified UMAP**: Color points by subgroup, look for patterns
2. **Subgroup probe accuracy**: Does concept encoding differ by subgroup?
3. **SHAP analysis by subgroup**: Are different features important for different groups?

**Example**:
```python
# Analyze by subgroup
for subgroup in ['group_A', 'group_B', 'group_C']:
    mask = (metadata['subgroup'] == subgroup)
    X_sub = X[mask]
    y_sub = y[mask]

    acc = model.evaluate(X_sub, y_sub)
    probe_acc = train_probe(X_sub, concept_labels[mask])

    print(f"{subgroup}: acc={acc:.2%}, probe={probe_acc:.2%}")
```

---

### Problem: "Model predictions don't make intuitive sense"

**Interpretability diagnosis**:

1. **Grad-CAM/SHAP**: What features is it actually using?
   - If it's using background instead of object → data leakage or spurious correlation

2. **Adversarial examples**: Can you easily fool it?
   - High adversarial vulnerability → not learning robust features

3. **Probes for sanity concepts**: Does it encode basic concepts?
   ```python
   # For image classifier, test if it encodes:
   - "contains object" (vs background-only)
   - "object location" (is it spatially aware?)
   - "object size"
   ```

---

## Performance Considerations

### Scaling to Large Models

**Problem**: Interpretability methods can be slow on large models.

**Solutions**:

1. **Sample intelligently**: Don't analyze all data
   ```python
   # Stratified sampling
   from sklearn.model_selection import StratifiedShuffleSplit
   splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
   idx, _ = next(splitter.split(X, y))
   X_sample = X[idx]
   ```

2. **Use faster approximations**:
   - SHAP: Use `TreeExplainer` (fast) instead of `KernelExplainer` (slow)
   - Integrated Gradients: Reduce steps (e.g., 50 instead of 300)
   - UMAP: Reduce `n_neighbors` for speed

3. **Parallelize**: Most libraries support batch processing
   ```python
   # Captum supports batching
   attributions = integrated_gradients.attribute(
       input_batch,  # (batch_size, ...)
       target=target_batch
   )
   ```

4. **Cache activations**: Extract once, analyze many times
   ```python
   # Save activations to disk
   torch.save({
       'activations': X,
       'labels': y,
       'metadata': metadata
   }, 'cached_activations.pt')

   # Load for multiple analyses
   data = torch.load('cached_activations.pt')
   ```

---

### GPU vs CPU

**General guidance**:

| Method | Best Device | Why |
|--------|-------------|-----|
| Grad-CAM | GPU | Requires backward pass |
| Integrated Gradients | GPU | Many forward passes |
| SHAP (model-based) | GPU | Model inference |
| SHAP (Kernel) | CPU | Not model-dependent |
| Probes (training) | GPU if large | Matrix operations |
| UMAP/t-SNE | CPU | Not GPU-optimized (usually) |
| Mutual information | CPU | k-NN algorithms |

---

## Pitfalls and How to Avoid Them

### Pitfall 1: Confirmation Bias

**Problem**: Finding what you expect to find, ignoring contradictions.

**Solution**:
- Always include negative controls (shuffled labels)
- Pre-register hypotheses before analyzing
- Look for counter-evidence
- Have someone else review interpretations

---

### Pitfall 2: Over-interpreting Noise

**Problem**: Seeing patterns in random fluctuations.

**Solution**:
- Bootstrap confidence intervals
- Test statistical significance
- Average over multiple random seeds
- Compare to random baseline

```python
# Example: Bootstrap probe accuracy
accuracies = []
for i in range(100):  # 100 bootstrap samples
    X_boot, y_boot = resample(X_train, y_train)
    probe = LogisticRegression().fit(X_boot, y_boot)
    accuracies.append(probe.score(X_test, y_test))

mean_acc = np.mean(accuracies)
ci_95 = np.percentile(accuracies, [2.5, 97.5])
print(f"Accuracy: {mean_acc:.3f} [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
```

---

### Pitfall 3: Ignoring Confounds

**Problem**: Mistaking correlation for causation.

**Solution**:
- Use causal interventions (activation patching)
- Test with ablations
- Control for confounding variables
- Validate with multiple datasets

---

### Pitfall 4: Method Artifacts

**Problem**: Seeing artifacts of the analysis method, not the model.

**Solution**:
- Sanity checks (randomization test, model parameter randomization)
- Use multiple methods (triangulation)
- Understand method assumptions

**Example sanity check**:
```python
# Does Grad-CAM work with randomized model?
model_random = copy.deepcopy(model)
for param in model_random.parameters():
    param.data = torch.randn_like(param.data)

attr_real = grad_cam.attribute(input, target=target)
attr_random = grad_cam.attribute_with_model(model_random, input, target=target)

# attr_real should be very different from attr_random
# If they're similar, Grad-CAM might just be showing input structure
```

---

## Quick Reference Cheat Sheet

```
┌────────────────────────────────────────────────────────────┐
│                  INTERPRETABILITY CHEAT SHEET               │
├────────────────────────────────────────────────────────────┤
│  TASK                    │ TOOL              │ TIME        │
├──────────────────────────┼───────────────────┼─────────────┤
│  Explain image pred      │ Grad-CAM          │ 10 min      │
│  Explain text pred       │ Attention / IG    │ 15 min      │
│  Explain tabular pred    │ SHAP              │ 20 min      │
│  Test concept encoding   │ Linear probe      │ 30 min      │
│  Visualize structure     │ UMAP              │ 15 min      │
│  Compare models          │ CKA               │ 20 min      │
│  Track training          │ PHATE on ckpts    │ 30 min      │
│  Find feature importance │ SHAP / IG         │ 20-60 min   │
│  Deep mechanistic understanding │ Circuits   │ Days        │
├──────────────────────────┴───────────────────┴─────────────┤
│  ALWAYS INCLUDE:                                            │
│  ✓ Baseline/control comparisons                            │
│  ✓ Statistical significance testing                         │
│  ✓ Multiple methods for validation                          │
│  ✓ Sanity checks (randomization tests)                     │
└────────────────────────────────────────────────────────────┘
```

---

## Next Steps

**You now have practical recipes for common interpretability tasks!**

### Go Deeper

- **[Methods](../2_methods/)**: Detailed guides for each technique
- **[Tutorials](../6_tutorials/)**: Full working notebooks
- **[Case Studies](../5_case_studies/)**: Real-world applications

### Expand Your Toolkit

- **[Tools](../4_tools_and_libraries/)**: Learn more libraries (TransformerLens, etc.)
- **[Architecture Guides](../3_architectures/)**: Architecture-specific best practices

### Build Foundation

- **[Foundations](../1_foundations/)**: Understand the theory behind the tools
- **[Glossary](glossary.md)**: Build your interpretability vocabulary

---

**Questions?** Return to the [main landing page](README.md) or explore related guides.

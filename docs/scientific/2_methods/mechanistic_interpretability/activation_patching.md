# Activation Patching (Causal Intervention)

A comprehensive guide to activation patching, a causal method for understanding information flow and identifying circuits in neural networks.

---

## Overview

**Problem**: Correlation-based methods (probes, attention visualization) show what representations contain, but not whether they're actually used.

**Solution**: Activation patching directly intervenes on activations to establish causal relationships.

**Core idea**: Replace (patch) activations during forward pass, measure effect on output.

**Also known as**: Causal tracing, activation intervention, path patching, interchange intervention

This guide covers theory, implementation, and applications for mechanistic interpretability.

---

## Why Activation Patching Matters

### Causality vs Correlation

**Probing finds correlations**:
- "Layer 5 encodes sentiment information"
- But does the model actually use it?

**Patching establishes causation**:
- Replace layer 5 activations → sentiment prediction changes
- Therefore layer 5 causally affects sentiment prediction

### Identifying Computational Circuits

**Circuit**: A subset of model components that implement a specific computation

**Patching workflow**:
1. Hypothesize a circuit (e.g., "layer 3 neuron 47 detects proper nouns")
2. Patch that neuron's activation
3. Measure effect on downstream behavior
4. If large effect → neuron is part of circuit

### Debugging Model Behavior

**Example**: Model makes incorrect prediction on specific input

**Patching helps answer**:
- Which layers contribute to the error?
- What information is missing at the failure point?
- Can we fix behavior by patching specific activations?

---

## Basic Activation Patching

### Single Activation Replacement

**Simplest form**: Replace one activation with a chosen value

```python
import torch
import torch.nn as nn

def patch_activation(model, target_layer, target_index, patch_value, input_data):
    """Replace a single activation during forward pass.

    Args:
        model: PyTorch model
        target_layer: Name of layer to patch (e.g., 'layer3')
        target_index: Which activation to patch (neuron/position index)
        patch_value: Value to replace with
        input_data: Input to model

    Returns:
        output: Model output with patched activation
        original_output: Model output without patching
    """
    activations = {}
    patched_activations = {}

    # Hook to capture activation
    def capture_hook(name):
        def hook(module, input, output):
            activations[name] = output.clone()
        return hook

    # Hook to patch activation
    def patch_hook(name):
        def hook(module, input, output):
            output = output.clone()
            # Patch specific index
            if len(output.shape) == 3:  # (batch, seq, hidden)
                output[0, target_index] = patch_value
            elif len(output.shape) == 2:  # (batch, hidden)
                output[0, target_index] = patch_value
            patched_activations[name] = output
            return output
        return hook

    # Get original output
    with torch.no_grad():
        original_output = model(input_data)

    # Register hooks and run with patching
    layer = dict(model.named_modules())[target_layer]
    handle = layer.register_forward_hook(patch_hook(target_layer))

    with torch.no_grad():
        patched_output = model(input_data)

    handle.remove()

    return patched_output, original_output

# Example usage
# model = load_model()
# input_data = tokenize("The cat sat on the mat")
# patched_out, original_out = patch_activation(
#     model, target_layer='transformer.h.5',
#     target_index=10, patch_value=torch.zeros(768),
#     input_data=input_data
# )
# effect = (patched_out - original_out).abs().max()
# print(f"Max effect of patching: {effect:.4f}")
```

### Patching with Counterfactual Activations

**More common**: Replace activations from one input (corrupted) with activations from another (clean)

```python
def counterfactual_patch(model, layer_name, clean_input, corrupted_input, patch_position=None):
    """Patch corrupted run with clean activations.

    Classic activation patching: run model on corrupted input, but
    replace activations at specific layer with those from clean run.

    Args:
        model: PyTorch model
        layer_name: Layer to patch
        clean_input: Input that produces desired behavior
        corrupted_input: Input with undesired behavior
        patch_position: Position to patch (None = all positions)

    Returns:
        patched_output: Output when corrupted input uses clean activations
        clean_output: Normal output for clean input
        corrupted_output: Normal output for corrupted input
    """
    # Storage for activations
    clean_cache = {}
    corrupted_cache = {}

    def capture_activation(cache, name):
        def hook(module, input, output):
            cache[name] = output.clone().detach()
        return hook

    def patch_activation(name):
        def hook(module, input, output):
            # Replace with clean activation
            patched = output.clone()
            if patch_position is not None:
                # Patch specific position
                patched[:, patch_position] = clean_cache[name][:, patch_position]
            else:
                # Patch all
                patched = clean_cache[name]
            return patched
        return hook

    layer = dict(model.named_modules())[layer_name]

    # Run clean input (capture activations)
    capture_handle = layer.register_forward_hook(capture_activation(clean_cache, layer_name))
    with torch.no_grad():
        clean_output = model(clean_input)
    capture_handle.remove()

    # Run corrupted input normally
    with torch.no_grad():
        corrupted_output = model(corrupted_input)

    # Run corrupted input with patching
    patch_handle = layer.register_forward_hook(patch_activation(layer_name))
    with torch.no_grad():
        patched_output = model(corrupted_input)
    patch_handle.remove()

    return patched_output, clean_output, corrupted_output

# Example: Indirect Object Identification (IOI) task
# clean: "When John and Mary went to the store, Mary gave a drink to"
# corrupted: "When John and Susan went to the store, Mary gave a drink to"
# Question: Which layer recovers correct answer ("John") when patched?

# clean_tokens = tokenize("When John and Mary went to the store, Mary gave a drink to")
# corrupt_tokens = tokenize("When John and Susan went to the store, Mary gave a drink to")
#
# patched_out, clean_out, corrupt_out = counterfactual_patch(
#     model, layer_name='transformer.h.8',
#     clean_input=clean_tokens, corrupted_input=corrupt_tokens
# )
#
# # Measure restoration of correct answer
# clean_logit = clean_out[0, -1, john_token_id]
# corrupt_logit = corrupt_out[0, -1, john_token_id]
# patched_logit = patched_out[0, -1, john_token_id]
#
# recovery = (patched_logit - corrupt_logit) / (clean_logit - corrupt_logit)
# print(f"Patching layer 8 recovers {recovery*100:.1f}% of clean performance")
```

---

## Systematic Patching Studies

### Layer-wise Patching

**Question**: Which layer is most important for a task?

```python
def layer_patching_sweep(model, clean_input, corrupted_input, target_metric_fn):
    """Patch each layer sequentially to find critical layers.

    Args:
        model: Model to analyze
        clean_input: Clean input
        corrupted_input: Corrupted input
        target_metric_fn: Function to compute metric from output
                         (e.g., lambda out: out[0, -1, target_token])

    Returns:
        layer_effects: Dictionary mapping layer name to recovery percentage
    """
    # Get baseline metrics
    with torch.no_grad():
        clean_output = model(clean_input)
        corrupted_output = model(corrupted_input)

    clean_metric = target_metric_fn(clean_output)
    corrupted_metric = target_metric_fn(corrupted_output)

    # Iterate through layers
    layer_effects = {}

    for layer_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):  # Adjust based on your model
            continue

        print(f"Patching {layer_name}...")

        patched_output, _, _ = counterfactual_patch(
            model, layer_name, clean_input, corrupted_input
        )

        patched_metric = target_metric_fn(patched_output)

        # Compute recovery percentage
        if clean_metric != corrupted_metric:
            recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
        else:
            recovery = torch.tensor(0.0)

        layer_effects[layer_name] = recovery.item()

        print(f"  Recovery: {recovery.item()*100:.1f}%")

    return layer_effects

# Visualize results
def plot_layer_effects(layer_effects):
    """Plot which layers matter most."""
    import matplotlib.pyplot as plt

    layers = list(layer_effects.keys())
    effects = list(layer_effects.values())

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layers)), effects)
    plt.xlabel('Layer')
    plt.ylabel('Recovery Percentage')
    plt.title('Effect of Patching Each Layer')
    plt.xticks(range(len(layers)), layers, rotation=45, ha='right')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axhline(y=1, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# Usage
# def metric_fn(output):
#     return output[0, -1, target_token_id]  # Logit for target token
#
# effects = layer_patching_sweep(model, clean_input, corrupt_input, metric_fn)
# plot_layer_effects(effects)
```

### Position-wise Patching

**Question**: Which token positions matter?

```python
def position_patching_sweep(model, layer_name, clean_input, corrupted_input, target_metric_fn):
    """Patch each position independently to find critical positions.

    Args:
        model: Model
        layer_name: Which layer to patch
        clean_input: Clean input tokens
        corrupted_input: Corrupted input tokens
        target_metric_fn: Metric function

    Returns:
        position_effects: Array of recovery per position
    """
    seq_len = clean_input.shape[1]

    # Baselines
    with torch.no_grad():
        clean_output = model(clean_input)
        corrupted_output = model(corrupted_input)

    clean_metric = target_metric_fn(clean_output)
    corrupted_metric = target_metric_fn(corrupted_output)

    position_effects = []

    for pos in range(seq_len):
        print(f"Patching position {pos}...")

        patched_output, _, _ = counterfactual_patch(
            model, layer_name, clean_input, corrupted_input,
            patch_position=pos
        )

        patched_metric = target_metric_fn(patched_output)

        recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
        position_effects.append(recovery.item())

    return position_effects

# Visualize
def plot_position_effects(position_effects, tokens):
    """Plot effect of patching each position."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(position_effects)), position_effects)
    plt.xlabel('Token Position')
    plt.ylabel('Recovery Percentage')
    plt.title('Effect of Patching Each Position')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

# Usage
# effects = position_patching_sweep(model, 'transformer.h.8', clean_input, corrupt_input, metric_fn)
# plot_position_effects(effects, token_strings)
```

### Neuron-level Patching

**Fine-grained**: Patch individual neurons

```python
def neuron_patching_sweep(model, layer_name, clean_input, corrupted_input, target_metric_fn):
    """Patch each neuron independently.

    Warning: Expensive for large hidden dimensions.

    Args:
        model: Model
        layer_name: Layer to analyze
        clean_input, corrupted_input: Inputs
        target_metric_fn: Metric function

    Returns:
        neuron_effects: (hidden_dim,) array of effects
    """
    # Get hidden dimension
    layer = dict(model.named_modules())[layer_name]

    # Capture clean activations
    clean_cache = {}

    def capture_hook(name):
        def hook(module, input, output):
            clean_cache[name] = output.clone().detach()
        return hook

    handle = layer.register_forward_hook(capture_hook(layer_name))
    with torch.no_grad():
        clean_output = model(clean_input)
    handle.remove()

    hidden_dim = clean_cache[layer_name].shape[-1]

    # Baselines
    with torch.no_grad():
        corrupted_output = model(corrupted_input)

    clean_metric = target_metric_fn(clean_output)
    corrupted_metric = target_metric_fn(corrupted_output)

    neuron_effects = []

    for neuron_idx in range(hidden_dim):
        if neuron_idx % 100 == 0:
            print(f"Progress: {neuron_idx}/{hidden_dim}")

        # Patch this neuron only
        def patch_hook(module, input, output):
            output = output.clone()
            output[..., neuron_idx] = clean_cache[layer_name][..., neuron_idx]
            return output

        handle = layer.register_forward_hook(patch_hook)
        with torch.no_grad():
            patched_output = model(corrupted_input)
        handle.remove()

        patched_metric = target_metric_fn(patched_output)
        recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)
        neuron_effects.append(recovery.item())

    return torch.tensor(neuron_effects)

# Find most important neurons
def find_top_neurons(neuron_effects, k=20):
    """Identify most important neurons."""
    top_indices = torch.topk(neuron_effects, k=k).indices
    top_effects = neuron_effects[top_indices]

    print(f"Top {k} neurons:")
    for idx, effect in zip(top_indices, top_effects):
        print(f"  Neuron {idx}: {effect*100:.1f}% recovery")

    return top_indices, top_effects

# Usage
# neuron_effects = neuron_patching_sweep(model, 'transformer.h.5', clean_input, corrupt_input, metric_fn)
# top_neurons, top_effects = find_top_neurons(neuron_effects, k=20)
```

---

## Path Patching

**Advanced**: Patch paths through network, not just layers

### Concept

**Standard patching**: Patch all of layer L
**Path patching**: Patch only the path from layer L to specific downstream component

**Why**: Isolate specific information flows

### Implementation

```python
def path_patch(model, source_layer, target_layer, clean_input, corrupted_input, target_metric_fn):
    """Patch the path from source layer to target layer.

    Only the connection from source → target uses clean activations;
    everything else uses corrupted activations.

    Args:
        model: Model
        source_layer: Layer where path starts
        target_layer: Layer where path ends
        clean_input, corrupted_input: Inputs
        target_metric_fn: Metric function

    Returns:
        recovery: How much the path matters
    """
    # This is a simplified version; real implementation requires
    # careful tracking of computational graph

    # Capture clean activations at source
    clean_activations = {}

    def capture_clean(name):
        def hook(module, input, output):
            clean_activations[name] = output.clone().detach()
        return hook

    source_module = dict(model.named_modules())[source_layer]
    handle = source_module.register_forward_hook(capture_clean(source_layer))
    with torch.no_grad():
        clean_output = model(clean_input)
    handle.remove()

    clean_metric = target_metric_fn(clean_output)

    # Run corrupted normally
    with torch.no_grad():
        corrupted_output = model(corrupted_input)
    corrupted_metric = target_metric_fn(corrupted_output)

    # Path patch: inject clean activations at source, but only
    # for computation that flows to target

    # This requires model-specific implementation
    # For transformers: patch attention from source position to target position
    # For fully connected: patch specific weight pathways

    # Simplified: patch source layer entirely
    # (Real path patching is more surgical)

    def patch_source(module, input, output):
        return clean_activations[source_layer]

    handle = source_module.register_forward_hook(patch_source)
    with torch.no_grad():
        patched_output = model(corrupted_input)
    handle.remove()

    patched_metric = target_metric_fn(patched_output)

    recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric)

    print(f"Path from {source_layer} to {target_layer}: {recovery*100:.1f}% recovery")

    return recovery.item()

# For real path patching in transformers, see:
# - Goldowsky-Dill et al. (2023): Localizing Model Behavior with Path Patching
```

---

## Activation Addition

**Related technique**: Add, rather than replace, activations

### Concept

**Patching**: `activation = clean_activation`
**Addition**: `activation = activation + steering_vector`

**Use case**: Steer model behavior in specific direction

### Implementation

```python
def activation_steering(model, layer_name, steering_vector, input_data, strength=1.0):
    """Add a steering vector to activations.

    Args:
        model: Model
        layer_name: Layer to steer
        steering_vector: Direction to add (e.g., "happiness" direction)
        input_data: Input
        strength: Magnitude of steering

    Returns:
        steered_output: Output with steering applied
    """
    def steering_hook(module, input, output):
        # Add steering vector
        output = output + strength * steering_vector
        return output

    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(steering_hook)

    with torch.no_grad():
        steered_output = model(input_data)

    handle.remove()

    return steered_output

# Example: Steer toward positive sentiment
# positive_examples = ... # Hidden states from positive reviews
# negative_examples = ... # Hidden states from negative reviews
#
# # Compute steering direction
# steering_vector = (positive_examples.mean(dim=0) - negative_examples.mean(dim=0))
# steering_vector = steering_vector / steering_vector.norm()
#
# # Steer neutral input toward positive
# neutral_input = tokenize("This product is")
# steered_output = activation_steering(model, 'transformer.h.10', steering_vector, neutral_input, strength=5.0)
# print(generate_from_output(steered_output))  # Should be more positive
```

---

## Applications to Circuit Discovery

### Case Study: Indirect Object Identification

**Task**: "When John and Mary went to the store, John gave a drink to ___"
**Correct**: Mary

**Circuit hypothesis**: Name mover heads copy "Mary" to output

**Validation via patching**:
1. Corrupt input: Replace "Mary" with "Susan" early in sentence
2. Patch attention heads one by one
3. Measure restoration of correct answer ("Mary")
4. Heads with high restoration are part of circuit

```python
def find_name_mover_heads(model, clean_prompt, corrupted_prompt, target_name_token):
    """Identify attention heads that move names to output.

    Args:
        model: Transformer model
        clean_prompt: Original prompt
        corrupted_prompt: Prompt with name changed
        target_name_token: Token ID for correct name

    Returns:
        head_effects: Dictionary of (layer, head) -> recovery score
    """
    def metric_fn(output):
        # Logit for target name at final position
        return output[0, -1, target_name_token]

    head_effects = {}

    # Iterate through all attention heads
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    for layer in range(n_layers):
        for head in range(n_heads):
            print(f"Testing layer {layer}, head {head}")

            # Patch this head's output
            # (Implementation requires accessing head outputs specifically)

            layer_name = f'transformer.h.{layer}.attn'
            # Simplified: patch entire attention layer
            # Real implementation would patch specific head

            patched_output, clean_output, corrupt_output = counterfactual_patch(
                model, layer_name, clean_prompt, corrupted_prompt
            )

            clean_metric = metric_fn(clean_output)
            corrupt_metric = metric_fn(corrupt_output)
            patched_metric = metric_fn(patched_output)

            recovery = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)
            head_effects[(layer, head)] = recovery.item()

    # Find top heads
    sorted_heads = sorted(head_effects.items(), key=lambda x: x[1], reverse=True)

    print("\nTop name mover heads:")
    for (layer, head), effect in sorted_heads[:10]:
        print(f"  Layer {layer}, Head {head}: {effect*100:.1f}% recovery")

    return head_effects

# Usage
# clean = tokenize("When John and Mary went to the store, John gave a drink to")
# corrupt = tokenize("When John and Susan went to the store, John gave a drink to")
# mary_token_id = tokenizer.encode("Mary")[0]
#
# head_effects = find_name_mover_heads(model, clean, corrupt, mary_token_id)
```

---

## Best Practices

### Choosing Baselines

**Clean vs corrupted inputs should differ minimally**

Good:
- Clean: "The cat sat on the mat"
- Corrupted: "The dog sat on the mat"

Bad:
- Clean: "The cat sat on the mat"
- Corrupted: "Random tokens"

**Why**: Want to isolate specific computational difference

### Interpreting Recovery Scores

**Recovery = (patched - corrupted) / (clean - corrupted)**

- 0%: Patching has no effect (layer not involved)
- 100%: Patching fully restores clean behavior (layer is critical)
- >100%: Patching overcorrects (can happen with nonlinear interactions)
- Negative: Patching makes things worse

**Beware**: High recovery doesn't mean layer is sufficient, only that it's necessary

### Multiple Testing Correction

**Problem**: Testing many positions/neurons/heads → false positives

**Solution**: Apply multiple testing correction (Bonferroni, FDR)

```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_testing(effects, alpha=0.05):
    """Apply FDR correction to patching results.

    Args:
        effects: Dictionary or array of effect sizes
        alpha: Significance threshold

    Returns:
        significant_indices: Which tests pass correction
    """
    # Convert to array if needed
    if isinstance(effects, dict):
        keys = list(effects.keys())
        values = np.array(list(effects.values()))
    else:
        values = np.array(effects)
        keys = list(range(len(values)))

    # Compute p-values (assuming effect ~ t-distribution, simplified)
    # In practice, use bootstrap or permutation tests

    # Apply FDR correction
    # rejected, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    # For now, simple threshold on effect size
    threshold = np.percentile(values, 95)  # Top 5%
    significant = values > threshold

    significant_keys = [k for k, s in zip(keys, significant) if s]

    print(f"Significant components after correction: {len(significant_keys)}")

    return significant_keys

# Usage
# significant_neurons = correct_multiple_testing(neuron_effects, alpha=0.05)
```

### Computational Efficiency

**Problem**: Patching every neuron is expensive

**Solutions**:
1. **Hierarchical search**: First patch layers, then positions, then neurons
2. **Batching**: Patch multiple components simultaneously if independent
3. **Early stopping**: Stop if no components show large effects
4. **Importance sampling**: Focus on high-variance neurons

---

## Tools and Libraries

### TransformerLens

**Recommended library** for patching in transformers

```python
# pip install transformer-lens

from transformer_lens import HookedTransformer

# Load model with hooks
model = HookedTransformer.from_pretrained('gpt2-small')

# Activation patching utilities built-in
from transformer_lens import patching

# Run patching experiment
results = patching.get_act_patch_resid_pre(
    model=model,
    clean_tokens=clean_tokens,
    corrupted_tokens=corrupt_tokens,
    metric=metric_fn
)

# Visualize
patching.plot_patch_effect(results)
```

### Nnsight

**General-purpose intervention library**

```python
# pip install nnsight

from nnsight import NNsight

model = NNsight(your_pytorch_model)

with model.trace(input_data):
    # Intervene on activations
    model.layer5.output = clean_activations
    output = model.output.save()
```

---

## Advanced Topics

### Gradient-Based Patching

**Combine with gradients**: Find optimal patch

```python
def optimal_patch_direction(model, layer_name, corrupted_input, target_metric_fn, n_steps=100, lr=0.01):
    """Find optimal direction to patch for maximum effect.

    Args:
        model: Model
        layer_name: Layer to patch
        corrupted_input: Input to improve
        target_metric_fn: Metric to optimize
        n_steps: Optimization steps
        lr: Learning rate

    Returns:
        optimal_direction: Best direction to patch
    """
    layer = dict(model.named_modules())[layer_name]

    # Initialize learnable patch direction
    hidden_dim = get_hidden_dim(layer)
    patch_direction = torch.zeros(hidden_dim, requires_grad=True)

    optimizer = torch.optim.Adam([patch_direction], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Apply patch
        def patch_hook(module, input, output):
            return output + patch_direction

        handle = layer.register_forward_hook(patch_hook)
        output = model(corrupted_input)
        handle.remove()

        # Compute loss (negative metric to maximize)
        metric = target_metric_fn(output)
        loss = -metric

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: Metric = {metric.item():.4f}")

    return patch_direction.detach()

# Usage: Find direction that maximizes target token probability
# optimal_dir = optimal_patch_direction(model, 'transformer.h.8', corrupt_input, metric_fn)
```

### Distributed Patching

**Patch multiple components simultaneously**

Useful for identifying whether circuit components work together or independently.

---

## Further Reading

**Foundational papers**:
- Vig et al. (2020): "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
- Meng et al. (2022): "Locating and Editing Factual Associations in GPT"
- Wang et al. (2023): "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"
- Goldowsky-Dill et al. (2023): "Localizing Model Behavior with Path Patching"

**Related methods**:
- [Linear Probes](../probing/linear_probes.md) - Correlation-based analysis
- [Circuits](circuits.md) - Full circuit reverse-engineering
- [What is Interpretability?](../../1_foundations/what_is_interpretability.md) - Causality in interpretability

**Tools**:
- TransformerLens documentation: https://transformerlensorg.github.io/TransformerLens/
- Nnsight documentation: https://nnsight.net/

**Full bibliography**: [References](../../references/bibliography.md)

---

**Return to**: [Methods](../README.md) | [Main Handbook](../../0_start_here/README.md)

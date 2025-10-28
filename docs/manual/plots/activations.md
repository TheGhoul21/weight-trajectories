# CNN activation maps (Grad-CAM)

Produced by: `./wt.sh visualize --viz-type activations --checkpoint-dir <run>`
Backed by: `src/visualize_trajectories.py`

## Purpose

Visualize which board regions the CNN focuses on when making predictions. Uses Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight influential spatial locations for policy decisions or value estimates.

## Grad-CAM algorithm

### Overview
Grad-CAM produces a heatmap showing which input pixels contribute most to a target output (policy move or value). It works by:
1. Identifying which CNN feature maps are important (via gradients)
2. Weighting those feature maps by importance
3. Projecting back to input resolution

### Step-by-step computation

**1. Register hooks on target layer**

Target layer: Last CNN layer (final ResNet block output)
```python
def forward_hook(module, input, output):
    activations = output.detach()  # Store feature maps

def backward_hook(module, grad_input, grad_output):
    gradients = grad_output[0].detach()  # Store gradients

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)
```

**2. Forward pass**
```python
board_tensor.requires_grad_(True)
policy_logits, value, hidden = model(board_tensor)

# Activations captured by hook: shape (1, channels, H, W)
# For ResNet output: typically (1, 64, 6, 7) or similar
```

**3. Select target scalar**
```python
if target == 'policy':
    # Focus on specific move (user-specified or predicted)
    focus_move = move if move is not None else argmax(policy_logits)
    target_scalar = policy_logits[0, focus_move]  # Single logit
else:  # target == 'value'
    target_scalar = value[0]  # Scalar value estimate
```

**4. Backward pass**
```python
target_scalar.backward()

# Gradients captured by hook: shape (1, channels, H, W)
# Shows how each feature map element affects target_scalar
```

**5. Compute channel importance weights**

Global Average Pooling over spatial dimensions:
```python
weights = gradient.mean(dim=(2, 3), keepdim=True)  # (1, channels, 1, 1)
```

**Interpretation**: `weights[k]` = importance of channel k for the target
- Positive weight: channel k supports the prediction
- Negative weight: channel k opposes the prediction

**6. Weighted combination of feature maps**
```python
cam = (weights * activations).sum(dim=1)  # Sum over channels
# cam shape: (1, 1, H, W)
```

This produces a coarse spatial attention map at CNN feature resolution.

**7. Apply ReLU**
```python
cam = ReLU(cam)  # Zero out negative contributions
```

Only keeps regions that **increase** the target (policy logit or value).

**8. Upsample to board resolution**
```python
cam = F.interpolate(cam, size=(6, 7), mode='bilinear')
# Resize from feature map size to Connect Four board size
```

**9. Normalize to [0, 1]**
```python
cam = (cam - cam.min()) / (cam.max() - cam.min() + epsilon)
```

**Result**: Heatmap where 1.0 = most important cell, 0.0 = least important

### Why this works

**Intuition**:
- If increasing feature map F[k, i, j] increases the target, then:
  - Gradient ∂target/∂F[k,i,j] is positive (large)
  - That spatial location (i, j) in that channel is important
- Weight by channel importance, sum across channels → spatial importance map

**Compared to raw gradients**:
- Raw gradients: noisy, hard to interpret
- Grad-CAM: smooth, channel-aggregated, spatially coherent

## What it shows
- For a few random board states, overlays a Grad-CAM heatmap on the 6×7 grid
- Target can be policy (per-move logit; choose a move via --activation-move or default to predicted) or value

Artifacts
- visualizations/activations/activation_000.png, activation_001.png, ...
  - Titles show target type, focus move (and its probability) and predicted value
  - Turn indicator is printed, and axes label columns 0..6 and rows 5..0

How to read
- **Color intensity**: brighter cells contributed more to the prediction (policy logit or value). Cool/neutral colours carry little weight.
- **Policy target**
  - Expect heat concentrated on winning threats, blocking moves, or long columns/diagonals tied to the selected action.
  - If the highlighted move looks unreasonable, double-check the policy probability in the title—low-confidence selections often produce diffuse maps.
- **Value target**
  - Bright streaks typically trace potential connect-fours (for/against the current player).
  - Watch for heat on opponent pieces: strong value intuition should highlight both offensive opportunities and defensive liabilities.
- **Turn indicator** reminds you whose perspective the map reflects; heat should align with the acting player’s threats.

Common diagnostics
- **Feature blindness**: flat/diffuse heatmaps across many boards —> the CNN isn’t focusing on specific patterns.
- **Over-focus**: always lighting up the last move played —> the network might be shortcutting via move history rather than global structure.
- **Inconsistent policy/value maps**: if policy heat ignores the regions that the value map cares about, the actor/critic components are disagreeing.

Options
- --n-boards: pool size of random boards to sample from
- --activation-target [policy|value], --activation-move index, --activation-max-examples

# Weight Trajectory Analysis with PHATE

**Idea in Brief**: Training checkpoints form a time-ordered curve through weight
space. PHATE treats that curve as a manifold, preserving both local continuity
and global phase changes so that learning dynamics become visible. Use it to
spot when optimisation takes shortcuts, stalls, or branches across ablations.

## Overview

Training a neural network traces a path through high-dimensional parameter space. Understanding this trajectory reveals learning dynamics: whether training proceeds smoothly or exhibits phase transitions, whether different architectural components co-evolve or learn at different rates, and how weight changes relate to performance improvements. PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding) provides a method for visualizing these trajectories while preserving their temporal structure.

## Why PHATE for Training Trajectories

Parameter evolution during training has specific characteristics that influence embedding choice. Checkpoints are temporally ordered and typically few in number (20-50 samples). Changes should be continuous—adjacent epochs should have similar weights—making trajectory preservation essential. Regime transitions (phase changes in learning dynamics) should be visually apparent.

PHATE addresses these requirements through diffusion-based distance computation. Unlike PCA (purely linear), PHATE captures nonlinear structure. Unlike t-SNE (which can tear trajectories apart to optimize local clustering), PHATE explicitly preserves both local neighborhoods and global manifold structure. The diffusion process acts as a denoising operator, revealing underlying trends despite noisy per-epoch fluctuations.

## Data Flow

1. **Checkpoint ingestion**
   - The script scans a checkpoint directory for `weights_epoch_*.pt` files, ordered by the epoch embedded in the file name.
   - Each checkpoint is loaded with `torch.load` so that both `state_dict` and metadata (for example, stored epoch number) can be inspected.

2. **Weight extraction**
   - CNN weights: all convolutional kernels that live under the ResNet backbone are flattened and concatenated per checkpoint, forming a matrix of shape `(n_checkpoints, n_cnn_params)`.
   - GRU weights: analogous handling for all GRU matrices/biases yields `(n_checkpoints, n_gru_params)`.

3. **Representation snapshots**
   - When asked to plot board states or full game trajectories, `create_model` rebuilds the network (`ResNetGRUConnect4`) and restores the chosen checkpoint.
   - Each supplied board tensor is pushed through the model; the hidden state returned by the GRU serves as the latent representation for PHATE.

4. **Embedding with PHATE**
   - For each matrix, fit `phate.PHATE(n_components=2, knn=k, t=10)`.
   - Choose `knn` consistent with sample count to maintain a well‑posed affinity graph.

5. **Plotting**
   - Weight trajectories are displayed with a grey backbone line, coloured points for epoch ordering, and start/end markers.
   - Minimal and final validation-loss checkpoints are highlighted automatically (orange diamond and red star) when `training_history.json` is present.
   - Representation plots mirror this layout, while game trajectories additionally draw a polylined path to emphasize temporal order.
   - Optional file outputs land in `visualizations/` (or any directory specified through `--output-dir`).

## CLI Usage

```bash
python -m src.visualize_trajectories \
  --checkpoint-dir checkpoints/k3_c16_gru8_20251022_193955 \
  --viz-type all \
  --output-dir visualizations/20251022_run
```

`--viz-type` accepts `all`, `cnn`, `gru`, `boards`, or `summary`. The `boards` mode uses randomly generated boards by default. Point it to curated samples by replacing the call to `generate_random_boards` in your own driver script.

## Practical Notes

- PHATE fit scales roughly as O(n_samples^2) in the number of checkpoints; tens of checkpoints are practical.
- CPU is sufficient; CUDA is not required.
- When weight vectors are very wide, apply a compact PCA (≤ min(32, n_samples−1)) before PHATE to reduce variance dominated by scale.
- Expose `knn`, `t`, and kernel decay to tune local/global structure and readability across runs.

## Ablation Comparison

Use the new ablation helpers to embed checkpoints from multiple runs in a single PHATE space. The script pads shorter weight vectors with zeros so every run shares a consistent feature dimension before fitting the joint embedding.

```bash
python -m src.visualize_trajectories \
   --viz-type ablation-cnn \
   --ablation-dirs checkpoints/k3_c16_gru8_20251022_193955 \
                            checkpoints/k3_c64_gru8_20251023_052225 \
                            checkpoints/k3_c256_gru8_20251023_055910
```

- `ablation-cnn` compares convolutional kernels; `ablation-gru` targets recurrent weights.
- A shared embedding across runs clarifies qualitatively different routes through parameter space even with differing dimensionalities (after padding/alignment).

### Checkpoint Filtering

- `--epoch-min` / `--epoch-max` / `--epoch-step` let you focus on any window of the training history (e.g., only the first 20 epochs or every fifth checkpoint). All visual modes respect these filters, including ablations and joint plots.

## CNN Activation Maps

Generate Grad-CAM heatmaps that highlight which board squares drive the CNN features for a checkpoint:

```bash
python -m src.visualize_trajectories \
   --checkpoint-dir checkpoints/k3_c16_gru8_20251024_033336 \
   --viz-type activations \
   --activation-target policy \
   --activation-max-examples 6 \
   --output-dir visualizations/activations_demo
```

- Outputs overlay Grad‑CAM intensity on top of board states to indicate spatial saliency for policy/value predictions.

## Joint CNN/GRU Trajectory

Plot both weight families in the same PHATE space for a single configuration. The CNN curve uses filled markers, the GRU uses outlined markers at the same epochs.

```bash
python -m src.visualize_trajectories \
   --checkpoint-dir checkpoints/k3_c16_gru8_20251024_033336 \
   --viz-type joint \
   --joint-center anchor \
   --output-dir visualizations/joint_k3_c16_gru8
```

- Centering options help align curves for readability (anchor at epoch zero; optional path‑length normalization).

## Extensions and Alternatives

**Custom trajectory sources**: Replace `generate_random_boards` with handcrafted board sequences or load saved tensors from `data/` directory to analyze specific game scenarios.

**Animation**: `matplotlib.animation.FuncAnimation` can animate the 2D trajectory, showing weight evolution as a movie. The import is already available in the visualization script.

**Alternative embedding methods**: While PHATE is the default for trajectory preservation, UMAP (faster, good global structure) and t-SNE (emphasizes local clustering) can serve as sensitivity checks. Comparing embeddings across methods validates that observed structure is not an artifact of the projection algorithm.

---

## Documentation Cross-References

**Mathematical foundations**: [Theoretical Foundations](theoretical_foundations.md#4-manifold-learning-and-trajectory-embedding) covers the manifold hypothesis, PHATE algorithm details, diffusion geometry, and comparative analysis with PCA, t-SNE, and UMAP.

**Learning dynamics connection**: [Theoretical Foundations](theoretical_foundations.md#6-learning-dynamics-how-attractors-emerge) explains how weight trajectories in parameter space correspond to evolution of attractor landscapes in hidden state space.

**Practical applications**: [Case Studies](case_studies.md) presents Maheswaranathan's line attractor analysis using PCA (case study #2) and Yang's compositional PCA for multi-task RNNs (case study #5), demonstrating trajectory analysis in different contexts.

**Implementation status**: [GRU Observability: Intuition, Methods, and Recommendations](gru_observability_literature.md) summarizes current capabilities and planned extensions.

**Methodological references**: [References](references.md) section "Manifold Learning & Trajectory Embedding" provides citations for PHATE (Moon et al. 2019), T-PHATE (Rübel et al. 2023), UMAP (McInnes et al. 2018), and related methods.

Temporal axes and T‑PHATE
- Two orthogonal time axes matter for learning dynamics: training epochs and in‑game timesteps. We visualize both: (i) fix a game step and follow its hidden state across epochs; (ii) fix an epoch (e.g., min val loss) and follow hidden states across game steps.
- T‑PHATE improves temporal structure by combining delay embeddings and a temporal affinity kernel, yielding clearer phase transitions and smoother manifold trajectories.

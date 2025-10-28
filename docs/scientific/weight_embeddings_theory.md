# Weight Trajectory Visualization

This note summarizes the intent and internals of `src/visualize_trajectories.py`, the script that turns training checkpoints into PHATE embeddings and diagnostic plots.

## Why PHATE

PHATE captures progressive changes in high-dimensional data and tends to produce smoother trajectories than PCA or t-SNE when sampling is sparse (a handful of checkpoints). The script adapts the `knn` parameter automatically so that very small checkpoint collections (for example, only five epochs) still yield stable embeddings without overfitting.

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
   - For every weight or representation matrix, the script instantiates `phate.PHATE(n_components=2, knn=k, t=10)`.
   - `knn` is capped by the sample count (`max(2, min(max_knn, n_samples - 1))`) to keep the kernel well-posed even in tiny datasets.

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

- The PHATE fit scales at roughly `O(n_samples^2)`; expect a few seconds for 50+ checkpoints.
- GPU acceleration is not required; the script pins everything to CPU unless CUDA is available.
- If you add BatchNorm weight visualizations later, make sure to filter them separately so they do not overwhelm the embedding with scale differences.
- When flattened weights exceed roughly five thousand dimensions, the CLI now auto-runs a PCA down to at most 32 components (and never more than the checkpoint count minus one) before PHATE. Override this with `--phate-n-pca <k>` if you want a different compression budget—the request will be clipped to the valid range.
- Advanced PHATE knobs are exposed: `--phate-knn` widens or shrinks the local graph, `--phate-t` extends diffusion for clearer global structure, and `--phate-decay` adjusts how quickly affinities fall off. Combine them with `--ablation-center normalize` or `anchor` to keep multi-run plots legible.
- Automate whole sweeps with `scripts/run_visualization_suite.py`, which reads a JSON config of CLI argument sets, executes each visualization, and writes a markdown recap with embedded figures.

## Ablation Comparison

Use the new ablation helpers to embed checkpoints from multiple runs in a single PHATE space. The script pads shorter weight vectors with zeros so every run shares a consistent feature dimension before fitting the joint embedding.

```bash
python -m src.visualize_trajectories \
   --viz-type ablation-cnn \
   --ablation-dirs checkpoints/k3_c16_gru8_20251022_193955 \
                            checkpoints/k3_c64_gru8_20251023_052225 \
                            checkpoints/k3_c256_gru8_20251023_055910
```

- `ablation-cnn` compares the convolutional kernels; `ablation-gru` targets the recurrent weights.
- Each trajectory retains its own start/end markers so progression is easy to follow, and the legend lists the directory plus parsed hyperparameters.
- The shared embedding makes it clear when larger models take qualitatively different routes through parameter space, even if they use more weights than the smaller variants.
- Append `--ablation-animate` to produce a GIF alongside the static PNG (requires Pillow for GIF export).
- If `training_history.json` exists per run, the combined plot reuses the orange diamond/red star convention to mark the epochs with minimal and final validation loss for each configuration.
- When one run dominates the scale, pass `--ablation-center anchor` (or `normalize`) to align every trajectory at its starting point (optionally scale by path length) so smaller drifts remain visible.

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

- Results land in `<output-dir>/activations/activation_XXX.png`, each overlaying the Grad-CAM intensity on top of the board state.
- `--activation-target value` switches the explanation target to the scalar value head; use `--activation-move <col>` to focus on a specific policy output rather than the arg-max.
- Random boards are sampled by default; swap in curated states by passing your own list to `visualize_cnn_activations` (for example by loading tensors from disk before calling the helper).

## Joint CNN/GRU Trajectory

Plot both weight families in the same PHATE space for a single configuration. The CNN curve uses filled markers, the GRU uses outlined markers at the same epochs.

```bash
python -m src.visualize_trajectories \
   --checkpoint-dir checkpoints/k3_c16_gru8_20251024_033336 \
   --viz-type joint \
   --joint-center anchor \
   --output-dir visualizations/joint_k3_c16_gru8
```

- `--joint-center` mirrors the ablation centering options so you can anchor the curves at epoch zero or renormalize their path length.
- The legend indicates how to read the shared val-loss markers: filled diamonds/stars correspond to the CNN, outlined versions to the GRU evaluated at the same epoch.

## Extending

- Swap in handcrafted board trajectories by replacing `generate_random_boards` or by loading saved tensors from the `data/` directory.
- Add animation by reusing `matplotlib.animation.FuncAnimation` on the 2D embeddings—the scaffolding (`FuncAnimation` import) is already in place.
- Experiment with different dimensionality reducers (`UMAP`, `TSNE`) by introducing new helper methods alongside the PHATE ones and wiring them into the CLI through an additional flag.

# GRU observability + probes

Produced by: `./wt.sh observability analyze` → `scripts/analyze_gru_observability_results.py`
Inputs from: `./wt.sh observability extract` → `diagnostics/gru_observability/<model>/`

Figures
- **gate_mean_trajectories.png** – two stacked line plots (update mean, reset mean) per model across epochs; legend lists the nine architectures.
- **timescale_heatmap.png** – heatmap of the median GRU integration timescale at the final epoch, indexed by channels (rows) and GRU size (columns).
- **phate_epoch_XXX_<feature>.png** – 3×3 grid (one panel per model) of PHATE embeddings of hidden samples at epoch `XXX`; colour encodes the requested board feature (e.g. `move_index`).
- **probe_accuracy.png** – logistic-probe accuracy over epochs; hue differentiates features, line style differentiates GRU size.
- **probe_results.csv** – long-form table `[model, epoch, feature, accuracy, f1, channels, gru, kernel]` for spreadsheet work.

What to look for
- **Gate trajectories**
  - Update mean climbing toward 1 ⇒ the hidden state leans more on its previous value (longer memories).
  - Update mean collapsing toward 0 ⇒ rapid overwriting; expect this early in training or for under-capacity models.
  - Reset mean near 0 ⇒ gates shut off historical contribution; near 1 ⇒ the model keeps injecting past information.
  - Convergence/flattening pinpoints the epoch where dynamics stabilise.
- **Timescale heatmap**
  - Darker cells (higher values) indicate eigenvalues closer to the unit circle ⇒ slower dynamics / longer integration.
  - Compare across GRU sizes to see where extra capacity really buys you longer memory.
- **PHATE grids**
  - Smooth gradients across the embedding reveal hidden-state progression with the coloured feature (useful for `move_index`).
  - Distinct lobes or clusters for tactical features (`immediate_win_*`) imply the GRU has carved separate attractors for those situations.
  - Late-epoch spirals or sharp turns often coincide with overfitting.
- **Probe accuracy**
  - Upward trajectories show increasingly linearly-decodable features.
  - Plateaus signal saturation; if accuracy later drops, suspect representation drift/overfitting.

Axes & encodings
- Gate plots: x = epoch, y = mean gate value; one line per model.
- Heatmap: colour = median `1 / |log(|λ|)|` of the GRU candidate recurrence matrix.
- PHATE: axes are the first two PHATE components; colourbar = feature value.
- Probe accuracy: x = epoch, y = accuracy; hue = feature choice; linestyle = GRU size.

Interpretation tips
- Update mean ≈ 1 → “keep” behaviour; ≈ 0 → “write” behaviour.
- Median timescale ↑ → the GRU sustains information longer (good for planning ahead in Connect Four).
- PHATE clusters for discrete features (e.g. `current_player`) indicate clear separability in hidden space.
- Probe accuracy rising with epoch → hidden state grows more informative; sudden drops warn of degradation.

Knobs
- analyze: --embedding-epochs, --embedding-feature, --probe-epochs, --probe-features, --max-hidden-samples, --palette, --skip-embedding, --skip-probing
- extract prerequisites: see the Observability command page for sampling controls

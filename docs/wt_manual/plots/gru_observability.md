# GRU observability + probes

Produced by: `./wt.sh observability analyze` → `scripts/analyze_gru_observability_results.py`
Inputs from: `./wt.sh observability extract` → `diagnostics/gru_observability/<model>/`

Figures
- gate_mean_trajectories.png: Two stacked line plots (update mean, reset mean) per model across epochs; legend lists models
- timescale_heatmap.png: Heatmap of median GRU timescale at final epoch, indexed by channels (rows) and GRU (cols)
- phate_epoch_XXX_<feature>.png: 3×3 grid (models) of PHATE embeddings of hidden samples at a given epoch; color encodes `<feature>` (e.g., move_index)
- probe_accuracy.png: Line plot of logistic probe accuracy over epochs for selected features; style varies by GRU size
- probe_results.csv: Accuracy/F1 table with columns [model, epoch, feature, accuracy, f1, channels, gru, kernel]

Axes/encodings
- Gate trajectories: x=epoch, y=mean gate value; each model is a separate line
- Timescale heatmap: color = median 1/|log(|eig|)| of GRU candidate recurrence (W_hn)
- PHATE grids: per-model 2D embedding; axes are PHATE 1/2; colorbar shows feature values
- Probes: x=epoch, y=accuracy; hue=feature; style=GRU

Reading the plots
- Update gate mean near 1 keeps more of previous state; near 0 writes more new info
- Higher median timescales suggest slower dynamics (longer memory)
- PHATE clusters for discrete features (e.g., current_player) indicate linearly separable encoding
- Probe accuracy rising with epoch implies more informative hidden states; plateaus/declines can flag overfitting

Knobs
- analyze: --embedding-epochs, --embedding-feature, --probe-epochs, --probe-features, --max-hidden-samples, --palette, --skip-embedding, --skip-probing
- extract prerequisites: see the Observability command page for sampling controls

# GRU mutual information

Produced by: `./wt.sh observability analyze` (invokes `scripts/compute_hidden_mutual_info.py`)
Inputs from: `diagnostics/gru_observability/<model>/hidden_samples/epoch_XXX.npz`

Artifacts (under `visualizations/gru_observability/`)
- mi_results.csv: Long-form table of mean MI per model/epoch/feature with type ∈ {classification, regression}
- mi_heatmap_final.png: Heatmap of MI at each model’s final epoch; rows=features, columns=models
- mi_trends.png: Small-multiples of MI vs epoch per feature; hue=model
- mi_per_dimension_<model>.png: Heatmap showing MI per hidden dimension for each feature at the model’s final epoch; ★ marks top dimension
- mi_dimension_values_<model>.png: For each feature, plot values of the highest-MI hidden dimension (violin for binary features; scatter for continuous)
- mi_metadata.json: Parameters used for the run

Reading the plots
- Higher MI indicates stronger statistical dependence between hidden vectors and the given board feature
- Classification features (current_player, immediate wins, three-in-row) use mutual_info_classif; continuous use mutual_info_regression
- Per-dimension heatmaps help identify specialized units; repeated ★ in same column suggests a dominant unit
- Dimension value plots reveal separability (binary) or correlation (continuous)

Knobs
- --features list, --max-samples, --seed, --output-dir; respects hidden sample availability from the extract step

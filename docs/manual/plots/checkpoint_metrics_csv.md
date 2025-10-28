# Checkpoint metrics CSV

Produced by: `./wt.sh metrics` → `scripts/compute_checkpoint_metrics.py`
Location: `diagnostics/checkpoint_metrics/<run>_metrics.csv`

Columns
- epoch: Training epoch number (from checkpoint metadata or filename)
- weight_norm: L2 norm of the flattened weight vector for the selected component (cnn|gru|all)
- step_norm: L2 norm of the weight delta from previous checkpoint; blank for the first row
- step_cosine: Cosine similarity between consecutive weight vectors; blank for the first row
  - +1: perfectly aligned steps, 0: orthogonal, −1: reversal
- relative_step: step_norm / previous weight_norm (scale-free step size); blank for the first row
- repr_total_variance: Sum of squared singular values of hidden representations over a fixed board set (if enabled via --board-source)
- repr_topK_ratio columns (repr_top1_ratio, repr_top2_ratio, ...): Proportion of total variance captured by top-k singular components (k up to --top-singular-values)

How it’s computed
- Weights are flattened by filtering state_dict keys:
  - cnn: keys containing 'resnet' and 'weight'
  - gru: keys containing 'gru' and 'weight'
  - all: any key containing 'weight'
- If `--board-source` is random or dataset, representation stats are computed by forwarding a fixed set of boards through each checkpoint to collect GRU hidden vectors, then computing SVD on the centered matrix.

Typical expectations
- weight_norm usually grows over training, more for larger GRU; excessive growth often correlates with overfitting
- step_norm tends to be larger early, then decay
- step_cosine usually hovers positive; sharp sign flips can indicate oscillations
- relative_step declines as weights grow or learning rates decay
- repr_total_variance increases as the model uses more hidden capacity; declines or instability can indicate collapse
- repr_top1_ratio near 0.5+ implies severe representation collapse (dominant dimension)

Upstream knobs
- --component [cnn|gru|all]
- --epoch-min/--epoch-max/--epoch-step to sub-sample epochs
- --board-source [none|random|dataset], --board-count, --board-dataset, --board-seed
- --top-singular-values (how many repr_topK_ratio columns to include)

How to use it
- **Weight norms**
  - Plot `weight_norm` over epochs to check capacity growth; steep late increases often precede validation loss spikes.
  - Compare CNN vs GRU components: diverging slopes highlight which block is still changing.
- **Step statistics**
  - `step_norm` + `relative_step` show how aggressively weights move. Sudden spikes usually coincide with learning-rate restarts or instability.
  - `step_cosine` near +1 signals consistent updates; oscillations between + and − mean the optimiser is fighting the curvature.
- **Representation SVD metrics**
  - `repr_total_variance` ↑ means the hidden space is expanding; persistent drops can indicate collapse or saturation.
  - `repr_top1_ratio`/`repr_topK_ratio` close to 1 denote a low-rank embedding (bad); healthy models spread variance across many components.

Typical workflows
- **Early overfitting detection**: monitor `step_norm` and `relative_step`—if they flatten while `weight_norm` still rises, the optimiser is taking tiny steps but drifting into sharper minima.
- **Architecture comparisons**: aggregate `repr_total_variance` across models to quantify which GRU size uses more representational capacity.
- **Checkpoint selection**: pick epochs where `repr_top1_ratio` is low (diverse representations) and `step_cosine` remains stable.

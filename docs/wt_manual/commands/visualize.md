# wt visualize

Run the unified visualization suite or launch the interactive wizard.

- `./wt.sh visualize wizard` → interactive shell wizard for end-to-end analysis
- `./wt.sh visualize [args]` → non-interactive batch via `scripts/run_visualization_suite.py`

Batch mode (run_visualization_suite.py)
- --config [json, REQUIRED] config describing experiments to run using `python -m src.visualize_trajectories`
- --report [path] optional override for markdown report path
- --python [path, default current interpreter] interpreter used to spawn subprocess

Reads
- Checkpoints, training_history.json per model

Writes
- Visualizations to the output folders referenced by your JSON
- Markdown report enumerating commands and produced images

Related CLI (src.visualize_trajectories)
- See `python -m src.visualize_trajectories --help` for options such as:
  - --checkpoint-dir / --ablation-dirs, --output-dir, --viz-type
  - --epoch-min/--epoch-max/--epoch-step
  - PHATE tuning (phate-n-pca, phate-knn, phate-t, phate-decay)
  - Grad-CAM activations controls

Artifacts documented at: [Unified visualization suite](../plots/visualize_unified.md) and [CNN activation maps](../plots/activations.md).

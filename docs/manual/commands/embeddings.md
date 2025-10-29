# wt embeddings

Overview
- Computes low‑dimensional embeddings of weight trajectories and optional hidden‑state representations for sanity checks and qualitative comparisons.

How to Run
- `./wt.sh embeddings [args]` (non‑interactive) or `./wt.sh embeddings wizard` (interactive)

- `./wt.sh embeddings wizard` → interactive selector (scripts/run_embedding_wizard.sh)
- `./wt.sh embeddings [args]` → non-interactive (scripts/analyze_weight_embeddings.py)

Options (analyze_weight_embeddings.py)
- --checkpoint-dirs [one or more, REQUIRED]
- --component [cnn|gru|all, default cnn]
- --methods [list, default pca tsne umap phate]
- --output-dir [path, default visualizations/simple_embeddings]
- --epoch-min / --epoch-max / --epoch-step [default 1]
- --random-seed [int, default 0]
- --annotate [flag]
- --export-csv [flag]
- --require-channel / --require-gru / --require-kernel [int filters]
- --compare [flag] build overlay plots/gifs across runs
- --animate [flag] and --animate-fps [int, default 4]

Board representations (optional)
- --board-representations [flag] also embed GRU hidden states on board samples
- --board-source [random|dataset, default random]
- --board-count [int, default 120]
- --board-dataset [path] when source=dataset (.pt or .json)
- --board-seed [int, default 0]
- --representation-methods [list] defaults to --methods

Reads
- Checkpoint weights; optionally dataset/boards for GRU hidden extraction

Writes
- Per-run plots under `<output-dir>/<run>/`:
  - `{component}_{method}.png` (+ optional `_anim.gif` and `.csv`)
  - `repr_{method}.png` if board representations enabled
- Overlay comparisons under `<output-dir>/comparisons/` when `--compare`

See plot reference: [Weight/representation embeddings](manual/plots/embeddings_weights)

# wt cka

Overview
- Computes Centered Kernel Alignment (CKA) similarity between models’ representations (GRU or CNN) on a fixed set of boards.
- Answers whether different capacities converge to similar internal encodings.

How to Run
- `./wt.sh cka [args]` (non‑interactive) or `./wt.sh cka wizard` (interactive)

- `./wt.sh cka wizard` → friendly interactive runner (scripts/run_cka_wizard.sh)
- `./wt.sh cka [args]` → direct run (scripts/compute_cka_similarity.py)

Options and defaults
- --checkpoint-dir [path, default checkpoints/save_every_3]
- --epochs [list[int], default 3 10 30 60 100]
- --epoch-step [int, optional] if set, generates 3..99 by step then appends 100
- --output-dir [path, default visualizations]
- --num-boards [int, default 64] synthetic test boards
- --seed [int, default 42]
- --device [cpu|cuda, default cpu]
- --animate [flag] heatmap animation across epochs
- --animate-fps [int, default 3]
- --animate-format [gif|mp4, default gif]
- --representation [gru|cnn, default gru]

Reads
- All 9 model folders for the given epoch(s); loads weights and extracts representations on fixed board set.

Writes (under `<output-dir>/<representation>/`)
- `cka_{rep}_similarity_epoch_{E}.png` heatmap per epoch
- `cka_{rep}_clustered_epoch_{E}.png` dendrogram/clustered heatmap
- `cka_{rep}_evolution.png` pairwise evolution for selected pairs
- `cka_{rep}_similarity_epoch_{E}.csv` raw matrices
- Optional: `cka_{rep}_heatmap_animation.gif|mp4`

Plot explainers: see [CKA Similarity](manual/plots/cka)

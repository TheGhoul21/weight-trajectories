# wt dataset

Generate Connect-4 datasets from the bundled AlphaZero implementation.

Subcommands
- flat → scripts/generate_connect4_dataset.py
- sequential (alias: seq) → scripts/generate_sequential_dataset.py

Inputs and outputs
- Reads: AlphaZero code under `dataset/Alpha-Zero-algorithm-for-Connect-4-game`, may write temp files in its `data/`.
- Writes: `.pt` dataset at the requested path and a sidecar `.json` with metadata.

Usage
- Flat positions: `./wt.sh dataset flat [options]`
- Sequential games: `./wt.sh dataset sequential [options]`

Options (flat)
- --num-games [int, default 10000]
- --simulations [int, default 200] MCTS sims per move
- --output [path, default data/connect4_10k_games.pt]
- --cpus [int, default 4]
- --test-run [flag] generate ~20–100 mini set
- --save-numpy [flag] also save .npz
- --seed [int, default 0] reproducible self-play seed

Output schema (flat)
- `.pt` dict with:
  - states FloatTensor (N, 3, 6, 7)
  - policies FloatTensor (N, 7)
  - values FloatTensor (N, 1)
  - metadata dict (counts, timings, versions)

Options (sequential)
- --num-games [int, default 1000]
- --simulations [int, default 250]
- --output [path, default data/connect4_sequential_1k_games.pt]
- --cpus [int, default 4]
- --test-run [flag] (~20 games)
- --seed [int, default 0] reproducible gameplay seed

Output schema (sequential)
- `.pt` dict with:
  - games: list of length N
    - each game: dict with tensors
      - states (T, 3, 6, 7)
      - policies (T, 7)
      - values (T, 1)
  - metadata dict: generation_time_seconds, counts, win stats, etc.
- Sidecar JSON: same stats for easy inspection (includes `seed`).

Notes
- Sequential format is required for GRU training and downstream observability analyses.
- Both generators subsample or augment safely and print expected run stats; seeds ensure repeated runs yield identical datasets.

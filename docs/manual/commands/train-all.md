# wt train-all

Overview
- Trains the 3 GRU sizes {8, 32, 128} across your chosen kernel size and channel set; produces the 3×3 sweep used by comparison plots.

How to Run
- `./wt.sh train-all [options]`

Maps to: wrapper around `src/train.py` with batching and sensible batch size tweaks.

Defaults (environment overrideable)
- WT_DATASET → dataset path [default data/connect4_10k_games.pt]
- WT_EPOCHS → total epochs [default 100]
- WT_SAVE_EVERY → save period [default 10]
- WT_KERNEL_SIZE → kernel [default 3]

CLI overrides
- --data [path]
- --epochs [int]
- --save-every [int]
- --kernel-size [int]
- -- … (anything after `--` is forwarded to src/train.py)

Behavior
- Trains GRU sizes in sequence: 8, 32, 128. CNN channels fixed to {16, 64, 256}.
- Batch size auto tuned: 32 for GRU≥128, else 64.
- Warns if dataset path is missing.

Outputs
- Three checkpoint directories with the standard naming scheme, each with weights and training_history.json.

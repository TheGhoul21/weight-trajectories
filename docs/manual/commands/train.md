# wt train

Train a single ResNet+GRU model and save checkpoints every N epochs.

Maps to: `python src/train.py`

Reads
- Dataset `.pt` (flat or sequential)

Writes
- `checkpoints/<run_name>/weights_epoch_XXXX.pt`
- `checkpoints/<run_name>/best_model.pt`
- `checkpoints/<run_name>/training_history.json`

Run name
- `k{kernel}_c{channels}_gru{hidden}_{timestamp}`

Key options and defaults
- --data [path] REQUIRED if not using --config
- --config [yaml] optional preset file; with --model key
- --model [key] model key inside YAML; supports `includes`
- --batch-size [int, default 64]
- --cnn-channels [list[int], default [64]] one run per channel value
- --gru-hidden [int, default 32]
- --kernel-size [int, default 3]
- --epochs [int, default 100]
- --lr [float, default 0.001]
- --policy-weight [float, default 1.0]
- --value-weight [float, default 1.0]
- --save-every [int, default 10]
- --checkpoint-dir [path, default checkpoints]
- --seed [int, default 0] reproducible initialization/split seed

Training behavior
- Splits dataset 90/10 into train/val; supports sequential batching with padding.
- Tracks losses per epoch; saves weights at intervals and at the end; best_model.pt updated on improved val loss. Each run is seeded deterministically (history JSON records the exact seed).

Artifacts used by other tools
- Checkpoint weights → metrics, embeddings, CKA, observability, fixed-point analyses.
- training_history.json → unified visualizations and report.

YAML config mode
- Allows templated experiments via `includes`; CLI flags act as global defaults.

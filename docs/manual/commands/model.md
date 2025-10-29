# wt model

Overview
- Smoke‑tests the ResNet+GRU architecture across ablation configs; prints shapes and parameter counts.

How to Run
- `./wt.sh model`

Maps to: `python src/model.py`

What it does
- Iterates kernel ∈ {3, 6}, channels ∈ {16, 64, 256}, GRU ∈ {8, 32, 128}.
- Builds the model, runs a dummy forward on random boards, prints tensor shapes and parameter counts.

Inputs/outputs
- Reads: none (no dataset)
- Writes: stdout only (no files)

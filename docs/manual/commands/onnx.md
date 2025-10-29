# wt onnx

Overview
- Exports the ResNet+GRU architecture to ONNX for inspection or use in external tools.

How to Run
- `./wt.sh onnx [args]` (invokes `scripts/export_model_onnx.py`)

Backed by: `scripts/export_model_onnx.py`

Options
- --cnn-channels [int, default 64]
- --gru-hidden [int, default 32]
- --kernel-size [int, default 3]
- --seq-len [int, default 30] dummy sequence length for the export input
- --batch-size [int, default 1]
- --output [path, default artifacts/model.onnx]
- --opset [int, default 17]

Behavior
- Builds the model via `src.model.create_model(cnn_channels, gru_hidden, kernel_size)`
- Exports with dynamic axes for batch and sequence dims:
  - input: (batch, sequence, 3, 6, 7)
  - outputs: policy (batch, sequence, 7) and value (batch, sequence, 1)

Reads
- None (uses a randomly initialized model – not training weights)

Writes
- ONNX file at `--output`

Tips
- Increase `--seq-len` to reflect typical unroll lengths if consumers expect that shape
- Keep opset ≥ 13 for modern runtimes; default 17 is broadly compatible

"""Export ResNet+GRU Connect-4 model to ONNX for visualization tools."""
import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export model architecture to ONNX")
    parser.add_argument("--cnn-channels", type=int, default=64,
                        help="Number of CNN channels (default: 64)")
    parser.add_argument("--gru-hidden", type=int, default=32,
                        help="GRU hidden size (default: 32)")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Convolution kernel size (default: 3)")
    parser.add_argument("--seq-len", type=int, default=30,
                        help="Sequence length to embed in dummy input (default: 30)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to use for dummy input (default: 1)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.onnx"),
                        help="Path for the exported ONNX file")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = create_model(args.cnn_channels, args.gru_hidden, args.kernel_size)
    model.eval()

    dummy_input = torch.randn(args.batch_size, args.seq_len, 3, 6, 7)

    print("Exporting model to ONNX...")
    print(f"  CNN channels: {args.cnn_channels}")
    print(f"  GRU hidden:  {args.gru_hidden}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Output:      {output_path}")

    dynamic_axes = {
        "input": {0: "batch", 1: "sequence"},
        "policy": {0: "batch", 1: "sequence"},
        "value": {0: "batch", 1: "sequence"}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes=dynamic_axes
    )

    print("✓ Export complete")
if __name__ == "__main__":
    main()

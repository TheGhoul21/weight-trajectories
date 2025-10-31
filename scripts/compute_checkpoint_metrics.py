#!/usr/bin/env python
"""Compute diagnostic statistics from saved checkpoints.

This utility complements the embedding visualizations by emitting tabular
metrics that can be consumed quickly inside the workflow notebook/reports.
It focuses on two kinds of signals:

* Weight-space dynamics: norms, step sizes, and cosine similarities between
  consecutive checkpoints for a chosen component (CNN, GRU, or all weights).
* Representation variance: optional singular-value summaries of latent vectors
  produced by each checkpoint over a fixed set of board states.

Results are written to a CSV per run under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from src.model import create_model
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.model import create_model  # type: ignore


def _parse_model_config(checkpoint_dir: Path) -> Dict[str, int]:
    """Extract kernel/channels/gru sizes from directory name."""
    parts = checkpoint_dir.name.split("_")
    config: Dict[str, int] = {}
    for part in parts:
        if part.startswith("k") and part[1:].isdigit():
            config["kernel"] = int(part[1:])
        elif part.startswith("c") and part[1:].isdigit():
            config["channels"] = int(part[1:])
        elif part.startswith("gru") and part[3:].isdigit():
            config["gru_hidden"] = int(part[3:])
    return config


def _gather_checkpoint_files(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    """Collect (epoch, path) pairs sorted by epoch."""
    pairs: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.glob("weights_epoch_*.pt"):
        name = path.stem
        try:
            epoch = int(name.replace("weights_epoch_", ""))
        except ValueError:
            continue
        pairs.append((epoch, path))
    pairs.sort(key=lambda item: item[0])
    return pairs


def _filter_epochs(
    pairs: List[Tuple[int, Path]],
    epoch_min: Optional[int],
    epoch_max: Optional[int],
    stride: int,
) -> List[Tuple[int, Path]]:
    stride = max(1, stride)
    selected: List[Tuple[int, Path]] = []
    for epoch, path in pairs:
        if epoch_min is not None and epoch < epoch_min:
            continue
        if epoch_max is not None and epoch > epoch_max:
            continue
        selected.append((epoch, path))
    return selected[::stride]


def _flatten_weights(state_dict: Dict[str, torch.Tensor], component: str) -> np.ndarray:
    """Return concatenated weight vector for the desired component."""
    buckets: List[np.ndarray] = []
    for name, tensor in state_dict.items():
        if "weight" not in name:
            continue
        if component == "cnn" and "resnet" not in name:
            continue
        if component == "gru" and "gru" not in name:
            continue
        buckets.append(tensor.detach().cpu().numpy().ravel())
    if not buckets:
        raise ValueError(f"No weights collected for component '{component}'.")
    return np.concatenate(buckets).astype(np.float32, copy=False)


def _load_weight_matrix(
    checkpoint_dir: Path,
    component: str,
    epoch_min: Optional[int],
    epoch_max: Optional[int],
    stride: int,
) -> Tuple[np.ndarray, List[int], List[Path]]:
    """Stack checkpoint weights into a (N, D) matrix."""
    all_pairs = _gather_checkpoint_files(checkpoint_dir)
    if not all_pairs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    filtered = _filter_epochs(all_pairs, epoch_min, epoch_max, stride)
    if not filtered:
        raise ValueError("No checkpoints remain after applying epoch filters.")

    weights: List[np.ndarray] = []
    epochs: List[int] = []
    paths: List[Path] = []

    for epoch, path in filtered:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        weights.append(_flatten_weights(state_dict, component))
        epochs.append(int(checkpoint.get("epoch", epoch)))
        paths.append(path)

    matrix = np.stack(weights, axis=0)
    return matrix, epochs, paths


def _coerce_board_tensor(sample) -> Optional[torch.Tensor]:
    """Try to normalize a candidate sample into a (B, 3, 6, 7) float tensor.

    Notes:
    - Accepts either a single board (3, 6, 7) -> unsqueezed to (1, 3, 6, 7),
      or a batch/sequence (B, 3, 6, 7).
    - Avoids attempting to convert ragged lists (which become numpy.object_)
      so that callers can recurse into their items instead.
    """
    # Torch tensors: handle directly and validate shape
    if isinstance(sample, torch.Tensor):
        tensor = sample.detach().cpu()
        if tensor.ndim == 3 and tensor.shape == (3, 6, 7):
            return tensor.unsqueeze(0).to(dtype=torch.float32)
        if tensor.ndim == 4 and tensor.shape[1:] == (3, 6, 7):
            return tensor.to(dtype=torch.float32)
        return None

    # Numpy arrays: guard against object dtype before converting to torch
    if isinstance(sample, np.ndarray):
        # Skip ragged/object arrays; let the caller recurse instead
        if sample.dtype == np.object_:
            return None
        try:
            return _coerce_board_tensor(torch.from_numpy(sample))
        except (TypeError, ValueError):
            return None

    # Python sequences: only attempt numeric coercion when not ragged
    if isinstance(sample, (list, tuple)):
        try:
            arr = np.asarray(sample)
        except Exception:
            return None
        # If this produced an object array, it's likely a ragged/nested structure
        if isinstance(arr, np.ndarray) and arr.dtype != np.object_:
            return _coerce_board_tensor(arr)
        return None

    return None


def _generate_random_boards(count: int, seed: int) -> List[torch.Tensor]:
    rng = np.random.default_rng(seed)
    boards: List[torch.Tensor] = []
    for _ in range(count):
        board = np.zeros((3, 6, 7), dtype=np.float32)
        n_pieces = int(rng.integers(0, 22))
        for _ in range(n_pieces):
            player = int(rng.integers(0, 2))
            column = int(rng.integers(0, 7))
            for row in range(5, -1, -1):
                if board[0, row, column] == 0 and board[1, row, column] == 0:
                    board[player, row, column] = 1.0
                    break
        board[2] = rng.integers(0, 2, size=(6, 7))
        boards.append(torch.from_numpy(board).unsqueeze(0))
    return boards


def _load_dataset_boards(dataset_path: Path, max_count: int, seed: int) -> List[torch.Tensor]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    content = torch.load(dataset_path, map_location="cpu", weights_only=False)
    collected: List[torch.Tensor] = []

    def visit(node) -> None:
        if len(collected) >= max_count:
            return
        tensor = _coerce_board_tensor(node)
        if tensor is not None:
            collected.append(tensor)
            return
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
                if len(collected) >= max_count:
                    break
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)
                if len(collected) >= max_count:
                    break

    visit(content)
    if not collected:
        raise ValueError("Could not extract board tensors from dataset contents.")

    if len(collected) > max_count:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(collected), size=max_count, replace=False))
        collected = [collected[int(idx)] for idx in indices]

    return collected


def _prepare_boards(source: str, count: int, seed: int, dataset: Optional[Path]) -> List[torch.Tensor]:
    if count <= 0:
        raise ValueError("Board count must be positive.")
    if source == "none":
        return []
    if source == "random":
        return _generate_random_boards(count, seed)
    if source == "dataset":
        if dataset is None:
            raise ValueError("Dataset path required when --board-source=dataset.")
        return _load_dataset_boards(dataset, count, seed)
    raise ValueError(f"Unsupported board source '{source}'.")


def _forward_hidden_vectors(
    checkpoint_path: Path,
    config: Dict[str, int],
    boards: Sequence[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    required = {"kernel", "channels", "gru_hidden"}
    if not required.issubset(config):
        raise ValueError(
            f"Directory name must encode kernel/channels/gru, got {checkpoint_path.parent.name}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = create_model(config["channels"], config["gru_hidden"], config["kernel"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    representations: List[np.ndarray] = []
    with torch.no_grad():
        for board in boards:
            tensor = board.to(device=device, dtype=torch.float32)
            _, _, hidden = model(tensor)
            # GRU returns hidden as (num_layers, batch, hidden_size)
            # We want a fixed-length vector per input board sample.
            # If batch>1 (e.g., when a tensor packs multiple boards), pool across batch.
            if hidden.dim() == 3:
                # Remove layer dimension
                hidden2d = hidden.squeeze(0)  # (batch, hidden)
                if hidden2d.dim() == 1:
                    vec = hidden2d
                else:
                    # Average across batch for a single representation
                    vec = hidden2d.mean(dim=0)
            else:
                # Fallback: flatten then reshape if possible
                vec = hidden.view(-1)
            representations.append(vec.cpu().numpy())

    return np.stack(representations, axis=0).astype(np.float32, copy=False)


def _singular_value_stats(matrix: np.ndarray, top_k: int) -> Tuple[float, List[float]]:
    if matrix.shape[0] < 2:
        return 0.0, []
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if not np.isfinite(centered).all():
        return 0.0, []
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0:
        return 0.0, []
    total = float(np.sum(singular_values ** 2))
    if total <= 0.0:
        return 0.0, []
    ratios = (singular_values ** 2) / total
    clipped = ratios[:top_k].tolist()
    return total, clipped


def compute_weight_metrics(weights: np.ndarray) -> Dict[str, np.ndarray]:
    norms = np.linalg.norm(weights, axis=1)
    deltas = np.diff(weights, axis=0) if weights.shape[0] > 1 else np.zeros((0, weights.shape[1]), dtype=np.float32)
    step_sizes = np.linalg.norm(deltas, axis=1)

    cosines = np.full_like(step_sizes, fill_value=np.nan, dtype=np.float32)
    if deltas.shape[0] > 0:
        for idx in range(deltas.shape[0]):
            prev = weights[idx]
            curr = weights[idx + 1]
            prev_norm = norms[idx]
            curr_norm = norms[idx + 1]
            denom = prev_norm * curr_norm
            if denom > 0:
                cosines[idx] = np.dot(prev, curr) / denom

    rel_steps = np.full_like(step_sizes, fill_value=np.nan, dtype=np.float32)
    if step_sizes.size > 0:
        for idx, value in enumerate(step_sizes):
            base = norms[idx]
            if base > 0:
                rel_steps[idx] = value / base

    return {
        "norm": norms,
        "step_norm": step_sizes,
        "step_cosine": cosines,
        "relative_step": rel_steps,
    }


def _write_csv(
    save_path: Path,
    epochs: Sequence[int],
    weight_metrics: Dict[str, np.ndarray],
    repr_totals: Sequence[Optional[float]],
    repr_ratios: Sequence[Optional[List[float]]],
) -> None:
    headers = [
        "epoch",
        "weight_norm",
        "step_norm",
        "step_cosine",
        "relative_step",
        "repr_total_variance",
    ]
    max_ratio_len = max((len(r) for r in repr_ratios if r is not None), default=0)
    ratio_headers = [f"repr_top{i+1}_ratio" for i in range(max_ratio_len)]
    headers.extend(ratio_headers)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx, epoch in enumerate(epochs):
            row: List[object] = [
                epoch,
                float(weight_metrics["norm"][idx]),
            ]
            if idx == 0:
                row.extend(["", "", ""])  # step_norm, step_cosine, relative_step
            else:
                row.append(float(weight_metrics["step_norm"][idx - 1]))
                cosine = weight_metrics["step_cosine"][idx - 1]
                row.append("" if math.isnan(float(cosine)) else float(cosine))
                rel = weight_metrics["relative_step"][idx - 1]
                row.append("" if math.isnan(float(rel)) else float(rel))

            total = repr_totals[idx]
            row.append("" if total is None else float(total))

            ratios = repr_ratios[idx]
            for pos in range(max_ratio_len):
                if ratios is None or pos >= len(ratios):
                    row.append("")
                else:
                    row.append(float(ratios[pos]))

            writer.writerow(row)


def analyze_run(
    checkpoint_dir: Path,
    component: str,
    epoch_min: Optional[int],
    epoch_max: Optional[int],
    stride: int,
    board_states: Sequence[torch.Tensor],
    device: torch.device,
    top_singular_values: int,
    output_dir: Path,
) -> Path:
    matrix, epochs, paths = _load_weight_matrix(checkpoint_dir, component, epoch_min, epoch_max, stride)
    weight_metrics = compute_weight_metrics(matrix)

    repr_totals: List[Optional[float]] = [None] * len(epochs)
    repr_ratios: List[Optional[List[float]]] = [None] * len(epochs)

    config = _parse_model_config(checkpoint_dir)
    if board_states:
        for idx, path in enumerate(paths):
            reps = _forward_hidden_vectors(path, config, board_states, device)
            total, ratios = _singular_value_stats(reps, top_singular_values)
            repr_totals[idx] = total
            repr_ratios[idx] = ratios

    save_path = output_dir / f"{checkpoint_dir.name}_metrics.csv"
    _write_csv(save_path, epochs, weight_metrics, repr_totals, repr_ratios)
    return save_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute diagnostic metrics from checkpoints.")
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        required=True,
        help="One or more checkpoint directories under checkpoints/save_every_3/...",
    )
    parser.add_argument(
        "--component",
        choices=["cnn", "gru", "all"],
        default="gru",
        help="Model component whose weights should be tracked.",
    )
    parser.add_argument("--epoch-min", type=int, help="Earliest epoch to include (inclusive).")
    parser.add_argument("--epoch-max", type=int, help="Latest epoch to include (inclusive).")
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Stride used when sampling checkpoints (default keeps every snapshot).",
    )
    parser.add_argument(
        "--board-source",
        choices=["none", "random", "dataset"],
        default="none",
        help="If provided, evaluates hidden-state variance on a fixed board set.",
    )
    parser.add_argument(
        "--board-count",
        type=int,
        default=16,
        help="Number of boards to sample when --board-source!=none.",
    )
    parser.add_argument("--board-dataset", type=Path, help="Dataset path when board-source=dataset.")
    parser.add_argument("--board-seed", type=int, default=37, help="Seed for board sampling.")
    parser.add_argument(
        "--top-singular-values",
        type=int,
        default=4,
        help="Number of leading singular-value ratios to report for representations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("diagnostics/checkpoint_metrics"),
        help="Destination directory for CSV outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    board_states: List[torch.Tensor] = []
    if args.board_source != "none":
        board_states = _prepare_boards(args.board_source, args.board_count, args.board_seed, args.board_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir

    checkpoint_dirs = [Path(path) for path in args.checkpoint_dirs]
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        csv_path = analyze_run(
            checkpoint_dir=checkpoint_dir,
            component=args.component,
            epoch_min=args.epoch_min,
            epoch_max=args.epoch_max,
            stride=args.epoch_step,
            board_states=board_states,
            device=device,
            top_singular_values=max(1, args.top_singular_values),
            output_dir=output_dir,
        )
        print(f"Wrote metrics for {checkpoint_dir} to {csv_path}")


if __name__ == "__main__":
    main()


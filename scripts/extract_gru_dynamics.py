#!/usr/bin/env python3
"""
GRU observability analysis utilities.

For each checkpointed model, this script extracts:
  * GRU gate statistics (update/reset) aggregated over sampled game trajectories
  * Hidden state samples suitable for probing or low-dimensional embeddings
  * Eigenvalue/timescale summaries of the recurrent candidate-weight matrix

Outputs:
  diagnostics/gru_observability/<model>/metrics.csv
  diagnostics/gru_observability/<model>/unit_gate_stats.csv
  diagnostics/gru_observability/<model>/hidden_samples/epoch_XXX.npz
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

import sys

# Ensure project root on path for src imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import create_model  # type: ignore


@dataclass
class GateAccumulator:
    hidden_size: int

    total_steps: int = 0
    sum_all: torch.Tensor | None = None
    sum_sq_all: torch.Tensor | None = None
    sum_per_unit: torch.Tensor | None = None
    sum_sq_per_unit: torch.Tensor | None = None

    def update(self, gate_values: torch.Tensor) -> None:
        if self.sum_all is None:
            device = gate_values.device
            self.sum_all = torch.tensor(0.0, device=device)
            self.sum_sq_all = torch.tensor(0.0, device=device)
            self.sum_per_unit = torch.zeros(self.hidden_size, device=device)
            self.sum_sq_per_unit = torch.zeros(self.hidden_size, device=device)

        self.total_steps += gate_values.size(0)
        self.sum_all += gate_values.sum()
        self.sum_sq_all += (gate_values ** 2).sum()
        self.sum_per_unit += gate_values.sum(dim=0)
        self.sum_sq_per_unit += (gate_values ** 2).sum(dim=0)

    def finalize(self) -> Dict[str, np.ndarray | float]:
        if self.total_steps == 0 or self.sum_all is None or self.sum_sq_all is None:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "unit_mean": np.full(self.hidden_size, np.nan, dtype=np.float32),
                "unit_std": np.full(self.hidden_size, np.nan, dtype=np.float32),
            }

        count = self.total_steps * self.hidden_size
        mean = (self.sum_all / count).item()
        var = (self.sum_sq_all / count) - mean ** 2
        var = max(var.item(), 0.0) if isinstance(var, torch.Tensor) else max(var, 0.0)

        unit_mean = (self.sum_per_unit / self.total_steps).cpu().numpy()
        unit_var = (self.sum_sq_per_unit / self.total_steps) - (
            torch.from_numpy(unit_mean).to(self.sum_per_unit.device) ** 2
        )
        unit_var = torch.clamp(unit_var, min=0.0).cpu().numpy()

        return {
            "mean": mean,
            "std": math.sqrt(var),
            "unit_mean": unit_mean.astype(np.float32),
            "unit_std": np.sqrt(unit_var.astype(np.float32)),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GRU observability metrics from checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/save_every_3",
        help="Base directory containing per-model checkpoint folders.",
    )
    parser.add_argument(
        "--dataset",
        default="data/connect4_sequential_10k_games.pt",
        help="Path to sequential Connect-4 dataset (.pt).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=256,
        help="Maximum number of games to sample per model for statistics.",
    )
    parser.add_argument(
        "--sample-hidden",
        type=int,
        default=1500,
        help="Number of hidden states to retain per epoch via reservoir sampling.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run analysis on (cpu or cuda).",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/gru_observability",
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress per checkpoint.",
    )
    return parser.parse_args()


def list_model_dirs(base_dir: Path) -> List[Path]:
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and "gru" in p.name)


def list_checkpoints(model_dir: Path) -> List[Tuple[int, Path]]:
    checkpoints: List[Tuple[int, Path]] = []
    for ckpt in model_dir.glob("weights_epoch_*.pt"):
        try:
            epoch = int(ckpt.stem.split("_")[-1])
        except ValueError:
            continue
        checkpoints.append((epoch, ckpt))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def load_dataset(dataset_path: Path, max_games: int) -> List[Dict[str, torch.Tensor]]:
    raw = torch.load(dataset_path, weights_only=False)
    if "games" not in raw:
        raise ValueError("Dataset must be sequential with a 'games' list.")
    games: List[Dict[str, torch.Tensor]] = raw["games"][:max_games]
    return games


def board_from_state(state: torch.Tensor) -> np.ndarray:
    yellow = (state[0] > 0.5).cpu().numpy()
    red = (state[1] > 0.5).cpu().numpy()
    board = np.zeros((6, 7), dtype=np.int8)
    board[yellow] = 1
    board[red] = 2
    return board


def get_current_player(state: torch.Tensor) -> int:
    turn_value = float(state[2, 0, 0])
    return 1 if turn_value < 0.5 else 2


def valid_moves(board: np.ndarray) -> List[int]:
    return [col for col in range(7) if board[0, col] == 0]


def drop_piece(board: np.ndarray, col: int, player: int) -> np.ndarray:
    new_board = board.copy()
    for row in range(5, -1, -1):
        if new_board[row, col] == 0:
            new_board[row, col] = player
            break
    return new_board


def check_four(board: np.ndarray, player: int) -> bool:
    # Horizontal
    for row in range(6):
        for col in range(4):
            block = board[row, col : col + 4]
            if np.all(block == player):
                return True
    # Vertical
    for row in range(3):
        for col in range(7):
            block = board[row : row + 4, col]
            if np.all(block == player):
                return True
    # Diagonal down-right
    for row in range(3):
        for col in range(4):
            if all(board[row + i, col + i] == player for i in range(4)):
                return True
    # Diagonal up-right
    for row in range(3, 6):
        for col in range(4):
            if all(board[row - i, col + i] == player for i in range(4)):
                return True
    return False


def count_three_in_row(board: np.ndarray, player: int) -> int:
    count = 0
    def window_has_three(cells: Iterable[int]) -> bool:
        arr = np.array(list(cells))
        return np.count_nonzero(arr == player) == 3 and np.count_nonzero(arr == 0) == 1

    # Horizontal windows
    for row in range(6):
        for col in range(4):
            if window_has_three(board[row, col : col + 4]):
                count += 1
    # Vertical
    for row in range(3):
        for col in range(7):
            if window_has_three(board[row : row + 4, col]):
                count += 1
    # Diagonals
    for row in range(3):
        for col in range(4):
            if window_has_three(board[row + np.arange(4), col + np.arange(4)]):
                count += 1
    for row in range(3, 6):
        for col in range(4):
            if window_has_three(board[row - np.arange(4), col + np.arange(4)]):
                count += 1
    return count


def compute_board_features(state: torch.Tensor, move_index: int) -> Dict[str, float]:
    board = board_from_state(state)
    current = get_current_player(state)
    opponent = 3 - current
    yellow_count = int(np.count_nonzero(board == 1))
    red_count = int(np.count_nonzero(board == 2))
    v_moves = valid_moves(board)

    immediate_current = any(
        check_four(drop_piece(board, col, current), current) for col in v_moves
    )
    immediate_opponent = any(
        check_four(drop_piece(board, col, opponent), opponent) for col in v_moves
    )

    center_col = board[:, 3]
    center_current = int(np.count_nonzero(center_col == current))
    center_opponent = int(np.count_nonzero(center_col == opponent))

    return {
        "move_index": float(move_index),
        "current_player": float(current),
        "yellow_count": float(yellow_count),
        "red_count": float(red_count),
        "piece_diff": float(yellow_count - red_count),
        "valid_moves": float(len(v_moves)),
        "immediate_win_current": float(int(immediate_current)),
        "immediate_win_opponent": float(int(immediate_opponent)),
        "center_control_current": float(center_current),
        "center_control_opponent": float(center_opponent),
        "three_in_row_current": float(count_three_in_row(board, current)),
        "three_in_row_opponent": float(count_three_in_row(board, opponent)),
    }


def split_gru_parameters(gru: nn.GRU) -> Dict[str, torch.Tensor]:
    weight_ih = gru.weight_ih_l0.detach()
    weight_hh = gru.weight_hh_l0.detach()
    bias_ih = gru.bias_ih_l0.detach()
    bias_hh = gru.bias_hh_l0.detach()

    hidden_size = gru.hidden_size

    def slc(t: torch.Tensor, idx: int) -> torch.Tensor:
        return t[idx * hidden_size : (idx + 1) * hidden_size]

    # PyTorch orders gates as reset, input(z), new
    W_ir = slc(weight_ih, 0)
    W_iz = slc(weight_ih, 1)
    W_in = slc(weight_ih, 2)

    W_hr = slc(weight_hh, 0)
    W_hz = slc(weight_hh, 1)
    W_hn = slc(weight_hh, 2)

    b_ir = slc(bias_ih, 0)
    b_iz = slc(bias_ih, 1)
    b_in = slc(bias_ih, 2)

    b_hr = slc(bias_hh, 0)
    b_hz = slc(bias_hh, 1)
    b_hn = slc(bias_hh, 2)

    return {
        "W_ir": W_ir,
        "W_iz": W_iz,
        "W_in": W_in,
        "W_hr": W_hr,
        "W_hz": W_hz,
        "W_hn": W_hn,
        "b_ir": b_ir,
        "b_iz": b_iz,
        "b_in": b_in,
        "b_hr": b_hr,
        "b_hz": b_hz,
        "b_hn": b_hn,
    }


def analyze_checkpoint(
    model: nn.Module,
    games: List[Dict[str, torch.Tensor]],
    sample_hidden: int,
    device: torch.device,
    rng: random.Random,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
    model.eval()
    model.to(device)

    gru: nn.GRU = model.gru
    hidden_size = gru.hidden_size
    params = split_gru_parameters(gru)

    update_acc = GateAccumulator(hidden_size)
    reset_acc = GateAccumulator(hidden_size)

    samples: List[Dict[str, np.ndarray]] = []
    sample_count = 0

    total_steps = 0

    with torch.no_grad():
        for game in games:
            states = game["states"].to(device).float()  # (seq_len, 3, 6, 7)
            seq_len = states.size(0)

            features = model.resnet(states).view(seq_len, -1)
            h_t = torch.zeros(hidden_size, device=device)

            for t in range(seq_len):
                x_t = features[t]
                linear_r = torch.matmul(params["W_ir"], x_t) + params["b_ir"] + torch.matmul(params["W_hr"], h_t) + params["b_hr"]
                linear_z = torch.matmul(params["W_iz"], x_t) + params["b_iz"] + torch.matmul(params["W_hz"], h_t) + params["b_hz"]
                r_t = torch.sigmoid(linear_r)
                z_t = torch.sigmoid(linear_z)
                n_t = torch.tanh(
                    torch.matmul(params["W_in"], x_t)
                    + params["b_in"]
                    + r_t * (torch.matmul(params["W_hn"], h_t) + params["b_hn"])
                )
                h_next = (1 - z_t) * n_t + z_t * h_t

                update_acc.update(z_t.unsqueeze(0))
                reset_acc.update(r_t.unsqueeze(0))
                total_steps += 1

                if sample_count < sample_hidden:
                    keep = True
                else:
                    j = rng.randint(0, total_steps - 1)
                    keep = j < sample_hidden

                if keep:
                    feats = compute_board_features(states[t].cpu(), move_index=t)
                    sample_entry = {
                        "hidden": h_next.detach().cpu().numpy().astype(np.float32),
                        "update_gate": z_t.detach().cpu().numpy().astype(np.float32),
                        "reset_gate": r_t.detach().cpu().numpy().astype(np.float32),
                        "features": np.array([feats[k] for k in sorted(feats.keys())], dtype=np.float32),
                        "feature_names": np.array(sorted(feats.keys())),
                        "step_index": np.array([t], dtype=np.int32),
                    }
                    if sample_count >= sample_hidden:
                        replace_idx = rng.randint(0, sample_hidden - 1)
                        samples[replace_idx] = sample_entry
                    else:
                        samples.append(sample_entry)
                    sample_count = min(sample_count + 1, sample_hidden)

                h_t = h_next

    eigen_stats, unit_eigen = compute_timescale_statistics(params["W_hn"])
    update_stats = update_acc.finalize()
    reset_stats = reset_acc.finalize()

    metrics = {
        "update_gate_mean": update_stats["mean"],
        "update_gate_std": update_stats["std"],
        "reset_gate_mean": reset_stats["mean"],
        "reset_gate_std": reset_stats["std"],
        "steps_evaluated": float(total_steps),
        "max_abs_eigen": eigen_stats["max_abs_eig"],
        "median_abs_eigen": eigen_stats["median_abs_eig"],
        "median_timescale": eigen_stats["median_tau"],
        "max_timescale": eigen_stats["max_tau"],
    }

    unit_metrics = {
        "update_gate_unit_mean": update_stats["unit_mean"],
        "update_gate_unit_std": update_stats["unit_std"],
        "reset_gate_unit_mean": reset_stats["unit_mean"],
        "reset_gate_unit_std": reset_stats["unit_std"],
    }

    return metrics, unit_metrics, unit_eigen, samples


def compute_timescale_statistics(weight_hn: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    matrix = weight_hn.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(matrix)
    abs_vals = np.abs(eigvals)

    epsilon = 1e-6
    mask = (abs_vals > epsilon) & (abs_vals < 1 - epsilon)
    safe_vals = abs_vals[mask]

    if safe_vals.size > 0:
        taus = 1.0 / np.maximum(np.abs(np.log(safe_vals)), epsilon)
        median_tau = float(np.median(taus))
        max_tau = float(np.max(taus))
    else:
        median_tau = float("nan")
        max_tau = float("nan")

    eigen_stats = {
        "max_abs_eig": float(np.max(abs_vals)),
        "median_abs_eig": float(np.median(abs_vals)),
        "median_tau": median_tau,
        "max_tau": max_tau,
    }

    per_unit = {
        "eigenvalues_real": eigvals.real.astype(np.float32),
        "eigenvalues_imag": eigvals.imag.astype(np.float32),
        "eigenvalues_abs": abs_vals.astype(np.float32),
    }
    return eigen_stats, per_unit


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Infer architecture from path name
    import re

    match = re.search(r"k(\d+)_c(\d+)_gru(\d+)", str(checkpoint_path))
    if not match:
        raise ValueError(f"Cannot parse architecture from path {checkpoint_path}")
    kernel_size = int(match.group(1))
    channels = int(match.group(2))
    gru_hidden = int(match.group(3))

    model = create_model(channels, gru_hidden, kernel_size=kernel_size)
    state_key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    return model


def ensure_output_dirs(base: Path, model_name: str) -> Tuple[Path, Path]:
    model_dir = base / model_name
    hidden_dir = model_dir / "hidden_samples"
    model_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, hidden_dir


def write_metrics_csv(model_dir: Path, rows: List[Dict[str, float]]) -> None:
    import pandas as pd  # Local import to avoid mandatory dependency if unused

    df = pd.DataFrame(rows)
    df.sort_values("epoch", inplace=True)
    df.to_csv(model_dir / "metrics.csv", index=False)


def write_unit_stats(model_dir: Path, unit_rows: List[Dict[str, float]]) -> None:
    import pandas as pd

    df = pd.DataFrame(unit_rows)
    df.sort_values(["epoch", "unit"], inplace=True)
    df.to_csv(model_dir / "unit_gate_stats.csv", index=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    base_dir = Path(args.checkpoint_dir)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    games = load_dataset(dataset_path, args.max_games)
    feature_names_sorted = sorted(compute_board_features(games[0]["states"][0], 0).keys())

    device = torch.device(args.device)

    for model_dir in list_model_dirs(base_dir):
        model_name = model_dir.name
        if args.verbose:
            print(f"\n=== Analyzing {model_name} ===")

        checkpoints = list_checkpoints(model_dir)
        if not checkpoints:
            if args.verbose:
                print("  No checkpoints found, skipping.")
            continue

        model_out_dir, hidden_out_dir = ensure_output_dirs(output_dir, model_name)

        metrics_rows: List[Dict[str, float]] = []
        unit_rows: List[Dict[str, float]] = []

        for epoch, ckpt_path in checkpoints:
            if args.verbose:
                print(f"  Epoch {epoch:3d}: loading {ckpt_path.name}")
            model = load_model(ckpt_path, device)
            metrics, unit_metrics, eigen_per_unit, samples = analyze_checkpoint(
                model, games, args.sample_hidden, device, rng
            )

            metrics_row = {"epoch": epoch}
            metrics_row.update(metrics)
            metrics_rows.append(metrics_row)

            unit_records: Dict[Tuple[int, int], Dict[str, float]] = {}
            for stat_name, values in unit_metrics.items():
                if isinstance(values, np.ndarray):
                    for idx, val in enumerate(values):
                        key = (epoch, idx)
                        entry = unit_records.setdefault(key, {"epoch": epoch, "unit": idx})
                        entry[stat_name] = float(val)
            unit_rows.extend(unit_records.values())

            sample_path = hidden_out_dir / f"epoch_{epoch:03d}.npz"
            sample_payload = {
                "hidden": np.stack([s["hidden"] for s in samples], axis=0) if samples else np.empty((0, model.gru_hidden_size)),
                "update_gate": np.stack([s["update_gate"] for s in samples], axis=0) if samples else np.empty((0, model.gru_hidden_size)),
                "reset_gate": np.stack([s["reset_gate"] for s in samples], axis=0) if samples else np.empty((0, model.gru_hidden_size)),
                "features": np.stack([s["features"] for s in samples], axis=0) if samples else np.empty((0, len(feature_names_sorted))),
                "feature_names": np.array(feature_names_sorted),
                "step_index": np.concatenate([s["step_index"] for s in samples]) if samples else np.empty((0,), dtype=np.int32),
            }
            np.savez_compressed(sample_path, **sample_payload)

            eigen_path = model_out_dir / f"epoch_{epoch:03d}_eigenvalues.npz"
            np.savez_compressed(eigen_path, **eigen_per_unit)

        if metrics_rows:
            write_metrics_csv(model_out_dir, metrics_rows)
        if unit_rows:
            write_unit_stats(model_out_dir, unit_rows)

        if args.verbose:
            print(f"  -> Outputs written to {model_out_dir}")


if __name__ == "__main__":
    main()

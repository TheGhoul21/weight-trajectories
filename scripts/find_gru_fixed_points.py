#!/usr/bin/env python3
"""Fixed-point analysis for GRU checkpoints.

Finds hidden-state fixed points h* such that GRU(f, h*) ≈ h* for a selected
set of board contexts. Also estimates stability by inspecting the Jacobian of
the GRU transition and stores per-epoch summaries for downstream attractor
analysis.
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
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import create_model  # type: ignore


@dataclass
class FixedPointResult:
    context_index: int
    residual: float
    spectral_radius: float
    classification: str
    hidden: np.ndarray
    eigenvalues: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find GRU fixed points across checkpoints.")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/save_every_3",
        help="Base directory containing per-model checkpoints.",
    )
    parser.add_argument(
        "--dataset",
        default="data/connect4_sequential_10k_games.pt",
        help="Sequential Connect-4 dataset (.pt) used to sample board contexts.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=12,
        help="Maximum number of board contexts to analyse per model.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=8,
        help="Random restarts per context for fixed-point search (including zero init).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Maximum optimisation steps per restart.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Residual tolerance (mean squared error) to accept a fixed point.",
    )
    parser.add_argument(
        "--epoch-min",
        type=int,
        help="Minimum epoch to analyse (inclusive).",
    )
    parser.add_argument(
        "--epoch-max",
        type=int,
        help="Maximum epoch to analyse (inclusive).",
    )
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Stride when iterating checkpoints (e.g., 3 -> analyse every third epoch).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run analysis on (cpu or cuda).",
    )
    parser.add_argument(
        "--output-dir",
        default="diagnostics/gru_fixed_points",
        help="Directory where fixed-point summaries will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for context sampling and restarts.",
    )
    args, _ = parser.parse_known_args()
    return args


def list_model_dirs(base_dir: Path) -> List[Path]:
    return sorted(p for p in base_dir.iterdir() if p.is_dir() and "gru" in p.name)


def list_checkpoints(model_dir: Path) -> List[Tuple[int, Path]]:
    ckpts: List[Tuple[int, Path]] = []
    for path in model_dir.glob("weights_epoch_*.pt"):
        try:
            epoch = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        ckpts.append((epoch, path))
    ckpts.sort(key=lambda item: item[0])
    return ckpts


def filter_checkpoints(
    checkpoints: List[Tuple[int, Path]],
    epoch_min: int | None,
    epoch_max: int | None,
    epoch_step: int,
) -> List[Tuple[int, Path]]:
    epoch_step = max(1, epoch_step)
    filtered: List[Tuple[int, Path]] = []
    for epoch, path in checkpoints:
        if epoch_min is not None and epoch < epoch_min:
            continue
        if epoch_max is not None and epoch > epoch_max:
            continue
        filtered.append((epoch, path))
    if epoch_step > 1:
        filtered = filtered[::epoch_step]
    return filtered


def parse_model_name(name: str) -> Dict[str, int]:
    parts: Dict[str, int] = {}
    for token in name.split("_"):
        if token.startswith("k"):
            parts["kernel"] = int(token[1:])
        elif token.startswith("c"):
            parts["channels"] = int(token[1:])
        elif token.startswith("gru"):
            parts["gru"] = int(token[3:])
    return parts


def load_dataset_states(dataset_path: Path) -> List[torch.Tensor]:
    data = torch.load(dataset_path, weights_only=False)
    if "games" not in data:
        raise ValueError("Dataset must contain a 'games' list (sequential format).")
    states: List[torch.Tensor] = []
    games = data["games"]
    for game in games:
        seq = game["states"].float()
        for state in seq:
            states.append(state)
    return states


def sample_context_states(states: List[torch.Tensor], max_contexts: int, seed: int) -> List[torch.Tensor]:
    rng = random.Random(seed)
    indices = list(range(len(states)))
    rng.shuffle(indices)
    selected: List[torch.Tensor] = []
    for idx in indices:
        selected.append(states[idx])
        if len(selected) >= max_contexts:
            break
    return selected


def board_from_state(state: torch.Tensor) -> np.ndarray:
    yellow = (state[0] > 0.5).cpu().numpy()
    red = (state[1] > 0.5).cpu().numpy()
    board = np.zeros((6, 7), dtype=np.int8)
    board[yellow] = 1
    board[red] = 2
    return board


def get_current_player(state: torch.Tensor) -> int:
    return 1 if float(state[2, 0, 0]) < 0.5 else 2


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
    for row in range(6):
        for col in range(4):
            if np.all(board[row, col : col + 4] == player):
                return True
    for row in range(3):
        for col in range(7):
            if np.all(board[row : row + 4, col] == player):
                return True
    for row in range(3):
        for col in range(4):
            if all(board[row + i, col + i] == player for i in range(4)):
                return True
    for row in range(3, 6):
        for col in range(4):
            if all(board[row - i, col + i] == player for i in range(4)):
                return True
    return False


def count_three_in_row(board: np.ndarray, player: int) -> int:
    def window_has_three(cells: Iterable[int]) -> bool:
        arr = np.array(list(cells))
        return np.count_nonzero(arr == player) == 3 and np.count_nonzero(arr == 0) == 1

    total = 0
    for row in range(6):
        for col in range(4):
            if window_has_three(board[row, col : col + 4]):
                total += 1
    for row in range(3):
        for col in range(7):
            if window_has_three(board[row : row + 4, col]):
                total += 1
    for row in range(3):
        for col in range(4):
            if window_has_three(board[row + np.arange(4), col + np.arange(4)]):
                total += 1
    for row in range(3, 6):
        for col in range(4):
            if window_has_three(board[row - np.arange(4), col + np.arange(4)]):
                total += 1
    return total


def compute_board_features(state: torch.Tensor, move_index: int) -> Dict[str, float]:
    board = board_from_state(state)
    current = get_current_player(state)
    opponent = 3 - current
    yellow_count = int(np.count_nonzero(board == 1))
    red_count = int(np.count_nonzero(board == 2))
    valid = valid_moves(board)
    immediate_current = any(check_four(drop_piece(board, col, current), current) for col in valid)
    immediate_opponent = any(check_four(drop_piece(board, col, opponent), opponent) for col in valid)
    center_col = board[:, 3]
    center_current = int(np.count_nonzero(center_col == current))
    center_opponent = int(np.count_nonzero(center_col == opponent))

    return {
        "move_index": float(move_index),
        "current_player": float(current),
        "yellow_count": float(yellow_count),
        "red_count": float(red_count),
        "piece_diff": float(yellow_count - red_count),
        "valid_moves": float(len(valid)),
        "immediate_win_current": float(int(immediate_current)),
        "immediate_win_opponent": float(int(immediate_opponent)),
        "center_control_current": float(center_current),
        "center_control_opponent": float(center_opponent),
        "three_in_row_current": float(count_three_in_row(board, current)),
        "three_in_row_opponent": float(count_three_in_row(board, opponent)),
    }


def gru_step(model: nn.Module, feature: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    feature = feature.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
    h0 = h.unsqueeze(0).unsqueeze(1)  # (1, 1, hidden_dim)
    output, h_next = model.gru(feature, h0)
    return h_next.squeeze(0).squeeze(0)


def optimise_fixed_point(
    model: nn.Module,
    feature: torch.Tensor,
    h_init: torch.Tensor,
    max_iter: int,
    tol: float,
) -> Tuple[torch.Tensor | None, float]:
    h = torch.nn.Parameter(h_init.clone())
    optimiser = torch.optim.Adam([h], lr=0.05)
    best_residual = math.inf
    best_h: torch.Tensor | None = None

    for _ in range(max_iter):
        optimiser.zero_grad()
        h_next = gru_step(model, feature, h)
        diff = h_next - h
        loss = (diff * diff).mean()
        loss.backward()
        optimiser.step()

        residual = float(loss.detach())
        if residual < best_residual:
            best_residual = residual
            best_h = h.detach().clone()
        if residual < tol:
            break

    if best_h is None:
        return None, float(best_residual)

    # Final check with no grad
    with torch.no_grad():
        final_res = float(((gru_step(model, feature, best_h) - best_h) ** 2).mean())
    return best_h, final_res


def deduplicate_fixed_points(points: List[torch.Tensor], threshold: float = 1e-3) -> List[torch.Tensor]:
    unique: List[torch.Tensor] = []
    for candidate in points:
        if not unique:
            unique.append(candidate)
            continue
        if min(torch.norm(candidate - existing).item() for existing in unique) < threshold:
            continue
        unique.append(candidate)
    return unique


def compute_jacobian(model: nn.Module, feature: torch.Tensor, h_fp: torch.Tensor) -> torch.Tensor:
    def func(h_param: torch.Tensor) -> torch.Tensor:
        return gru_step(model, feature, h_param)

    return torch.autograd.functional.jacobian(func, h_fp, vectorize=True)


def classify_fixed_point(jacobian: torch.Tensor) -> Tuple[str, float, np.ndarray]:
    eigvals = np.linalg.eigvals(jacobian.detach().cpu().numpy())
    spectral_radius = float(np.abs(eigvals).max())
    if spectral_radius < 1 - 1e-3:
        cls = "stable"
    elif spectral_radius > 1 + 1e-3:
        cls = "unstable"
    else:
        cls = "marginal"
    return cls, spectral_radius, eigvals


def find_fixed_points_for_context(
    model: nn.Module,
    feature: torch.Tensor,
    restarts: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> List[torch.Tensor]:
    hidden_size = model.gru.hidden_size
    inits = [torch.zeros(hidden_size, device=feature.device)]
    for _ in range(max(0, restarts - 1)):
        init = torch.randn(hidden_size, device=feature.device) * 0.1
        inits.append(init)

    candidates: List[torch.Tensor] = []
    for init in inits:
        fp, residual = optimise_fixed_point(model, feature, init, max_iter, tol)
        if fp is None:
            continue
        if residual > tol * 10:
            continue
        candidates.append(fp)

    return deduplicate_fixed_points(candidates)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def ensure_output_dirs(base: Path, model_name: str) -> Path:
    model_dir = base / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "epochs").mkdir(exist_ok=True)
    return model_dir


def load_or_create_contexts(
    output_dir: Path,
    dataset_path: Path,
    max_contexts: int,
    seed: int,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    ctx_states_path = output_dir / "contexts.pt"
    ctx_meta_path = output_dir / "contexts.json"
    if ctx_states_path.exists() and ctx_meta_path.exists():
        states = torch.load(ctx_states_path)
        meta = json.loads(ctx_meta_path.read_text())
        return states, meta

    all_states = load_dataset_states(dataset_path)
    selected = sample_context_states(all_states, max_contexts, seed)
    states_tensor = torch.stack(selected, dim=0)
    metadata: List[Dict[str, float]] = []
    for idx, state in enumerate(selected):
        metadata.append(compute_board_features(state, move_index=idx))

    torch.save(states_tensor, ctx_states_path)
    ctx_meta_path.write_text(json.dumps(metadata, indent=2))
    return states_tensor, metadata


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = parse_model_name(ckpt_path.parent.name)
    model = create_model(meta.get("channels", 64), meta.get("gru", 32), meta.get("kernel", 3))
    key = "state_dict" if "state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[key])
    model.to(device)
    model.eval()
    return model


def compute_context_features(model: nn.Module, states: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        inputs = states.to(device)
        features = model.resnet(inputs).view(inputs.size(0), -1)
    return features


def main() -> None:
    args = parse_args()
    checkpoint_base = Path(args.checkpoint_dir)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    context_states, context_metadata = load_or_create_contexts(
        output_dir, dataset_path, args.max_contexts, args.seed
    )

    for model_dir in list_model_dirs(checkpoint_base):
        model_name = model_dir.name
        print(f"=== Fixed-point analysis for {model_name} ===")
        model_out_dir = ensure_output_dirs(output_dir, model_name)
        summary_rows: List[Dict[str, object]] = []

        checkpoints = list_checkpoints(model_dir)
        checkpoints = filter_checkpoints(
            checkpoints, args.epoch_min, args.epoch_max, args.epoch_step
        )

        for epoch, ckpt_path in checkpoints:
            print(f"  Epoch {epoch:3d}: {ckpt_path.name}")
            model = load_model_from_checkpoint(ckpt_path, device)
            context_features = compute_context_features(model, context_states, device)

            epoch_results: List[FixedPointResult] = []

            for ctx_index, feature in enumerate(context_features):
                feature = feature.to(device).detach()
                fixed_points = find_fixed_points_for_context(
                    model,
                    feature,
                    restarts=args.restarts,
                    max_iter=args.max_iter,
                    tol=args.tolerance,
                    rng=rng,
                )

                for fp in fixed_points:
                    fp = fp.detach()
                    try:
                        jac = compute_jacobian(model, feature, fp)
                        classification, spectral_radius, eigvals = classify_fixed_point(jac)
                    except RuntimeError as exc:
                        print(f"    Jacobian failed (ctx {ctx_index}): {exc}")
                        continue

                    with torch.no_grad():
                        diff = gru_step(model, feature, fp) - fp
                        residual = float((diff * diff).mean().item())
                    epoch_results.append(
                        FixedPointResult(
                            context_index=ctx_index,
                            residual=residual,
                            spectral_radius=spectral_radius,
                            classification=classification,
                            hidden=tensor_to_numpy(fp),
                            eigenvalues=eigvals.astype(np.complex64),
                        )
                    )

            if not epoch_results:
                continue

            hidden_array = np.stack([res.hidden for res in epoch_results], axis=0)
            residuals = np.array([res.residual for res in epoch_results], dtype=np.float32)
            spectral = np.array([res.spectral_radius for res in epoch_results], dtype=np.float32)
            classifications = np.array([res.classification for res in epoch_results], dtype="<U16")
            contexts = np.array([res.context_index for res in epoch_results], dtype=np.int32)
            eigvals = np.stack([res.eigenvalues for res in epoch_results], axis=0)

            epoch_path = model_out_dir / "epochs" / f"epoch_{epoch:03d}_fixed_points.npz"
            np.savez_compressed(
                epoch_path,
                hidden=hidden_array,
                residual=residuals,
                spectral_radius=spectral,
                classification=classifications,
                context_index=contexts,
                eigvals_real=eigvals.real,
                eigvals_imag=eigvals.imag,
            )

            for idx, res in enumerate(epoch_results):
                summary_rows.append(
                    {
                        "model": model_name,
                        "epoch": epoch,
                        "context_index": res.context_index,
                        "classification": res.classification,
                        "spectral_radius": res.spectral_radius,
                        "residual": res.residual,
                        **parse_model_name(model_name),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = model_out_dir / "fixed_points_summary.csv"
            summary_df.sort_values(["epoch", "context_index"], inplace=True)
            summary_df.to_csv(summary_path, index=False)
            print(f"  → Summary saved to {summary_path}")

    (output_dir / "contexts_metadata.json").write_text(json.dumps(context_metadata, indent=2))
    print("Fixed-point analysis complete.")


if __name__ == "__main__":
    main()

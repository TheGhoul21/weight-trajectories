import argparse
import json
import sys
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from umap import UMAP as _UMAP  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    try:
        from umap.umap_ import UMAP as _UMAP  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        _UMAP = None

import phate

try:
    from src.model import create_model
except ModuleNotFoundError:  # pragma: no cover - script executed outside package context
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.model import create_model


def _parse_model_config(checkpoint_dir: Path) -> Dict[str, int]:
    """Extract kernel/channels/hidden size from the checkpoint directory name."""
    match = re.search(r"k(\d+)_c(\d+)_gru(\d+)", checkpoint_dir.name)
    if not match:
        return {}
    return {
        "kernel": int(match.group(1)),
        "channels": int(match.group(2)),
        "gru_hidden": int(match.group(3)),
    }


def _gather_checkpoint_files(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    """Return list of (epoch, path) pairs sorted by epoch."""
    checkpoint_files: List[Tuple[int, Path]] = []
    for path in checkpoint_dir.glob("weights_epoch_*.pt"):
        match = re.search(r"weights_epoch_(\d+)\.pt", path.name)
        if not match:
            continue
        checkpoint_files.append((int(match.group(1)), path))
    checkpoint_files.sort(key=lambda item: item[0])
    return checkpoint_files


def _filter_epochs(pairs: List[Tuple[int, Path]], epoch_min: Optional[int],
                   epoch_max: Optional[int], stride: int) -> List[Tuple[int, Path]]:
    """Subset checkpoints by epoch range and stride."""
    stride = max(1, stride)
    filtered: List[Tuple[int, Path]] = []
    for epoch, path in pairs:
        if epoch_min is not None and epoch < epoch_min:
            continue
        if epoch_max is not None and epoch > epoch_max:
            continue
        filtered.append((epoch, path))
    return filtered[::stride]


def _flatten_weights(state_dict: Dict[str, torch.Tensor], component: str) -> np.ndarray:
    """Flatten selected weights into a single vector."""
    selected: List[np.ndarray] = []
    for key, value in state_dict.items():
        if "weight" not in key:
            continue
        if component == "cnn" and "resnet" not in key:
            continue
        if component == "gru" and "gru" not in key:
            continue
        if component == "all":
            pass
        else:
            if component not in {"cnn", "gru", "all"}:
                continue
        selected.append(value.detach().cpu().numpy().ravel())
    if not selected:
        raise ValueError(f"No weights collected for component '{component}'.")
    return np.concatenate(selected).astype(np.float32, copy=False)


def _load_run(checkpoint_dir: Path, component: str, epoch_min: Optional[int],
              epoch_max: Optional[int], stride: int) -> Tuple[np.ndarray, List[int]]:
    """Load checkpoints and return stacked weight matrix and epoch list."""
    checkpoint_pairs = _gather_checkpoint_files(checkpoint_dir)
    if not checkpoint_pairs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}.")

    filtered = _filter_epochs(checkpoint_pairs, epoch_min, epoch_max, stride)
    if not filtered:
        raise ValueError("No checkpoints remain after filtering by epoch range.")

    weights: List[np.ndarray] = []
    epochs: List[int] = []
    for epoch, path in filtered:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        weights.append(_flatten_weights(state_dict, component))
        epochs.append(int(checkpoint.get("epoch", epoch)))

    matrix = np.stack(weights)
    return matrix, epochs


def _compute_pca(weights: np.ndarray, random_state: int) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(weights)


def _compute_tsne(weights: np.ndarray, random_state: int) -> np.ndarray:
    n_samples = weights.shape[0]
    if n_samples < 3:
        raise ValueError("t-SNE requires at least three snapshots.")
    base = min(30.0, max(5.0, (n_samples - 1) / 3.0))
    max_valid = max(2.0, n_samples - 1.0)
    perplexity = min(base, max_valid - 1e-6)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=random_state)
    return tsne.fit_transform(weights)


def _compute_umap(weights: np.ndarray, random_state: int) -> np.ndarray:
    if _UMAP is None:
        raise ImportError("umap-learn is required for UMAP embeddings.")
    if weights.shape[0] < 2:
        raise ValueError("UMAP requires at least two snapshots.")
    neighbors = min(15, max(2, weights.shape[0] - 1))
    reducer = _UMAP(n_components=2, n_neighbors=neighbors,
                    min_dist=0.1, metric="euclidean", random_state=random_state)
    return reducer.fit_transform(weights)


def _compute_phate(weights: np.ndarray, random_state: int) -> np.ndarray:
    if weights.shape[0] < 2:
        raise ValueError("PHATE requires at least two snapshots.")
    neighbors = max(2, min(5, weights.shape[0] - 1))
    op = phate.PHATE(n_components=2, knn=neighbors, random_state=random_state, verbose=False)
    return op.fit_transform(weights)


_EMBEDDERS = {
    "pca": _compute_pca,
    "tsne": _compute_tsne,
    "umap": _compute_umap,
    "phate": _compute_phate,
}


def _plot_embedding(embedding: np.ndarray, epochs: List[int], title: str,
                    annotate: bool, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = colors.Normalize(vmin=min(epochs), vmax=max(epochs))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=epochs, cmap="viridis",
                         norm=norm, s=80, edgecolors="black", linewidths=0.6, alpha=0.85)

    ax.scatter(embedding[0, 0], embedding[0, 1], c="green", s=160, marker="o",
               edgecolors="black", linewidths=1.2, zorder=10, label=f"Start (epoch {epochs[0]})")
    ax.scatter(embedding[-1, 0], embedding[-1, 1], c="red", s=180, marker="*",
               edgecolors="black", linewidths=1.4, zorder=10, label=f"End (epoch {epochs[-1]})")

    if annotate:
        step = max(1, len(epochs) // 10)
        for idx in range(0, len(epochs), step):
            ax.annotate(f"Ep {epochs[idx]}", (embedding[idx, 0], embedding[idx, 1]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.annotate(f"Ep {epochs[-1]}", (embedding[-1, 0], embedding[-1, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Epoch")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _create_checkpoint_animation(embedding: np.ndarray, epochs: Sequence[int],
                                 title: str, save_path: Path, fps: int = 4) -> None:
    """Create a GIF that reveals points checkpoint-by-checkpoint.

    embedding: (n,2) array
    epochs: sequence of epoch labels (len n)
    """
    n = embedding.shape[0]
    if n < 2:
        print("Not enough points to animate; skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # Plot the backbone (light gray)
    ax.plot(embedding[:, 0], embedding[:, 1], '-', color='0.85', linewidth=1.5)

    scatter = ax.scatter([], [], s=110, edgecolors='black', linewidths=0.6, zorder=5)
    start_marker = ax.scatter([], [], s=200, c='green', marker='o', edgecolors='black', zorder=8)
    end_marker = ax.scatter([], [], s=220, c='red', marker='*', edgecolors='black', zorder=9)

    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.25)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        start_marker.set_offsets(np.empty((0, 2)))
        end_marker.set_offsets(np.empty((0, 2)))
        return scatter, start_marker, end_marker

    def update(frame):
        upto = frame + 1
        pts = embedding[:upto]
        scatter.set_offsets(pts)
        start_marker.set_offsets(pts[0:1])
        end_marker.set_offsets(pts[-1:])
        # annotate current epoch near the last point
        return scatter, start_marker, end_marker

    interval = max(40, int(1000 / max(1, fps)))
    anim = FuncAnimation(fig, update, frames=list(range(n)), init_func=init,
                         interval=interval, blit=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"  Saved animation to {save_path}")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        print(f"  Failed to save animation {save_path}: {exc}")
    finally:
        plt.close(fig)


def _write_points_csv(embedding: np.ndarray, ids: Iterable[int], save_path: Path,
                      id_name: str = "epoch") -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"{id_name},component_1,component_2\n"
    with save_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for idx, point in zip(ids, embedding):
            handle.write(f"{idx},{point[0]},{point[1]}\n")


def _format_run_title(checkpoint_dir: Path, component: str, method: str) -> str:
    config = _parse_model_config(checkpoint_dir)
    descriptor = checkpoint_dir.name
    if config:
        descriptor = (f"k={config['kernel']} c={config['channels']} "
                      f"gru={config['gru_hidden']}")
    return f"{method.upper()} | {descriptor} | {component.upper()}"


def analyze_run(checkpoint_dir: Path, component: str, methods: List[str],
                epoch_min: Optional[int], epoch_max: Optional[int], stride: int,
                random_state: int, annotate: bool, save_dir: Path,
                export_csv: bool, label: str, config: Dict[str, int],
                board_states: Optional[Sequence[torch.Tensor]] = None,
                board_methods: Optional[List[str]] = None,
                animate: bool = False, animate_fps: int = 4) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    weights, epochs = _load_run(checkpoint_dir, component, epoch_min, epoch_max, stride)
    weights = weights.astype(np.float32, copy=False)
    print(
        f"Loaded {len(epochs)} checkpoints from {checkpoint_dir} with feature dim {weights.shape[1]} "
        f"for component {component}."
    )

    weight_results: Dict[str, Dict[str, object]] = {}
    for method in methods:
            compute = _EMBEDDERS[method]
            embedding = compute(weights, random_state=random_state)
            title = _format_run_title(checkpoint_dir, component, method)
            figure_path = save_dir / checkpoint_dir.name / f"{component}_{method}.png"
            _plot_embedding(embedding, epochs, title=title, annotate=annotate,
                            save_path=figure_path)
            if export_csv:
                csv_path = save_dir / checkpoint_dir.name / f"{component}_{method}.csv"
                _write_points_csv(embedding, epochs, csv_path, id_name="epoch")
            print(f"Saved {method} embedding for {checkpoint_dir} to {figure_path}")

            # Optionally create an animation that reveals points in order
            if animate:
                anim_path = save_dir / checkpoint_dir.name / f"{component}_{method}_anim.gif"
                try:
                    _create_checkpoint_animation(embedding, epochs, title, anim_path, fps=animate_fps)
                except Exception as exc:  # pragma: no cover - runtime errors
                    print(f"  Animation failed for {checkpoint_dir} {method}: {exc}")

            weight_results[method] = {
                "embedding": embedding,
                "sequence": epochs,
                "label": label,
                "directory": checkpoint_dir,
                "sequence_prefix": "Ep ",
            }

    representation_results: Dict[str, Dict[str, object]] = {}
    if board_states and board_methods:
        try:
            repr_matrix = _extract_board_representations(checkpoint_dir, config, board_states)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"Skipping representation embedding for {checkpoint_dir}: {exc}")
        else:
            sample_indices = list(range(1, repr_matrix.shape[0] + 1))
            for method in board_methods:
                compute = _EMBEDDERS[method]
                embedding = compute(repr_matrix, random_state=random_state)
                title = f"{method.upper()} | {label} | board representations"
                figure_path = save_dir / checkpoint_dir.name / f"repr_{method}.png"
                _plot_representation_embedding(embedding, sample_indices, title,
                                               annotate, figure_path)
                if export_csv:
                    csv_path = save_dir / checkpoint_dir.name / f"repr_{method}.csv"
                    _write_points_csv(embedding, sample_indices, csv_path, id_name="sample")
                if animate:
                    anim_path = save_dir / checkpoint_dir.name / f"repr_{method}_anim.gif"
                    try:
                        _create_checkpoint_animation(embedding, sample_indices, title, anim_path, fps=animate_fps)
                    except Exception as exc:  # pragma: no cover - runtime errors
                        print(f"  Representation animation failed for {checkpoint_dir} {method}: {exc}")
                representation_results[method] = {
                    "embedding": embedding,
                    "sequence": sample_indices,
                    "label": label,
                    "directory": checkpoint_dir,
                    "sequence_prefix": "#",
                }

    return weight_results, representation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate low-dimensional embeddings of weight trajectories."
    )
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        required=True,
        help="One or more checkpoint directories (e.g. checkpoints/save_every_3/...)",
    )
    parser.add_argument(
        "--component",
        choices=["cnn", "gru", "all"],
        default="cnn",
        help="Model component to analyze (filters parameters by name).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(_EMBEDDERS.keys()),
        default=["pca", "tsne", "umap", "phate"],
        help="Which embeddings to compute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualizations/simple_embeddings"),
        help="Directory for plots and CSV exports.",
    )
    parser.add_argument("--epoch-min", type=int, help="Earliest epoch to include (inclusive).")
    parser.add_argument("--epoch-max", type=int, help="Latest epoch to include (inclusive).")
    parser.add_argument(
        "--epoch-step",
        type=int,
        default=1,
        help="Stride when sampling checkpoints (1 keeps all snapshots).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=0, help="Random seed for stochastic embeddings."
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate a subset of epochs directly on the plots.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Write embedding coordinates to CSV alongside plots.",
    )
    parser.add_argument(
        "--require-channel",
        type=int,
        help="If provided, only analyze checkpoint dirs whose parsed channel count matches.",
    )
    parser.add_argument(
        "--require-gru",
        type=int,
        help="If provided, only analyze checkpoint dirs whose parsed GRU size matches.",
    )
    parser.add_argument(
        "--require-kernel",
        type=int,
        help="If provided, only analyze checkpoint dirs whose parsed kernel size matches.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Create overlay plots that compare all selected runs per embedding method.",
    )
    parser.add_argument(
        "--board-representations",
        action="store_true",
        help="Also compute embeddings of GRU hidden states for sampled board positions.",
    )
    parser.add_argument(
        "--board-source",
        choices=["random", "dataset"],
        default="random",
        help="Source for board positions when --board-representations is enabled.",
    )
    parser.add_argument(
        "--board-count",
        type=int,
        default=120,
        help="Number of board positions to embed for representation analysis.",
    )
    parser.add_argument(
        "--board-dataset",
        type=Path,
        help="Path to dataset file when --board-source=dataset (supports .pt or .json).",
    )
    parser.add_argument(
        "--board-seed",
        type=int,
        default=0,
        help="Random seed for random board generation or dataset subsampling.",
    )
    parser.add_argument(
        "--representation-methods",
        nargs="+",
        choices=sorted(_EMBEDDERS.keys()),
        help="Embedding methods to use for board representations (defaults to --methods).",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create per-run animations (GIF) that reveal checkpoints sequentially.",
    )
    parser.add_argument(
        "--animate-fps",
        type=int,
        default=4,
        help="Frames per second for generated animations (default: 4).",
    )
    return parser.parse_args()


def _format_label(directory: Path, config: Dict[str, int]) -> str:
    if not config:
        return directory.name
    parts = []
    if "kernel" in config:
        parts.append(f"k{config['kernel']}")
    if "channels" in config:
        parts.append(f"c{config['channels']}")
    if "gru_hidden" in config:
        parts.append(f"gru{config['gru_hidden']}")
    return " ".join(parts)


def _plot_comparison(method: str, component: str,
                     entries: List[Dict[str, object]], annotate: bool,
                     save_path: Path) -> None:
    if len(entries) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.colormaps["tab20"].resampled(max(len(entries), 1))
    colors_list = cmap(np.linspace(0, 1, len(entries))) if entries else np.empty((0, 4))

    for idx, entry in enumerate(entries):
        embedding = entry["embedding"]
        sequence = entry["sequence"]
        label = entry["label"]
        prefix = entry.get("sequence_prefix", "")
        color = colors_list[idx]

        ax.plot(embedding[:, 0], embedding[:, 1], "-", color=color, linewidth=2, alpha=0.7)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=[color], s=70, alpha=0.8,
                   edgecolors="black", linewidths=0.6, label=label)

        ax.scatter(embedding[0, 0], embedding[0, 1], c=[color], s=150, marker="o",
                   edgecolors="black", linewidths=1.2, zorder=10)
        ax.scatter(embedding[-1, 0], embedding[-1, 1], c=[color], s=170, marker="*",
                   edgecolors="black", linewidths=1.3, zorder=10)

        if annotate:
            step = max(1, len(sequence) // 6)
            for j in range(0, len(sequence), step):
                ax.annotate(f"{prefix}{sequence[j]}", (embedding[j, 0], embedding[j, 1]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
            ax.annotate(f"{prefix}{sequence[-1]}", (embedding[-1, 0], embedding[-1, 1]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_title(f"{method.upper()} comparison ({component})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _create_comparison_animation(method: str, component: str,
                                 entries: List[Dict[str, object]], save_path: Path,
                                 fps: int = 4) -> None:
    """Animate comparison of multiple runs. Each run provides 'embedding' and 'sequence' lists.

    The animation iterates over the union of epochs (or sequence ids) and reveals each run up to
    that epoch. Saves a GIF to save_path.
    """
    # Build unified sorted list of unique epoch markers across runs
    all_epochs = sorted({int(ep) for entry in entries for ep in entry["sequence"]})
    if len(all_epochs) < 2:
        print("Not enough distinct epochs for comparison animation; skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.colormaps["tab20"].resampled(max(len(entries), 1))
    # Fix axes limits up front so initial empty artists don't lock autoscale to (0,1)
    all_x = np.concatenate([np.asarray(e["embedding"])[:, 0] for e in entries])
    all_y = np.concatenate([np.asarray(e["embedding"])[:, 1] for e in entries])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    # Add a small margin
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = 0.05 * x_span if x_span > 0 else 1.0
    y_pad = 0.05 * y_span if y_span > 0 else 1.0
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    color_samples = cmap(np.linspace(0, 1, len(entries))) if entries else np.empty((0, 4))

    lines = []
    scatters = []
    start_markers = []
    end_markers = []
    labels = []
    for idx, entry in enumerate(entries):
        emb = np.asarray(entry["embedding"])  # (n,2)
        color = color_samples[idx]
        # faint backbone to provide context and help autoscale visually
        ax.plot(emb[:, 0], emb[:, 1], '-', color=color, linewidth=1.2, alpha=0.15, zorder=1)
        line, = ax.plot([], [], '-', color=color, linewidth=2, alpha=0.7, zorder=3)
        scat = ax.scatter([], [], c=[color], s=60, edgecolors='black', linewidths=0.5, zorder=4)
        start = ax.scatter([], [], c=[color], s=140, marker='o', edgecolors='black', zorder=5)
        end = ax.scatter([], [], c=[color], s=160, marker='*', edgecolors='black', zorder=6)
        lines.append((line, emb))
        scatters.append((scat, emb))
        start_markers.append(start)
        end_markers.append(end)
        labels.append(entry.get('label', f'run{idx}'))

    ax.set_title(f"{method.upper()} comparison ({component})")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)

    epoch_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=12, va='top')

    def update(frame_idx):
        epoch = all_epochs[frame_idx]
        epoch_text.set_text(f'Epoch {epoch}')
        for i, ((line, emb), (scat, _), start, end) in enumerate(zip(lines, scatters, start_markers, end_markers)):
            seq = entries[i]['sequence']
            # find last index where sequence <= epoch
            idxs = [j for j, e in enumerate(seq) if int(e) <= epoch]
            if not idxs:
                line.set_data([], [])
                scat.set_offsets(np.empty((0, 2)))
                start.set_offsets(np.empty((0, 2)))
                end.set_offsets(np.empty((0, 2)))
                continue
            upto = idxs[-1] + 1
            pts = emb[:upto]
            line.set_data(pts[:, 0], pts[:, 1])
            scat.set_offsets(pts)
            start.set_offsets(pts[0:1])
            end.set_offsets(pts[-1:])
        return [l for l, _ in lines] + [s for s, _ in scatters] + start_markers + end_markers + [epoch_text]

    interval = max(40, int(1000 / max(1, fps)))
    anim = FuncAnimation(fig, update, frames=len(all_epochs), interval=interval, blit=False)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"  Saved comparison animation to {save_path}")
    except Exception as exc:
        print(f"  Failed to save comparison animation {save_path}: {exc}")
    finally:
        plt.close(fig)


def _coerce_board_tensor(sample) -> Optional[torch.Tensor]:
    """Convert various sample formats into a (1, 3, 6, 7) float tensor."""
    if isinstance(sample, torch.Tensor):
        tensor = sample.detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4 and tensor.shape[1:] == (3, 6, 7):
            return tensor.to(dtype=torch.float32)
        return None

    if isinstance(sample, np.ndarray):
        return _coerce_board_tensor(torch.from_numpy(sample))

    if isinstance(sample, (list, tuple)):
        return _coerce_board_tensor(np.asarray(sample))

    return None


def _load_dataset_boards(dataset_path: Path, max_count: int, seed: int) -> List[torch.Tensor]:
    """Load board tensors from a dataset file (.pt or .json)."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".pt":
        content = torch.load(dataset_path, map_location="cpu", weights_only=False)
    elif suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as fh:
            content = json.load(fh)
    else:
        raise ValueError("Unsupported dataset format; use .pt or .json")

    collected: List[torch.Tensor] = []

    def visit(node) -> None:
        if len(collected) >= max_count:
            return
        if isinstance(node, torch.Tensor):
            tensor = node.detach().cpu().to(dtype=torch.float32)
            if tensor.ndim == 3 and tensor.shape == (3, 6, 7):
                collected.append(tensor.unsqueeze(0))
                return
            if tensor.ndim == 4 and tensor.shape[1:] == (3, 6, 7):
                for idx in range(tensor.shape[0]):
                    if len(collected) >= max_count:
                        break
                    collected.append(tensor[idx:idx + 1])
                return
            if tensor.ndim == 5 and tensor.shape[-3:] == (3, 6, 7):
                reshaped = tensor.view(-1, 3, 6, 7)
                for idx in range(reshaped.shape[0]):
                    if len(collected) >= max_count:
                        break
                    collected.append(reshaped[idx:idx + 1])
                return
            return

        if isinstance(node, np.ndarray):
            visit(torch.from_numpy(node))
            return

        if isinstance(node, (list, tuple)):
            for item in node:
                if len(collected) >= max_count:
                    break
                visit(item)
            return

        if isinstance(node, dict):
            for key in ("states", "boards", "board_states", "samples", "data"):
                if key in node:
                    visit(node[key])
                    if len(collected) >= max_count:
                        return
            for value in node.values():
                if len(collected) >= max_count:
                    break
                visit(value)

    visit(content)

    if not collected:
        raise ValueError("Could not extract board tensors from dataset contents.")

    if len(collected) > max_count:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(collected), size=max_count, replace=False)
        collected = [collected[int(idx)] for idx in sorted(indices)]

    return collected[:max_count]


def _generate_random_boards(count: int, seed: int) -> List[torch.Tensor]:
    """Generate random-but-valid board occupation tensors."""
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
        board[2, :, :] = rng.integers(0, 2)
        boards.append(torch.from_numpy(board).unsqueeze(0))
    return boards


def _prepare_board_states(source: str, count: int, dataset_path: Optional[Path],
                          seed: int) -> List[torch.Tensor]:
    if count <= 0:
        raise ValueError("Board count must be positive")
    if source == "random":
        return _generate_random_boards(count, seed)
    if source == "dataset":
        if dataset_path is None:
            raise ValueError("--board-dataset must be provided when --board-source=dataset")
        return _load_dataset_boards(dataset_path, count, seed)
    raise ValueError(f"Unknown board source '{source}'")


def _extract_board_representations(checkpoint_dir: Path, config: Dict[str, int],
                                   board_states: Sequence[torch.Tensor]) -> np.ndarray:
    required = {"kernel", "channels", "gru_hidden"}
    if not required.issubset(config):
        raise ValueError("Checkpoint directory name must encode kernel/channels/gru sizes")

    checkpoint_pairs = _gather_checkpoint_files(checkpoint_dir)
    if not checkpoint_pairs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_path = checkpoint_pairs[-1][1]
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = create_model(config["channels"], config["gru_hidden"], config["kernel"])
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    representations: List[np.ndarray] = []
    with torch.no_grad():
        for board in board_states:
            state = board.to(device=device, dtype=torch.float32)
            _, _, hidden = model(state)
            vector = hidden.squeeze().view(-1).cpu().numpy()
            representations.append(vector)

    if not representations:
        raise ValueError("No representations were extracted from the provided boards")

    return np.stack(representations, axis=0).astype(np.float32, copy=False)


def _plot_representation_embedding(embedding: np.ndarray, indices: List[int], title: str,
                                   annotate: bool, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = colors.Normalize(vmin=min(indices), vmax=max(indices))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=indices, cmap="plasma",
                         norm=norm, s=70, edgecolors="black", linewidths=0.5, alpha=0.9)

    if annotate:
        step = max(1, len(indices) // 12)
        for idx in range(0, len(indices), step):
            ax.annotate(f"#{indices[idx]}", (embedding[idx, 0], embedding[idx, 1]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sample index")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    checkpoint_dirs = [Path(path) for path in args.checkpoint_dirs]
    for path in checkpoint_dirs:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    filtered_runs: List[Tuple[Path, Dict[str, int]]] = []
    for path in checkpoint_dirs:
        config = _parse_model_config(path)
        if args.require_channel is not None:
            if config.get("channels") != args.require_channel:
                print(f"Skipping {path} (channel {config.get('channels')} != {args.require_channel})")
                continue
        if args.require_gru is not None:
            if config.get("gru_hidden") != args.require_gru:
                print(f"Skipping {path} (gru {config.get('gru_hidden')} != {args.require_gru})")
                continue
        if args.require_kernel is not None:
            if config.get("kernel") != args.require_kernel:
                print(f"Skipping {path} (kernel {config.get('kernel')} != {args.require_kernel})")
                continue
        filtered_runs.append((path, config))

    if not filtered_runs:
        raise ValueError("No checkpoint directories remaining after applying filters.")

    board_methods: List[str] = []
    board_states: Optional[List[torch.Tensor]] = None
    if args.board_representations:
        board_methods = args.representation_methods or args.methods
        board_states = _prepare_board_states(
            source=args.board_source,
            count=args.board_count,
            dataset_path=args.board_dataset,
            seed=args.board_seed,
        )
        print(f"Prepared {len(board_states)} board states for representation embeddings.")

    comparison_weights: Dict[str, List[Dict[str, object]]] = {method: [] for method in args.methods}
    comparison_repr: Dict[str, List[Dict[str, object]]] = {method: [] for method in board_methods}

    for directory, config in filtered_runs:
        label = _format_label(directory, config)
        weight_results, repr_results = analyze_run(
            directory,
            component=args.component,
            methods=args.methods,
            epoch_min=args.epoch_min,
            epoch_max=args.epoch_max,
            stride=args.epoch_step,
            random_state=args.random_seed,
            annotate=args.annotate,
            save_dir=args.output_dir,
            export_csv=args.export_csv,
            label=label,
            config=config,
            board_states=board_states,
            board_methods=board_methods,
            animate=args.animate,
            animate_fps=args.animate_fps,
        )
        for method, payload in weight_results.items():
            comparison_weights[method].append(payload)
        for method, payload in repr_results.items():
            comparison_repr[method].append(payload)

    if args.compare and len(filtered_runs) > 1:
        for method, entries in comparison_weights.items():
            if len(entries) < 2:
                continue
            comparison_path = args.output_dir / "comparisons" / f"{args.component}_{method}_comparison.png"
            component_label = f"{args.component} weights"
            _plot_comparison(method, component_label, entries, args.annotate, comparison_path)
            print(f"Saved comparison plot for {method} at {comparison_path}")
            if args.animate:
                anim_path = args.output_dir / "comparisons" / f"{args.component}_{method}_comparison.gif"
                try:
                    _create_comparison_animation(method, component_label, entries, anim_path, fps=args.animate_fps)
                except Exception as exc:
                    print(f"  Comparison animation failed for {method}: {exc}")

        for method, entries in comparison_repr.items():
            if len(entries) < 2:
                continue
            comparison_path = args.output_dir / "comparisons" / f"board_repr_{method}_comparison.png"
            _plot_comparison(method, "board representations", entries, args.annotate, comparison_path)
            print(f"Saved board representation comparison plot for {method} at {comparison_path}")
            if args.animate:
                anim_path = args.output_dir / "comparisons" / f"board_repr_{method}_comparison.gif"
                try:
                    _create_comparison_animation(method, "board representations", entries, anim_path, fps=args.animate_fps)
                except Exception as exc:
                    print(f"  Board comparison animation failed for {method}: {exc}")

    print(f"\nDone. Outputs are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

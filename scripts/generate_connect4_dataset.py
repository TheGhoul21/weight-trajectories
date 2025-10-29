#!/usr/bin/env python3
"""
High-Quality Connect-4 Dataset Generator
=========================================

Generates a dataset of high-quality Connect-4 games using a trained AlphaZero model.
The dataset is optimized for PyTorch training with proper tensor formatting.

Usage:
    python generate_connect4_dataset.py --num-games 10000 --output data/connect4_10k_games.pt

Output Format:
    PyTorch .pt file containing a dictionary:
    {
        'states': torch.FloatTensor,      # Shape: (N, 3, 6, 7) - board states
        'policies': torch.FloatTensor,    # Shape: (N, 7) - MCTS policy targets
        'values': torch.FloatTensor,      # Shape: (N, 1) - game outcome values
        'metadata': dict                  # Generation parameters and statistics
    }
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add the AlphaZero repository to Python path
REPO_ROOT = Path(__file__).parent.parent
ALPHAZERO_PATH = REPO_ROOT / "dataset" / "Alpha-Zero-algorithm-for-Connect-4-game"
sys.path.insert(0, str(ALPHAZERO_PATH))

import numpy as np
import torch

# Import from AlphaZero repo
import config
from main_functions import load_or_create_neural_net, self_play

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _repro import seed_everything


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate high-quality Connect-4 game dataset"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10000,
        help="Number of games to generate (default: 10000)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=200,
        help="MCTS simulations per move for high quality (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/connect4_10k_games.pt",
        help="Output file path (default: data/connect4_10k_games.pt)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs to use (default: 4, safe for most systems)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Generate only 100 games as a test",
    )
    parser.add_argument(
        "--save-numpy",
        action="store_true",
        help="Also save in NumPy .npz format",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for self-play generation (default: 0)",
    )

    return parser.parse_args()


def raw_data_to_pytorch(raw_data):
    """
    Convert raw AlphaZero data format to PyTorch-friendly tensors.

    Args:
        raw_data: numpy array of shape (N, 134)
                  [0:42]    - Yellow board (flattened 6x7)
                  [42:84]   - Red board (flattened 6x7)
                  [84:126]  - Turn indicator (42 values)
                  [126:133] - Policy π (7 values)
                  [133]     - Value z (1 value)

    Returns:
        states: torch.FloatTensor of shape (N, 3, 6, 7)
        policies: torch.FloatTensor of shape (N, 7)
        values: torch.FloatTensor of shape (N, 1)
    """
    N = raw_data.shape[0]

    # Extract components
    yellow_boards = raw_data[:, 0:42].reshape(N, 6, 7)
    red_boards = raw_data[:, 42:84].reshape(N, 6, 7)
    turn_indicators = raw_data[:, 84:126].reshape(N, 6, 7)
    policies = raw_data[:, 126:133]
    values = raw_data[:, 133:134]

    # Stack into (N, 3, 6, 7) tensor: [yellow_plane, red_plane, turn_plane]
    states = np.stack([yellow_boards, red_boards, turn_indicators], axis=1)

    # Convert to PyTorch tensors
    states_tensor = torch.FloatTensor(states)
    policies_tensor = torch.FloatTensor(policies)
    values_tensor = torch.FloatTensor(values)

    return states_tensor, policies_tensor, values_tensor


def generate_dataset(num_games, simulations, cpus=None, seed=0):
    """
    Generate a high-quality dataset using self-play.

    Args:
        num_games: Total number of games to generate
        simulations: Number of MCTS simulations per move
        cpus: Number of CPUs to use (None = use config default)
        seed: Random seed used for reproducibility

    Returns:
        raw_data: numpy array of game positions
        stats: dictionary of generation statistics
    """
    print("="*80)
    print("High-Quality Connect-4 Dataset Generation")
    print("="*80)

    # Load the trained model
    print("\n[1/4] Loading trained AlphaZero model...")
    model = load_or_create_neural_net()
    model.eval()
    print(f"✓ Model loaded: {type(model).__name__}")

    # Configure generation parameters
    if cpus is not None:
        original_cpus = config.CPUS
        config.CPUS = cpus
        print(f"✓ Using {cpus} CPUs (overriding config default: {original_cpus})")
    else:
        print(f"✓ Using {config.CPUS} CPUs from config")

    # Warn about resource usage
    if config.CPUS > 8:
        print(f"\n⚠️  WARNING: Using {config.CPUS} CPUs will be very CPU-intensive!")
        print("   Consider using --cpus 4-8 for normal usage.")

    # Calculate actual games (must be divisible by CPUs)
    games_per_iteration = (num_games // config.CPUS)
    actual_games = games_per_iteration * config.CPUS

    print(f"\n[2/4] Generation parameters:")
    print(f"  - Target games: {num_games}")
    print(f"  - Actual games: {actual_games} ({games_per_iteration} iterations × {config.CPUS} CPUs)")
    print(f"  - MCTS simulations per move: {simulations}")
    print(f"  - Dirichlet noise: {config.dirichlet_for_self_play}")
    print(f"  - Data augmentation (flipping): {config.data_extension}")
    print(f"  - Seed: {seed}")
    print(f"  - Expected positions: ~{actual_games * 21 * (2 if config.data_extension else 1)}")

    # Generate games
    print(f"\n[3/4] Generating {actual_games} high-quality games...")
    print("(This may take a while depending on CPU count and simulations)")

    start_time = datetime.now()

    raw_data, winp1, winp2, draws, first_player_ratio = self_play(
        player=model,
        self_play_loop_number=games_per_iteration,
        CPUs=config.CPUS,
        sim_number=simulations,
        cpuct=config.CPUCT,
        tau=config.tau_self_play,
        tau_zero=config.tau_zero_self_play,
        use_dirichlet=config.dirichlet_for_self_play,
        seed=seed,
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Compute statistics
    total_games = winp1 + winp2 + draws
    stats = {
        'generation_time_seconds': duration,
        'total_games': int(total_games),
        'actual_games_generated': int(actual_games),
        'total_positions': int(raw_data.shape[0]),
        'positions_per_game': float(raw_data.shape[0] / actual_games),
        'player1_wins': int(winp1),
        'player2_wins': int(winp2),
        'draws': int(draws),
        'first_player_win_rate': float(first_player_ratio),
        'mcts_simulations': int(simulations),
        'cpus_used': int(config.CPUS),
        'data_augmentation': bool(config.data_extension),
        'dirichlet_noise': bool(config.dirichlet_for_self_play),
        'seed': int(seed),
    }

    print(f"\n✓ Generation complete in {duration:.1f} seconds")
    print(f"\n[4/4] Dataset statistics:")
    print(f"  - Total positions: {stats['total_positions']:,}")
    print(f"  - Positions per game: {stats['positions_per_game']:.1f}")
    print(f"  - Player 1 wins: {stats['player1_wins']} ({100*winp1/total_games:.1f}%)")
    print(f"  - Player 2 wins: {stats['player2_wins']} ({100*winp2/total_games:.1f}%)")
    print(f"  - Draws: {stats['draws']} ({100*draws/total_games:.1f}%)")
    print(f"  - First player advantage: {100*first_player_ratio:.1f}%")

    return raw_data, stats


def save_dataset(raw_data, stats, output_path, save_numpy=False):
    """
    Save dataset in PyTorch format (and optionally NumPy).

    Args:
        raw_data: Raw numpy array from self-play
        stats: Statistics dictionary
        output_path: Path to save the .pt file (relative to project root)
        save_numpy: Whether to also save .npz format
    """
    output_path = Path(output_path)

    # If path is relative, make it relative to REPO_ROOT (not current dir)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting to PyTorch format...")
    states, policies, values = raw_data_to_pytorch(raw_data)

    print(f"  - States shape: {states.shape}")
    print(f"  - Policies shape: {policies.shape}")
    print(f"  - Values shape: {values.shape}")

    # Add tensor shapes to metadata
    stats['tensor_shapes'] = {
        'states': list(states.shape),
        'policies': list(policies.shape),
        'values': list(values.shape),
    }
    stats['generation_timestamp'] = datetime.now().isoformat()
    stats['pytorch_version'] = torch.__version__
    stats['numpy_version'] = np.__version__
    stats.setdefault('seed', None)

    # Save PyTorch format
    print(f"\nSaving PyTorch dataset to: {output_path}")
    dataset = {
        'states': states,
        'policies': policies,
        'values': values,
        'metadata': stats,
    }
    torch.save(dataset, output_path)
    print(f"✓ Saved PyTorch dataset ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # Save metadata JSON
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        # Convert non-serializable types
        json_stats = {k: (str(v) if isinstance(v, (torch.Size, np.ndarray)) else v)
                      for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")

    # Optionally save NumPy format
    if save_numpy:
        numpy_path = output_path.with_suffix('.npz')
        print(f"\nSaving NumPy dataset to: {numpy_path}")
        np.savez_compressed(
            numpy_path,
            states=states.numpy(),
            policies=policies.numpy(),
            values=values.numpy(),
            **stats
        )
        print(f"✓ Saved NumPy dataset ({numpy_path.stat().st_size / 1024**2:.1f} MB)")

    return output_path


def main():
    """Main entry point."""
    args = parse_args()

    # Handle test run
    if args.test_run:
        print("⚠️  TEST RUN MODE: Generating only 20 games with 2 CPUs")
        args.num_games = 20
        args.cpus = 2  # Use only 2 CPUs for testing
        args.simulations = min(args.simulations, 30)  # Reduce sims for faster testing
        if not args.output.endswith('_test.pt'):
            base = args.output.replace('.pt', '')
            args.output = f"{base}_test.pt"

    seed_everything(args.seed, deterministic_torch=False)

    # Change to AlphaZero directory for data/ folder access
    original_dir = os.getcwd()
    os.chdir(ALPHAZERO_PATH)

    try:
        # Create data directory if needed
        data_dir = ALPHAZERO_PATH / "data"
        data_dir.mkdir(exist_ok=True)

        # Generate dataset
        raw_data, stats = generate_dataset(
            num_games=args.num_games,
            simulations=args.simulations,
            cpus=args.cpus,
            seed=args.seed,
        )

        # Change back to original directory for saving
        os.chdir(original_dir)

        # Save dataset
        output_path = save_dataset(
            raw_data=raw_data,
            stats=stats,
            output_path=args.output,
            save_numpy=args.save_numpy
        )

        print("\n" + "="*80)
        print("✓ DATASET GENERATION COMPLETE!")
        print("="*80)
        print(f"\nDataset saved to: {output_path.absolute()}")
        print(f"\nTo load in PyTorch:")
        print(f"  dataset = torch.load('{output_path}')")
        print(f"  states = dataset['states']    # Shape: {stats['tensor_shapes']['states']}")
        print(f"  policies = dataset['policies']  # Shape: {stats['tensor_shapes']['policies']}")
        print(f"  values = dataset['values']    # Shape: {stats['tensor_shapes']['values']}")

    finally:
        # Always restore directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

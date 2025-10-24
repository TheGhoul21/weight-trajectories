"""
Generate Connect-4 dataset with GAME SEQUENCES preserved

This version keeps games as sequences instead of flattening all positions.
Essential for training GRU models that need temporal information.

Key difference from generate_connect4_dataset.py:
- Old: Returns (N_positions, 3, 6, 7) - flat list of all positions
- New: Returns list of games, each game is (seq_len, 3, 6, 7) - preserves temporal order
"""

import sys
from pathlib import Path
import argparse
import pickle
import json
from datetime import datetime

import numpy as np
import torch

# Add AlphaZero code to path
REPO_ROOT = Path(__file__).parent.parent
ALPHAZERO_DIR = REPO_ROOT / "dataset" / "Alpha-Zero-algorithm-for-Connect-4-game"
sys.path.insert(0, str(ALPHAZERO_DIR))

import config
from main_functions import load_or_create_neural_net
from multiprocessing import Process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Connect-4 dataset with preserved game sequences (for GRU training)"
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=250,
        help="MCTS simulations per move (default: 250, higher = better quality but slower)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/connect4_sequential_1k_games.pt",
        help="Output file path (default: data/connect4_sequential_1k_games.pt)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs to use (default: 4)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Generate only 20 games as a test",
    )

    return parser.parse_args()


def play_one_game_sequential(player, sim_number, cpuct, tau, tau_zero, use_dirichlet, game_index):
    """
    Play one game and return the sequence of positions

    Modified from onevsonegame to return data per game instead of flattening
    """
    import random
    from Game_bitboard import Game
    from MCTS_NN import MCTS_NN

    # Safety first
    random.seed()
    np.random.seed()

    game_sequence = []  # Will store (state, policy, value) for each move

    gameover = 0
    turn = 0

    while gameover == 0:
        turn += 1

        # Initialize tree on first turn
        if turn == 1:
            game = Game()
            tree = MCTS_NN(player, use_dirichlet)
            rootnode = tree.createNode(game.state)
            currentnode = rootnode

        # Run MCTS simulations
        for _ in range(sim_number):
            tree.simulate(currentnode, cpuct)

        # Get move probabilities from MCTS
        visits = []
        moves = []

        for child in currentnode.children:
            visits.append(child.N ** (1/tau))
            moves.append(child.move)

        all_visits = np.asarray(visits)
        probvisit = all_visits / np.sum(all_visits)
        child_cols = [game.convert_move_to_col_index(move) for move in moves]

        # Create policy distribution (7 columns)
        child_cols = np.asarray(child_cols, dtype=int)
        policy = np.zeros(config.L)
        policy[child_cols] = probvisit

        # Get current state
        flatten_state = game.state_flattener(currentnode.state)

        # Store this position (value will be filled in after game ends)
        game_sequence.append({
            'state': flatten_state,
            'policy': policy,
            'value': 0  # Placeholder, will update after game ends
        })

        # Take a move
        if turn < tau_zero:
            currentnode = np.random.choice(currentnode.children, p=probvisit)
        else:
            max_idx = np.random.choice(np.where(all_visits == np.max(all_visits))[0])
            currentnode = currentnode.children[max_idx]

        # Reinit tree for next turn
        game = Game(currentnode.state)
        tree = MCTS_NN(player, use_dirichlet)
        rootnode = tree.createNode(game.state)
        currentnode = rootnode

        gameover = currentnode.isterminal()

    # Game has terminated - determine winner
    game = Game(currentnode.state)
    gameover, winner = game.gameover()

    # Determine game outcome
    if winner == 1:  # Yellow wins
        outcome = 1
    elif winner == 2:  # Red wins
        outcome = -1
    else:  # Draw
        outcome = 0

    # Update values for all positions
    # Value is from current player's perspective, alternates each turn
    for i, position in enumerate(game_sequence):
        # Player alternates: odd turns = yellow (1), even turns = red (-1)
        current_player = 1 if (i + 1) % 2 == 1 else -1

        # Value from current player's perspective
        if outcome == 0:  # Draw
            position['value'] = 0
        elif outcome == current_player:  # Current player wins
            position['value'] = 1
        else:  # Current player loses
            position['value'] = -1

    # Optionally apply data augmentation (horizontal flip)
    augmented_games = [game_sequence]

    if config.data_extension:
        flipped_sequence = []
        for position in game_sequence:
            # Flip the board state horizontally
            state = position['state']
            yellow = state[0:42].reshape(6, 7)
            red = state[42:84].reshape(6, 7)
            turn = state[84:126].reshape(6, 7)

            yellow_flipped = np.fliplr(yellow).flatten()
            red_flipped = np.fliplr(red).flatten()
            turn_flipped = np.fliplr(turn).flatten()

            state_flipped = np.concatenate([yellow_flipped, red_flipped, turn_flipped])

            # Flip policy (column order reversed)
            policy_flipped = np.flip(position['policy'].copy())

            flipped_sequence.append({
                'state': state_flipped,
                'policy': policy_flipped,
                'value': position['value']
            })

        augmented_games.append(flipped_sequence)

    # Save to file for parallel collection
    filename = f'./data/sequential_game_{game_index}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump({
            'games': augmented_games,
            'outcome': outcome,
            'winner': winner,
            'length': len(game_sequence)
        }, f)

    return len(augmented_games)


def generate_sequential_dataset(num_games, simulations=250, cpus=4):
    """
    Generate dataset with preserved game sequences

    Returns:
        games: List of game sequences
        stats: Statistics dictionary
    """
    print("="*80)
    print("Sequential Connect-4 Dataset Generation (for GRU Training)")
    print("="*80)

    # Load model
    print("\n[1/4] Loading trained AlphaZero model...")
    model = load_or_create_neural_net()
    model.eval()
    print(f"✓ Model loaded: {type(model).__name__}")

    # Configure
    config.CPUS = cpus
    print(f"✓ Using {cpus} CPUs")

    # Calculate actual games (must be divisible by CPUs)
    games_per_iteration = num_games // cpus
    actual_games = games_per_iteration * cpus

    print(f"\n[2/4] Generation parameters:")
    print(f"  - Target games: {num_games}")
    print(f"  - Actual games: {actual_games}")
    print(f"  - MCTS simulations: {simulations}")
    print(f"  - Data augmentation: {config.data_extension}")

    # Generate games
    print(f"\n[3/4] Generating {actual_games} games...")

    start_time = datetime.now()

    all_games = []
    total_positions = 0
    game_lengths = []
    outcomes = {'yellow_wins': 0, 'red_wins': 0, 'draws': 0}

    import tqdm

    for iteration in tqdm.tqdm(range(games_per_iteration)):
        # Parallel game generation
        procs = []

        for cpu_idx in range(cpus):
            game_idx = iteration * cpus + cpu_idx
            proc = Process(
                target=play_one_game_sequential,
                args=(model, simulations, config.CPUCT, config.tau_self_play,
                      config.tau_zero_self_play, config.dirichlet_for_self_play, game_idx)
            )
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        # Collect results
        for cpu_idx in range(cpus):
            game_idx = iteration * cpus + cpu_idx
            filename = f'./data/sequential_game_{game_idx}.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Add game(s) to dataset
            for game_seq in data['games']:
                all_games.append(game_seq)
                total_positions += len(game_seq)
                game_lengths.append(len(game_seq))

            # Track outcomes
            if data['outcome'] == 1:
                outcomes['yellow_wins'] += 1
            elif data['outcome'] == -1:
                outcomes['red_wins'] += 1
            else:
                outcomes['draws'] += 1

            # Clean up temp file
            Path(filename).unlink()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Compute statistics
    n_games_with_augmentation = len(all_games)

    stats = {
        'generation_time_seconds': duration,
        'original_games': actual_games,
        'total_games_with_augmentation': n_games_with_augmentation,
        'total_positions': total_positions,
        'avg_game_length': np.mean(game_lengths),
        'min_game_length': int(np.min(game_lengths)),
        'max_game_length': int(np.max(game_lengths)),
        'yellow_wins': outcomes['yellow_wins'],
        'red_wins': outcomes['red_wins'],
        'draws': outcomes['draws'],
        'mcts_simulations': simulations,
        'cpus_used': cpus,
        'data_augmentation': config.data_extension,
    }

    print(f"\n✓ Generation complete in {duration:.1f} seconds")
    print(f"\n[4/4] Dataset statistics:")
    print(f"  - Original games: {actual_games}")
    print(f"  - Games with augmentation: {n_games_with_augmentation}")
    print(f"  - Total positions: {total_positions:,}")
    print(f"  - Avg game length: {stats['avg_game_length']:.1f} moves")
    print(f"  - Game length range: [{stats['min_game_length']}, {stats['max_game_length']}]")
    print(f"  - Yellow wins: {outcomes['yellow_wins']} ({100*outcomes['yellow_wins']/actual_games:.1f}%)")
    print(f"  - Red wins: {outcomes['red_wins']} ({100*outcomes['red_wins']/actual_games:.1f}%)")
    print(f"  - Draws: {outcomes['draws']} ({100*outcomes['draws']/actual_games:.1f}%)")

    return all_games, stats


def save_sequential_dataset(games, stats, output_path):
    """
    Save dataset in PyTorch format with preserved sequences

    Format:
        - games: List of N games
        - Each game: List of M positions
        - Each position: {'state': (126,), 'policy': (7,), 'value': scalar}
    """
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting to PyTorch tensors...")

    # Convert each game to tensors
    games_tensor = []
    for game in games:
        game_states = []
        game_policies = []
        game_values = []

        for position in game:
            # Parse state (126,) -> (3, 6, 7)
            state = position['state']
            yellow = state[0:42].reshape(6, 7)
            red = state[42:84].reshape(6, 7)
            turn = state[84:126].reshape(6, 7)
            board_state = np.stack([yellow, red, turn], axis=0)

            game_states.append(board_state)
            game_policies.append(position['policy'])
            game_values.append([position['value']])

        games_tensor.append({
            'states': torch.FloatTensor(np.array(game_states)),  # (seq_len, 3, 6, 7)
            'policies': torch.FloatTensor(np.array(game_policies)),  # (seq_len, 7)
            'values': torch.FloatTensor(np.array(game_values))  # (seq_len, 1)
        })

    print(f"  - Number of games: {len(games_tensor)}")
    print(f"  - Example game shapes:")
    print(f"    - States: {games_tensor[0]['states'].shape}")
    print(f"    - Policies: {games_tensor[0]['policies'].shape}")
    print(f"    - Values: {games_tensor[0]['values'].shape}")

    # Add metadata
    stats['generation_timestamp'] = datetime.now().isoformat()
    stats['pytorch_version'] = torch.__version__
    stats['format'] = 'sequential'
    stats['format_description'] = 'List of games, each game is a sequence of (state, policy, value) tuples'

    # Save
    print(f"\nSaving sequential dataset to: {output_path}")
    dataset = {
        'games': games_tensor,
        'metadata': stats,
    }
    torch.save(dataset, output_path)
    print(f"✓ Saved sequential dataset ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # Save metadata JSON
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json_stats = {k: (str(v) if isinstance(v, (torch.Size, np.ndarray)) else v)
                      for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_path}")

    return output_path


def main():
    """Main entry point."""
    args = parse_args()

    # Handle test run
    if args.test_run:
        print("⚠️  TEST RUN MODE: Generating only 20 games with 2 CPUs")
        args.num_games = 20
        args.cpus = 2
        args.output = "data/connect4_sequential_test.pt"

    # Generate dataset
    games, stats = generate_sequential_dataset(
        num_games=args.num_games,
        simulations=args.simulations,
        cpus=args.cpus
    )

    # Save dataset
    output_path = save_sequential_dataset(games, stats, args.output)

    print("\n" + "="*80)
    print("✓ Dataset generation complete!")
    print(f"  Output: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

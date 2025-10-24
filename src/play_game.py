"""
Interactive Connect-4 game: Human vs AI
Play against your trained model
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import create_model


class Connect4Game:
    """Simple Connect-4 game engine"""

    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)  # 0=empty, 1=yellow, 2=red
        self.current_player = 1  # 1=yellow (human), 2=red (AI)

    def reset(self):
        """Reset the board"""
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def get_valid_moves(self):
        """Return list of valid column indices (0-6)"""
        return [col for col in range(7) if self.board[0, col] == 0]

    def make_move(self, col):
        """
        Drop piece in column col
        Returns: True if successful, False if invalid
        """
        if col not in self.get_valid_moves():
            return False

        # Find lowest empty row in column
        for row in range(5, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                break

        # Switch player
        self.current_player = 3 - self.current_player  # Toggle 1<->2
        return True

    def check_winner(self):
        """
        Check for winner
        Returns: 0 (no winner), 1 (yellow wins), 2 (red wins), -1 (draw)
        """
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if self.board[row, col] != 0 and \
                   self.board[row, col] == self.board[row, col+1] == \
                   self.board[row, col+2] == self.board[row, col+3]:
                    return self.board[row, col]

        # Check vertical
        for row in range(3):
            for col in range(7):
                if self.board[row, col] != 0 and \
                   self.board[row, col] == self.board[row+1, col] == \
                   self.board[row+2, col] == self.board[row+3, col]:
                    return self.board[row, col]

        # Check diagonal /
        for row in range(3, 6):
            for col in range(4):
                if self.board[row, col] != 0 and \
                   self.board[row, col] == self.board[row-1, col+1] == \
                   self.board[row-2, col+2] == self.board[row-3, col+3]:
                    return self.board[row, col]

        # Check diagonal \
        for row in range(3):
            for col in range(4):
                if self.board[row, col] != 0 and \
                   self.board[row, col] == self.board[row+1, col+1] == \
                   self.board[row+2, col+2] == self.board[row+3, col+3]:
                    return self.board[row, col]

        # Check draw
        if len(self.get_valid_moves()) == 0:
            return -1

        return 0

    def board_to_tensor(self):
        """
        Convert board to model input format: (1, 3, 6, 7)
        Channels: [yellow_pieces, red_pieces, current_turn]
        """
        yellow = (self.board == 1).astype(np.float32)
        red = (self.board == 2).astype(np.float32)
        turn = np.full((6, 7), self.current_player - 1, dtype=np.float32)  # 0 or 1

        state = np.stack([yellow, red, turn], axis=0)  # (3, 6, 7)
        return torch.from_numpy(state).unsqueeze(0)  # (1, 3, 6, 7)

    def display(self):
        """Print the board"""
        print("\n  " + " ".join(str(i) for i in range(7)))
        print("  " + "-" * 13)

        for row in range(6):
            row_str = "| "
            for col in range(7):
                cell = self.board[row, col]
                if cell == 0:
                    row_str += ". "
                elif cell == 1:
                    row_str += "Y "
                else:
                    row_str += "R "
            print(row_str + "|")
        print("  " + "-" * 13)


class AIPlayer:
    """AI player using trained model"""

    def __init__(self, checkpoint_path, device):
        """Load model from checkpoint"""
        print(f"Loading model from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract architecture params from checkpoint path or use defaults
        # Format: k{kernel}_c{channels}_gru{gru}/...
        import re
        match = re.search(r'k(\d+)_c(\d+)_gru(\d+)', checkpoint_path)
        if match:
            kernel_size = int(match.group(1))
            cnn_channels = int(match.group(2))
            gru_hidden = int(match.group(3))
        else:
            # Defaults
            kernel_size = 3
            cnn_channels = 64
            gru_hidden = 32

        self.model = create_model(cnn_channels, gru_hidden, kernel_size)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()
        self.device = device

        print(f"  Model loaded: k={kernel_size}, c={cnn_channels}, gru={gru_hidden}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    def get_move(self, game):
        """
        Get AI's move choice
        Returns: column index (0-6)
        """
        with torch.no_grad():
            state = game.board_to_tensor().to(self.device)
            policy_probs, value = self.model.predict(state)

            # Get valid moves and their probabilities
            valid_moves = game.get_valid_moves()
            valid_probs = policy_probs[0, valid_moves].cpu().numpy()

            # Choose move with highest probability
            best_idx = np.argmax(valid_probs)
            chosen_col = valid_moves[best_idx]

            print(f"\nAI thinking...")
            print(f"  Win probability: {value.item():.3f}")
            print(f"  Move probabilities: {dict(zip(valid_moves, valid_probs))}")
            print(f"  Chosen column: {chosen_col}")

            return chosen_col


def play_game(checkpoint_path, human_first=True):
    """Main game loop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game = Connect4Game()
    ai = AIPlayer(checkpoint_path, device)

    print("\n" + "=" * 50)
    print("Connect-4: Human vs AI")
    print("=" * 50)
    print(f"You are {'YELLOW (Y)' if human_first else 'RED (R)'}")
    print(f"AI is {'RED (R)' if human_first else 'YELLOW (Y)'}")
    print("Enter column number (0-6) to drop your piece")
    print("=" * 50)

    # If AI goes first
    if not human_first:
        game.display()
        col = ai.get_move(game)
        game.make_move(col)

    # Main game loop
    while True:
        game.display()

        # Check for game over
        winner = game.check_winner()
        if winner != 0:
            if winner == -1:
                print("\nGame Over: DRAW!")
            elif winner == 1:
                print("\nGame Over: YELLOW WINS!")
            else:
                print("\nGame Over: RED WINS!")
            break

        # Human's turn
        if game.current_player == 1 if human_first else game.current_player == 2:
            valid_moves = game.get_valid_moves()
            print(f"\nYour turn! Valid columns: {valid_moves}")

            while True:
                try:
                    col = int(input("Enter column (0-6): "))
                    if col in valid_moves:
                        game.make_move(col)
                        break
                    else:
                        print(f"Invalid column! Choose from {valid_moves}")
                except (ValueError, KeyboardInterrupt):
                    print("\nGame aborted!")
                    return

        # AI's turn
        else:
            col = ai.get_move(game)
            game.make_move(col)

    # Ask to play again
    print("\n" + "=" * 50)
    response = input("Play again? (y/n): ")
    if response.lower() == 'y':
        play_game(checkpoint_path, human_first)


def main():
    parser = argparse.ArgumentParser(description="Play Connect-4 against AI")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--ai-first", action="store_true",
                        help="AI plays first (default: human plays first)")

    args = parser.parse_args()

    play_game(args.checkpoint, human_first=not args.ai_first)


if __name__ == "__main__":
    main()

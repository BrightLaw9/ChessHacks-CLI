"""
Continuous overnight training script for ChessHacks model.

This script plays games against Stockfish, collects training data,
and continuously updates the model weights. It's designed to run for
extended periods (e.g., overnight) with periodic checkpoints.

Usage:
    python train_overnight.py [--stockfish-depth 15] [--games 1000] [--batch-size 128]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import chess
import chess.engine
import numpy as np
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import deque

from .chess_engine import ChessEngine
from .network_architecture import ChessNetwork
from .board_state import board_to_tensor
from .config import MODEL_PATH, BATCH_SIZE, LEARNING_RATE, NUM_SIMULATIONS
from .MCTSAlgorithm import MCTS, MCTSNode


# Configure logging
def setup_logging(log_dir="logs"):
    """Setup logging to file and console"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


logger = logging.getLogger(__name__)


class TrainingBuffer:
    """Stores game data for training"""
    
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.boards = deque(maxlen=max_size)
        self.policy_targets = deque(maxlen=max_size)
        self.value_targets = deque(maxlen=max_size)
    
    def add_position(self, board, mcts_policy, game_result, player_color):
        """
        Add a position to the buffer.
        
        Args:
            board: chess.Board object
            mcts_policy: dict of move -> prior probability from MCTS
            game_result: 1.0 (win), -1.0 (loss), 0.0 (draw) from perspective
            player_color: bool (True=White, False=Black)
        """
        # Convert board to tensor
        board_tensor = board_to_tensor(board)
        
        # Create policy target (4096-size array)
        policy_target = np.zeros(4096, dtype=np.float32)
        for move_uci, prob in mcts_policy.items():
            move = chess.Move.from_uci(move_uci)
            from_square = move.from_square
            to_square = move.to_square
            move_idx = from_square * 64 + to_square
            policy_target[move_idx] = prob
        
        # Normalize policy
        policy_sum = policy_target.sum()
        if policy_sum > 0:
            policy_target /= policy_sum
        
        # Calculate value from player's perspective
        if board.turn == player_color:
            value_target = game_result
        else:
            value_target = -game_result
        
        self.boards.append(board_tensor)
        self.policy_targets.append(policy_target)
        self.value_targets.append(np.array([value_target], dtype=np.float32))
    
    def get_batch(self, batch_size):
        """Get random batch from buffer"""
        if len(self) < batch_size:
            return None
        
        indices = np.random.choice(len(self), batch_size, replace=False)
        boards = torch.FloatTensor(np.array([self.boards[i] for i in indices]))
        policies = torch.FloatTensor(np.array([self.policy_targets[i] for i in indices]))
        values = torch.FloatTensor(np.array([self.value_targets[i] for i in indices]))
        
        return boards, policies, values
    
    def __len__(self):
        return len(self.boards)


class OvernightTrainer:
    """Main training loop for continuous learning"""
    
    def __init__(self, stockfish_path="stockfish", stockfish_depth=15, 
                 device=None, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
        """
        Initialize the trainer.
        
        Args:
            stockfish_path: Path to Stockfish executable
            stockfish_depth: Depth for Stockfish to search
            device: torch device (cuda/cpu)
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize network
        self.network = ChessNetwork().to(self.device)
        if os.path.exists(MODEL_PATH):
            self.network.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            logger.info(f"Loaded model from {MODEL_PATH}")
        
        self.network.train()  # Set to training mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize MCTS engine
        self.chess_engine = ChessEngine(num_simulations=NUM_SIMULATIONS, device=self.device)
        
        # Training buffer
        self.buffer = TrainingBuffer(max_size=10000)
        
        # Statistics
        self.games_played = 0
        self.total_positions = 0
        self.training_steps = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        
        # Checkpointing
        self.checkpoint_frequency = 10  # Save every N games
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.should_exit = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}. Finishing current game and saving...")
        self.should_exit = True
    
    def play_game_vs_stockfish(self):
        """
        Play a single game against Stockfish.
        Our model plays White, Stockfish plays Black.
        
        Returns:
            tuple: (game_positions_data, result)
                   result: 1.0 (our win), -1.0 (our loss), 0.0 (draw)
        """
        board = chess.Board()
        game_data = []
        
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                move_count = 0
                
                # Before starting each game:
                if self.games_played % 2 == 0:
                    white_player = "our"
                    black_player = "stockfish"
                else:
                    white_player = "stockfish"
                    black_player = "our"

                while not board.is_game_over():
                    print(board.move_stack)
                    player = white_player if board.turn else black_player

                    # -----------------------------------------
                    # OUR ENGINE MOVE
                    # -----------------------------------------
                    if player == "our":
                        best_move = self.chess_engine.get_best_move(board)

                        if best_move is None:
                            logger.warning("No legal moves available!")
                            break

                        # MCTS rollout and training data collection
                        root = MCTSNode(board)
                        for _ in range(NUM_SIMULATIONS):
                            node = root
                            search_path = [node]

                            while not node.is_leaf() and not node.board.is_game_over():
                                node = self.chess_engine.mcts._select_child(node)
                                search_path.append(node)

                            if not node.board.is_game_over():
                                value = self.chess_engine.mcts._expand_node(node)
                            else:
                                value = self.chess_engine.mcts._result_to_value(
                                    node.board.result(), board.turn
                                )

                            self.chess_engine.mcts._backpropagate(search_path, value)

                        # Build policy
                        mcts_policy = {}
                        total_visits = sum(child.visit_count for child in root.children.values())
                        if total_visits > 0:
                            for move, child in root.children.items():
                                mcts_policy[move.uci()] = child.visit_count / total_visits

                        # Save training data
                        game_data.append({
                            'board': board.copy(),
                            'policy': mcts_policy,
                            'player_color': chess.WHITE if white_player == "our" else chess.BLACK
                        })

                        board.push(best_move)

                        # -----------------------------------------
                        # STOCKFISH MOVE
                        # -----------------------------------------
                    else:
                        result = engine.play(board, chess.engine.Limit(depth=self.stockfish_depth))
                        board.push(result.move)
                            
                    move_count += 1
            
        except Exception as e:
            logger.error(f"Error during game: {e}")
            return game_data, 0.0
            
        print("Ran ", move_count, " number of moves")
        # Determine result
        result_str = board.result()
        if result_str == "1-0":
            game_result = 1.0
            self.white_wins += 1
        elif result_str == "0-1":
            game_result = -1.0
            self.black_wins += 1
        else:
            game_result = 0.0
            self.draws += 1
        
        return game_data, game_result
    
    def train_on_batch(self, boards, policies, values):
        """Train network on a batch of data"""
        boards = boards.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        # Forward pass
        policy_output, value_output = self.network(boards)
        
        # Compute losses
        policy_loss = nn.CrossEntropyLoss()(policy_output, policies.argmax(dim=1))
        value_loss = nn.MSELoss()(value_output, values)
        
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def save_checkpoint(self, tag=""):
        """Save model checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"model_{self.games_played}_games_{timestamp}{tag}.pth"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        torch.save(self.network.state_dict(), checkpoint_path)
        
        # Also update the main model
        # torch.save(self.network.state_dict(), MODEL_PATH)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def log_stats(self):
        """Log training statistics"""
        if self.games_played > 0:
            win_rate = self.white_wins / self.games_played * 100
            logger.info(
                f"Games: {self.games_played} | "
                f"Positions: {self.total_positions} | "
                f"Win Rate: {win_rate:.1f}% | "
                f"W:{self.white_wins} B:{self.black_wins} D:{self.draws} | "
                f"Training Steps: {self.training_steps}"
            )
    
    def run(self, num_games=None, max_duration_hours=None):
        """
        Run the training loop.
        
        Args:
            num_games: Maximum number of games to play (None = infinite)
            max_duration_hours: Maximum duration in hours (None = infinite)
        """
        logger.info("=" * 80)
        logger.info("Starting Overnight Training Loop")
        logger.info(f"Device: {self.device}")
        logger.info(f"Stockfish Depth: {self.stockfish_depth}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            while not self.should_exit:
                # Check duration limit
                if max_duration_hours:
                    elapsed_hours = (time.time() - start_time) / 3600
                    if elapsed_hours > max_duration_hours:
                        logger.info(f"Reached max duration of {max_duration_hours} hours")
                        break
                
                # Check game limit
                if num_games and self.games_played >= num_games:
                    logger.info(f"Reached max games: {num_games}")
                    break
                
                logger.info(f"\n--- Game {self.games_played + 1} ---")
                
                # Play a game
                game_data, result = self.play_game_vs_stockfish()
                self.games_played += 1
                
                result_str = {1.0: "WIN", -1.0: "LOSS", 0.0: "DRAW"}.get(result, "UNKNOWN")
                logger.info(f"Game result: {result_str}")
                
                # Add positions to buffer
                for position in game_data:
                    self.buffer.add_position(
                        position['board'],
                        position['policy'],
                        result,
                        position['player_color']
                    )
                    self.total_positions += 1
                
                # Train on buffer
                if len(self.buffer) >= self.batch_size:
                    batch = self.buffer.get_batch(self.batch_size)
                    if batch:
                        boards, policies, values = batch
                        losses = self.train_on_batch(boards, policies, values)
                        self.training_steps += 1
                        
                        logger.info(
                            f"Training step {self.training_steps} | "
                            f"Loss: {losses['total_loss']:.4f} | "
                            f"Policy: {losses['policy_loss']:.4f} | "
                            f"Value: {losses['value_loss']:.4f}"
                        )
                
                # Save checkpoint periodically
                if self.games_played % self.checkpoint_frequency == 0:
                    self.save_checkpoint()
                    self.log_stats()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
        
        finally:
            # Final checkpoint and stats
            logger.info("\n" + "=" * 80)
            logger.info("Training Complete - Final Stats")
            logger.info("=" * 80)
            self.save_checkpoint("_final")
            self.log_stats()
            
            elapsed_time = time.time() - start_time
            elapsed_hours = elapsed_time / 3600
            games_per_hour = self.games_played / elapsed_hours if elapsed_hours > 0 else 0
            
            logger.info(f"Total time: {elapsed_hours:.2f} hours")
            logger.info(f"Games per hour: {games_per_hour:.2f}")
            logger.info(f"Model saved to: {MODEL_PATH}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Continuous overnight training against Stockfish"
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        # default=15,
        default=2,
        help="Stockfish search depth (default: 15)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Number of games to play (default: infinite)"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Maximum training duration in hours (default: infinite)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="C:\\Users\\lawre\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish.exe",
        help="Path to Stockfish executable (default: stockfish)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create trainer
    trainer = OvernightTrainer(
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Run training
    trainer.run(
        num_games=args.games,
        max_duration_hours=args.hours
    )


if __name__ == "__main__":
    main()

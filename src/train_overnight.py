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
import random
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import deque
import math  # NEW

from .chess_engine import ChessEngine
from .network_architecture import ChessNetwork
from .board_state import board_to_tensor
from .config import MODEL_PATH, BATCH_SIZE, GAMES_PER_CHECKPOINT, LEARNING_RATE, NUM_SIMULATIONS
from .MCTSAlgorithm import MCTS, MCTSNode


# ==============================
# Stockfish-based helper funcs
# ==============================

LOGISTIC_K = 0.00368208   # from lichess-style mapping
CP_CLIP = 1000             # clip centipawns
POLICY_TAU = 200.0         # temperature for policy softmax
LAMBDA_VALUE = 1.0         # weight on value loss vs policy loss


def cp_to_value(cp, k=LOGISTIC_K, clip=CP_CLIP):
    """
    Convert Stockfish cp eval → [-1,1] value using logistic mapping.
    """
    cp = max(-clip, min(clip, cp))
    p = 1.0 / (1.0 + math.exp(-k * cp))  # 0–1
    value = 2.0 * p - 1.0  # → [-1,1]
    return value


def build_sf_policy_vector(board, engine, depth, clip=CP_CLIP, tau=POLICY_TAU, vec_size=4096):
    """
    Build a 4096-d policy target vector using Stockfish:
    - For each legal move, evaluate resulting position in cp.
    - Softmax over scaled cp's to get probabilities.
    - Map probabilities into a 4096 vector using (from * 64 + to) indexing.

    Returns:
        policy_vec (np.ndarray of shape [4096], float32) or None if no legal moves.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    cp_vals = []
    for move in legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        cp = info["score"].pov(board.turn).score(mate_score=100000)
        board.pop()
        cp = max(-clip, min(clip, cp))
        cp_vals.append(cp)

    cp_tensor = torch.tensor(cp_vals, dtype=torch.float32)
    logits = cp_tensor / tau
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

    policy_vec = np.zeros(vec_size, dtype=np.float32)
    for move, p in zip(legal_moves, probs):
        idx = move.from_square * 64 + move.to_square
        policy_vec[idx] = p

    policy_sum = policy_vec.sum()
    if policy_sum > 0:
        policy_vec /= policy_sum

    return policy_vec


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
        self.board_fens = deque(maxlen=max_size)  # NEW: store FENs for SF evaluation
    
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
        
        # Create policy target (4096-size array) from MCTS policy (original)
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
        
        # Calculate value from player's perspective (original AlphaZero-style)
        if board.turn == player_color:
            value_target = game_result
        else:
            value_target = -game_result
        
        self.boards.append(board_tensor)
        self.policy_targets.append(policy_target)
        self.value_targets.append(np.array([value_target], dtype=np.float32))
        self.board_fens.append(board.fen())  # NEW: store FEN for SF targets
    
    def get_batch(self, batch_size):
        """Get random batch from buffer"""
        if len(self) < batch_size:
            return None
        
        indices = np.random.choice(len(self), batch_size, replace=False)
        boards = torch.FloatTensor(np.array([self.boards[i] for i in indices]))
        policies = torch.FloatTensor(np.array([self.policy_targets[i] for i in indices]))
        values = torch.FloatTensor(np.array([self.value_targets[i] for i in indices]))
        fens = [self.board_fens[i] for i in indices]  # NEW
        
        return boards, policies, values, fens
    
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
        self.checkpoint_frequency = GAMES_PER_CHECKPOINT  # Save every N games
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.should_exit = False

        self.current_game_positions = []   # <-- store (board_tensor, sf_policy, sf_value)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}. Finishing current game and saving...")
        self.should_exit = True
    
    def play_game_vs_stockfish(self):
        """
        Play a single game against Stockfish.
        Our model plays White, Stockfish plays Black.
        
        Returns:
            tuple: (game_positions_data, result, move_list)
                   result: 1.0 (our win), -1.0 (our loss), 0.0 (draw)
        """
        board = chess.Board()
        game_data = []
        move_list = []  # store SANs/UCIs of moves for per-game logging
        
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

                # Random injection: choose how many moves until the next injected random move
                next_inject = random.choice([3, 5, 7])

                while not board.is_game_over():
                    # print(board.move_stack)
                    player = white_player if board.turn else black_player

                    # Decrement injection counter and, if zero, inject a random legal move
                    next_inject -= 1
                    if next_inject <= 0:
                        legal = list(board.legal_moves)
                        if legal:
                            injected = random.choice(legal)
                            logger.info(f"Injecting random move for {player}: {injected.uci()}")
                            board.push(injected)
                            # reset counter and continue to next player's turn without recording
                            next_inject = random.choice([3, 5, 7])
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
                            move_count += 1
                            continue

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

                        
                        # record move SAN (fallback to UCI) and log SAN so logs match saved move files
                        try:
                            san = board.san(best_move)
                        except Exception:
                            san = best_move.uci()
                        move_list.append(san)
                        logger.info(f"Move {move_count + 1}: ChessEngine (White) plays {san} ({best_move.uci()}) | Top MCTS moves: {sorted(mcts_policy.items(), key=lambda x: x[1], reverse=True)[:3]}")
                        # Train the classifier immediately after selecting the move
                        try:
                            self.collect_sf_labels(board, best_move)
                        except Exception as e:
                            logger.error(f"train_after_move failed: {e}")

                        print("Move made: ", best_move.uci())
                        board.push(best_move)

                        # -----------------------------------------
                        # STOCKFISH MOVE
                        # -----------------------------------------
                    else:
                        result = engine.play(board, chess.engine.Limit(depth=self.stockfish_depth))
                        # compute SAN for consistency with white moves and saved move files
                        try:
                            san = board.san(result.move)
                        except Exception:
                            san = result.move.uci()
                        move_list.append(san)
                        logger.info(f"Move {move_count + 1}: Stockfish (Black) plays {san} ({result.move.uci()})")
                        print("Stockfish move: ", result.move.uci())
                        board.push(result.move)
                            
                    move_count += 1
            
        except Exception as e:
            logger.error(f"Error during game: {e}")
            return game_data, 0.0, move_list
            
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
        
        return game_data, game_result, move_list
    
    def train_on_batch(self, boards, policies, values, board_fens):
        """
        Train network on a batch of data.

        Now uses Stockfish as:
          - Value target (cp → logistic [-1,1])
          - Policy target (softmax over SF move evals → 4096 vector)

        Falls back to original MCTS policy/value targets when SF targets
        can't be built (e.g., no legal moves).
        """
        boards = boards.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        batch_size = boards.size(0)

        # Forward pass
        policy_output, value_output = self.network(boards)
        # policy_output: [B, 4096] logits
        # value_output:  [B, 1]     predicted value in [-1,1] (ideally)

        # --- Build Stockfish targets for this batch ---
        sf_policy_targets = np.zeros_like(policies.cpu().numpy(), dtype=np.float32)
        sf_value_targets = np.zeros_like(values.cpu().numpy(), dtype=np.float32)

        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                for i, fen in enumerate(board_fens):
                    board = chess.Board(fen)

                    # 1) Value target from SF eval of current position
                    try:
                        info = engine.analyse(board, chess.engine.Limit(depth=self.stockfish_depth))
                        cp = info["score"].pov(board.turn).score(mate_score=100000)
                        sf_value_targets[i, 0] = cp_to_value(cp)
                    except Exception as e:
                        # Fallback to original self-play value if SF fails
                        sf_value_targets[i, 0] = values[i].item()

                    # 2) Policy target from SF eval of next positions
                    try:
                        policy_vec = build_sf_policy_vector(
                            board,
                            engine,
                            depth=self.stockfish_depth,
                            clip=CP_CLIP,
                            tau=POLICY_TAU,
                            vec_size=policies.shape[1]
                        )
                        if policy_vec is None:
                            # fallback: use MCTS policy target
                            sf_policy_targets[i] = policies[i].cpu().numpy()
                        else:
                            sf_policy_targets[i] = policy_vec
                    except Exception as e:
                        # fallback: use MCTS policy target
                        sf_policy_targets[i] = policies[i].cpu().numpy()

        except Exception as e:
            logger.error(f"Error during Stockfish evaluation for batch: {e}")
            # If SF completely fails, just train on MCTS targets
            sf_policy_targets = policies.cpu().numpy()
            sf_value_targets = values.cpu().numpy()

        sf_policy_targets = torch.FloatTensor(sf_policy_targets).to(self.device)
        sf_value_targets = torch.FloatTensor(sf_value_targets).to(self.device)

        # --- Policy loss: cross-entropy between SF distribution and predicted logits ---
        log_probs = torch.log_softmax(policy_output, dim=1)  # [B, 4096]
        policy_loss = -(sf_policy_targets * log_probs).sum(dim=1).mean()

        # --- Value loss: MSE between SF value target and predicted ---
        value_loss = nn.MSELoss()(value_output, sf_value_targets)

        total_loss = policy_loss + LAMBDA_VALUE * value_loss
        print("Policy loss: ", policy_loss, "; Value loss: ", value_loss, "; Total Loss: ", total_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # CRITICAL: Sync updated weights to chess engine so it uses the latest model
        self.chess_engine.network.load_state_dict(self.network.state_dict())
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
    def collect_sf_labels(self, board_before, move_chosen):
        """
        Only COLLECT labels, DO NOT TRAIN.
        This avoids BatchNorm(batch=1) crash and greatly improves performance.
        """

        # Clone positions
        board_b = board_before.copy()
        board_a = board_before.copy()
        board_a.push(move_chosen)

        # Convert board to tensor
        board_tensor = board_to_tensor(board_b).astype(np.float32)

        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:

                # --- VALUE TARGET ---
                info = engine.analyse(board_a, chess.engine.Limit(depth=self.stockfish_depth))
                cp = info["score"].pov(board_a.turn).score(mate_score=100000)
                sf_value = cp_to_value(cp)

                # --- POLICY TARGET ---
                sf_policy_vec = build_sf_policy_vector(
                    board_b,
                    engine,
                    depth=self.stockfish_depth,
                    vec_size=4096
                )

                # fallback if no legal moves
                if sf_policy_vec is None:
                    legal = list(board_b.legal_moves)
                    sf_policy_vec = np.zeros(4096, np.float32)
                    p = 1.0 / len(legal)
                    for mv in legal:
                        idx = mv.from_square * 64 + mv.to_square
                        sf_policy_vec[idx] = p

        except Exception as e:
            logger.error(f"SF evaluation failed: {e}")
            return

        # ADD to per-game buffer
        self.current_game_positions.append(
            (board_tensor, sf_policy_vec, sf_value)
        )
        #print(self.current_game_positions)
        print("Stockfish evaluation", sf_value)

    def train_after_game(self):
        """
        Run ONE batch update after the game using all collected positions.
        This keeps BatchNorm stable and improves convergence.
        """
        if len(self.current_game_positions) == 0:
            return

        boards = torch.tensor(
            [b for (b, _, _) in self.current_game_positions],
            dtype=torch.float32
        ).to(self.device)

        policies = torch.tensor(
            [p for (_, p, _) in self.current_game_positions],
            dtype=torch.float32
        ).to(self.device)

        values = torch.tensor(
            [[v] for (_, _, v) in self.current_game_positions],
            dtype=torch.float32
        ).to(self.device)

        # forward
        policy_logits, value_pred = self.network(boards)

        # losses
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
        value_loss = nn.MSELoss()(value_pred, values)
        total_loss = policy_loss + LAMBDA_VALUE * value_loss

        # backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        # sync to MCTS
        self.chess_engine.network.load_state_dict(self.network.state_dict())

        logger.info(
            f"[Post-Game Training] Total Loss={total_loss.item():.4f} | "
            f"policy={policy_loss.item():.4f} | value={value_loss.item():.4f}"
        )

        # clear buffer
        self.current_game_positions = []


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
                
                # Play a game (returns game data, result, and list of moves)
                game_data, result, move_list = self.play_game_vs_stockfish()
                self.train_after_game()
                # game index for logging (next game number)
                game_index = self.games_played + 1
                # Save per-game move list for inspection
                try:
                    Path("logs").mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    moves_file = Path("logs") / f"moves_game_{game_index}_{timestamp}.txt"
                    # Convert moves into numbered string: "1. e4 e5 2. Nf3 Nc6 ..."
                    numbered = []
                    for i in range(0, len(move_list), 2):
                        white_move = move_list[i]
                        black_move = move_list[i+1] if i+1 < len(move_list) else ""
                        ply = f"{(i//2)+1}. {white_move} {black_move}".strip()
                        numbered.append(ply)
                    moves_text = " ".join(numbered)
                    with open(moves_file, "w") as f:
                        f.write(moves_text + "\n")
                    logger.info(f"Saved moves for game {game_index} to {moves_file}")
                except Exception as e:
                    logger.warning(f"Could not write moves file for game {game_index}: {e}")

                self.games_played += 1
                
                result_str = {1.0: "WIN", -1.0: "LOSS", 0.0: "DRAW"}.get(result, "UNKNOWN")
                logger.info(f"Game result: {result_str}")
                
                # Add positions to buffer
                # for position in game_data:
                #     self.buffer.add_position(
                #         position['board'],
                #         position['policy'],
                #         result,
                #         position['player_color']
                #     )
                #     self.total_positions += 1
                
                # Train on buffer
                # if len(self.buffer) >= self.batch_size:
                #     batch = self.buffer.get_batch(self.batch_size)
                #     if batch:
                #         boards, policies, values, fens = batch
                #         losses = self.train_on_batch(boards, policies, values, fens)
                #         self.training_steps += 1
                        
                #         logger.info(
                #             f"Training step {self.training_steps} | "
                #             f"Loss: {losses['total_loss']:.4f} | "
                #             f"Policy: {losses['policy_loss']:.4f} | "
                #             f"Value: {losses['value_loss']:.4f} | "
                #             f"Weights synced to chess engine ✓"
                #         )
                
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
        default=15,
        # default=2,
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
    
    print("Creating trainer...")
    # Create trainer
    trainer = OvernightTrainer(
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("Running training")
    # Run training
    trainer.run(
        num_games=args.games,
        max_duration_hours=args.hours
    )


if __name__ == "__main__":
    main()

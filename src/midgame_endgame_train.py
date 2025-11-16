import os
import chess.pgn
import numpy as np
import torch
from torch import nn, optim
from collections import defaultdict
from board_state import board_to_tensor
from network_architecture import ChessNetwork

# -------------------------------
# Game Phase Classification
# -------------------------------
def classify_game_phase(board):
    """Classify position as opening, midgame, or endgame."""
    move_count = board.fullmove_number
    piece_count = len(board.piece_map())

    # Check for specific piece patterns
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))

    # Endgame: few pieces or no queens and few rooks
    if piece_count <= 10 or (queens == 0 and rooks <= 2):
        return "endgame"
    # Opening: early moves with most pieces
    elif move_count <= 12 and piece_count >= 28:
        return "opening"
    # Everything else is midgame
    else:
        return "midgame"


def is_position_worth_learning(board, move_number):
    """Filter out positions that aren't useful for learning."""
    if move_number < 5:
        return False
    if len(list(board.legal_moves)) < 3:
        return False
    if board.is_checkmate() or board.is_stalemate():
        return False
    return True


# -------------------------------
# Load PGN games with filtering
# -------------------------------
def load_pgn_games(pgn_folder, min_elo=1800, max_games_per_file=1000):
    """Load games with quality filtering."""
    games = []
    for filename in os.listdir(pgn_folder):
        if filename.endswith(".pgn"):
            path = os.path.join(pgn_folder, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                i = 0
                loaded = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    i += 1
                    try:
                        white_elo = int(game.headers.get("WhiteElo", 0))
                        black_elo = int(game.headers.get("BlackElo", 0))
                        avg_elo = (white_elo + black_elo) / 2
                        if avg_elo >= min_elo:
                            games.append(game)
                            loaded += 1
                    except Exception:
                        pass
                    if loaded >= max_games_per_file:
                        break
                    if i % 100 == 0:
                        print(f"Scanned {i} games from {filename}, loaded {loaded}")
                print(f"✓ Finished {filename}: loaded {loaded}/{i} games (avg ELO >= {min_elo})")
    return games


# -------------------------------
# Enhanced training data preparation
# -------------------------------
def prepare_training_data(
    games,
    focus_phase="all",  # "opening", "midgame", "endgame", or "all"
    opening_weight=0.3,
    midgame_weight=0.4,
    endgame_weight=0.3,
    skip_early_moves=5
):
    """Prepare training data with phase-aware sampling and discounted value targets."""
    training_data = {"opening": [], "midgame": [], "endgame": []}
    phase_stats = defaultdict(int)

    for j, game in enumerate(games, 1):
        board = chess.Board()
        outcome = game.headers.get("Result", "*")
        if outcome not in ["1-0", "0-1", "1/2-1/2"]:
            continue

        moves_list = list(game.mainline_moves())
        total_moves = len(moves_list)
        if total_moves < 15:
            continue

        for move_number, move in enumerate(moves_list, start=1):
            if move_number <= skip_early_moves:
                board.push(move)
                continue
            if not is_position_worth_learning(board, move_number):
                board.push(move)
                continue

            phase = classify_game_phase(board)
            phase_stats[phase] += 1

            # Discounted value target
            moves_to_end = total_moves - move_number
            discount_factor = np.exp(-moves_to_end / 20.0)
            if outcome == "1-0":
                base_value = 1.0 if board.turn == chess.WHITE else -1.0
            elif outcome == "0-1":
                base_value = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                base_value = 0.0
            value_target = base_value * discount_factor

            # Position tensor
            position_tensor = board_to_tensor(board)

            # Policy target: uniform over legal moves with slight preference for actual move
            policy_target = np.zeros(64 * 64, dtype=np.float32)
            legal_moves = list(board.legal_moves)
            for legal_move in legal_moves:
                idx = legal_move.from_square * 64 + legal_move.to_square
                policy_target[idx] = 1.0 / len(legal_moves)
            actual_idx = move.from_square * 64 + move.to_square
            policy_target[actual_idx] += 0.3
            policy_target /= policy_target.sum()

            training_data[phase].append({
                "position": position_tensor,
                "policy_target": policy_target,
                "value_target": value_target,
                "move_number": move_number,
                "phase": phase
            })

            board.push(move)

        if j % 50 == 0:
            print(f"Processed {j}/{len(games)} games")
            print(f"  Opening: {len(training_data['opening'])}, "
                  f"Midgame: {len(training_data['midgame'])}, "
                  f"Endgame: {len(training_data['endgame'])}")

    # Print statistics
    print("\n" + "="*60)
    print("📊 Training Data Statistics:")
    for phase in ["opening", "midgame", "endgame"]:
        print(f"{phase.capitalize()}: {len(training_data[phase]):,} positions")
    print("="*60 + "\n")

    # Combine based on weights if using "all"
    if focus_phase == "all":
        combined = []
        total_positions = sum(len(training_data[p]) for p in ["opening", "midgame", "endgame"])
        for phase, weight in [("opening", opening_weight), ("midgame", midgame_weight), ("endgame", endgame_weight)]:
            phase_data = training_data[phase]
            if len(phase_data) == 0:
                continue
            target_samples = int(total_positions * weight)
            if len(phase_data) < target_samples:
                samples = np.random.choice(len(phase_data), target_samples, replace=True)
            else:
                samples = np.random.choice(len(phase_data), target_samples, replace=False)
            combined.extend([phase_data[i] for i in samples])
        return combined
    elif focus_phase in training_data:
        return training_data[focus_phase]
    else:
        return training_data["opening"] + training_data["midgame"] + training_data["endgame"]


# -------------------------------
# Main training
# -------------------------------
def train(
    pgn_folder="./pgn_folder",
    focus_phase="all",
    opening_weight=0.2,
    midgame_weight=0.4,
    endgame_weight=0.4,
    epochs=5,
    batch_size=64,
    learning_rate=1e-4,
    model_path="models/trained_model.pth"
):
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using device:", device)
    if device.type == "cuda":
        print("   GPU:", torch.cuda.get_device_name(0))

    # Load games
    print("\n📖 Loading PGN games...")
    games = load_pgn_games(pgn_folder, min_elo=1800, max_games_per_file=1000)
    print(f"✓ Total quality games loaded: {len(games)}\n")
    if len(games) == 0:
        print("❌ No games loaded! Check your PGN folder.")
        return

    # Prepare training data
    print(f"⚙  Preparing training data (focus: {focus_phase})...")
    print(f"   Weights → Opening: {opening_weight:.0%}, Midgame: {midgame_weight:.0%}, Endgame: {endgame_weight:.0%}")
    training_data = prepare_training_data(
        games,
        focus_phase=focus_phase,
        opening_weight=opening_weight,
        midgame_weight=midgame_weight,
        endgame_weight=endgame_weight
    )
    print(f"✓ Total training positions: {len(training_data):,}\n")
    if len(training_data) == 0:
        print("❌ No training data prepared!")
        return

    # Initialize network
    network = ChessNetwork().to(device)
    if os.path.exists(model_path):
        try:
            network.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded existing model from {model_path}")
        except Exception as e:
            print(f"⚠  Could not load model: {e}\n   Starting fresh.")

    network.train()
    optimizer = optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Training loop
    print("="*60)
    print("🚀 Starting Training")
    print("="*60)

    for epoch in range(epochs):
        epoch_losses = []

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]

            # Convert batch to tensors
            positions = torch.from_numpy(np.stack([np.array(d["position"], dtype=np.float32) for d in batch])).to(device)
            policy_targets = torch.from_numpy(np.stack([np.array(d["policy_target"], dtype=np.float32) for d in batch])).to(device)
            value_targets = torch.from_numpy(np.array([[float(d["value_target"])] for d in batch], dtype=np.float32)).to(device)

            # Forward pass
            policy_pred, value_pred = network(positions)
            policy_loss = -torch.mean(torch.sum(policy_targets * torch.log_softmax(policy_pred, dim=1), dim=1))
            value_loss = value_loss_fn(value_pred, value_targets)
            total_loss = 0.7 * policy_loss + 0.3 * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(total_loss.item())

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\n✓ Epoch {epoch+1}/{epochs} completed | Avg Loss: {avg_epoch_loss:.4f}")
        scheduler.step(avg_epoch_loss)

    # Save model
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(network.state_dict(), model_path)
    print("="*60)
    print(f"✅ Model saved to {model_path}")
    print("="*60)


# -------------------------------
# Run training with arguments
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train chess network from PGN games")
    parser.add_argument("--pgn-folder", type=str, default="./pgn_folder")
    parser.add_argument("--focus", type=str, default="all", choices=["all", "opening", "midgame", "endgame"])
    parser.add_argument("--opening-weight", type=float, default=0.2)
    parser.add_argument("--midgame-weight", type=float, default=0.4)
    parser.add_argument("--endgame-weight", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-path", type=str, default="models/midgame_model.pth")
    args = parser.parse_args()

    train(
        pgn_folder=args.pgn_folder,
        focus_phase=args.focus,
        opening_weight=args.opening_weight,
        midgame_weight=args.midgame_weight,
        endgame_weight=args.endgame_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_path=args.model_path
    )
#!/usr/bin/env python3
"""Aggregate training logs and per-game move files into CSV summaries.

Produces:
 - logs/games_summary.csv
 - logs/move_level.csv

Usage: python scripts/aggregate_training_stats.py
"""
import re
import ast
from pathlib import Path
import csv

LOG_DIR = Path("logs")


def find_latest_training_log():
    logs = sorted(LOG_DIR.glob("training_*.log"))
    return logs[-1] if logs else None


def parse_training_log(log_path):
    text = log_path.read_text()
    # Split into game chunks by the marker '--- Game'
    chunks = re.split(r"--- Game (\d+) ---", text)
    # chunks will be like [pre, game1_index, game1_text, game2_index, game2_text, ...]
    games = []
    if len(chunks) < 3:
        return games

    # Iterate pairs
    it = iter(chunks)
    pre = next(it)
    while True:
        try:
            game_idx = next(it)
            chunk = next(it)
        except StopIteration:
            break

        game_index = int(game_idx)
        # Extract moves lines
        move_lines = []
        for m in re.finditer(r"Move (\d+): (.*?)$", chunk, flags=re.M):
            move_no = int(m.group(1))
            rest = m.group(2)
            move_lines.append((move_no, rest.strip()))

        # Result
        res_match = re.search(r"Game result: (WIN|LOSS|DRAW|UNKNOWN)", chunk)
        result = res_match.group(1) if res_match else "UNKNOWN"

        # Last training step info in this chunk (if any)
        train_match = None
        for t in re.finditer(r"Training step (\d+) \| Loss: ([0-9\.]+) \| Policy: ([0-9\.]+) \| Value: ([0-9\.]+).*(Weights synced to chess engine)?", chunk):
            train_match = t

        training = None
        if train_match:
            training = {
                'step': int(train_match.group(1)),
                'loss': float(train_match.group(2)),
                'policy_loss': float(train_match.group(3)),
                'value_loss': float(train_match.group(4)),
            }

        games.append({
            'game_index': game_index,
            'result': result,
            'move_lines': move_lines,
            'training': training,
            'chunk_text': chunk,
        })

    return games


def parse_top_mcts(rest):
    # Look for "Top MCTS moves: [ ... ]"
    m = re.search(r"Top MCTS moves: (\[.*\])", rest)
    if not m:
        return []
    try:
        lst = ast.literal_eval(m.group(1))
        # Expect list of (uci, prob)
        return lst
    except Exception:
        return []


def read_moves_file(game_index):
    # Find file matching moves_game_{game_index}_*.txt
    files = sorted(LOG_DIR.glob(f"moves_game_{game_index}_*.txt"))
    if not files:
        return []
    text = files[-1].read_text().strip()
    if not text:
        return []
    parts = text.split()
    # text is like: '1. e4 e5 2. Nf3 Nc6 ...' -> extract moves (every token that's not '1.' '2.' etc)
    moves = [p for p in parts if not re.match(r"^\d+\.$", p)]
    return moves


def main():
    LOG_DIR.mkdir(exist_ok=True)
    log_path = find_latest_training_log()
    if not log_path:
        print("No training log found in logs/")
        return

    games = parse_training_log(log_path)
    if not games:
        print("No games parsed from log")
        return

    # Write games summary
    games_csv = LOG_DIR / "games_summary.csv"
    moves_csv = LOG_DIR / "move_level.csv"

    with open(games_csv, "w", newline='') as gf, open(moves_csv, "w", newline='') as mf:
        gw = csv.writer(gf)
        mw = csv.writer(mf)
        gw.writerow(["game_index", "result", "training_step", "training_loss", "policy_loss", "value_loss", "num_moves"])
        mw.writerow(["game_index", "ply", "side", "san", "uci", "top1_uci", "top1_prob", "top2_uci", "top2_prob", "top3_uci", "top3_prob"])

        for g in sorted(games, key=lambda x: x['game_index']):
            moves = read_moves_file(g['game_index'])
            num_moves = len(moves)
            train = g['training'] or {}
            gw.writerow([
                g['game_index'],
                g['result'],
                train.get('step', ''),
                train.get('loss', ''),
                train.get('policy_loss', ''),
                train.get('value_loss', ''),
                num_moves,
            ])

            # Build move-level entries from parsed move_lines and moves file
            # Use move_lines to extract top MCTS moves
            for (move_no, rest) in g['move_lines']:
                # Determine side: odd move_no = White, even = Black
                side = 'W' if move_no % 2 == 1 else 'B'
                # SAN from moves list: index = move_no-1
                san = moves[move_no-1] if move_no-1 < len(moves) else ''
                # Extract UCI move token after the word 'plays'
                uci_match = re.search(r"plays\s+([^\s|]+)", rest)
                uci = uci_match.group(1) if uci_match else ''
                top = parse_top_mcts(rest)
                top_vals = []
                for i in range(3):
                    if i < len(top):
                        top_vals.extend([top[i][0], top[i][1]])
                    else:
                        top_vals.extend(['', ''])

                mw.writerow([
                    g['game_index'],
                    move_no,
                    side,
                    san,
                    uci,
                    top_vals[0], top_vals[1], top_vals[2], top_vals[3], top_vals[4], top_vals[5]
                ])

    print(f"Wrote {games_csv} and {moves_csv}")


if __name__ == '__main__':
    main()

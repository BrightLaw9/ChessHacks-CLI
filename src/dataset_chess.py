# dataset_chess.py
import os
import chess.pgn
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import zstandard as zstd
import io

from .board_state import board_to_tensor  # must return numpy array or tensor

def _legal_moves_policy(board):
    """Return a numpy float32 vector length 4096 (64x64) as in your original script."""
    policy = np.zeros(64*64, dtype=np.float32)
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return policy
    prob = 1.0 / len(legal_moves)
    for m in legal_moves:
        idx = m.from_square * 64 + m.to_square
        policy[idx] = prob
    return policy

class ChessIterablePGN(IterableDataset):
    """
    IterableDataset that streams PGN files from a folder.
    It shards files across world ranks and DataLoader workers to avoid duplication.
    Yields tuples: (position_tensor:float32 numpy array, policy_target:float32 numpy array, value_target:float32 scalar)
    """

    def __init__(self, pgn_folder, min_elo=0, skip_early_moves=5, max_games_per_file=None):
        super().__init__()
        self.pgn_folder = pgn_folder
        self.min_elo = min_elo
        self.skip_early_moves = skip_early_moves
        self.max_games_per_file = max_games_per_file

        # Precompute list of pgn file paths (sorted for deterministic sharding)
        files = [os.path.join(pgn_folder, f) for f in os.listdir(pgn_folder) if f.endswith(".pgn") or f.endswith(".pgn.zst") or f.endswith(".zst")]
        files.sort()
        self.files = files


    def _files_for_worker(self):
        """
        Determine which subset of files this DataLoader worker should read.
        We shard on (global rank, world_size) if present via env vars; otherwise only on worker id.
        """
        worker_info = get_worker_info()
        # Worker-level indexing
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # DDP rank/world_size (if set in env)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Total shards = world_size * num_workers
        total_shards = world_size * num_workers
        shard_id = rank * num_workers + worker_id

        # Deterministic round-robin file assignment
        selected = [p for i, p in enumerate(self.files) if (i % total_shards) == shard_id]
        print(f"[DIAGNOSTIC] _files_for_worker: total_files={len(self.files)}, world_size={world_size}, num_workers={num_workers}, rank={rank}, worker_id={worker_id}, shard_id={shard_id}/{total_shards}, assigned_files={len(selected)}")
        return selected

    def _iter_file(self, path):
        """
        Iterate games & moves in a .pgn or .pgn.zst file.
        Streams using zstandard decompression if needed.
        Yields (position_tensor, policy, value).
        """

        # ----- choose correct file handle -----
        if path.endswith(".zst"):
            # Streaming Zstandard decompression
            fh = open(path, "rb")
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            stream_reader = dctx.stream_reader(fh)
            # chess.pgn.read_game requires text mode → wrap as text IO
            text_f = io.TextIOWrapper(stream_reader, encoding="utf-8", errors="ignore")
        else:
            # Normal PGN
            text_f = open(path, "r", encoding="utf-8", errors="ignore")

        # ----- process file -----
        games_read = 0
        try:
            while True:
                game = chess.pgn.read_game(text_f)
                if game is None:  # EOF
                    break

                games_read += 1

                # Optional headers
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                    avg_elo = (white_elo + black_elo) / 2
                except Exception:
                    avg_elo = 0
                # DISABLED for diagnostics: if avg_elo < self.min_elo:
                #     continue

                moves_list = list(game.mainline_moves())
                # DISABLED for diagnostics: if len(moves_list) < 6:
                #     continue

                board = chess.Board()
                outcome = game.headers.get("Result", "*")
                if outcome not in ["1-0", "0-1", "1/2-1/2"]:
                    continue

                total_moves = len(moves_list)
                for move_number, move in enumerate(moves_list, start=1):
                    # DISABLED for diagnostics: if move_number <= self.skip_early_moves:
                    #     board.push(move)
                    #     continue

                    # DISABLED for diagnostics: if len(list(board.legal_moves)) < 2:
                    #     board.push(move)
                    #     continue
                    # DISABLED for diagnostics: if board.is_checkmate() or board.is_stalemate():
                    #     board.push(move)
                    #     continue

                    board.push(move)
                    pos_tensor = board_to_tensor(board)

                    # policy target
                    policy = _legal_moves_policy(board)
                    actual_idx = move.from_square * 64 + move.to_square
                    policy[actual_idx] += 0.3
                    policy = policy / (policy.sum() + 1e-12)

                    # final result as value target
                    if outcome == "1-0":
                        value = 1.0
                    elif outcome == "0-1":
                        value = -1.0
                    else:
                        value = 0.0

                    yield (
                        pos_tensor,
                        policy.astype(np.float32),
                        np.array([value], dtype=np.float32)
                    )

                    if self.max_games_per_file is not None and games_read >= self.max_games_per_file:
                        return

        finally:
            # Close files safely
            print("Games read: ", games_read)
            text_f.close()
            if path.endswith(".zst"):
                fh.close()


    def __iter__(self):
        print("[DIAGNOSTIC] __iter__: starting iterator")
        files = self._files_for_worker()
<<<<<<< HEAD
        print(f"[DIAGNOSTIC] __iter__: files_for_worker returned {len(files)} files")
        if not files:
            # If this worker got no files (e.g., fewer files than shards), return empty iterator
            print("[DIAGNOSTIC] __iter__: WARNING - no files assigned to this worker, returning empty iterator")
=======
        rank = os.environ.get("RANK", "0")
        if not files:
            print(f"[dataset][rank {rank}] No files assigned to this worker! pgn_folder={self.pgn_folder}")
>>>>>>> 41bbe7e (Modal Files Fixes)
            return iter(())
        print(f"[dataset][rank {rank}] Processing {len(files)} files. Example: {files[:3]}")
        for path in files:
            print(f"[DIAGNOSTIC] __iter__: processing file {path}")
            yield from self._iter_file(path)
        print("[DIAGNOSTIC] __iter__: iterator complete")

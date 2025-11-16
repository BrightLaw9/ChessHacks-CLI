# train_ddp_stream.py
import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataset_chess import ChessIterablePGN
from .network_architecture import ChessNetwork  # user-provided
import sys
import logging
# board_state.board_to_tensor is used from dataset; not needed here

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)



def collate_batch(batch):
    """
    Collate function: batch is list of tuples (pos_np, policy_np, value_np).
    Convert to torch tensors and stack.
    """
    positions = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    policies = torch.from_numpy(np.stack([b[1] for b in batch])).float()
    values = torch.from_numpy(np.stack([b[2] for b in batch])).float()
    return positions, policies, values

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_loop(rank, world_size, args):
    """
    Each process executes this function.
    rank should match local GPU id (0...world_size-1).
    """
    try:
        logger.warning("Entering training loop!")
        print("Enter train loop", flush=True)
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
        # Initialize process group (NCCL backend for GPUs)
        os.environ.setdefault("MASTER_ADDR", args.master_addr)
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        device = torch.device(f"cuda:{rank}")
        print(f"[rank {rank}] Using device {device}")

        # Set seeds (optionally offset by rank for determinism)
        setup_seed(42 + rank)

        # Dataset and DataLoader: iterable dataset will shard based on RANK and DataLoader worker id
        dataset = ChessIterablePGN(
            pgn_folder=args.pgn_folder,
            min_elo=args.min_elo,
            skip_early_moves=args.skip_early_moves,
            max_games_per_file=args.max_games_per_file,
        )
        print(f"[rank {rank}] Dataset created. Files found in {args.pgn_folder}: {len(dataset.files)} files", flush=True)
        if len(dataset.files) == 0:
            print(f"[rank {rank}] WARNING: No PGN files found! Checking if folder exists...", flush=True)
            print(f"[rank {rank}] Folder exists: {os.path.isdir(args.pgn_folder)}", flush=True)
            if os.path.isdir(args.pgn_folder):
                contents = os.listdir(args.pgn_folder)
                print(f"[rank {rank}] Folder contents: {contents}", flush=True)

        # Important: for IterableDataset, set shuffle False and don't use DistributedSampler.
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )
        print(f"[rank {rank}] DataLoader created with batch_size={args.batch_size}, num_workers={args.num_workers}", flush=True)

        # Model
        model = ChessNetwork().to(device)
        model.train()  # Ensure model is in training mode
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        value_loss_fn = nn.MSELoss()
        # policy loss computed via cross-entropy style using softmax + kl/neg-log-likelihood:
        # We'll compute negative log-likelihood of the target distribution
        model.train()

        global_step = 0
        epoch = -1  # Initialize in case loop doesn't run
        for epoch in range(args.epochs):
            print(f"[rank {rank}] Starting epoch {epoch+1}/{args.epochs}")
            epoch_losses = []
            for batch_idx, (positions, policies, values) in enumerate(dataloader):
                # Move to device
                positions = positions.to(device, non_blocking=True)  # shape [B, ...]
                policies = policies.to(device, non_blocking=True)    # shape [B, 4096]
                values = values.to(device, non_blocking=True)        # shape [B, 1]

                # Forward
                policy_logits, value_pred = model(positions)  # policy_logits: [B, 4096], value_pred: [B, 1] or [B]
                # Policy loss: KL / cross-entropy between target distribution and model logits
                log_probs = torch.log_softmax(policy_logits, dim=1)
                # Avoid log(0) by clamping
                policy_loss = -torch.sum(policies * log_probs, dim=1).mean()

                value_loss = value_loss_fn(value_pred.view(-1,1), values)
                loss = args.policy_weight * policy_loss + args.value_weight * value_loss

                print(f"Loss at {batch_idx}: ", loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                global_step += 1

                # Logging (only rank 0)
                if (batch_idx + 1) % args.log_every == 0 and rank == 0:
                    avg = sum(epoch_losses[-args.log_every:]) / len(epoch_losses[-args.log_every:])
                    logger.warning(f"[rank {rank}] Epoch {epoch+1} step {batch_idx+1} | loss {avg:.4f}")
                    print(f"[rank {rank}] Epoch {epoch+1} step {batch_idx+1} | loss {avg:.4f}", flush=True)
                    torch.save(model.module.state_dict(), model_filename)
                    print(f"[rank {rank}] Saved checkpoint to {model_filename}", flush=True)

            # Optionally save checkpoint at epoch end (rank 0 only)
            if rank == 0:
                os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
                model_filename = args.model_path + f"_epoch_{epoch}.pth"
                torch.save(model.module.state_dict(), model_filename)
                print(f"[rank {rank}] Saved checkpoint to {model_filename}", flush=True)

    finally:
        # Cleanup
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        
        print(f"[rank {rank}] Training finished. (epoch={epoch})", flush=True)
        
        # Only save if we actually trained (epoch >= 0)
        if epoch >= 0:
            os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
            model_filename = args.model_path + f"_epoch_{epoch}_rank_{rank}.pth"
            if 'model' in locals():
                torch.save(model.module.state_dict(), model_filename)
                print(f"[rank {rank}] Saved checkpoint to {model_filename}", flush=True)


def spawn_training(world_size, args):
    """
    Spawn processes for local multi-GPU training.
    Each child receives rank 0..world_size-1.
    """
    mp.spawn(train_loop, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn-folder", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-path", type=str, default="models/ddp_model.pth")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--policy-weight", type=float, default=0.7)
    parser.add_argument("--value-weight", type=float, default=0.3)
    parser.add_argument("--min-elo", type=int, default=1800)
    parser.add_argument("--skip-early-moves", type=int, default=5)
    parser.add_argument("--max-games-per-file", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--world-size", type=int, default=8, help="number of GPUs/processes to spawn")
    parser.add_argument("--num-nodes", type=int, default=8, help="set >1 for multi-node; not covered here")

    args = parser.parse_args()
    # quick validation
    assert os.path.isdir(args.pgn_folder), f"PGN folder not found: {args.pgn_folder}"
    if args.world_size > 1:
        spawn_training(args.world_size, args)
    else:
        # Single process (no mp.spawn) - still uses the same train_loop signature
        train_loop(0, 1, args)

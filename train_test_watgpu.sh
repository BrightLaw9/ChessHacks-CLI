#!/bin/bash
#SBATCH --job-name=chesshacks-test
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-test-%j.out
#SBATCH -e logs/slurm-test-%j-err.out

# ChessHacks Quick Test on WATGPU
# This runs a brief test to validate the setup (20 minutes)
# Submit with: sbatch train_test_watgpu.sh

set -e

WORK_DIR="/home/$USER/ChessHacks-CLI"
cd "$WORK_DIR" || exit 1

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate chesshacks

mkdir -p logs checkpoints

echo "Starting quick test (20 minutes, depth 12)..."

# Quick test: depth 12, 20 minutes
python -m src.train_overnight \
    --stockfish-depth 12 \
    --hours 0.33 \
    --batch-size 128 \
    --learning-rate 0.0005

echo "✓ Test complete"

#!/bin/bash
#SBATCH --job-name=chesshacks-10games
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-10games-%j.out
#SBATCH -e logs/slurm-10games-%j-err.out

# Quick 10-game training run to observe improvements

set -e

WORK_DIR="/u201/a44chhab/research/chessml/ChessHacks-CLI"
cd "$WORK_DIR" || exit 1

echo "Working directory: $(pwd)"

echo "Testing environment with: conda run -n chesshacks python -c ..."
conda run -n chesshacks python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo ""

# Make output dirs
mkdir -p logs checkpoints

STOCKFISH_PATH="/u201/a44chhab/research/chessml/chessengine/stockfish/stockfish-ubuntu-x86-64-avx2"
if [ ! -f "$STOCKFISH_PATH" ]; then
    echo "ERROR: Stockfish not found at $STOCKFISH_PATH"
    exit 1
fi
chmod +x "$STOCKFISH_PATH"

# Run training for 10 games
conda run -n chesshacks python -m src.train_overnight \
    --stockfish-path "$STOCKFISH_PATH" \
    --stockfish-depth 12 \
    --games 10 \
    --batch-size 64 \
    --learning-rate 0.0005

EXIT_CODE=$?

echo "Training exited with code $EXIT_CODE"

echo "Check logs: $WORK_DIR/logs"
echo "Per-game moves will be in files named logs/moves_game_*.txt"

exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=chesshacks-test-quick
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j-err.out

# ChessHacks Quick Test Training (15 minutes)
# Purpose: Verify training works, moves are made, and weights update
# Submit with: sbatch train_test_quick.sh

set -e

echo "================================================"
echo "ChessHacks Quick Test Training (15 min)"
echo "================================================"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo ""

# Set working directory
WORK_DIR="/home/$USER/ChessHacks-CLI"
cd "$WORK_DIR" || exit 1

echo "Working directory: $(pwd)"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate chesshacks || {
    echo "ERROR: Could not activate conda environment 'chesshacks'"
    exit 1
}

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create output directories
mkdir -p logs checkpoints

# Check Stockfish
if ! command -v stockfish &> /dev/null; then
    echo "ERROR: Stockfish not found in PATH"
    exit 1
fi

echo "Stockfish: $(which stockfish)"
echo ""

# Training parameters - SHORT for testing
STOCKFISH_DEPTH=8            # Fast, just for testing
NUM_HOURS=0.25               # 15 minutes = 0.25 hours
BATCH_SIZE=64                # Smaller batch for speed
LEARNING_RATE=0.0005

echo "================================================"
echo "Test Configuration"
echo "================================================"
echo "Stockfish Depth: $STOCKFISH_DEPTH (fast)"
echo "Duration: ${NUM_HOURS} hours (15 minutes)"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Goal: Play a few games, verify moves logged, check weight updates"
echo "================================================"
echo ""

# Store model hash before training
BEFORE_HASH=$(md5sum src/models/trained_model.pth | awk '{print $1}')
BEFORE_TIME=$(stat -c %Y src/models/trained_model.pth)
echo "Model before: MD5=$BEFORE_HASH, Timestamp=$(date -d @$BEFORE_TIME)"
echo ""

# Start timestamp
START_TIME=$(date)
echo "Training started: $START_TIME"
echo ""

# Run training
python -m src.train_overnight \
    --stockfish-depth "$STOCKFISH_DEPTH" \
    --hours "$NUM_HOURS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE"

EXIT_CODE=$?

# End timestamp
END_TIME=$(date)
echo ""
echo "Training completed: $END_TIME"
echo ""

# Check if model was updated
AFTER_HASH=$(md5sum src/models/trained_model.pth | awk '{print $1}')
AFTER_TIME=$(stat -c %Y src/models/trained_model.pth)
echo "Model after: MD5=$AFTER_HASH, Timestamp=$(date -d @$AFTER_TIME)"
echo ""

if [ "$BEFORE_HASH" = "$AFTER_HASH" ]; then
    echo "⚠️  WARNING: Model hash unchanged - weights may not have updated!"
else
    echo "✓ Model hash changed - weights were updated!"
fi

echo ""
echo "Check logs with: tail -100 logs/slurm-$SLURM_JOB_ID.out"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training exited with code $EXIT_CODE"
fi

exit $EXIT_CODE

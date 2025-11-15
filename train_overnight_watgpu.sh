#!/bin/bash
#SBATCH --job-name=chesshacks-train
#SBATCH --time=07:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j-err.out
#SBATCH --mail-user=YOUR_EMAIL_HERE@uwaterloo.ca
#SBATCH --mail-type=ALL

# ChessHacks Overnight Training on WATGPU
# 
# This SLURM script runs model training against Stockfish for 7 hours
# Submit with: sbatch train_overnight_watgpu.sh
# Monitor with: squeue -u $USER
# Cancel with: scancel <jobid>

set -e

echo "================================================"
echo "ChessHacks Training on WATGPU (SLURM)"
echo "================================================"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo ""

# Set working directory
WORK_DIR="/home/$USER/ChessHacks-CLI"
cd "$WORK_DIR" || exit 1

echo "Working directory: $(pwd)"
echo ""

# Activate conda environment
# IMPORTANT: Replace 'chesshacks' with your actual conda environment name
echo "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate chesshacks || {
    echo "ERROR: Could not activate conda environment 'chesshacks'"
    echo "Available environments:"
    conda env list
    exit 1
}

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create output directories
mkdir -p logs checkpoints

# Check if Stockfish is available
if ! command -v stockfish &> /dev/null; then
    echo "ERROR: Stockfish not found in PATH"
    echo "Please install Stockfish or add it to your PATH"
    exit 1
fi

echo "Stockfish found: $(which stockfish)"
echo "Stockfish version: $(stockfish --version 2>&1 | head -1)"
echo ""

# Training parameters
STOCKFISH_DEPTH=16           # Balanced for WATGPU
NUM_HOURS=7                  # Use full SLURM allocation
BATCH_SIZE=256
LEARNING_RATE=0.0005

echo "================================================"
echo "Training Configuration"
echo "================================================"
echo "Stockfish Depth: $STOCKFISH_DEPTH"
echo "Max Duration: ${NUM_HOURS} hours"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "================================================"
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

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training successful"
else
    echo "✗ Training exited with code $EXIT_CODE"
fi

echo ""
echo "Model saved to: $WORK_DIR/src/models/trained_model.pth"
echo "Checkpoints: $WORK_DIR/checkpoints/"
echo "Logs: $WORK_DIR/logs/"

exit $EXIT_CODE

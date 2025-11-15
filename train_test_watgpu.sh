#!/bin/bash
#SBATCH --job-name=chesshacks-test
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o logs/slurm-test-%j.out
#SBATCH -e logs/slurm-test-%j-err.out

# ChessHacks Quick Test on WATGPU
# This runs a brief test to verify training and weight updates (15 minutes, 3 games)
# Submit with: sbatch train_test_watgpu.sh

set -e

echo "================================================"
echo "ChessHacks Quick Training Test"
echo "================================================"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

WORK_DIR="/u201/a44chhab/research/chessml/ChessHacks-CLI"
cd "$WORK_DIR" || exit 1

echo "Working directory: $(pwd)"
echo ""

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

mkdir -p logs checkpoints

# Verify Stockfish exists
STOCKFISH_PATH="/u201/a44chhab/research/chessml/chessengine/stockfish/stockfish-ubuntu-x86-64-avx2"
if [ ! -f "$STOCKFISH_PATH" ]; then
    echo "ERROR: Stockfish not found at $STOCKFISH_PATH"
    exit 1
fi

chmod +x "$STOCKFISH_PATH"
echo "Stockfish found: $STOCKFISH_PATH"

# Use conda run to execute in chesshacks environment
echo ""
echo "Testing environment with: conda run -n chesshacks python -c ..."
conda run -n chesshacks python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo ""

# Capture model file info BEFORE
echo "Model file BEFORE training:"
ls -lh src/models/trained_model.pth
MODEL_TIME_BEFORE=$(stat -c%y src/models/trained_model.pth 2>/dev/null | cut -d'.' -f1 || echo "unknown")
MODEL_SIZE_BEFORE=$(stat -c%s src/models/trained_model.pth 2>/dev/null || echo "0")
echo ""

echo "================================================"
echo "Training Configuration"
echo "================================================"
echo "Stockfish: $STOCKFISH_PATH"
echo "Stockfish Depth: 10 (fast for testing)"
echo "Max Games: 3 (quick test)"
echo "Batch Size: 32 (small for quick iteration)"
echo "Learning Rate: 0.0005"
echo "================================================"
echo ""

START_TIME=$(date)
echo "Training started: $START_TIME"
echo ""

# Quick test: 3 games only with verbose logging
conda run -n chesshacks python -m src.train_overnight \
    --stockfish-path "$STOCKFISH_PATH" \
    --stockfish-depth 10 \
    --games 3 \
    --batch-size 32 \
    --learning-rate 0.0005

EXIT_CODE=$?

echo ""
echo "Training completed with exit code: $EXIT_CODE"
echo ""

# Capture model file info AFTER
echo "Model file AFTER training:"
ls -lh src/models/trained_model.pth
MODEL_TIME_AFTER=$(stat -c%y src/models/trained_model.pth 2>/dev/null | cut -d'.' -f1 || echo "unknown")
MODEL_SIZE_AFTER=$(stat -c%s src/models/trained_model.pth 2>/dev/null || echo "0")
echo ""

echo "================================================"
echo "Model File Verification"
echo "================================================"
echo "Size BEFORE: $MODEL_SIZE_BEFORE bytes"
echo "Size AFTER:  $MODEL_SIZE_AFTER bytes"
echo "Modified BEFORE: $MODEL_TIME_BEFORE"
echo "Modified AFTER:  $MODEL_TIME_AFTER"

if [ "$MODEL_SIZE_BEFORE" != "$MODEL_SIZE_AFTER" ] && [ "$MODEL_SIZE_AFTER" != "0" ]; then
    echo "✓ Model file was updated (weights changed!)"
else
    echo "⚠ Model file size unchanged (check logs for errors)"
fi
echo "================================================"
echo ""

if [ -f "logs/training_*.log" ]; then
    echo "Latest training log (last 30 lines):"
    echo "---"
    tail -30 logs/training_*.log
    echo "---"
    echo ""
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Quick test completed successfully!"
else
    echo "✗ Quick test exited with code $EXIT_CODE (check logs)"
fi

echo ""
echo "Model: $WORK_DIR/src/models/trained_model.pth"
echo "Checkpoints: $WORK_DIR/checkpoints/"
echo "Logs: $WORK_DIR/logs/"
echo ""

exit $EXIT_CODE

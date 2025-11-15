#!/bin/bash
# quick_test.sh
# Quick test of training pipeline (runs for 5 minutes)

set -e

echo "================================================"
echo "ChessHacks - Quick Training Test"
echo "================================================"
echo ""

# Navigate to project
cd "$(dirname "$0")"

# Check Stockfish
if ! command -v stockfish &> /dev/null; then
    echo "ERROR: Stockfish not installed!"
    echo "Run: brew install stockfish"
    exit 1
fi

echo "✓ Stockfish found"

# Activate venv
if [ -f "newenv/bin/activate" ]; then
    source newenv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Create directories
mkdir -p logs checkpoints

echo ""
echo "Starting 5-minute test run..."
echo "This will play ~3-5 games against Stockfish at depth 12"
echo ""

# Run test
python -m src.train_overnight \
    --stockfish-depth 12 \
    --hours 0.083 \
    2>&1 | tee logs/test_run.log

echo ""
echo "✓ Test complete! Check logs and checkpoints directories"

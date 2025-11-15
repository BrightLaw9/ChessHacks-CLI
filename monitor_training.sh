#!/bin/bash
# monitor_training.sh
# Monitor and check status of training jobs

echo "================================================"
echo "ChessHacks Training Monitor"
echo "================================================"
echo ""

# Check for running training processes
echo "Active training processes:"
ps aux | grep train_overnight | grep -v grep || echo "  (none)"

echo ""
echo "Latest log files:"
ls -lhtr logs/training_*.log 2>/dev/null | tail -5 || echo "  (no logs)"

echo ""
echo "Model checkpoints:"
ls -lhtr checkpoints/model_*.pth 2>/dev/null | tail -5 || echo "  (no checkpoints)"

echo ""
echo "Latest training log output (last 50 lines):"
echo "================================================"
if [ -f "$(ls -t logs/training_*.log 2>/dev/null | head -1)" ]; then
    tail -50 "$(ls -t logs/training_*.log 2>/dev/null | head -1)"
else
    echo "(no logs found)"
fi

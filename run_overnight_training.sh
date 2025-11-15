#!/bin/bash
# run_overnight_training.sh
# Launch script for overnight model training

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}ChessHacks Overnight Training Launcher${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if Stockfish is installed
if ! command -v stockfish &> /dev/null; then
    echo -e "${RED}ERROR: Stockfish not found!${NC}"
    echo "Please install Stockfish:"
    echo "  brew install stockfish"
    exit 1
fi

echo -e "${GREEN}✓ Stockfish found: $(which stockfish)${NC}"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/newenv/bin/activate" ]; then
    source "$PROJECT_ROOT/newenv/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}Warning: Could not find virtual environment at $PROJECT_ROOT/newenv${NC}"
fi

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/checkpoints"

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_ROOT/logs/training_${TIMESTAMP}.log"

echo -e "${GREEN}✓ Log file: $LOG_FILE${NC}"

# Parse arguments
STOCKFISH_DEPTH=15
NUM_GAMES=""
DURATION_HOURS=""
BATCH_SIZE=""
LEARNING_RATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --depth)
            STOCKFISH_DEPTH="$2"
            shift 2
            ;;
        --games)
            NUM_GAMES="--games $2"
            shift 2
            ;;
        --hours)
            DURATION_HOURS="--hours $2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="--batch-size $2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="--learning-rate $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Starting training with:${NC}"
echo -e "${GREEN}  Stockfish Depth: $STOCKFISH_DEPTH${NC}"
[ -n "$NUM_GAMES" ] && echo -e "${GREEN}  Max Games: ${NUM_GAMES#--games }${NC}"
[ -n "$DURATION_HOURS" ] && echo -e "${GREEN}  Max Duration: ${DURATION_HOURS#--hours } hours${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Run training
cd "$PROJECT_ROOT"

python -m src.train_overnight \
    --stockfish-depth "$STOCKFISH_DEPTH" \
    $NUM_GAMES \
    $DURATION_HOURS \
    $BATCH_SIZE \
    $LEARNING_RATE \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo -e "${GREEN}================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training exited with code $EXIT_CODE${NC}"
fi
echo -e "${GREEN}================================================${NC}"

exit $EXIT_CODE

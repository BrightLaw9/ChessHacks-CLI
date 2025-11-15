# Overnight Training Guide

This guide explains how to set up and run continuous model training against Stockfish overnight.

> **For WATGPU (University of Waterloo GPU Cluster)**: See [WATGPU_TRAINING_GUIDE.md](WATGPU_TRAINING_GUIDE.md) for SLURM-specific instructions.

## Overview

The `train_overnight.py` script:
- Plays chess games against Stockfish
- Collects training data from each game (board positions and MCTS policy)
- Continuously updates your neural network weights
- Saves checkpoints periodically
- Runs indefinitely or for a specified duration

Your model learns to:
1. **Predict moves** that match Stockfish's decisions (policy learning)
2. **Evaluate positions** accurately (value learning)

## Prerequisites

### 1. Install Stockfish

```bash
# macOS
brew install stockfish

# Verify installation
which stockfish
stockfish --version
```

### 2. Ensure Python Environment is Set Up

```bash
cd /Users/0591a/development/projects/ChessHacks/ChessHacks-CLI

# Activate virtual environment
source newenv/bin/activate

# Verify dependencies are installed
pip install -r requirements.txt
```

## Quick Start

### Option A: Manual Run (Recommended for First Test)

```bash
cd /Users/0591a/development/projects/ChessHacks/ChessHacks-CLI

# Activate environment
source newenv/bin/activate

# Run training for 2 hours with depth-15 Stockfish
python -m src.train_overnight --hours 2 --stockfish-depth 15
```

### Option B: Using the Launch Script

```bash
cd /Users/0591a/development/projects/ChessHacks/ChessHacks-CLI

# Make script executable
chmod +x run_overnight_training.sh

# Run with default settings (Stockfish depth 15)
./run_overnight_training.sh

# Or with custom duration and depth
./run_overnight_training.sh --depth 18 --hours 8 --games 100
```

## Usage Examples

### Example 1: Train for 8 Hours (Typical Overnight)

```bash
python -m src.train_overnight --hours 8 --stockfish-depth 15
```

### Example 2: Play 100 Games Regardless of Time

```bash
python -m src.train_overnight --games 100 --stockfish-depth 12
```

### Example 3: Stronger Training (Deeper Stockfish, But Slower)

```bash
python -m src.train_overnight --hours 4 --stockfish-depth 20
```

### Example 4: Faster Training (Shallower Stockfish)

```bash
python -m src.train_overnight --hours 8 --stockfish-depth 12
```

### Example 5: Custom Learning Rate and Batch Size

```bash
python -m src.train_overnight \
    --hours 8 \
    --stockfish-depth 15 \
    --batch-size 256 \
    --learning-rate 0.0005
```

## Command-Line Arguments

```
--hours HOURS              Maximum training duration in hours (default: infinite)
--games NUM                Maximum number of games to play (default: infinite)
--stockfish-depth DEPTH    Stockfish search depth (default: 15)
                          Lower = faster training, less opponent strength
                          Higher = slower training, stronger opponent
--batch-size SIZE          Training batch size (default: 256)
--learning-rate LR         Learning rate for optimizer (default: 0.001)
--stockfish-path PATH      Path to Stockfish executable (default: stockfish)
```

## Automated Scheduling

### Option 1: macOS `launchd` (Recommended)

This runs training automatically at 10 PM every day for 8 hours.

```bash
# Copy the plist file to Launch Agents
cp com.chesshacks.overnight-training.plist ~/Library/LaunchAgents/

# Load the job
launchctl load ~/Library/LaunchAgents/com.chesshacks.overnight-training.plist

# Verify it's loaded
launchctl list | grep chesshacks

# Disable it later with
launchctl unload ~/Library/LaunchAgents/com.chesshacks.overnight-training.plist
```

**Edit the plist to customize:**
- Change `<integer>22</integer>` (22 = 10 PM) to your preferred hour
- Change `<string>8</string>` in hours argument to train for different duration

### Option 2: `cron` Job

```bash
# Edit crontab
crontab -e

# Add this line to run at 10 PM daily:
0 22 * * * cd /Users/0591a/development/projects/ChessHacks/ChessHacks-CLI && ./run_overnight_training.sh --depth 15 --hours 8

# View your cron jobs
crontab -l

# Remove a cron job
crontab -e  # and delete the line
```

### Option 3: Manual Start Before Sleep

```bash
# Before going to sleep, run:
./run_overnight_training.sh --depth 15 --hours 8
```

The script will continue training even if you close your terminal or logout.

## Monitoring Training

### Check Current Status

```bash
# View active training processes
ps aux | grep train_overnight

# View latest logs
./monitor_training.sh

# Real-time monitoring
tail -f logs/training_$(date +%Y%m%d)_*.log
```

### View Training Results

Training creates two directories:
- `logs/` - Detailed training logs with losses, win rates, statistics
- `checkpoints/` - Model checkpoints saved every 50 games

```bash
# List logs
ls -lh logs/

# List checkpoints
ls -lh checkpoints/

# View latest checkpoint
ls -lhtr checkpoints/ | tail -1

# View latest log (last 100 lines)
tail -100 logs/training_$(ls -t logs/ | head -1 | cut -d_ -f2-)
```

## Understanding the Training Output

Each log line shows:

```
Games: 25 | Positions: 412 | Win Rate: 48.0% | W:12 B:13 D:0 | Training Steps: 150
```

- **Games**: Number of completed games
- **Positions**: Total board positions collected (training data points)
- **Win Rate**: Percentage of games won against Stockfish
- **W:12 B:13 D:0**: Wins, losses, draws
- **Training Steps**: Number of neural network training iterations

Lower loss values = better learning:
```
Training step 150 | Loss: 0.8234 | Policy: 0.5123 | Value: 0.3111
```

- **Loss**: Total training loss
- **Policy**: Cross-entropy loss for move prediction
- **Value**: MSE loss for position evaluation

## Model Files

The trained model is saved to: `src/models/trained_model.pth`

Checkpoints are saved to: `checkpoints/model_<games>_games_<timestamp>.pth`

## Stopping Training

### Graceful Shutdown (Recommended)

```bash
# Press Ctrl+C in terminal
# The script will:
# 1. Finish current game
# 2. Save final checkpoint
# 3. Log final statistics

# Or kill the process:
pkill -f train_overnight
```

### Hard Kill (Last Resort)

```bash
killall -9 python
```

## Tips for Optimal Overnight Training

### 1. **Balance Speed vs. Strength**
- Depth 12-14: ~3-5 min/game, good for quantity
- Depth 15-17: ~5-10 min/game, balanced
- Depth 18-20: ~10-20 min/game, very strong but slow

For an 8-hour session:
- Depth 12 → ~80-100 games
- Depth 15 → ~30-50 games  
- Depth 18 → ~15-30 games

### 2. **Monitor Win Rate**
- Expect 20-40% win rate against strong Stockfish
- If climbing above 40%, your model is improving!
- If stuck below 20%, try shallower Stockfish depth

### 3. **Keep Stockfish Depth < MCTS Simulations**
- Current: 100 MCTS simulations vs Stockfish depth 15
- This is a reasonable balance

### 4. **Restart Weekly**
```bash
# Keep old checkpoints
cp src/models/trained_model.pth src/models/trained_model_backup_$(date +%Y%m%d).pth

# Training will continue from latest weights
./run_overnight_training.sh --hours 8
```

### 5. **Check System Resources**
```bash
# Monitor CPU/GPU usage during training
top
# or
watch -n 1 nvidia-smi  # if using CUDA
```

## Troubleshooting

### Error: "Stockfish not found"
```bash
brew install stockfish
which stockfish
# If still not in PATH, use: python -m src.train_overnight --stockfish-path /usr/local/bin/stockfish
```

### Error: "Model file not found"
- This is OK! The script will create a new model and save it.

### Error: "CUDA out of memory"
```bash
# Reduce batch size
python -m src.train_overnight --batch-size 64 --hours 8
```

### Training is Very Slow
- Reduce Stockfish depth: `--stockfish-depth 12`
- Reduce batch size: `--batch-size 128`
- Check CPU load with `top`

### Process Dies Unexpectedly
```bash
# Check logs for errors
tail -100 logs/training_*.log

# Ensure virtual environment is working
source newenv/bin/activate
python -c "import torch; print(torch.__version__)"
```

## Advanced: Custom Training Settings

### Edit `src/config.py` for permanent changes:

```python
NUM_SIMULATIONS = 100        # More = stronger but slower
C_PUCT = 1.5                 # Exploration constant (1.0-2.0)
BATCH_SIZE = 256             # Larger = more stable, slower
LEARNING_RATE = 0.001        # Lower = more conservative
```

## FAQ

**Q: Can I interrupt and resume training?**
A: Yes! Training saves checkpoints every 50 games. Just run the script again.

**Q: What if I close my terminal?**
A: The process continues running in the background. Use `ps aux | grep train_overnight` to check.

**Q: How much disk space needed?**
A: Each checkpoint is ~50-100 MB. Logs are small (~1 MB/game). Budget ~2 GB for a week of training.

**Q: Can I use a different chess engine?**
A: Yes, modify `train_overnight.py` to use a different UCI engine (Arena, Komodo, etc.)

**Q: How often should I run this?**
A: Every night! Your model improves incrementally. Consistency matters more than duration.

## Next Steps

1. **Test it out**: Run for 1 hour manually first
2. **Check results**: Review logs and checkpoint
3. **Schedule it**: Set up launchd for automatic nightly runs
4. **Monitor**: Check logs weekly for improvement
5. **Tune parameters**: Adjust Stockfish depth based on your hardware

# ChessHacks Continuous Training - Complete Guide

This document provides a complete overview of continuous model training for ChessHacks, with support for both local machines and WATGPU cluster.

## Quick Navigation

- **Local Machine Setup**: See [OVERNIGHT_TRAINING_GUIDE.md](OVERNIGHT_TRAINING_GUIDE.md)
- **WATGPU Setup**: See [WATGPU_SETUP_GUIDE.md](WATGPU_SETUP_GUIDE.md)
- **WATGPU Training**: See [WATGPU_TRAINING_GUIDE.md](WATGPU_TRAINING_GUIDE.md)

## Overview

Your model learns by playing chess games against Stockfish:

1. **Play Game**: Model (using MCTS) vs Stockfish
2. **Collect Data**: Positions, move predictions, outcomes
3. **Train**: Update neural network weights
4. **Save**: Checkpoint every 50 games
5. **Repeat**: Continuously improve

## System Requirements

### Local Machine
- **RAM**: 16+ GB
- **GPU**: Optional but recommended (CUDA capable)
- **Stockfish**: Installed (via `brew install stockfish`)
- **Python**: 3.10+
- **Storage**: 2-5 GB for logs and checkpoints

### WATGPU Cluster
- **GPU**: 1-4 GPUs available
- **Memory**: 16-128 GB per job
- **CPUs**: 4-16 cores per job
- **Storage**: Home directory available

## Three Ways to Train

### Option 1: Local Machine (Overnight)

Best for: Personal laptop/desktop, testing

```bash
# Simple one-liner
python -m src.train_overnight --hours 8 --stockfish-depth 15
```

**Pros**: No setup needed, control local behavior
**Cons**: Computer must stay on, limited by local resources

### Option 2: Local Machine (Scheduled)

Best for: Automatic nightly runs on your machine

```bash
# macOS: Uses launchd
# Linux/Windows: Uses cron or Task Scheduler

# See OVERNIGHT_TRAINING_GUIDE.md for setup
```

**Pros**: Automatic, runs daily
**Cons**: Still requires your machine to be on

### Option 3: WATGPU Cluster (Recommended)

Best for: University users, unlimited runtime, shared resources

```bash
# SSH to WATGPU
ssh username@watgpu.cs.uwaterloo.ca

# Submit training job (7 hours)
sbatch train_overnight_watgpu.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-*.out
```

**Pros**: Professional infrastructure, scheduled jobs, high-performance GPUs, email notifications
**Cons**: Requires University account, SLURM learning curve

## Training Performance Comparison

| Method | Stockfish Depth | Games/Night | Total Resources | Best For |
|--------|-----------------|-------------|-----------------|----------|
| Local (MacBook M1) | 12 | 20-30 | Medium | Testing |
| Local (Gaming PC) | 15 | 40-60 | High | Personal training |
| WATGPU (1 GPU) | 16 | 40-60 | Shared | Serious training |
| WATGPU (Multiple jobs) | 16 | 100-200+ | Shared | Maximum learning |

## File Structure

```
ChessHacks-CLI/
├── src/
│   ├── train_overnight.py           # Main training script
│   ├── chess_engine.py              # MCTS + Neural network
│   ├── MCTSAlgorithm.py
│   ├── network_architecture.py
│   ├── models/
│   │   └── trained_model.pth        # Trained weights (updates nightly)
│   └── ...
├── OVERNIGHT_TRAINING_GUIDE.md      # Local machine guide
├── WATGPU_SETUP_GUIDE.md            # WATGPU initial setup
├── WATGPU_TRAINING_GUIDE.md         # WATGPU training reference
├── train_overnight_watgpu.sh        # SLURM job script (7 hours)
├── train_test_watgpu.sh             # SLURM test script (20 min)
├── run_overnight_training.sh        # Local launcher (macOS/Linux)
├── monitor_training.sh              # Local monitoring
├── quick_test.sh                    # Local quick test (5 min)
├── com.chesshacks.overnight-training.plist  # macOS scheduling
├── logs/                            # Training logs (auto-created)
├── checkpoints/                     # Model checkpoints (auto-created)
└── requirements.txt
```

## Model Training Flow

```
┌─────────────────────────────┐
│ Start Training              │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 1. Play Game vs Stockfish   │
│    - Model (MCTS) vs SF     │
│    - Collect move data      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 2. Store Training Data      │
│    - Board states           │
│    - MCTS policies          │
│    - Game result            │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. Train Neural Network     │
│    - Supervised learning    │
│    - Policy loss            │
│    - Value loss             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. Update Weights           │
│    - Backpropagation        │
│    - Gradient descent       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 5. Save Checkpoint (every   │
│    50 games + final)        │
└──────────────┬──────────────┘
               │
               ▼
      ┌────────┴────────┐
      │ Continue?       │
      └────────┬────────┘
               │
         ┌─────┴─────┐
         │           │
      YES│           │NO
         │           │
         ▼           ▼
    [Loop]      [Finish]
```

## Command Quick Reference

### Local Machine

```bash
# Test (5 minutes)
./quick_test.sh

# Full training (8 hours)
./run_overnight_training.sh

# Monitor
./monitor_training.sh

# Custom parameters
python -m src.train_overnight \
    --hours 8 \
    --stockfish-depth 15 \
    --batch-size 256 \
    --learning-rate 0.0005
```

### WATGPU Cluster

```bash
# SSH to cluster
ssh username@watgpu.cs.uwaterloo.ca

# Test (20 minutes)
sbatch train_test_watgpu.sh

# Full training (7 hours)
sbatch train_overnight_watgpu.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-*.out

# Check resources
sresources

# Cancel job
scancel <jobid>
```

## Training Parameters Explained

### Stockfish Depth
- **Lower (10-12)**: Fast games (~2-3 min), weaker opponent, ~15-30 games/hour
- **Balanced (14-16)**: ~5-10 min/game, ~6-12 games/hour ⭐ **Recommended**
- **Higher (18-20)**: Slow games (~15-30 min), strong opponent, ~2-4 games/hour

### Batch Size
- **64**: Low memory, slower convergence
- **128**: Balanced, slower training
- **256**: Recommended, good learning
- **512**: Fast training, needs high memory

### Learning Rate
- **0.0001**: Conservative, slow learning
- **0.0005**: Balanced ⭐ **Default**
- **0.001**: Aggressive, may oscillate
- **Higher**: Risk of divergence

### Duration
- **Local**: 8 hours typical (before sleep)
- **WATGPU**: 7 hours typical (cluster max for single job)
- Chain multiple jobs for continuous training

## Monitoring Progress

### Check Win Rate
```bash
# View logs
tail logs/training_*.log

# Look for:
# Games: 50 | Win Rate: 35.0% | W:18 B:19 D:13
```

Expected win rates:
- Depth 12 Stockfish: 40-50% (your model should be competitive)
- Depth 16 Stockfish: 20-35% (strong opponent)
- Depth 20 Stockfish: 10-20% (very strong opponent)

### Check Model Improvement
```bash
# Compare checkpoints
ls -lh checkpoints/

# Size indicates training time
# Weights update with each training step
```

### View Training Losses
```bash
# Grep for loss values
grep "Loss:" logs/training_*.log

# Should decrease over time (lower is better)
```

## Troubleshooting Matrix

| Problem | Local | WATGPU |
|---------|-------|--------|
| Stockfish not found | `brew install stockfish` | `conda install -c conda-forge stockfish` |
| Out of memory | Reduce `--batch-size` | `sbatch --mem=16GB` |
| GPU memory full | Reduce batch size | Reduce batch size or use different GPU |
| Very slow | Reduce `--stockfish-depth` | Reduce depth or increase CPUs |
| Job keeps pending | N/A | Try `sbatch --qos=low` |
| Can't SSH to WATGPU | N/A | Load SSH key at authman.uwaterloo.ca |

## Best Practices

### 1. Always Test First
```bash
# Local: ./quick_test.sh
# WATGPU: sbatch train_test_watgpu.sh
```

### 2. Start Conservative
```bash
# Lower depth = faster iteration for tuning
--stockfish-depth 12
```

### 3. Monitor Regularly
```bash
# Check logs frequently
# Look for errors, losses, win rates
```

### 4. Save Backups
```bash
# Before major changes
cp src/models/trained_model.pth src/models/trained_model_backup_$(date +%Y%m%d).pth
```

### 5. Chain Jobs for Continuity
```bash
# Multiple jobs ensure continuous learning
sbatch train_overnight_watgpu.sh  # Job 1 (7 hours)
# After completion:
sbatch train_overnight_watgpu.sh  # Job 2 (7 hours)
# Model improves progressively
```

## Expected Results After One Week

- **20-30 games total** from local machine
- **200-300 games total** from WATGPU (1 job/night)
- **Model improvement**: Measurable on same Stockfish depth
- **Win rate**: May stabilize or slowly increase
- **Loss values**: Should decrease overall trend

## Advanced: Custom Training

### Multiple Stockfish Depths
Create different scripts for variety:
```bash
# train_fast.sh: depth 12 (many games)
# train_balanced.sh: depth 16 (good mix)
# train_strong.sh: depth 20 (strong opponent)

# Run rotation for diverse training
sbatch train_fast.sh && sbatch train_balanced.sh && sbatch train_strong.sh
```

### Adaptive Learning
Adjust learning rate based on progress:
```bash
# Early training: higher LR (0.001)
# After 100 games: medium LR (0.0005)
# After 500 games: lower LR (0.0001)
```

## Support & Documentation

### ChessHacks-Specific
- Local guide: [OVERNIGHT_TRAINING_GUIDE.md](OVERNIGHT_TRAINING_GUIDE.md)
- WATGPU setup: [WATGPU_SETUP_GUIDE.md](WATGPU_SETUP_GUIDE.md)
- WATGPU training: [WATGPU_TRAINING_GUIDE.md](WATGPU_TRAINING_GUIDE.md)

### External Resources
- WATGPU docs: https://watgpu.cs.uwaterloo.ca/
- SLURM docs: https://slurm.schedmd.com/pdfs/summary.pdf
- Stockfish: https://www.stockfishchess.org/
- PyTorch: https://pytorch.org/docs/stable/index.html

## Summary: Getting Started

### For Local Machine
```bash
1. cd ChessHacks-CLI
2. source newenv/bin/activate
3. ./quick_test.sh              # Test first
4. ./run_overnight_training.sh  # Run training
```

### For WATGPU
```bash
1. ssh username@watgpu.cs.uwaterloo.ca
2. cd ~/ChessHacks-CLI
3. sbatch train_test_watgpu.sh         # Test first
4. sbatch train_overnight_watgpu.sh    # Run training
```

---

**Last updated**: November 2025
**Status**: Ready for continuous training

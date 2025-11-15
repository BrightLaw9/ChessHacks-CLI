# ChessHacks Overnight Training - Setup Summary

## ✅ What Has Been Created

Your ChessHacks project now has **complete overnight training infrastructure** with full WATGPU SLURM support!

### 🐍 Python Scripts

1. **`src/train_overnight.py`** (Main Training Script)
   - Plays games against Stockfish
   - Collects training data from MCTS decisions
   - Continuously updates model weights
   - Saves checkpoints every 50 games
   - Graceful shutdown with signal handling
   - Detailed logging with statistics
   - **Features**: Replay buffer, batch training, loss tracking, model persistence

### 🚀 WATGPU/SLURM Scripts

2. **`train_overnight_watgpu.sh`** (Primary Training Job)
   - 7-hour SLURM job (max for watgpu)
   - 1 GPU, 32GB RAM, 8 CPU cores
   - Stockfish depth 16 (balanced)
   - Expected: 40-60 games per run
   - Email notifications on start/end/fail
   - **Submit**: `sbatch train_overnight_watgpu.sh`

3. **`train_test_watgpu.sh`** (Quick Test Job)
   - 20-minute SLURM test
   - 1 GPU, 16GB RAM, 4 CPU cores
   - Stockfish depth 12 (fast)
   - Validates setup before full training
   - **Submit**: `sbatch train_test_watgpu.sh`

### 📚 Documentation

4. **`WATGPU_SETUP_GUIDE.md`** (New!)
   - Step-by-step initial setup
   - SSH key configuration
   - Conda environment creation
   - Interactive GPU testing
   - Troubleshooting common issues
   - File transfer instructions

5. **`WATGPU_TRAINING_GUIDE.md`** (New!)
   - Complete SLURM command reference
   - Job submission and monitoring
   - Resource allocation strategies
   - GPU management
   - Chaining jobs for continuous training
   - Email notifications setup
   - Performance tuning by hardware

6. **`OVERNIGHT_TRAINING_GUIDE.md`** (Updated)
   - Local machine training (macOS/Linux)
   - Automated scheduling via launchd/cron
   - Monitoring and troubleshooting
   - Now includes reference to WATGPU guide

7. **`TRAINING_COMPLETE_GUIDE.md`** (New!)
   - Overview of all three training methods
   - Quick navigation to all guides
   - Command reference for both local and WATGPU
   - Performance comparison table
   - Training flow diagrams
   - Advanced techniques and best practices

### 🔧 Local Machine Scripts (for reference)

8. **`run_overnight_training.sh`** - Launcher for local machine
9. **`monitor_training.sh`** - Log monitoring tool
10. **`quick_test.sh`** - 5-minute local test
11. **`com.chesshacks.overnight-training.plist`** - macOS scheduling

### 📁 Auto-Generated Directories

When training runs, these will be created:
- `logs/` - Detailed training logs
- `checkpoints/` - Model checkpoints (every 50 games)

## 🚀 Quick Start

### For WATGPU (Recommended)

```bash
# 1. SSH to WATGPU
ssh your_username@watgpu.cs.uwaterloo.ca

# 2. Follow WATGPU_SETUP_GUIDE.md (one-time setup)
cd ~/ChessHacks-CLI
conda create -n chesshacks python=3.10
conda activate chesshacks
pip install -r requirements.txt

# 3. Run quick test (20 minutes)
sbatch train_test_watgpu.sh

# 4. Monitor
squeue -u $USER
tail -f logs/slurm-*.out

# 5. Submit full training (7 hours)
sbatch train_overnight_watgpu.sh

# 6. Check results
tail -100 logs/slurm-*.out
ls -lh checkpoints/
```

### For Local Machine

```bash
# 1. Activate environment
source newenv/bin/activate

# 2. Run quick test
./quick_test.sh

# 3. Run full training
./run_overnight_training.sh --hours 8 --depth 15

# 4. Monitor
./monitor_training.sh
```

## 📊 Training Comparison

| Method | Duration | Games/Run | GPU | Resources | When |
|--------|----------|-----------|-----|-----------|------|
| Local Test | 5 min | ~3 | None | Your machine | Testing |
| WATGPU Test | 20 min | ~10 | 1 | Shared cluster | Setup validation |
| WATGPU Full | 7 hours | 40-60 | 1 | Shared cluster | **Recommended** |
| Local Full | 8 hours | 20-30 | Optional | Your machine | Personal use |

## 🎯 What Happens During Training

1. **Game Play**: Model (MCTS) plays vs Stockfish
2. **Data Collection**: Positions and move decisions stored
3. **Training**: Neural network learns from collected data
4. **Weight Update**: Model improves iteratively
5. **Checkpointing**: Progress saved every 50 games
6. **Logging**: Detailed stats on wins, losses, learning curves

## 📈 Expected Progress

### After 1 Week (WATGPU, 7 hrs/night):
- ~200-300 games
- Model improving measurably
- Clear win rate trend

### After 1 Month:
- ~1000 games
- Strong measurable improvement
- Model may beat Stockfish at shallow depths

## 🔗 Which Guide to Follow?

- **First time on WATGPU?** → [WATGPU_SETUP_GUIDE.md](WATGPU_SETUP_GUIDE.md)
- **Running training on WATGPU?** → [WATGPU_TRAINING_GUIDE.md](WATGPU_TRAINING_GUIDE.md)
- **Running on local machine?** → [OVERNIGHT_TRAINING_GUIDE.md](OVERNIGHT_TRAINING_GUIDE.md)
- **Want overview of all methods?** → [TRAINING_COMPLETE_GUIDE.md](TRAINING_COMPLETE_GUIDE.md)

## 🛠️ Key Features

✅ **Automatic checkpointing** - Save model every 50 games
✅ **Training statistics** - Track wins, losses, learning curves
✅ **Graceful shutdown** - Save work on Ctrl+C
✅ **Flexible duration** - Train for hours or specific number of games
✅ **SLURM integration** - Job scheduling, email notifications
✅ **Continuous learning** - Chain multiple jobs for non-stop training
✅ **Easy monitoring** - Real-time log viewing and status checks
✅ **GPU accelerated** - PyTorch with CUDA support

## 🔧 Customization

All scripts support parameters:

```bash
# WATGPU
sbatch --mem=64GB --time=08:00:00 train_overnight_watgpu.sh

# Local
python -m src.train_overnight --hours 8 --batch-size 512 --learning-rate 0.0001
```

Edit scripts directly to change defaults:
- Stockfish depth
- Training duration
- Batch size
- Learning rate
- GPU allocation
- Memory allocation

## 🚨 Troubleshooting

**WATGPU SSH issues?**
- Load SSH key: https://authman.uwaterloo.ca/

**Stockfish not found?**
- WATGPU: `conda install -c conda-forge stockfish`
- Local: `brew install stockfish`

**Out of memory?**
- Reduce batch size: `--batch-size 64`
- Request more memory: `sbatch --mem=64GB`

**Job stays pending?**
- Try: `sbatch --qos=low train_overnight_watgpu.sh`

See full troubleshooting in the relevant guide!

## 📞 Support

- **WATGPU Issues**: watgpu-admin@lists.uwaterloo.ca
- **PyTorch Issues**: https://pytorch.org/docs
- **SLURM Issues**: https://slurm.schedmd.com/pdfs/summary.pdf

## 🎓 Next Steps

1. **Choose your platform**: WATGPU (cluster) or local machine
2. **Read the appropriate guide**: Setup or Training
3. **Run the test script first**: Validate your setup
4. **Submit the training job**: Watch logs and monitor
5. **Check results**: View logs and checkpoints
6. **Iterate**: Adjust parameters based on results

---

**Everything is ready! 🚀**

Start with:
- **WATGPU users**: `WATGPU_SETUP_GUIDE.md`
- **Local users**: `OVERNIGHT_TRAINING_GUIDE.md`

# ChessHacks Overnight Training on WATGPU

This guide explains how to train your chess model continuously on the WATGPU cluster using SLURM.

## Quick Start

### 1. SSH to WATGPU

```bash
ssh your_username@watgpu.cs.uwaterloo.ca
```

### 2. Setup (One-time)

```bash
# Clone/navigate to your repo
cd ~/ChessHacks-CLI

# Create conda environment
conda create -n chesshacks python=3.10
conda activate chesshacks

# Install dependencies
pip install -r requirements.txt

# Install Stockfish (if not globally available)
# On WATGPU compute nodes, stockfish may already be available
# Check with: stockfish --version
```

### 3. Submit Training Job

```bash
# Submit the training job (7 hours)
sbatch train_overnight_watgpu.sh

# You'll get output like:
# Submitted batch job 12345

# Check status
squeue -u $USER

# View logs (while running)
tail -f logs/slurm-12345.out
```

### 4. Monitor Progress

```bash
# View your jobs
squeue -u $USER

# View detailed job info
scontrol show job -dd 12345

# View cluster resources
sresources

# Check job output
cat logs/slurm-12345.out
cat logs/slurm-12345-err.out
```

## SLURM Job Scripts Provided

### `train_overnight_watgpu.sh` (Recommended)
- **Duration**: 7 hours (max for WATGPU)
- **GPU**: 1 GPU
- **Memory**: 32 GB
- **CPUs**: 8 cores
- **Stockfish Depth**: 16 (balanced)
- **Expected**: ~40-60 games

```bash
sbatch train_overnight_watgpu.sh
```

### `train_test_watgpu.sh` (Testing)
- **Duration**: 20 minutes
- **GPU**: 1 GPU
- **Memory**: 16 GB
- **CPUs**: 4 cores
- **Stockfish Depth**: 12 (fast)
- **Use**: To test setup before full training

```bash
sbatch train_test_watgpu.sh
```

## Customizing Job Scripts

Edit `train_overnight_watgpu.sh` to adjust parameters:

```bash
# Email notification
#SBATCH --mail-user=your.email@uwaterloo.ca

# Job duration (HH:MM:SS)
#SBATCH --time=07:00:00

# Memory allocation
#SBATCH --mem=32GB

# CPU cores
#SBATCH --cpus-per-task=8

# GPU count (1-4 typically)
#SBATCH --gres=gpu:1

# Output files
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j-err.out
```

### Adjust Training Parameters in Script

```bash
# Inside train_overnight_watgpu.sh, modify:
STOCKFISH_DEPTH=16           # 12-18 typical range
NUM_HOURS=7                  # Must be <= SLURM time
BATCH_SIZE=256               # 128, 256, or 512
LEARNING_RATE=0.0005         # 0.0001 to 0.001
```

## SLURM Command Reference

### Submit Jobs

```bash
# Submit regular job
sbatch train_overnight_watgpu.sh

# Submit with custom parameters (override script defaults)
sbatch --time=04:00:00 --mem=48GB train_overnight_watgpu.sh

# Submit low-priority job (gets more resources when available)
sbatch --qos=low train_overnight_watgpu.sh
```

### Monitor Jobs

```bash
# View your jobs
squeue -u $USER

# View only running jobs
squeue -u $USER -t RUNNING

# View only pending jobs
squeue -u $USER -t PENDING

# View detailed info for job 12345
scontrol show job -dd 12345

# View cluster status
sresources

# Watch queue (updates every 2 seconds)
watch 'squeue -u $USER'
```

### Manage Jobs

```bash
# Cancel a job
scancel 12345

# Cancel all your pending jobs
scancel -t PENDING -u $USER

# Cancel all your jobs
scancel -u $USER

# Requeue a job
scontrol requeue 12345
```

### Check Logs

```bash
# View stdout log
cat logs/slurm-12345.out

# View stderr log
cat logs/slurm-12345-err.out

# Real-time output (while running)
tail -f logs/slurm-12345.out

# View last 100 lines
tail -100 logs/slurm-12345.out

# View entire log
less logs/slurm-12345.out
```

## GPU and Resource Management

### Check Available Resources

```bash
# View all nodes and GPU availability
sresources

# Example output:
# watgpu[301-308]:
#   Nodes: 8
#   GPUs: 2 per node
#   Mem: 128GB per node
```

### Request Specific Resources

```bash
# For GPU-heavy training (multiple GPUs)
sbatch --gres=gpu:2 --mem=64GB train_overnight_watgpu.sh

# For very long training (72 hours max in batch mode)
sbatch --time=72:00:00 train_overnight_watgpu.sh

# For CPU-intensive preprocessing
sbatch --cpus-per-task=16 --mem=48GB train_overnight_watgpu.sh
```

### Monitor GPU Usage

```bash
# Interactive session to check GPU
salloc --gres=gpu:1 --mem=16GB --time=01:00:00
nvidia-smi
exit
```

## Common WATGPU Issues and Solutions

### Issue: "Conda command not found"

**Solution:**
```bash
# Load conda module
module load conda

# Or activate from conda path directly
source /opt/conda/etc/profile.d/conda.sh
```

### Issue: "Stockfish not found"

**Solution:**
```bash
# Install Stockfish in conda environment
conda install -c conda-forge stockfish

# Or add to your script
module load stockfish
```

### Issue: "Job stays in PENDING state"

**Reasons and solutions:**
```bash
# Check resource availability
sresources

# Try requesting fewer resources
sbatch --gres=gpu:1 --mem=16GB train_overnight_watgpu.sh

# Try different resource ratio
sbatch --mem=64GB --cpus-per-task=16 train_overnight_watgpu.sh

# Use lower QoS priority (runs faster when resources available)
sbatch --qos=low train_overnight_watgpu.sh
```

### Issue: "Out of memory" or "GPU out of memory"

**Solution:**
```bash
# Reduce batch size and memory allocation
sbatch --mem=16GB train_overnight_watgpu.sh

# Edit train_overnight_watgpu.sh and change:
BATCH_SIZE=64
# Or run directly with:
python -m src.train_overnight --batch-size 64 --hours 7
```

### Issue: "Job timed out"

**Solution:**
```bash
# Increase SLURM time limit (max 7 days = 168 hours for batch)
sbatch --time=24:00:00 train_overnight_watgpu.sh

# Or submit multiple jobs sequentially
# Job 2 will use the checkpoint from Job 1
sbatch train_overnight_watgpu.sh
# (wait for job 1 to complete)
sbatch train_overnight_watgpu.sh
```

## Advanced: Chaining Jobs

Run training continuously by chaining jobs:

```bash
# Create submission script
cat > run_training_chain.sh << 'EOF'
#!/bin/bash

for i in {1..30}; do
    echo "Submitting job iteration $i"
    JOB_ID=$(sbatch train_overnight_watgpu.sh | awk '{print $4}')
    echo "Job ID: $JOB_ID"
    
    # Submit next job to depend on current job
    sbatch --dependency=afterok:$JOB_ID train_overnight_watgpu.sh
done
EOF

chmod +x run_training_chain.sh
./run_training_chain.sh
```

This creates a queue of 30 jobs that run sequentially. Your model will continuously train!

## Advanced: Different Stockfish Depths

Create multiple scripts for different training intensities:

```bash
# Fast training (many games, weaker opponent)
# train_fast_watgpu.sh - depth 12, 8 hours

# Balanced training (moderate games/strength)
# train_overnight_watgpu.sh - depth 16, 7 hours

# Strong training (few games, strong opponent)
# train_strong_watgpu.sh - depth 20, 7 hours

# Submit in sequence
sbatch train_fast_watgpu.sh
sbatch train_strong_watgpu.sh
```

## Email Notifications

Set up email alerts for job completion:

1. Edit `train_overnight_watgpu.sh`:
```bash
#SBATCH --mail-user=your.email@uwaterloo.ca
#SBATCH --mail-type=ALL
```

2. Notification types:
- `BEGIN` - when job starts
- `END` - when job completes
- `FAIL` - when job fails
- `REQUEUE` - when job is requeued
- `ALL` - all of the above

## Best Practices

### 1. Test Before Full Submission

```bash
# Always run test first
sbatch train_test_watgpu.sh

# Check output
tail -f logs/slurm-test-*.out

# Then submit full training
sbatch train_overnight_watgpu.sh
```

### 2. Monitor First Job

```bash
# After submitting, watch the output
squeue -u $USER
tail -f logs/slurm-12345.out

# Cancel if issues arise
scancel 12345
```

### 3. Use Appropriate Depth

| Depth | Time/Game | Games/Hour | Use Case |
|-------|-----------|-----------|----------|
| 10-12 | 2-3 min   | 15-30     | Testing, fast iteration |
| 14-16 | 5-10 min  | 6-12      | Balanced (recommended) |
| 18-20 | 15-30 min | 2-4       | Strong training, fewer games |

### 4. Resource Allocation

For 7-hour sessions:

| GPU | Memory | CPUs | Depth | Expected Games |
|-----|--------|------|-------|-----------------|
| 1   | 16GB   | 4    | 12    | ~100-120        |
| 1   | 32GB   | 8    | 16    | ~40-60          |
| 1   | 48GB   | 12   | 18    | ~20-30          |
| 2   | 64GB   | 16   | 16    | ~80-120         |

### 5. Regular Backups

```bash
# Backup your model before major changes
cp src/models/trained_model.pth src/models/trained_model_backup_$(date +%Y%m%d).pth

# Check checkpoint directory
ls -lh checkpoints/
```

## Troubleshooting Checklist

- [ ] Can SSH to watgpu.cs
- [ ] Conda environment is set up and activated
- [ ] Dependencies installed (`pip list | grep -i torch`)
- [ ] Stockfish is available (`which stockfish`)
- [ ] Test job completes successfully
- [ ] Check output files for errors
- [ ] GPU memory is sufficient
- [ ] Logs are being written

## Next Steps

1. **Test**: Run `sbatch train_test_watgpu.sh` first
2. **Monitor**: Check `squeue -u $USER` frequently
3. **Review**: Check logs and checkpoint directory
4. **Train**: Submit `sbatch train_overnight_watgpu.sh`
5. **Chain**: Create multiple jobs for continuous training

## WATGPU Documentation

- Official docs: https://watgpu.cs.uwaterloo.ca/
- SLURM docs: https://watgpu.cs.uwaterloo.ca/slurm.html
- SLURM cheat sheet: https://slurm.schedmd.com/pdfs/summary.pdf

## Support

For WATGPU-specific issues: watgpu-admin@lists.uwaterloo.ca
For ChessHacks issues: Check the training logs and `OVERNIGHT_TRAINING_GUIDE.md`

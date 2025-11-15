# WATGPU Setup Guide

This guide covers initial setup on the WATGPU cluster for running ChessHacks training.

## Step 1: Get Access

1. Ensure you have an account on WATGPU
   - Faculty/Students: Contact watgpu-admin@lists.uwaterloo.ca
   - Load your SSH key at https://authman.uwaterloo.ca/

2. Add SSH key (if not already done):
   ```bash
   # Generate key (if you don't have one)
   ssh-keygen -t ed25519 -C "your_email@uwaterloo.ca"
   
   # Load at https://authman.uwaterloo.ca/
   cat ~/.ssh/id_ed25519.pub
   ```

## Step 2: Initial SSH and Setup

```bash
# SSH into WATGPU (from your local machine)
ssh your_username@watgpu.cs.uwaterloo.ca

# Note: Login node is for job submission only, not heavy computation
# Compute happens on watgpu[301-308] via SLURM
```

## Step 3: Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone your ChessHacks repository
git clone https://github.com/BrightLaw9/ChessHacks-CLI.git
cd ChessHacks-CLI

# Or if you have SSH keys set up for GitHub
git clone git@github.com:BrightLaw9/ChessHacks-CLI.git
cd ChessHacks-CLI
```

## Step 4: Create Conda Environment

```bash
# Ensure conda is available
module load conda
# OR
source /opt/conda/etc/profile.d/conda.sh

# Create environment (one time)
conda create -n chesshacks python=3.10 -y

# Activate it
conda activate chesshacks

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import chess; print('python-chess: OK')"
```

## Step 5: Check for Stockfish

```bash
# Check if stockfish is available system-wide
which stockfish
stockfish --version

# If not available, install via conda
conda install -c conda-forge stockfish -y

# Verify
which stockfish
```

## Step 6: Test Interactive GPU Access

```bash
# Request interactive resources (testing)
salloc --gres=gpu:1 --mem=16GB --cpus-per-task=4 --time=01:00:00

# Wait for allocation, then you'll be on a compute node:
# salloc: Nodes watgpu308 are ready for job

# Test GPU
nvidia-smi

# Test CUDA with PyTorch
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Exit when done
exit
```

## Step 7: Run Test Training Job

```bash
# Go back to your repository directory
cd ~/ChessHacks-CLI

# Make scripts executable
chmod +x train_test_watgpu.sh train_overnight_watgpu.sh

# Edit train_test_watgpu.sh and update the conda environment activation line
# Change: conda activate chesshacks (if needed)

# Submit test job (20 minutes)
sbatch train_test_watgpu.sh

# You'll see: Submitted batch job XXXXX

# Check status
squeue -u $USER

# Monitor output
tail -f logs/slurm-*.out
```

## Step 8: Configure Training Script

Edit `train_overnight_watgpu.sh` to customize:

```bash
# Email address for notifications
#SBATCH --mail-user=your.email@uwaterloo.ca

# Update conda environment name if different
conda activate chesshacks

# Adjust these parameters based on your resources
STOCKFISH_DEPTH=16           # 12=fast, 16=balanced, 20=strong
NUM_HOURS=7                  # 7 hours (max typical)
BATCH_SIZE=256               # 128, 256, or 512
LEARNING_RATE=0.0005         # 0.0001 to 0.001
```

## Step 9: Submit Full Training Job

```bash
# Submit 7-hour training job
sbatch train_overnight_watgpu.sh

# Monitor multiple ways
squeue -u $USER                    # Quick status
scontrol show job -dd <jobid>      # Detailed info
watch 'squeue -u $USER'            # Live updates

# Check logs
tail -100 logs/slurm-*.out
tail -100 logs/slurm-*-err.out
```

## Step 10: Set Up Email Notifications

Edit `train_overnight_watgpu.sh`:

```bash
#SBATCH --mail-user=your.email@uwaterloo.ca
#SBATCH --mail-type=ALL
```

You'll receive emails when:
- Job starts (BEGIN)
- Job completes (END)
- Job fails (FAIL)

## Quick Reference Commands

```bash
# From your local machine
ssh your_username@watgpu.cs.uwaterloo.ca

# Once logged in
module load conda                          # Load conda
source /opt/conda/etc/profile.d/conda.sh  # Activate conda
conda activate chesshacks                  # Activate environment
cd ~/ChessHacks-CLI                        # Go to repo

squeue -u $USER                            # View your jobs
sresources                                 # View cluster resources
sbatch train_overnight_watgpu.sh           # Submit job
scancel <jobid>                            # Cancel job
tail -f logs/slurm-*.out                   # Watch logs
```

## Directory Structure on WATGPU

```
~/ChessHacks-CLI/
├── src/
│   ├── train_overnight.py          # Main training script
│   ├── models/
│   │   └── trained_model.pth       # Your trained model
│   ├── chess_engine.py
│   ├── network_architecture.py
│   └── ...
├── train_overnight_watgpu.sh       # SLURM script (7 hours)
├── train_test_watgpu.sh            # SLURM test script (20 min)
├── logs/                           # Training logs (created automatically)
├── checkpoints/                    # Model checkpoints (created automatically)
├── requirements.txt
└── WATGPU_TRAINING_GUIDE.md        # This file
```

## Troubleshooting Setup

### Issue: Cannot SSH to WATGPU

```bash
# Check SSH key is loaded at https://authman.uwaterloo.ca/
# Try verbose SSH:
ssh -vvv your_username@watgpu.cs.uwaterloo.ca

# If it's a key permission issue:
chmod 600 ~/.ssh/id_*
chmod 700 ~/.ssh
```

### Issue: Conda not found

```bash
# Load conda module
module load conda

# Or check available modules
module avail conda

# If available, use full path
source /opt/conda/etc/profile.d/conda.sh
```

### Issue: PyTorch doesn't see GPU

```bash
# After salloc, check nvidia-smi first
nvidia-smi

# Then check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, try different CUDA version in pip install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory errors

```bash
# Reduce batch size in training script
BATCH_SIZE=64  # Instead of 256

# Or request more memory when submitting
sbatch --mem=48GB train_overnight_watgpu.sh
```

## File Transfer

### Upload files to WATGPU

```bash
# From your local machine
scp -r ~/path/to/local/file your_username@watgpu.cs.uwaterloo.ca:~/ChessHacks-CLI/
```

### Download checkpoints from WATGPU

```bash
# From your local machine
scp -r your_username@watgpu.cs.uwaterloo.ca:~/ChessHacks-CLI/checkpoints/ ~/local/path/
```

## WATGPU Resources

- **Main Documentation**: https://watgpu.cs.uwaterloo.ca/
- **SLURM Guide**: https://watgpu.cs.uwaterloo.ca/slurm.html
- **Cluster Status**: https://watgpu.cs.uwaterloo.ca/current_state.html
- **Support**: watgpu-admin@lists.uwaterloo.ca

## Next Steps

1. ✓ Set up SSH access
2. ✓ Clone repository
3. ✓ Create conda environment
4. ✓ Test GPU access
5. ✓ Run test training job
6. → Submit full 7-hour training job
7. → Monitor via SLURM commands
8. → Download checkpoints when complete

See [WATGPU_TRAINING_GUIDE.md](WATGPU_TRAINING_GUIDE.md) for detailed training instructions.

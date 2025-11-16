# modal_app.py
import modal
import os

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

GPU_TYPE = "A100"   # or "H100"
NUM_GPUS = 8        # change to 1, 4, 8, 16
PGN_VOLUME = modal.Volume.from_name("chess-datasets", create_if_missing=False)
CODE_VOLUME = modal.Volume.from_name("chess-code")  
MODEL_VOLUME = modal.Volume.from_name("chess-checkpoints", create_if_missing=True)

BATCH_SIZE = 512
EPOCHS = 1

POLICY_WEIGHT = 0.7
VALUE_WEIGHT = 0.3

IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git", "python3", "python3-pip")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "python-chess",
        "zstandard"
    )
)

app = modal.App("chess-ddp-training")

# ----------------------------------------------------------
# TRAIN FUNCTION
# ----------------------------------------------------------

@app.function(
    image=IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    volumes={
        "/app": CODE_VOLUME,   # mount your code here
        "/pgns": PGN_VOLUME,
        "/models": MODEL_VOLUME,
    },
    timeout=60 * 60 * 8,  # 8 hours (adjust as needed)
)
def train():
    """
    Runs multi-GPU DDP training inside one Modal container.
    """

    import subprocess
    import sys
    import os

    os.chdir("/app")  # ensure working directory
    print("Files on /app:", os.listdir("/app"))

    # Path inside container to PGNs
    pgn_path = "/pgns"

    # Path for checkpoints
    model_path_prefix = "/models/checkpoint_model"

    # Command to launch mp.spawn training
    cmd = [
        sys.executable,
        "-m",
        "app.train_ddp_stream",
        "--pgn-folder", pgn_path,
        "--world-size", str(NUM_GPUS),
        "--num-workers", "4",
        "--batch-size", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--model-path", model_path_prefix,
        "--min-elo", "1800",
        "--skip-early-moves", "5",
        "--policy-weight", str(POLICY_WEIGHT),
        "--value-weight", str(VALUE_WEIGHT),
    ]

    print("Launching training with command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    print("Training completed!")


# ----------------------------------------------------------
# ENTRYPOINT (modal deploy)
# ----------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Starting distributed training on Modal...")
    train.remote()

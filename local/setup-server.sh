#!/usr/bin/env bash
set -euo pipefail

# -------------- User configuration --------------
# Name for conda environment
ENV_NAME="lerobot-server"
# Hugging Face model repo ID or path
HF_MODEL="sengi/pi0fast-so100"
# Server binding settings
SERVER_HOST="0.0.0.0"
SERVER_PORT=5555
# -----------------------------------------------

# 1. Install Miniconda if missing
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
  echo "Installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
  bash miniconda.sh -b -p "$CONDA_DIR"
  rm miniconda.sh
fi
# Initialize conda
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 2. Create or ensure conda env exists
if ! conda env list | grep -q "^$ENV_NAME"; then
  echo "Creating conda environment '$ENV_NAME'..."
  if command -v mamba &>/dev/null; then
    mamba create -y -n "$ENV_NAME" python=3.12
  else
    conda create -y -n "$ENV_NAME" python=3.12
  fi
fi
conda activate "$ENV_NAME"
conda install ffmpeg -c conda-forge

# 3. Running within existing repository directory
echo "Using repository at $(pwd)"

# 4. Install project & dependencies
echo "Installing project dependencies..."
pip install --upgrade pip
pip install -e .[all] pyzmq

# 5. Download model from Hugging Face
echo "Downloading model '$HF_MODEL' into ./models..."
mkdir -p models
python - <<EOF
from huggingface_hub import snapshot_download
# save model files into local "models" folder
snapshot_download(repo_id="$HF_MODEL", cache_dir="models")
EOF

# 6. Launch server
echo "Starting remote inference server on port $SERVER_PORT..."
nohup python lerobot/scripts/remote_server.py \
  --policy_path "$HF_MODEL" \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  > server.log 2>&1 &

echo "Server started: logs at $(pwd)/server.log"

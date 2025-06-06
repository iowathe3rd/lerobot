#!/usr/bin/env bash
set -euo pipefail

# -------------- User configuration --------------
ENV_NAME="lerobot-server"           # conda environment name
HF_MODEL="sengi/pi0fast-so100"     # HF Hub model ID or local path
SERVER_HOST="0.0.0.0"              # server bind address
SERVER_PORT=5555                     # server port
# -----------------------------------------------

# 1. Install Miniconda if missing
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
  echo "Installing Miniconda..."
  # Download installer and checksum
  INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
  BASE_URL="https://repo.anaconda.com/miniconda"
  wget "$BASE_URL/$INSTALLER" -O "$INSTALLER"
  wget "$BASE_URL/$INSTALLER.sha256" -O "$INSTALLER.sha256"
  # Verify SHA256 checksum
  echo "Verifying installer integrity..."
  sha256sum -c "$INSTALLER.sha256"
  # Run silent install
  bash "$INSTALLER" -b -p "$CONDA_DIR"
  rm "$INSTALLER" "$INSTALLER.sha256"
fi

# 2. Initialize conda
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"

# 3. Create or activate conda env
if ! conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "Creating conda environment '$ENV_NAME' with Python 3.12..."
  if command -v mamba &>/dev/null; then
    mamba create -y -n "$ENV_NAME" python=3.12
  else
    conda create -y -n "$ENV_NAME" python=3.12
  fi
fi
conda activate "$ENV_NAME"
echo "Activated Python $(python --version)"

# 4. Install ffmpeg for video support
echo "Installing ffmpeg..."
conda install -y ffmpeg -c conda-forge

# 5. Install project dependencies
echo "Installing project dependencies..."
python -m pip install --upgrade pip
python -m pip install -e .[all] pyzmq

# 6. Download model into local 'models/' folder
echo "Downloading model '$HF_MODEL' into ./models..."
mkdir -p models
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$HF_MODEL", cache_dir="models")
EOF

# 7. Launch remote inference server
echo "Starting remote inference server at $SERVER_HOST:$SERVER_PORT..."
nohup python lerobot/scripts/remote_server.py \
    --policy_path "$HF_MODEL" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    > server.log 2>&1 &

echo "Server started. Logs: server.log"
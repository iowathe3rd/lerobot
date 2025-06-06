#!/usr/bin/env bash
set -euo pipefail

# -------------- User configuration --------------
VENV_DIR=".venv"                    # Python virtual environment directory
HF_MODEL="sengi/pi0fast-so100"     # HF Hub model ID or local path
SERVER_HOST="0.0.0.0"              # server bind address
SERVER_PORT=5555                    # server port
# -----------------------------------------------

# 1. Check Python version
REQUIRED_VERSION="3.10.0"
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3.10 or later"
    exit 1
fi

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Python version $REQUIRED_VERSION or later is required. Found version $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
# 2. Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install project dependencies
echo "Installing project dependencies..."
python -m pip install --upgrade pip
python -m pip install -e .[all] pyzmq

# 5. Download model into local 'models/' folder
echo "Downloading model '$HF_MODEL' into ./models..."
mkdir -p models
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$HF_MODEL", cache_dir="models")
EOF

# 6. Launch remote inference server
echo "Starting remote inference server at $SERVER_HOST:$SERVER_PORT..."
nohup python lerobot/scripts/remote_server.py \
    --policy_path "$HF_MODEL" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    > server.log 2>&1 &

echo "Server started. Logs are being written to: server.log"
echo "To view logs in real-time, run: tail -f server.log"
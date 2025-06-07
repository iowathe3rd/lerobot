# Docker image for GPU-based LeRobot inference on RunPod
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG HF_MODEL=sengi/pi0fast-so100

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    MUJOCO_GL="egl" \
    HF_MODEL="$HF_MODEL"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-serial \
  && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
  && python -m venv /opt/venv \
  && /opt/venv/bin/pip install --upgrade pip \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /lerobot

COPY . /lerobot

RUN /opt/venv/bin/pip install --no-cache-dir -e ".[all]" pyzmq huggingface_hub[cli]

EXPOSE 12278

ENTRYPOINT ["bash","-lc","if [ -n \"$HF_TOKEN\" ]; then huggingface-cli login --token \"$HF_TOKEN\"; fi && exec python lerobot/scripts/remote_server.py --host 0.0.0.0 --port 12278 --policy_path \"$HF_MODEL\""]

# syntax=docker/dockerfile:1
# Reproducible ML research env. Pin versions to match your paper's setup.
# Build:  docker build -t yolobert:latest .
# Run:    docker run --gpus all -it --rm -v $(pwd):/workspace -v /data:/data yolobert:latest

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/hf

# --- system deps (opencv, git, common CV libs) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- python deps (layer-cached: copy requirements first) ---
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# --- project code ---
COPY . .

# Non-root user (safer; matches many HPC/cluster policies)
ARG UID=1000
RUN useradd -m -u ${UID} researcher && chown -R researcher /workspace
USER researcher

CMD ["bash"]

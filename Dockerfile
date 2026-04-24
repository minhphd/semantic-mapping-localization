# ── Stage: ROS 2 Humble + GPU + spot_semantic_mapping ──────────────────────────
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ── OS deps ────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# ── Install Miniconda ──────────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# ── Accept conda ToS (required for Anaconda channels) ─────────────────────────
RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --channel https://repo.anaconda.com/pkgs/r

# ── Copy dependency specs first (Docker layer cache) ──────────────────────────
COPY environment.yml requirements.txt pyproject.toml ./

# ── Create the conda env (name "semantic_mapping" comes from environment.yml) ──
RUN conda env create -f environment.yml && conda clean -a -y

# Activate by prepending the env's bin to PATH
ENV PATH=/opt/conda/envs/semantic_mapping/bin:$PATH

# ── Pip-only deps not covered by the conda env ────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── JAX with CUDA 12 support ──────────────────────────────────────────────────
RUN pip install --no-cache-dir -U "jax[cuda12]" accelerate

# ── Boston Dynamics Spot SDK ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    bosdyn-client \
    bosdyn-mission \
    bosdyn-choreography-client

# ── Copy source and install the package ───────────────────────────────────────
COPY . .
RUN pip install --no-cache-dir -e .

# ── ROS entrypoint: sources ROS before any command ───────────────────────────
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'source /opt/ros/humble/setup.bash' \
    'exec "$@"' \
    > /ros_entrypoint.sh && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

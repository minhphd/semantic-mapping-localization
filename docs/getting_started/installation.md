# Installation

There are two supported installation paths. **Docker is strongly recommended** because it bundles the exact CUDA toolkit, PyTorch, and system libraries needed without conflicts.

---

## Option A — Docker (Recommended)

### 1. Install Docker and NVIDIA Container Toolkit

=== "Ubuntu 22.04"

    ```bash
    # Docker Engine
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER && newgrp docker

    # NVIDIA Container Toolkit
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
      | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

=== "Verify"

    ```bash
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
    ```

    You should see your GPU listed. If not, see the [Docker Prerequisites](../prerequisites/docker.md) page for troubleshooting.

### 2. Clone and Build

```bash
git clone https://github.com/mhpham26/deployment_sem_mapping
cd deployment_sem_mapping

# First time: build image (takes ~10 min)
./activate_docker.sh --build

# Subsequent runs: just launch
./activate_docker.sh
```

`activate_docker.sh` mounts the repo into `/workspace` inside the container and drops you into a bash shell with CUDA, PyTorch, and all dependencies available.

!!! note "Flags"
    | Flag | Effect |
    |------|--------|
    | `--build` | Force-rebuild the Docker image |
    | `--run` | Skip the build check, just start the container |
    | *(none)* | Check if image exists; build if missing, then start |

---

## Option B — Conda + pip

Use this if you prefer a native install or need to integrate with an existing environment.

### 1. Install Conda

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/) (faster solver).

### 2. Create the Environment

```bash
git clone https://github.com/mhpham26/deployment_sem_mapping
cd deployment_sem_mapping

# Create environment with all conda dependencies (PyTorch, Open3D, ROS 2...)
conda env create -f environment.yml

# Activate
conda activate semantic_mapping
```

### 3. Install Python Package

```bash
pip install -e .
```

The `-e` flag installs in *editable* mode, so changes to the source are immediately reflected without reinstalling.

### 4. Install PyTorch (if not already done via conda)

```bash
# CUDA 12.8
pip install -r requirement-torch.txt
```

---

## Downloading Model Weights

The pipeline requires several pre-trained model checkpoints. Download them and set their paths in `spot_semantic_mapping/configs/main_config.yml`.

| Model | Purpose | Download |
|-------|---------|----------|
| **YOLOv8** | Object detection | [Ultralytics Hub](https://docs.ultralytics.com/models/yolov8/) |
| **SAM 2.1** | Instance segmentation | [Meta AI](https://github.com/facebookresearch/segment-anything-2) |
| **DINOv2** (ViT-L/14) | VPR feature extraction | Auto-downloaded via HuggingFace |
| **SigLIP** | Visual-text similarity | Auto-downloaded via HuggingFace |
| **BLIP-2** | Object captioning | Auto-downloaded via HuggingFace |

!!! tip "HuggingFace auto-download"
    Models marked *"auto-downloaded"* are fetched on first use and cached in `~/.cache/huggingface/`. You only need to manually download YOLO and SAM 2 checkpoint files.

---

## API Keys

Some pipeline steps use external LLM APIs for relation extraction.

Create `spot_semantic_mapping/configs/api_keys.ini`:

```ini
[groq]
api_key = gsk_...

[openai]
api_key = sk-...
```

!!! warning "Keep this file secret"
    `api_keys.ini` is listed in `.gitignore` and will never be committed. Never hard-code API keys in source files.

---

## Verify Installation

Run this inside your container or activated conda environment:

```bash
python scripts/check_cuda.py
```

Expected output:
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

Then confirm the CLI entrypoint works:

```bash
build-scene-graph --help
```

---

## Serving Docs Locally Over SSH

If you are working on a remote server and want to view the MkDocs site in your **local** browser, forward the server port over SSH.

**On the remote server**, start MkDocs:

```bash
mkdocs serve --dev-addr 0.0.0.0:8000
```

**On your local machine**, open a new terminal and create the tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<server-address>
```

Replace `<user>` and `<server-address>` with your SSH credentials. Then open [http://localhost:8000](http://localhost:8000) in your local browser.

!!! tip "Persistent tunnel"
    Add `-N` to run the tunnel without opening a shell, and `-f` to push it to the background:
    ```bash
    ssh -fNL 8000:localhost:8000 <user>@<server-address>
    ```
    Kill it later with `pkill -f "8000:localhost:8000"`.

---

## Next Step

→ [Configure the pipeline](configuration.md) by editing `main_config.yml`.

# Prerequisites

Spot Semantic Mapping is a GPU-accelerated Python library that relies on several tools from the modern ML stack. If any of these are new to you, start with the pages below before attempting installation.

---

## What You Need

| Tool | Role in this project | Familiarity needed |
|------|---------------------|-------------------|
| **[Docker](docker.md)** | Reproducible environment with CUDA and all system deps | Basic |
| **[Conda](conda.md)** | Alternative environment manager; manages non-pip packages like PyTorch and Open3D | Basic |
| **[PyTorch & CUDA](pytorch.md)** | Neural network inference (YOLO, DINOv2, BLIP-2, SigLIP, SAM) | Intermediate |
| **[Open3D](open3d.md)** | 3D point cloud processing and visualisation | Basic |

---

## Skill-Level Guide

=== "Complete Beginner"
    1. Start with [Docker](docker.md) — it isolates everything so nothing breaks your system.
    2. Skim [Conda](conda.md) to understand what `environment.yml` does.
    3. Read the [PyTorch](pytorch.md) page at a high level — you don't need to train models, just run them.
    4. Proceed to [Installation](../getting_started/installation.md) using the Docker path.

=== "Comfortable with Python, new to ML"
    1. Read [PyTorch & CUDA](pytorch.md) carefully — understanding tensors and GPU memory matters here.
    2. Skim [Docker](docker.md) and [Conda](conda.md).
    3. Go directly to [Installation](../getting_started/installation.md).

=== "ML researcher, new to robotics"
    1. You already know PyTorch. Skim the [Open3D](open3d.md) page for point cloud operations.
    2. Proceed to [Installation](../getting_started/installation.md).

---

## External Learning Resources

| Resource | Topic | Format |
|----------|-------|--------|
| [Docker in 100 Seconds](https://www.youtube.com/watch?v=Gjnup-PuquQ) | Docker overview | Video (2 min) |
| [Conda docs — Getting Started](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) | Conda basics | Docs |
| [PyTorch — Learning the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) | PyTorch fundamentals | Tutorial |
| [Open3D Tutorial](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html) | Point cloud ops | Tutorial |
| [Fast.ai Practical Deep Learning](https://course.fast.ai/) | Deep learning foundations | Course (free) |

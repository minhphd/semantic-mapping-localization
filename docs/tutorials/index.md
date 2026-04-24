# Tutorials

Interactive Jupyter notebooks that walk through the full pipeline step by step. Each notebook is self-contained and can be run independently.

---

## Launch JupyterLab

=== "Docker"

    ```bash
    ./activate_docker.sh
    # Inside the container:
    jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
    ```

=== "Conda"

    ```bash
    conda activate semantic_mapping
    jupyter lab --ip 0.0.0.0 --port 8888
    ```

Then open `http://localhost:8888` and navigate to the `notebooks/` folder.

---

## Notebooks

<div class="grid cards" markdown>

-   **01 · Data Loading**

    Load Spot and iPhone RGB-D recordings, inspect camera intrinsics and poses, and visualise frame sequences and trajectories.

    **Time:** ~20 min &nbsp;·&nbsp; **Prereqs:** Installation complete, sample data available

    [→ Open](data_loading.md)

-   **02 · Detection & Segmentation**

    Run YOLO open-world detection on RGB frames, refine bounding boxes with SAM masks, and filter by confidence.

    **Time:** ~25 min &nbsp;·&nbsp; **Prereqs:** Notebook 01

    [→ Open](detection_segmentation.md)

-   **03 · Scene Graph Construction**

    Project detections to 3D, run the full tracking pipeline, caption objects with BLIP-2, extract spatial relations, and export the scene graph.

    **Time:** ~40 min &nbsp;·&nbsp; **Prereqs:** Notebooks 01–02

    [→ Open](scene_graph.md)

-   **04 · Localization (VPR)**

    Build a VPR database, embed a query image with DINOv2 + VLAD, retrieve top-k matches, and evaluate Recall@k.

    **Time:** ~30 min &nbsp;·&nbsp; **Prereqs:** Notebook 03 output

    [→ Open](localization.md)

</div>

---

## Sample Data

```
data/
├── iphone/     iPhone StrayScanner sequences
├── spot/       Spot robot captures
└── graph/      Pre-built scene graphs (start here for notebooks 03–04)
```

If you don't have hardware, use the pre-built graph in `data/graph/` to run notebooks 03–04 directly, or request sample data from the repository maintainer.

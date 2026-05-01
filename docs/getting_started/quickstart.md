# Quick Start

This guide walks you through building a scene graph from a sample recording using both the CLI and the Python API.

---

## Step 1 — Prepare Your Data

The pipeline expects a dataset directory in one of two formats:

=== "iPhone (StrayScanner)"

    Record a sequence with [StrayScanner](https://apps.apple.com/app/stray-scanner/id1557051662) and export it. The directory layout should be:

    ```
    my_scene/
    ├── color/           # RGB frames as .jpg  (000000.jpg, 000001.jpg, …)
    ├── depth/           # Depth frames as .png (16-bit, millimetres)
    ├── confidence/      # Depth confidence maps
    ├── camera_matrix.csv
    └── odometry.csv     # Camera poses (tx, ty, tz, qx, qy, qz, qw)
    ```

    !!! tip "Transferring from iPhone"
        Upload the exported dataset to the lab drive and copy it to your workstation before running the pipeline.

=== "Spot Robot"

    **Step 1a — Capture**

    Make sure your computer is connected to iRobot (with Spot also connected), then run the capture script. It will prompt for Spot's IP, username, and password and begin saving frames at a fixed interval. Walk Spot around the scene, then press `Ctrl+C` to stop.

    ```bash
    python -m spot_semantic_mapping.spot.collection.capture \
        --hostname <SPOT_IP> \
        --username <USERNAME> \
        --password <PASSWORD> \
        --outdir data/spot/my_scene \
        --interval 2.0
    ```

    This produces a directory with per-timestamp subdirectories, one per camera.

    !!! note
        `--hostname` is Spot's IP address on the iRobot network (e.g. `192.168.80.3`). `--interval` controls the capture cadence in seconds.

    **Step 1b — Convert to StrayScanner format**

    The rest of the pipeline expects StrayScanner layout. Run the conversion before moving on:

    ```python
    from spot_semantic_mapping.spot.dataset import SpotDataset

    ds = SpotDataset("data/spot/my_scene")
    ds.generate_strayscanner_dir(camera="hand_color")
    # Output is written back into data/spot/my_scene/ in StrayScanner layout.
    ```

    You can now use `data/spot/my_scene` as the `dataset_path` in Step 2.

---

## Step 2 — Build the Scene Graph

=== "CLI"

    ```bash
    build-scene-graph data/iphone/my_scene \
        --output_dir outputs/my_scene \
        --rotate -90        # Rotate depth frames (iPhone handheld recordings often need this)
    ```

    !!! tip "Do I need `--rotate`?"
        Only add `--rotate -90` if you recorded vertically on iPhone. Open a few frames first to check orientation — the goal is to pass right-side-up images to the detection model. Spot recordings do not need rotation.

=== "Python"

    ```python
    from spot_semantic_mapping.mapping.pipeline import main
    from spot_semantic_mapping.configs.loader import cfg

    tracker = main(
        dataset_path="data/iphone/my_scene",
        output_dir="outputs/my_scene",
        rotate=-90,
        floor_only=False,
        cfg=cfg,
    )

    print(f"Tracked {len(tracker.objects)} objects")
    ```

This will:

1. Load RGB-D frames and camera poses
2. Run YOLO + SAM detection on every frame
3. Project detections to 3D point clouds
4. Track objects across frames with `ObjectTracker3D`
5. Caption each object with BLIP-2
6. Extract spatial relations with an LLM
7. Write `outputs/my_scene/scene_graph.json` and `semantics.ply`

!!! tip "Expected runtime"
    On an RTX 3090, a 5000-frame iPhone sequence takes ~8–12 minutes end-to-end. It is relatively fast

---

## Step 3 — Inspect the Output

### Scene Graph JSON

```python
from spot_semantic_mapping.mapping.io import load_scene_graph

graph = load_scene_graph("outputs/my_scene/scene_graph.json")

for node in graph.nodes:
    print(f"[{node.id}] {node.label}  —  "{node.caption}"")
    print(f"       centre: {node.centroid}")
    print(f"       edges:  {[e.relation for e in node.edges]}")
```

### Coloured Point Cloud

Open `outputs/my_scene/semantics.ply` in [Open3D](http://www.open3d.org/) or [CloudCompare](https://cloudcompare.org/):

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("outputs/my_scene/semantics.ply")
o3d.visualization.draw_geometries([pcd])
```

!!! note "Lab computer with Docker"
    If port-tunnelling is enabled in Docker, the `o3d.visualization.draw_geometries()` call above will open an interactive 3D panel directly on your screen.

---

## Step 4 — Run Localization

!!! note "Simplified example"
    The code below shows the minimal Python API. For full Spot-in-the-loop localization — including live pose streaming and scene graph publishing over ROS2 — see `spot_semantic_mapping/spot/telemetry.py` and the [ROS2 & Spot Telemetry](../background/ros2.md) background page.

```python
from spot_semantic_mapping.localization.localizer import localize, prepare_embeddings
from spot_semantic_mapping.localization.dataset import build_database
from spot_semantic_mapping.mapping.io import load_scene_graph
from spot_semantic_mapping.configs.loader import cfg

graph = load_scene_graph("outputs/my_scene/scene_graph.json")

# Build a VPR database from training frames
db = build_database("data/iphone/my_scene", cfg=cfg)

# Localise a new query image
results = localize(
    query_image_path="query.jpg",
    database=db,
    scene_graph=graph,
    top_k=5,
    cfg=cfg,
)

for r in results:
    print(f"  Frame {r.frame_id}  score={r.score:.3f}  pose={r.pose}")
```

---

## What's Next?

- Dive into the [interactive Jupyter tutorials](../tutorials/index.md) for a step-by-step walkthrough of each pipeline stage.
- Read the [Concepts](../concepts/architecture.md) section to understand how the pipeline works under the hood.
- Browse the [API Reference](../api/index.md) for full module documentation.

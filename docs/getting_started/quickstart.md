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

=== "Spot Robot"

    Collect a dataset using the capture utilities:

    ```python
    from spot_semantic_mapping.spot.collection.capture import capture_frames

    capture_frames(output_dir="data/spot/my_scene", interval_s=0.5)
    ```

    This produces a directory with per-timestamp subdirectories, one per camera.

---

## Step 2 — Build the Scene Graph

=== "CLI"

    ```bash
    build-scene-graph data/iphone/my_scene \
        --output_dir outputs/my_scene \
        --rotate -90        # Rotate depth frames (iPhone handheld recordings often need this)
    ```

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
    On an RTX 3090, a 500-frame iPhone sequence takes ~8–12 minutes end-to-end (detection is the bottleneck).

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

---

## Step 4 — Run Localization

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

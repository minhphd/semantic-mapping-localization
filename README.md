# Spot Semantic Mapping

A 3D semantic scene graph construction system for Boston Dynamics Spot robots and iPhone RGBD sequences. Transforms raw sensor data into structured, queryable world representations with object detection, 3D reconstruction, multi-view captioning, and visual place recognition.

**Full documentation:** [mhpham26.github.io/deployment_sem_mapping](https://mhpham26.github.io/deployment_sem_mapping/)

---

## Overview

The system builds a persistent 3D scene graph from RGB-D video by:

1. Detecting objects with YOLO and refining masks with SAM2
2. Projecting detections into 3D point clouds
3. Tracking objects across frames with geometry + vision similarity
4. Captioning each object from multi-view crops using a VLM (Llama-4 / BLIP-2)
5. Extracting spatial relations (left_of, on_top_of, near, ...) via VLM voting
6. Exporting a JSON scene graph + semantic colored point cloud

Localization maps new observations back into the graph using DINOv2 + VLAD embeddings.

---

## Architecture

```
RGB-D Sensor (Spot / iPhone)
        |
        v
  +-----------+
  |  Decoding |  decoding.py / SpotDataset
  +-----------+
        |
        v
  +-----------+     +-----------+
  |   YOLO    |---->|   SAM2    |   detection.py + segmentation.py
  +-----------+     +-----------+
        |
        v
  +-------------------+
  |  3D Projection    |  core/geometry.py (apply_dbscan, Open3D)
  +-------------------+
        |
        v
  +-------------------+
  |  ObjectTracker3D  |  mapping/tracker.py (geometry + CLIP similarity)
  +-------------------+
        |
        v
  +-------------------+
  |  VLM Captioning   |  mapping/pipeline.py (caption_obj, Llama-4/BLIP-2)
  +-------------------+
        |
        v
  +-------------------+
  |  Relation Graph   |  mapping/relations.py (build_sparse_scene_graph_edges)
  +-------------------+
        |
        v
  scene_graph.json + semantic_cloud.ply
        |
        v
  +-------------------+
  |  Localization     |  localization/ (DINOv2 + VLAD)
  +-------------------+
        |
        v
  Retrieved subgraph for current position
```

---

## Package Structure

```
spot_semantic_mapping/
├── __init__.py
│
├── models/                  # Vision & language models
│   ├── detection.py         # YOLODetector (YOLOv8/v11 open-world)
│   ├── segmentation.py      # FastSAMPredictor, MobileSAMPredictor, SAM2Predictor
│   ├── captioning.py        # BlipCaptioner (multi-view BLIP-2)
│   ├── embedding.py         # DinoModel (DINOv2 patch/CLS), SiglipModel
│   └── llm.py               # GroqModel, OpenaiModel, OpenaiEmbedding
│
├── core/                    # Data structures & low-level utilities
│   ├── types.py             # ObjectNode, export_scene_graph
│   ├── geometry.py          # apply_dbscan, construct_top_down
│   ├── mask.py              # mask_iou, merge_overlapping_masks
│   ├── crops.py             # concat_crops_horizontal
│   ├── io.py                # load_intrinsics, load_poses, load_depth, load_conf
│   ├── jax_helper.py        # cosine_similarity_jax, cdist, vlad_aggregate
│   ├── logger.py            # build_logger, save_full_tracker, log_gpu_memory
│   ├── metrics.py           # compute_metrics_from_scores
│   ├── tsdf.py              # TSDF reconstruction
│   ├── visualization.py     # Open3D visualization helpers
│   └── dataloader.py        # PyTorch Dataset/DataLoader wrappers
│
├── mapping/                 # Scene graph construction
│   ├── pipeline.py          # main() — full end-to-end pipeline
│   ├── tracker.py           # ObjectTracker3D, MapObject, Detection, UnionFind
│   ├── relations.py         # RelationEdge, build_sparse_scene_graph_edges
│   ├── occupancy.py         # Occupancy field generation
│   ├── database.py          # Scene graph database construction
│   └── io.py                # Checkpoint load/save (load_full_tracker, save_full_tracker)
│
├── localization/            # Visual place recognition (VPR)
│   ├── encoder.py           # ImageEncoder (VLAD, domain-VLAD, GeMP, GAP, GeM)
│   ├── localizer.py         # localize(), retrieve_subgraphs(), prepare_embeddings()
│   ├── dataset.py           # Dataset construction and loading
│   └── evaluation.py        # Recall@k, median localization error
│
├── spot/                    # Spot robot interface
│   ├── agent.py             # SpotAgent, SpotAgentGraph, FastSpotGraphAgent
│   ├── environment.py       # SpotEnv (gym-like interface)
│   ├── dataset.py           # SpotDataset (offline data loading)
│   ├── decoding.py          # RGB + depth frame decoding
│   ├── telemetry.py         # ROS2 telemetry bridge (standard + fast)
│   └── collection/
│       ├── capture.py       # Capture frames at fixed intervals
│       └── plotting.py      # Real-time trajectory + scene plotting
│
└── configs/                 # Configuration
    ├── loader.py            # Config class + global cfg singleton
    ├── main_config.yml      # Master configuration file
    └── randlanet.yml        # RandLA-Net segmentation config
```

---

## Setup & Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate semantic_mapping
```

### 2. Install the package

```bash
pip install -e .
```

### 3. Model weights

Download the required checkpoints and update paths in `spot_semantic_mapping/configs/main_config.yml`:

| Model | Config key |
|---|---|
| DINOv2-large | `paths.dino_ckpt` |
| SigLIP | `paths.siglip_ckpt` |
| BLIP-2 Flan-T5-XL | `paths.blip_ckpt` |
| YOLOv8 (open-world) | `paths.yolo8_ckpt` |
| SAM2 | `paths.sam2_ckpt` |
| FastSAM | `paths.fastsam_ckpt` |
| MobileSAM | `paths.mobile_sam_ckpt` |

### 4. API keys

Create `spot_semantic_mapping/configs/api_keys.ini`:

```ini
[API_KEYS]
groq_api = your_groq_key
openai_api = your_openai_key
```

### 5. Spot credentials (optional, for live robot)

Create `spot_semantic_mapping/configs/spot_credentials.json`:

```json
{"username": "user", "password": "pass"}
```

### Alternative: Docker (recommended for reproducibility)

All dependencies — ROS 2 Humble, CUDA, conda env, Spot SDK — are baked into the image.

```bash
# First run: builds the image then launches an interactive shell
./activate_docker.sh

# Force a full rebuild (after editing Dockerfile or dependencies)
./activate_docker.sh --build

# Skip build check, just run
./activate_docker.sh --run
```

The repo is bind-mounted to `/app` inside the container, so code edits on the host are reflected immediately without rebuilding.

**What you still need to provide on the host** (bind-mounted automatically via `$PWD`):

| File | Purpose |
|---|---|
| `spot_semantic_mapping/configs/api_keys.ini` | Groq / OpenAI keys |
| `spot_semantic_mapping/configs/spot_credentials.json` | Spot robot login |
| Model weight files | Paths configured in `main_config.yml` |

---

## Documentation

Install the doc dependencies once:

```bash
pip install mkdocs-material
```

### Local machine

```bash
mkdocs serve
```

Open [http://localhost:8000](http://localhost:8000).

### Remote server (SSH tunnel)

Start the dev server on the remote, binding to all interfaces:

```bash
mkdocs serve --dev-addr 0.0.0.0:8000
```

On your **local machine**, forward the port:

```bash
ssh -L 8000:localhost:8000 <user>@<server-address>
```

Then open [http://localhost:8000](http://localhost:8000) in your local browser.

To run the tunnel in the background without opening a shell:

```bash
ssh -fNL 8000:localhost:8000 <user>@<server-address>
# kill it later with:
pkill -f "8000:localhost:8000"
```

### Build static site

```bash
mkdocs build        # outputs to site/
mkdocs gh-deploy    # publish to GitHub Pages
```

---

## Quickstart

### Build a scene graph from an iPhone/Spot recording

```python
from spot_semantic_mapping.configs.loader import cfg
from spot_semantic_mapping.mapping.pipeline import main

tracker = main(
    dataset_path="data/iphone/3578aa5730",
    rotate=-90,        # rotate frames to correct orientation
    floor_only=False,  # set True to keep only floor-level points
    cfg=cfg,
    ouput_dir="outputs/",
)
# tracker.objects  -> list of MapObject (id, class_name, pcd, crops, ...)
# tracker.edges    -> list of RelationEdge (src, dst, relation_type, ...)
```

Or from the command line:

```bash
build-scene-graph data/iphone/3578aa5730 --output_dir outputs/
```

### Load and query a saved scene graph

```python
from spot_semantic_mapping.mapping.io import load_scene_graph

graph = load_scene_graph("outputs/exp_20240101_120000/scene_graph.json")
for node in graph["nodes"]:
    print(node["class_name"], node["position"])
```

### Localize a query image

```python
from spot_semantic_mapping.localization.encoder import ImageEncoder
from spot_semantic_mapping.localization.localizer import localize, retrieve_subgraphs
from spot_semantic_mapping.models.embedding import DinoModel
from spot_semantic_mapping.configs.loader import cfg
import pickle, numpy as np

encoder = ImageEncoder(DinoModel(cfg))

with open("data/graph/vpr_database.pkl", "rb") as f:
    dataset = pickle.load(f)

db_emb = encoder.embed(dataset["db_images"], patches=True, agg_method="vlad", num_clusters=32)

query_image = np.array(...)  # (H, W, 3) RGB
q_emb = encoder.embed(np.array([query_image]), patches=True, agg_method="vlad", num_clusters=32, save=False)

scores, sorted_idx = localize(q_emb[0], db_emb)

graph = load_scene_graph("outputs/exp_20240101_120000/scene_graph.json")
subgraph = retrieve_subgraphs(dataset, sorted_idx, graph, top_k=5, window=3.0)
print(f"Located near: {[n['class_name'] for n in subgraph['nodes'].to_dict('records')]}")
```

---

## Module Reference

### `mapping.pipeline`
End-to-end pipeline: loads data, runs YOLO+SAM, projects to 3D, tracks objects, captions, extracts relations, exports. Entry point is `main()`.

### `mapping.tracker`
`ObjectTracker3D` associates detections across frames using spatial (point-cloud overlap) and visual (CLIP cosine similarity) similarity. `prune_and_merge_tracker_objects()` runs a final merge pass.

### `mapping.relations`
`build_sparse_scene_graph_edges()` builds a KNN candidate graph and refines edge labels with VLM voting from co-visible frames.

### `localization.encoder`
`ImageEncoder` extracts DINOv2 patch features and aggregates them into compact descriptors. Supports VLAD, domain-VLAD, GeMP, GAP, GeM aggregation.

### `localization.localizer`
`localize()` runs cosine similarity retrieval. `retrieve_subgraphs()` filters the scene graph to objects within a spatial window of the top-k retrieved positions.

### `spot.agent`
`SpotAgent` provides synchronous and async observation capture from 6 Spot cameras. `FastSpotGraphAgent` adds threaded fast/medium/slow data lanes for real-time deployment.

### `models`
All perception models are wrappers around HuggingFace Transformers or Ultralytics. `SiglipModel` provides both image and text embeddings used for tracking.

---

## Configuration

Key parameters in `configs/main_config.yml`:

| Section | Key | Description |
|---|---|---|
| `device` | — | CUDA device (`"cuda"` or `"cpu"`) |
| `segmentation` | `detector` | `"yolov8"` or `"yolov11"` |
| `segmentation` | `sam_backend` | `"sam2"`, `"mobilesam"`, or `"fastsam"` |
| `tracking` | `w_geo` | Weight for geometric similarity |
| `tracking` | `w_sem` | Weight for CLIP visual similarity |
| `tracking` | `match_threshold` | Min score to match detection to existing object |
| `pipeline` | `use_every_n_frames` | Frame subsampling rate |
| `pipeline` | `max_frames` | Max frames to process (`-1` = all) |
| `vpr` | `agg_method` | VPR aggregation: `"vlad"`, `"domain_vlad"`, `"gem"` |
| `vpr` | `num_clusters` | VLAD codebook size |
| `landmarks` | `classes` | Open-world detection class list |

---

## Tutorial Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/01_data_loading.ipynb`](notebooks/01_data_loading.ipynb) | Load Spot/iPhone recordings, visualize RGB-D frames and trajectory |
| [`notebooks/02_detection_segmentation.ipynb`](notebooks/02_detection_segmentation.ipynb) | Run YOLO detection + SAM mask refinement on sample frames |
| [`notebooks/03_scene_graph_construction.ipynb`](notebooks/03_scene_graph_construction.ipynb) | Step through the full pipeline: 3D projection → tracking → captioning → export |
| [`notebooks/04_localization_vpr.ipynb`](notebooks/04_localization_vpr.ipynb) | Build a VPR database, embed queries, match, and retrieve subgraphs |

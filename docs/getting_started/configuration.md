# Configuration

All runtime behavior is controlled through a single YAML file at:

```
spot_semantic_mapping/configs/main_config.yml
```

The config is loaded at import time via a global singleton:

```python
from spot_semantic_mapping.configs.loader import cfg

# Access any key
device = cfg["device"]           # "cuda"
yolo_path = cfg["yolo_checkpoint"]
```

---

## Full Config Reference

```yaml title="main_config.yml"
# ── Device ──────────────────────────────────────────────────────────────────
device: cuda                   # "cuda" or "cpu"

# ── Model Checkpoints ───────────────────────────────────────────────────────
yolo_checkpoint: path/to/yolo11x-worldv2.pt
sam2_checkpoint: path/to/sam2.1_hiera_large.pt
sam2_config: path/to/sam2.1_hiera_l.yaml
mobilesam_checkpoint: path/to/mobile_sam.pt
fastsam_checkpoint: path/to/FastSAM-x.pt
dino_model: facebook/dinov2-large   # HuggingFace model ID
siglip_model: google/siglip-large-patch16-384
blip2_model: Salesforce/blip2-opt-6.7b

# ── Camera Parameters ───────────────────────────────────────────────────────
depth_width: 256
depth_height: 192
rgb_width: 1920
rgb_height: 1440
max_depth: 20.0                # metres — points beyond this are discarded

# ── Detection & Segmentation ────────────────────────────────────────────────
detector: yolov11              # "yolov8" | "yolov11"
sam_backend: sam2              # "sam2" | "mobilesam" | "fastsam"
detection_conf: 0.25           # YOLO confidence threshold
nms_iou: 0.5                   # Non-maximum suppression IoU

# ── Object Classes (open-set) ────────────────────────────────────────────────
classes:
  - chair
  - table
  - door
  - window
  - cabinet
  - monitor
  - keyboard
  - robot
  # add any noun phrase — YOLO World supports open vocabulary

# ── 3D Tracking ─────────────────────────────────────────────────────────────
w_geo: 1.0                     # Geometric similarity weight (point cloud IOU)
w_sem: 1.0                     # Semantic similarity weight (CLIP embedding)
match_threshold: 0.4           # Min score to associate detection with existing object
merge_threshold: 0.7           # Min score to merge two tracked objects (loop closure)
min_views: 3                   # Discard objects seen fewer than N frames

# ── DBSCAN Point Cloud Clustering ────────────────────────────────────────────
dbscan_eps: 0.1                # Neighbourhood radius (metres)
dbscan_min_samples: 10         # Min points to form a cluster

# ── VPR / Localization ───────────────────────────────────────────────────────
vpr_aggregation: vlad          # "vlad" | "domain_vlad" | "gem" | "gap" | "gemp"
vlad_clusters: 32              # Number of VLAD cluster centres
vpr_top_k: 5                   # Return top-k nearest neighbours

# ── Relation Extraction ──────────────────────────────────────────────────────
relation_knn: 5                # Candidate neighbours per object
relation_votes: 3              # VLM votes required to confirm a relation
relation_llm: groq             # "groq" | "openai"
```

---

## Key Sections Explained

### Device

Set `device: cpu` only for debugging — the full pipeline requires a GPU for reasonable throughput. All PyTorch and FAISS operations respect this setting.

### Model Checkpoints

You must provide absolute or relative paths for YOLO and SAM. HuggingFace models (DINOv2, SigLIP, BLIP-2) are downloaded automatically on first use.

### Open-Set Classes

The `classes` list is passed verbatim to YOLO World as text prompts. You can add any noun phrase without retraining — e.g. `"charging cable"`, `"fire extinguisher"`, `"robot arm"`.

### Tracking Weights

`w_geo` and `w_sem` balance how geometric overlap (IOU of 3D point clouds) vs. visual appearance (CLIP similarity) contribute to the object association score:

```
score = (w_geo × iou + w_sem × cosine_sim) / (w_geo + w_sem)
```

Increase `w_sem` in texturally rich scenes; increase `w_geo` in geometrically distinctive environments.

### VPR Aggregation Methods

| Method | Description | Best for |
|--------|-------------|---------|
| `vlad` | Vector of Locally Aggregated Descriptors | General purpose |
| `domain_vlad` | VLAD with domain-adapted cluster centres | Same-scene retrieval |
| `gem` | Generalised Mean pooling | Fast, compact descriptor |
| `gap` | Global Average Pooling | Lightweight baseline |
| `gemp` | GeM with learned exponent | Tunable |

---

## Programmatic Override

You can override any config key at runtime without editing the YAML:

```python
from spot_semantic_mapping.configs.loader import cfg

cfg["device"] = "cpu"
cfg["detector"] = "yolov8"
```

---

## Next Step

→ [Build your first scene graph](quickstart.md)

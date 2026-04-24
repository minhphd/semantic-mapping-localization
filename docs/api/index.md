# API Reference

Complete reference for every module, class, and function in `spot_semantic_mapping`.

---

## Package Structure

```
spot_semantic_mapping/
├── configs/         Config loader and YAML files
├── core/            Data structures, geometry, I/O, utilities
├── models/          Model wrappers (detection, segmentation, captioning, embeddings, LLMs)
├── mapping/         Scene graph pipeline, tracker, relations, database
├── localization/    VPR encoder, localizer, evaluation
└── spot/            Boston Dynamics Spot robot interface
```

---

## Modules

| Module | Description |
|--------|-------------|
| [**Mapping Pipeline**](mapping.md) | End-to-end scene graph construction (`pipeline.py`, `tracker.py`, `relations.py`) |
| [**Localization**](localization.md) | VPR encoder, FAISS retrieval, subgraph extraction, evaluation |
| [**Models**](models.md) | YOLO, SAM, DINOv2, SigLIP, BLIP-2, LLM wrappers |
| [**Core Utilities**](core.md) | Geometry, masks, crops, I/O, metrics, TSDF, visualisation |
| [**Spot Interface**](spot.md) | Robot agents, dataset loading, telemetry, capture |
| [**Configuration**](config.md) | Config loader and YAML reference |

---

## Quick Reference

### Common Imports

```python
# Config
from spot_semantic_mapping.configs.loader import cfg

# Scene graph I/O
from spot_semantic_mapping.mapping.io import load_scene_graph, load_full_tracker
from spot_semantic_mapping.core.types import SceneGraph, ObjectNode

# Pipeline
from spot_semantic_mapping.mapping.pipeline import main

# Models
from spot_semantic_mapping.models.detection import YOLODetector
from spot_semantic_mapping.models.segmentation import SAM2Predictor, MobileSAMPredictor
from spot_semantic_mapping.models.captioning import BlipCaptioner
from spot_semantic_mapping.models.embedding import DinoModel, SiglipModel

# Localization
from spot_semantic_mapping.localization.localizer import localize
from spot_semantic_mapping.localization.dataset import build_database
from spot_semantic_mapping.localization.evaluation import compute_metrics_from_scores

# Geometry
from spot_semantic_mapping.core.geometry import apply_dbscan, construct_top_down
```

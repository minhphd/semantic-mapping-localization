# Core Utilities

Low-level data structures, geometry, I/O, and utility functions.

---

## Types · `core.types`

Core data structures for representing scene graphs and object nodes.

::: spot_semantic_mapping.core.types
    options:
      members:
        - ObjectNode
        - SceneGraph
        - export_scene_graph

---

## Geometry · `core.geometry`

3D geometric operations including point cloud back-projection, DBSCAN clustering, and floor-plane extraction.

::: spot_semantic_mapping.core.geometry
    options:
      members:
        - apply_dbscan
        - construct_top_down

---

## Masks · `core.mask`

Mask utility functions for computing overlap and merging predictions.

::: spot_semantic_mapping.core.mask
    options:
      members:
        - mask_iou
        - merge_overlapping_masks

---

## Image Crops · `core.crops`

::: spot_semantic_mapping.core.crops
    options:
      members:
        - concat_crops_horizontal

---

## I/O · `core.io`

File loading utilities for camera parameters, poses, and depth frames.

::: spot_semantic_mapping.core.io
    options:
      members:
        - load_intrinsics
        - load_poses
        - load_depth
        - load_conf

---

## JAX Helpers · `core.jax_helper`

GPU-accelerated numerical operations via JAX.

::: spot_semantic_mapping.core.jax_helper
    options:
      members:
        - cosine_similarity_jax
        - cdist
        - vlad_aggregate

---

## Metrics · `core.metrics`

::: spot_semantic_mapping.core.metrics
    options:
      members:
        - compute_metrics_from_scores

---

## Logger · `core.logger`

::: spot_semantic_mapping.core.logger
    options:
      members:
        - build_logger
        - save_full_tracker
        - log_gpu_memory

---

## TSDF · `core.tsdf`

Truncated Signed Distance Field for volumetric 3D reconstruction.

::: spot_semantic_mapping.core.tsdf

---

## Visualisation · `core.visualization`

Open3D-based 3D visualisation helpers.

::: spot_semantic_mapping.core.visualization

---

## DataLoader · `core.dataloader`

PyTorch Dataset and DataLoader wrappers for RGB-D sequences.

::: spot_semantic_mapping.core.dataloader

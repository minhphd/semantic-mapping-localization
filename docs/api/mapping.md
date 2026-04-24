# Mapping Pipeline

The mapping modules orchestrate the full scene graph construction pipeline.

---

## `mapping.pipeline`

::: spot_semantic_mapping.mapping.pipeline
    options:
      members:
        - main
        - run_yolo_sam
        - caption_obj
        - save_graph
        - save_semantics_cloud

---

## `mapping.tracker`

The tracker maintains persistent 3D objects across frames using joint geometric and semantic association.

::: spot_semantic_mapping.mapping.tracker
    options:
      members:
        - ObjectTracker3D
        - MapObject
        - Detection
        - UnionFind

---

## `mapping.relations`

Extracts directed spatial relations between objects using KNN candidate graphs and VLM voting.

::: spot_semantic_mapping.mapping.relations
    options:
      members:
        - build_sparse_scene_graph_edges
        - RelationEdge

---

## `mapping.io`

Serialisation and deserialisation of scene graphs and tracker checkpoints.

::: spot_semantic_mapping.mapping.io
    options:
      members:
        - load_scene_graph
        - load_full_tracker
        - save_full_tracker

---

## `mapping.database`

::: spot_semantic_mapping.mapping.database

---

## `mapping.occupancy`

::: spot_semantic_mapping.mapping.occupancy

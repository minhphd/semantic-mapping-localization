"""
spot_semantic_mapping
=====================

3D Semantic Scene Graph Construction for Boston Dynamics Spot.

Submodules
----------
models      : Vision and language models (YOLO, SAM, BLIP, SigLIP, LLM APIs)
core        : Data structures, geometry, I/O utilities
mapping     : Scene graph pipeline, object tracking, relation extraction
localization: Visual place recognition (VPR) with VLAD aggregation
spot        : Spot robot interface, dataset loading, ROS2 telemetry
configs     : YAML configuration loader
"""

__version__ = "0.2.0"

# Spot Interface

Modules for interfacing with Boston Dynamics Spot robots, data capture, and offline dataset loading.

---

## Agent · `spot.agent`

High-level robot agents for live operation.

::: spot_semantic_mapping.spot.agent
    options:
      members:
        - SpotAgent
        - SpotAgentGraph
        - FastSpotGraphAgent

---

## Environment · `spot.environment`

Gym-like interface for Spot, compatible with standard reinforcement learning frameworks.

::: spot_semantic_mapping.spot.environment
    options:
      members:
        - SpotEnv

---

## Dataset · `spot.dataset`

Offline dataset loading from recorded Spot captures.

::: spot_semantic_mapping.spot.dataset
    options:
      members:
        - SpotDataset

---

## Decoding · `spot.decoding`

RGB and depth frame decoding from Spot's proprietary formats.

::: spot_semantic_mapping.spot.decoding

---

## Telemetry · `spot.telemetry`

ROS 2 telemetry bridge for real-time data streaming.

::: spot_semantic_mapping.spot.telemetry

---

## Data Capture · `spot.collection`

Utilities for capturing and recording RGB-D sequences from Spot.

::: spot_semantic_mapping.spot.collection.capture
    options:
      members:
        - capture_frames

::: spot_semantic_mapping.spot.collection.plotting

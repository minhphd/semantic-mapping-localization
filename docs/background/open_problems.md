# Open Problems in Robotics

Two of the oldest and hardest unsolved problems in robotics are **localization** and **mapping**. This repo directly addresses both. Understanding why they are hard will help you appreciate every design decision in the codebase.

---

## The Core Challenge

A robot navigating the real world needs to answer two questions at all times:

```
1. Where am I?              ← Localization
2. What does the world look like?  ← Mapping
```

These questions are deeply intertwined: to build a map you need to know where you are, and to know where you are you need a map. This chicken-and-egg dependency is called the **SLAM problem** (Simultaneous Localization And Mapping).

```
The SLAM loop:

  ┌──────────────┐     need map to localize      ┌──────────────┐
  │              │ ─────────────────────────────► │              │
  │ Localization │                                │   Mapping    │
  │              │ ◄───────────────────────────── │              │
  └──────────────┘     need pose to map           └──────────────┘
```

---

## Localization

**Localization** is the problem of determining the robot's position and orientation (its **pose**) in a known environment.

### Why Not Just Use GPS?

GPS is accurate to ~3–5 meters outdoors with clear sky view. Inside a building, GPS signals are too weak. A 3-meter error is also far too large for a robot navigating a room with 1-meter-wide doorways.

### Dead Reckoning and Drift

The simplest approach: integrate the robot's velocity over time.

```
Dead reckoning:

  Start at [0, 0, 0°]
  Drive forward 1 m  → estimated position [1, 0, 0°]
  Turn left 90°      → estimated position [1, 0, 90°]
  Drive forward 1 m  → estimated position [1, 1, 90°]
  ...
```

Every sensor reading has a small error. Those errors **accumulate** — after driving 100 m, the position estimate can be off by several meters. This is called **drift**.

```
Drift over time:

  True path:       ─────────────────────────────────────►
  Estimated path:  ─────────────────/───────/──────────────
                                  drift     more drift

  Error grows roughly as √(distance traveled)
```

### Visual Place Recognition as a Fix

This repo solves drift by **matching the current camera image to a database** of previously seen frames. The database stores images with known positions. Finding the closest match gives an absolute position estimate — resetting the accumulated error.

```
VPR corrects drift:

  Dead reckoning estimate: "I think I'm at [3.2, 1.1]"
  VPR match: "This view matches Frame 47, which was at [3.0, 0.9]"
  Corrected estimate: [3.0, 0.9]
  Drift reset.
```

### Why Localization Is Hard

| Challenge | Why it's difficult |
|-----------|-------------------|
| Perceptual aliasing | Two different places look the same (long corridor with identical doors) |
| Dynamic environments | People, furniture rearranged — the map is outdated |
| Viewpoint change | Same place looks very different from a different angle |
| Lighting variation | Daylight vs artificial light drastically changes appearance |
| Computational cost | Comparing one query to millions of database images in real time |

---

## Mapping

**Mapping** is the problem of building a representation of the environment from sensor data.

### Levels of Representation

Not all maps are equally useful. There is a hierarchy from geometric to semantic:

```
Geometric maps (what shape is the world?):

  Occupancy grid    2D: which cells are free / occupied?
  Point cloud       3D: where are the surfaces?
  Mesh              3D: smooth triangle surface
  TSDF              3D: volumetric signed distance function

Semantic maps (what does the world mean?):

  Labeled point cloud   each point tagged with a class name
  Scene graph           objects as nodes, relations as edges
                        ← what this repo builds
```

### Why Not Just Stack Point Clouds?

A naive map is just a union of all back-projected depth frames. Problems:

```
Problem 1: Drift corrupts alignment

  Frame 1 (true pose):     chair here ●
  Frame 200 (drifted pose): chair here   ●
                                       ↑ ghost duplicate from drift

Problem 2: No semantic meaning

  A point cloud tells you where the surfaces are.
  It does not tell you: "that surface is a chair"
  or "that chair is next to a desk."
```

### The Scene Graph Approach

This repo builds a **semantic scene graph** — a structured map where:

- Every object is a node with a 3D position and a natural-language caption
- Every edge is a spatial relation between two objects

```
Semantic map (scene graph):

  [lamp]──near──►[desk]──on_top_of◄──[laptop]
                    │
                 left_of
                    │
                  [chair]
```

This representation is:
- **Queryable** — "find all objects near the desk"
- **Compact** — 50 object nodes instead of 5 million points
- **Actionable** — a robot can plan a path using object relations

### Why Semantic Mapping Is Hard

| Challenge | Why it's difficult |
|-----------|-------------------|
| Object permanence | The same chair seen from 100 different angles — is it 1 object or 100? |
| Occlusion | Objects partially hidden behind other objects |
| Instance disambiguation | Two identical chairs — tracking which is which |
| Relation ambiguity | Is the cup "near" the laptop or "on top of" the desk? |
| Scale | A real building has thousands of objects — needs to stay efficient |

---

## How This Repo Addresses Both Problems

```
Mapping:
  RGB-D frames + YOLO + SAM
       │
       ▼
  ObjectTracker3D   ← solves instance disambiguation with geometry + SigLIP
       │
       ▼
  BLIP-2 captioning + LLM relation extraction
       │
       ▼
  scene_graph.json  ← semantic map

Localization:
  New camera image
       │
       ▼
  DINOv2 + VLAD encoding
       │
       ▼
  Cosine similarity search against database
       │
       ▼
  Top-k matches → estimated position
       │
       ▼
  Subgraph retrieval  ← "what objects are near me right now?"
```

The two modules are intentionally separate: mapping runs offline (once, building the scene graph), and localization runs online (each time the robot needs to know where it is).

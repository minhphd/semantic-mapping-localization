# Intro to Computer Vision

Computer vision is what lets the robot understand what it sees. This repo uses three types of vision models: **detection**, **segmentation**, and **vision-language models**. Each one produces a different kind of output from an image.

---

## Object Detection

**Detection** answers: *what objects are in this image, and roughly where?*

The output is a list of **bounding boxes** — one rectangle per detected object:

```
Input: RGB image (H × W × 3)

Output:
  ┌─────────────────────────────┐
  │                             │
  │   ┌──────┐  chair  0.92     │
  │   │      │                  │
  │   └──────┘                  │
  │        ┌──────────┐         │
  │        │   desk   │  0.88   │
  │        └──────────┘         │
  └─────────────────────────────┘

Each detection = [x1, y1, x2, y2, class_name, confidence]
```

### YOLO

This repo uses **YOLO (You Only Look Once)**, which divides the image into a grid and predicts boxes in a single forward pass — no sliding window, no multiple passes. This makes it fast enough for real-time use.

```
YOLO grid (simplified):

  ┌───┬───┬───┬───┐
  │   │   │ ● │   │  ← each cell predicts:
  ├───┼───┼───┼───┤     - does an object center fall here?
  │   │ ● │   │   │     - if yes: box offset + class + confidence
  ├───┼───┼───┼───┤
  │   │   │   │   │
  └───┴───┴───┴───┘
```

### IoU — Intersection over Union

Used to measure how well two boxes overlap, and to remove duplicate detections:

\[
\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}
\]

```
  ┌──────────┐
  │  A       │
  │   ┌──────┼────┐
  │   │ A∩B  │  B │
  └───┼──────┘    │
      └───────────┘

  IoU = 0  → no overlap
  IoU = 1  → identical boxes
```

---

## Image Segmentation

**Segmentation** goes further than detection: instead of a box, it tells you exactly *which pixels* belong to the object — a **binary mask**.

```
Bounding box (detection):     Binary mask (segmentation):

  ┌────────────┐               F F F F F F F
  │            │               F T T T T F F
  │   chair    │     vs        F T T T T T F
  │            │               F F T T T F F
  └────────────┘               F F F F F F F

  Includes background pixels   Pixel-perfect boundary
```

### SAM — Segment Anything Model

This repo uses **SAM (Segment Anything)**, a foundation model from Meta that accepts a bounding box as a prompt and returns a tight binary mask:

```
SAM workflow:

  Image ──► ViT encoder ──► image embedding
                                  │
  Bbox  ──► decoder  ◄────────────┘
                │
                ▼
           binary mask (H × W)
```

The encoder runs once per frame. The decoder runs once per detected object. Clean masks mean clean point clouds — stray background pixels in the mask create noise in 3D.

---

## Vision-Language Models (VLMs)

A **VLM** understands both images and text together. The models in this repo:

### DINOv2

A Vision Transformer trained **without labels** (self-supervised) on 142M images. It produces powerful image embeddings used for place recognition and object tracking.

```
DINOv2 output:

  Image (224×224)
       │
       ▼
  Split into patches (16×16 each) → 196 tokens
       │
       ▼
  Transformer (12–24 layers of self-attention)
       │
       ├── CLS token  →  (768,) summary of the whole image
       └── Patch tokens → (196, 768) spatially-aware features
```

### SigLIP

Embeds images **and text** into the same vector space. A photo of a chair and the phrase "a chair" end up close together. Used in this repo to compare objects across frames.

```
SigLIP embedding space:

  image of chair ──► [0.3, −0.1, …]
                              ↕ close
  "a chair"      ──► [0.3, −0.1, …]
```

### BLIP-2

Takes an image and generates a **natural language caption**:

```
Crops of the chair from 3 different angles
       │
       ▼
  BLIP-2 (ViT + Q-Former + Flan-T5)
       │
       ▼
  "a black office chair with mesh back"
```

Each tracked object gets captioned from multiple views. The most frequent caption becomes the object's label in the scene graph.

---

## How They Chain Together in the Pipeline

```
RGB frame
    │
    ▼
YOLO ──► [bbox₁, bbox₂, bbox₃, ...]     (detection)
    │
    ▼
SAM  ──► [mask₁, mask₂, mask₃, ...]     (segmentation)
    │
    ▼
Apply mask to depth image
    │
    ▼
Back-project → 3D point cloud per object
    │
    ▼
SigLIP embedding per object crop         (for tracking)
    │
    ▼
ObjectTracker3D  ──► scene graph nodes
    │
    ▼
BLIP-2  ──► caption each node           (after mapping)
```

!!! tip "Config knobs"
    - `segmentation.detector`: `"yolov8"` or `"yolov11"`
    - `segmentation.sam_backend`: `"sam2"`, `"mobilesam"`, `"fastsam"`
    - `landmarks.classes`: the list of object categories to detect

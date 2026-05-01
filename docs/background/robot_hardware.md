# Robot Hardware

A robot is, at its core, a device that replaces a human doing something tedious, dangerous, or precise enough to require consistency. Spot was designed by Boston Dynamics to walk into places that wheeled robots cannot reach — rubble, stairs, uneven terrain — and to carry sensors into those places so something useful can happen. Before writing code that processes sensor data, it helps to know what that data actually is, how it was measured, and what assumptions are baked in. All of those assumptions will eventually bite you if you don't know they exist.

---

## Sensors

A **sensor** is a device that measures something about the physical world and converts it to a digital signal. Robots carry multiple sensors because no single sensor captures everything: a camera sees color and texture but cannot measure distance; a LiDAR measures distance but has no color; an IMU measures motion but goes blind to the outside world entirely.

```
Boston Dynamics Spot sensor suite (front view):

         ┌─────────────────────────────────────┐
         │                                     │
         │   front-left     front-right        │
         │   depth cam       depth cam         │   ← stereo + structured light
         │                                     │
         │         fisheye camera              │   ← wide-angle RGB
         │             (top)                   │
         │                                     │
         │       [HAND]  ← wrist camera:       │
         │          hand_color_image           │   ← high-res RGB + depth
         │          hand_depth                 │
         └─────────────────────────────────────┘

  Side cameras: left_fisheye, right_fisheye, back_fisheye
  IMU: 3-axis accelerometer + 3-axis gyroscope (bolted to chassis)
```

| Sensor | Measures | Output format |
|--------|---------|---------------|
| RGB camera | Color and texture | `(H, W, 3)` uint8 image |
| Depth camera | Distance per pixel | `(H, W)` float32, millimetres |
| IMU | Linear acceleration + angular velocity | 6-D vector at ~200 Hz |
| LiDAR | 3D distances via laser pulses | point cloud `(N, 3)` |
| Wheel/leg encoders | Joint angles and velocities | odometry estimates |

---

## Depth Cameras

A **depth camera** captures a depth image — every pixel stores a distance value instead of a color. Conceptually, it is like having a million tiny range-finders tiled across the field of view.

```
RGB image (colors):        Depth image (distances in meters):

  ┌──────────────┐           ┌──────────────┐
  │ 120 080 060  │           │ 0.0  0.0 0.0 │  ← sky/ceiling → 0 (invalid)
  │ 110 075 055  │           │ 0.0  1.7 1.8 │  ← wall at 1.7 m
  │ 100 070 060  │           │ 0.0  1.7 1.8 │
  └──────────────┘           └──────────────┘
    (R, G, B per pixel)        (meters per pixel)
```

### How Depth Sensors Work

**Structured light** (Spot body cameras): Project a grid of invisible infrared dots onto the scene. A second IR camera observes the grid. Where the dots land depends on how far away the surface is — closer objects shift the dots by a larger amount. This shift is called **disparity**, and it maps directly to depth.

**Time-of-flight / LiDAR** (iPhone LiDAR): Fire short infrared laser pulses and measure the time until the reflected pulse returns. Distance = (time × speed of light) / 2. Accurate to ~1 cm out to 5 m. Does not require a textured surface. Confused by mirrors and retroreflectors.

**Stereo** (some setups): Two cameras separated by a known baseline. Match the same surface point in both images. The horizontal shift (disparity) gives depth via triangulation. No active illumination required — works outdoors in bright sunlight where IR sensors are confused.

### Intrinsic Matrix K

The **intrinsic matrix** describes the lens geometry — it encodes how a 3D point in front of the camera projects onto a 2D pixel. Every time you convert a depth image to a 3D point cloud, you are using K.

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

| Symbol | Meaning |
|--------|---------|
| $f_x, f_y$ | Focal length in pixels (image scale per unit of depth) |
| $c_x, c_y$ | Principal point — where the optical axis pierces the sensor |

```
Projection (3D → pixel):          Back-projection (pixel + depth → 3D):

  3D point [X, Y, Z]                pixel (u, v) + depth d
         │                                  │
         ▼  divide by Z, apply K            ▼  apply K⁻¹, scale by d
  u = f_x · (X/Z) + c_x           X = (u - c_x) · d / f_x
  v = f_y · (Y/Z) + c_y           Y = (v - c_y) · d / f_y
                                   Z = d
```

```python
def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project a depth image to a (N, 3) point cloud."""
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))   # pixel grids, shape (H, W)
    valid = depth > 0                                  # mask out missing depth

    X = (u[valid] - cx) * depth[valid] / fx
    Y = (v[valid] - cy) * depth[valid] / fy
    Z = depth[valid]
    return np.stack([X, Y, Z], axis=1)    # (N, 3)
```

This is exactly what `core/geometry.py` does for every depth frame.

### Extrinsic Calibration: T_BC

The intrinsic matrix tells you where in the *camera frame* a 3D point lives. But the camera is mounted on the robot body at some offset and angle. The **extrinsic calibration** $T_{BC}$ (Body → Camera transform) is the rigid-body transform that maps a point in the body frame to the camera frame — and vice versa.

```
Coordinate chain for a single depth point:

  pixel (u, v) + depth d
         │  K⁻¹
         ▼
  point in camera frame p_C = [X, Y, Z]
         │  T_BC⁻¹  (= T_CB, camera to body)
         ▼
  point in body frame p_B
         │  T_WB  (body pose in world)
         ▼
  point in world frame p_W  ← what the planner sees
```

`T_BC` is determined during manufacturing (or re-calibrated periodically). It is stored in `intrinsics/intrinsics.json` for each Spot camera source.

---

## Poses and Orientations

A **position** is a 3D point `[x, y, z]` — where something is. A **pose** is position *plus* orientation — where it is and which way it faces. In code, a pose is always a `(4, 4)` homogeneous transform matrix (see [Linear Algebra in Python](linear_algebra.md)).

```
Pose matrix layout:

  ┌ r00  r01  r02 │ tx ┐
  │ r10  r11  r12 │ ty │  ← top-left 3×3  = rotation
  │ r20  r21  r22 │ tz │     top-right 3×1 = translation (= position of frame origin)
  └  0    0    0  │  1 ┘
```

```python
poses = load_poses(...)    # shape (T, 4, 4)  — one pose per frame

pose_t = poses[42]         # pose at frame 42
position = pose_t[:3, 3]   # [tx, ty, tz] — where the camera was
rotation = pose_t[:3, :3]  # 3×3 rotation matrix — which way it was pointing
```

### How Poses Are Captured

**iPhone (ARKit VIO):** Apple's ARKit fuses visual odometry (tracking feature points between frames) with IMU readings to estimate the device pose at every frame. This is called **Visual-Inertial Odometry (VIO)** — the camera provides heading corrections that stop the IMU from drifting, and the IMU fills in the gaps between camera frames. Each frame comes with a `(4, 4)` camera-to-world pose.

**Spot (state estimator):** Spot's on-board state estimator runs at ~200 Hz, fusing wheel/leg odometry, IMU, and foot contact forces. The body pose in the odometry frame is published as a SE(3) transform. The camera poses are derived by composing body pose × camera-to-body calibration.

```
Coordinate conventions:

  World frame (fixed):            Camera frame (OpenCV convention):
       z ↑                             ──────────► z (forward, into scene)
         │                            ╱
         │───── y                    ╱ x (right)
        ╱                            │
       x                             ▼ y (down — image rows increase downward)
```

This matters: if you blindly apply a transform without checking the convention, your point cloud ends up upside-down or mirrored. iPhone ARKit uses the OpenCV convention. Spot uses it too. Check before assuming.

---

## Depth Confidence Maps

Not every depth pixel is reliable. Reflective surfaces (windows, monitors, polished floors), strong specular highlights, and transparent materials all cause structured-light sensors to return incorrect or zero depth values.

The iPhone exports a **confidence map** alongside every depth frame:

```
Confidence values:
  0 = low   (likely invalid — discard before back-projecting)
  1 = medium
  2 = high  (reliable)

depth   shape: (H, W)  float32  in meters
conf    shape: (H, W)  uint8    values 0, 1, 2
```

```python
depth = load_depth(t)
conf  = load_conf(t)

depth[conf < 1] = 0.0    # zero out unreliable pixels
depth[depth > cfg["max_depth"]] = 0.0    # also clip at max range

pcd = depth_to_pointcloud(depth, K)   # now cleaner
```

Skipping the confidence mask is one of the most common sources of noisy point clouds. Reflective floors are particularly bad — they produce phantom points at incorrect depths that look like floating surfaces when visualised.

---

## Boston Dynamics Spot

Spot is a **quadruped** — four legs, not wheels. This matters because legged robots can navigate stairs, curbs, and uneven terrain, but their body pose is never perfectly stable: the body pitches and rolls slightly with each step. This pitch-roll motion is fully estimated by the state estimator and is baked into every pose reading, so the depth frames you receive are already corrected for body sway.

```
Spot top view:

          ┌─────────────────────────┐
  front   │  cam    cam    cam      │   ← front-left, front, front-right depth
  ───────►│                         │
          │                         │
          │  cam              cam   │   ← left and right depth
          │                         │
          │                [HAND]   │   ← arm-mounted hand camera (optional)
          └─────────────────────────┘

  All body cameras stream simultaneously.
  Hand camera is only available when the arm is deployed.
  Body pose estimated at ~200 Hz.
```

**Degrees of freedom:** Spot has 12 motorized joints (3 per leg). The body itself has 6 DoF in free space (3 translation + 3 rotation). The arm (optional) adds another 6 DoF. This complexity is why Spot's SDK exposes a high-level kinematic API rather than raw motor commands — you tell it "stand up" or "walk to waypoint X" and the controller handles the 12 joints.

**Hand camera:** The wrist-mounted `hand_color_image` and `hand_depth` cameras are the highest-resolution sensors on the robot and have the best depth quality (structured light at close range). For scene graph construction in tight indoor environments, they are often the best choice, which is why `generate_strayscanner_dir(camera="hand_color")` defaults to them.

!!! tip "Where this shows up in the repo"
    - `spot_semantic_mapping/spot/collection/capture.py` — streams and saves all camera sources with per-frame metadata
    - `spot_semantic_mapping/spot/dataset.py` — `SpotDataset` loads intrinsics, parses per-camera SE(3) poses, handles confidence maps
    - `core/geometry.py` — `depth_to_pointcloud()`, `transform_pointcloud()`
    - `core/io.py` — `load_poses()` returns `(T, 4, 4)` arrays
    - `spot_semantic_mapping/spot/telemetry.py` — publishes camera and pose data to ROS2 topics in real time

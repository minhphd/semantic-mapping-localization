# Robot Hardware

Before writing code that processes sensor data, it helps to know where that data comes from. This page covers the physical sensors this repo works with and how their outputs are represented in code.

---

## Sensors

A **sensor** is a device that measures something about the physical world and converts it to a digital signal. Robots carry multiple sensors because no single sensor captures everything.

```
Boston Dynamics Spot sensor suite:

  ┌─────────────────────────────┐
  │        Top camera           │  ← fisheye RGB
  │                             │
  │  Front-left    Front-right  │  ← stereo depth cameras
  │     depth          depth    │
  │                             │
  │   Left side    Right side   │  ← depth cameras
  │                             │
  │         IMU                 │  ← accelerometer + gyroscope
  └─────────────────────────────┘
```

| Sensor | Measures | Output |
|--------|---------|--------|
| RGB camera | Color and texture | `(H, W, 3)` image |
| Depth camera | Distance per pixel | `(H, W)` depth map |
| IMU | Acceleration + rotation rate | 6-D vector |
| LiDAR | 3D distances via laser | point cloud |
| Wheel encoders | How far wheels have turned | odometry |

---

## Depth Cameras

A **depth camera** captures a depth image — every pixel stores a distance value in meters instead of a color:

```
RGB image (colors):        Depth image (distances):

  ┌──────────────┐           ┌──────────────┐
  │ 120 080 060  │           │ 0.0  0.0 0.0 │
  │ 110 075 055  │           │ 0.0  1.7 1.8 │  ← object at 1.7 m
  │ 100 070 060  │           │ 0.0  1.7 1.8 │
  └──────────────┘           └──────────────┘
    (R, G, B per pixel)        (meters per pixel)
```

**How depth cameras work (structured light):** project an infrared dot pattern onto the scene. A second IR camera sees the distorted pattern — the distortion encodes depth.

**iPhone LiDAR:** fires near-infrared laser pulses and measures the round-trip time. Distance = (time × speed of light) / 2. Accurate to ~1 cm out to 5 m.

### Intrinsic Matrix K

The **intrinsic matrix** describes the geometry of the lens — it tells you how a 3D point in front of the camera maps to a pixel on the image:

\[
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\]

| Symbol | Meaning |
|--------|---------|
| \(f_x, f_y\) | Focal length in pixels |
| \(c_x, c_y\) | Principal point — where the optical axis hits the sensor (usually image center) |

```
Projection (3D → pixel):

  3D point P = [X, Y, Z]
       │
       ▼  divide by Z, apply K
  pixel u = f_x * (X/Z) + c_x
  pixel v = f_y * (Y/Z) + c_y
```

```
Back-projection (pixel + depth → 3D):

  pixel (u, v) + depth d
       │
       ▼  apply K⁻¹, multiply by d
  X = (u - c_x) * d / f_x
  Y = (v - c_y) * d / f_y
  Z = d
```

This back-projection is how every depth frame becomes a point cloud in `core/geometry.py`.

---

## Poses and Positions

A **position** is where the robot (or camera) is in space — a 3D point \([x, y, z]\).

A **pose** is position **plus** orientation — where is it and which way is it facing?

In code, a pose is a `(4, 4)` matrix (see [Linear Algebra in Python](linear_algebra.md)):

```
Pose matrix layout:

  ┌ r00  r01  r02 │ tx ┐
  │ r10  r11  r12 │ ty │  ← rotation (3×3) + translation (3×1)
  │ r20  r21  r22 │ tz │
  └  0    0    0  │  1 ┘

  Translation column [tx, ty, tz] = position of the camera in the world
  Rotation block [R] = which way the camera is pointing
```

```python
poses = load_poses(...)    # shape (T, 4, 4)  — one pose per frame

pose_t = poses[42]         # pose at frame 42
position = pose_t[:3, 3]   # [tx, ty, tz] — where the camera is
rotation = pose_t[:3, :3]  # 3×3 rotation matrix — which way it faces
```

### How Poses Are Captured

For iPhone data, Apple's ARKit tracks the device pose in real time using **Visual-Inertial Odometry (VIO)** — fusing camera motion estimation and IMU readings. Each frame comes with a `(4, 4)` pose matrix.

For Spot, the robot's on-board state estimator fuses wheel odometry + IMU + foot contact to produce a body pose, then the camera-to-body calibration gives camera poses.

```
Pose coordinate conventions:

  World frame (fixed):            Camera frame:
       z ↑                             z ──► (forward)
         │                            /
         │_____ y                    x
        /                            │
       x                             ↓ y (image rows go down)

  Camera pose T_WC maps camera → world coordinates.
```

---

## Depth Confidence Maps

Not every depth pixel is reliable. Reflective surfaces (glass, mirrors) confuse structured-light sensors. The iPhone exports a **confidence map** alongside depth:

```
Confidence values:
  0 = low   (discard)
  1 = medium
  2 = high  (keep)

depth   shape: (H, W)  float32 in meters
conf    shape: (H, W)  uint8   values 0/1/2
```

```python
depth = load_depth(t)
conf  = load_conf(t)
depth[conf < 1] = 0.0    # mask out unreliable pixels before back-projecting
```

---

## Boston Dynamics Spot

Spot is a quadruped (four-legged) robot. Key hardware relevant to this repo:

```
Spot top view:

          ┌─────────────────────┐
  front   │  cam   cam   cam   │   ← 3 depth cameras (front face)
  ───────►│                     │
          │     body            │
          │                     │
          │  cam           cam  │   ← 2 depth cameras (sides)
          └─────────────────────┘

  All 5 cameras stream simultaneously.
  6 DoF body pose estimated at ~200 Hz.
```

The `spot/` module in this repo connects to Spot via ROS 2 — the robot's driver publishes camera images and poses as ROS topics that our code subscribes to.

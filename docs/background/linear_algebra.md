# Linear Algebra in Python

You have probably seen vectors and matrices in math class. In robotics (or any Machine Learning and Deep Learning) code they show up *constantly*, and I cannot emphasize that enough. 

Forget scalars — they are boring, the most primitive units you will be dealing with from now on are **multi-dimensional** vectors. Why? You might ask. Because the world is **multi-dimensional**. Objects move in directions that cannot be explained with a single number. It is easy to say whether a car is moving north or south, but what if it is moving diagonally at some angle? Now you need 2 numbers. Robotics researchers deal with hardware that has real-world embodiment, so we use 3-dimensional vectors. Physicists deal with horrors outside of our collective comprehension, so they use 4+ dimensional vectors. Mathematicians use N-dimensional vectors because, well, Math.

I digressed. Here is how different linear algebra objects are represented in Python. We start with vectors.

---

## Vectors

A **vector** is a list of numbers. An N-dimensional vector contains N numbers. In 3D space it represents either a point (a location) or a direction (where something is pointing). This distinction matters. A point `[1, 2, 3]` and a direction `[1, 2, 3]` look identical as NumPy arrays. The difference is purely in how you use them. More on this when we get to homogeneous coordinates, where we finally force Python to tell the two apart.

$$
\mathbf{p} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}
$$

```python
import numpy as np

p = np.array([1.0, 2.0, 0.5])   # a 3D point (or direction, Python doesn't care)
print(p.shape)                   # (3,)
print(p.dtype)                   # float64
```

### Magnitude and Unit Vectors

The **magnitude** (or **norm**) of a vector is its length — the distance from the origin to the tip of the arrow. A **unit vector** is any vector scaled to have magnitude exactly 1. You will normalize vectors constantly: when computing angles, when comparing directions, when feeding data into anything that cares about orientation but not scale.

$$
\|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2 + v_z^2}, \qquad \hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}
$$

```python
v = np.array([3.0, 4.0, 0.0])

mag   = np.linalg.norm(v)          # 5.0  (3-4-5 triangle, classic)
v_hat = v / mag                    # [0.6, 0.8, 0.0]  — unit vector

# For large arrays of vectors, normalize all rows at once:
pts = np.random.randn(1000, 3)
norms = np.linalg.norm(pts, axis=1, keepdims=True)   # shape (1000, 1)
unit_pts = pts / norms                               # shape (1000, 3)
# keepdims=True is important — without it, norms is (1000,) and division breaks
```

One common trap: if a zero vector sneaks into your data, which it will eventually, dividing by its norm gives you `nan`. Defensive code:

```python
norms = np.linalg.norm(pts, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)   # replace 0 with 1 to avoid divide-by-zero
unit_pts = pts / norms
```

### Dot Product

The dot product measures **how aligned** two vectors are. If you squish one vector onto the other, the dot product is (roughly) how much of it survives. Perpendicular vectors contribute nothing to each other — dot product is zero. Parallel vectors contribute maximally.

$$
\mathbf{a} \cdot \mathbf{b} = a_x b_x + a_y b_y + a_z b_z = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta
$$

```python
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
np.dot(a, b)    # 0.0 — perpendicular, nothing in common

a = np.array([1.0, 0.0, 0.0])
b = np.array([1.0, 0.0, 0.0])
np.dot(a, b)    # 1.0 — identical unit vectors, fully aligned
```

```
θ = 0°   → dot = ‖a‖‖b‖   (same direction, maximum)
θ = 90°  → dot = 0          (perpendicular, zero contribution)
θ = 180° → dot = −‖a‖‖b‖  (opposite directions, negative)
```

**Recovering the angle** between two vectors:

$$
\theta = \arccos\!\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}\right)
$$

```python
def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return angle in radians between two vectors."""
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)   # guard against float rounding past ±1
    return float(np.arccos(cos_theta))

angle_between(np.array([1, 0, 0]), np.array([0, 1, 0]))  # 1.5707... ≈ π/2
```

**Dot product on batches** — again, no loops:

```python
a = np.random.randn(1000, 3)
b = np.random.randn(1000, 3)

# Dot product of each pair of rows
dots = np.sum(a * b, axis=1)      # shape (1000,)
# Or equivalently:
dots = np.einsum("ij,ij->i", a, b)
```

### Cross Product

The cross product only exists in 3D (and technically 7D, but if you are reading this you are not there yet). It produces a new vector **perpendicular to both inputs**. The magnitude equals the area of the parallelogram spanned by the two vectors.

$$
\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_y b_z - a_z b_y \\ a_z b_x - a_x b_z \\ a_x b_y - a_y b_x \end{bmatrix}
$$

```python
a = np.array([1.0, 0.0, 0.0])   # x-axis
b = np.array([0.0, 1.0, 0.0])   # y-axis
np.cross(a, b)                   # [0, 0, 1] — z-axis, perpendicular to both
np.cross(b, a)                   # [0, 0, -1] — order matters! cross product is anti-commutative
```

```
Right-hand rule:
  Point fingers along a, curl them toward b → thumb points in direction of a × b

  a = x-axis [1,0,0]
  b = y-axis [0,1,0]
  a × b = z-axis [0,0,1]  ✓
```

Use cases: computing surface normals from two edges of a triangle, computing torque, finding the rotation axis between two orientations.

---

## Matrices and Transforms

A **matrix** represents a transformation. Feed a vector in, get a transformed vector out. The most important transformation in robotics is the **4×4 rigid body transform**. It encodes both rotation and translation together in one clean package, so you never have to remember whether to apply the rotation or the translation first (the matrix does it in the right order automatically).

### Matrix–Vector Multiply

$$
\mathbf{y} = M\mathbf{x}
$$

```python
M = np.array([[0, -1, 0],   # 90° rotation around z-axis
              [1,  0, 0],
              [0,  0, 1]])

x = np.array([1.0, 0.0, 0.0])   # pointing along x
y = M @ x                        # [0, 1, 0] — now pointing along y
```

```
Matrix multiply rotates the vector:

  x = [1, 0, 0] ──► M ──► y = [0, 1, 0]

  Before:          After:
     y                y
     ↑             x→ ↑
     │                │
     └──── x→         └────
  (x-axis is separate)   (x-axis became y-axis)
```

**Applying a matrix to many vectors at once:**

```python
pts = np.random.randn(1000, 3)    # 1000 points

# Option 1: transpose trick
pts_rot = (M @ pts.T).T           # (3,3) @ (3,1000) → (3,1000), then back to (1000,3)

# Option 2: einsum — more explicit
pts_rot = np.einsum("ij,kj->ki", M, pts)   # "for each point k, compute M @ pts[k]"
```

### Rotation Matrices

A rotation matrix `R` is a 3×3 matrix that rotates vectors without stretching or reflecting them. It has two key properties:

- **Orthogonality**: `R.T @ R == I` (columns are mutually perpendicular unit vectors)
- **Determinant = +1**: which distinguishes a rotation from a reflection

Elementary rotations around each axis:

$$
R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad
R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}, \quad
R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}
$$

```python
def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

R = rot_z(np.pi / 2)

# Verify orthogonality
np.allclose(R.T @ R, np.eye(3))    # True
np.allclose(np.linalg.det(R), 1.0) # True

# The inverse of a rotation matrix is just its transpose — O(N²) copy vs O(N³) inversion
R_inv = R.T
```

`R.T` is the inverse because transposing swaps "rotate by θ" to "rotate by -θ". This is a massive computational win over calling `np.linalg.inv`, and you should always use it when you know `R` is a proper rotation matrix.

### Chaining Rotations

Multiple rotations compose by matrix multiplication. Order matters, rotations do not commute.

```python
# Rotate 45° around z, then 30° around y
R_combined = rot_y(np.radians(30)) @ rot_z(np.radians(45))
# ⚠️ This is NOT the same as:
R_wrong = rot_z(np.radians(45)) @ rot_y(np.radians(30))
```

```
Think of it as function composition: R_combined @ v = R_y(R_z(v))
The rightmost matrix is applied first.
```

---

## Homogeneous Coordinates and the 4×4 Transform

Here is a problem. A 3×3 rotation matrix can rotate a vector, but it cannot translate it. You cannot encode "move 5 units along x" as a 3×3 matrix multiply. If you try, you will find that multiplying any matrix by the zero vector always gives zero, so any transform expressed as a 3×3 matrix always maps the origin to the origin. That rules out translation.

The fix is **homogeneous coordinates**: append a `1` to every point, making it 4D. Now translation fits neatly into a 4×4 matrix, and both rotation and translation happen in one `@`.

$$
\tilde{\mathbf{p}} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} \quad \text{(a point)} \qquad
\tilde{\mathbf{d}} = \begin{bmatrix} x \\ y \\ z \\ 0 \end{bmatrix} \quad \text{(a direction)}
$$

The `1` vs `0` in the fourth slot is the formal way to say "point" vs "direction". Directions are not affected by translation (which is correct — the direction "north" doesn't shift when you move your location).

$$
T = \begin{bmatrix} R_{3\times3} & \mathbf{t}_{3\times1} \\ \mathbf{0}^\top & 1 \end{bmatrix},
\qquad
T\tilde{\mathbf{p}} = \begin{bmatrix} R\mathbf{p} + \mathbf{t} \\ 1 \end{bmatrix}
$$

```
4×4 layout:

  ┌ r00  r01  r02 │ tx ┐
  │ r10  r11  r12 │ ty │  ← top-left 3×3  = rotation
  │ r20  r21  r22 │ tz │     top-right 3×1 = translation
  └  0    0    0  │  1 ┘
```

```python
# Build a transform: rotate 90° around z, then translate by [1, 2, 0]
T = np.eye(4)
T[:3, :3] = rot_z(np.pi / 2)
T[:3,  3] = np.array([1.0, 2.0, 0.0])

# Apply to a single point
p_hom = np.array([1.0, 0.0, 0.0, 1.0])   # homogeneous point
p_out = T @ p_hom
p_out[:3]    # [1.0, 3.0, 0.0]
# Check: rot_z(90°) maps [1,0,0] → [0,1,0], then translate by [1,2,0] → [1, 3, 0] ✓

# Apply to a direction (w=0 — translation has no effect)
d_hom = np.array([1.0, 0.0, 0.0, 0.0])
d_out = T @ d_hom
d_out[:3]    # [0.0, 1.0, 0.0]  — only rotated, not translated ✓
```

**Apply a 4×4 transform to a batch of points:**

```python
pts = np.random.randn(1000, 3)

# Append a column of 1s to make (1000, 4)
pts_h = np.hstack([pts, np.ones((1000, 1))])   # shape (1000, 4)

# Transform all at once
pts_out = (T @ pts_h.T).T        # (4,4) @ (4,1000) → (4,1000), transpose → (1000,4)
pts_out = pts_out[:, :3]         # drop the homogeneous column → (1000, 3)
```

This is exactly how camera poses are stored in `core/io.py` — every pose is a `(4, 4)` NumPy array.

---

## Chaining Transforms

You have sensors on your robot. Many sensors. And every single one of them woke up today believing that it is the center of the universe.Your camera thinks the world starts at its lens. Your LiDAR thinks the world starts at its spinning mirror. Your IMU thinks the world starts wherever it was bolted to the chassis. They are all correct, from their own perspective, and they are all useless to each other until you reconcile this. A depth pixel at [0, 0, 2.5] in camera space means nothing to a motion planner that thinks in world coordinates. A LiDAR point at [1.2, 0.0, 0.4] in LiDAR space is in a completely different location than [1.2, 0.0, 0.4] in camera space, even though the numbers look identical.This is the problem coordinate transforms exist to solve. 

![gimbal-lock](./assets/gimbal-lock.png)

The solution is to pick one frame as the "world", the shared reference everyone agrees on, and express every sensor's output in that frame. You do that by multiplying through a chain of 4×4 matrices until the coordinate system matches the one you want. Here is the mental model: each 4×4 matrix is a bridge between two frames. To get from frame A to frame C, you walk the bridges.

$$
T_{WC} = T_{WB} \cdot T_{BC}
$$

*"Camera in world" = "Body in world" × "Camera on body"*

```python
T_WB = load_body_pose()      # robot body expressed in world frame
T_BC = calibration_offset()  # camera expressed in robot body frame
T_WC = T_WB @ T_BC           # camera expressed in world frame

# Inverse: go the other direction
T_CW = np.linalg.inv(T_WC)   # world expressed in camera frame

# If T is a pure rotation + translation (no scale, no shear),
# you can invert it more efficiently:
R = T_WC[:3, :3]
t = T_WC[:3,  3]
T_CW_fast = np.eye(4)
T_CW_fast[:3, :3] = R.T
T_CW_fast[:3,  3] = -(R.T @ t)
```

```
Sensor chain:

  World ──T_WB──► Body ──T_BC──► Camera
    │                               │
    └──────── T_WC = T_WB @ T_BC ──┘

To go backwards (Camera → World), take the inverse.
```

**Common mistake:** subscript order. `T_WC` means "the pose of C expressed in W", equivalently "the transform that takes a point in C and returns it in W". Not the other way around. This convention trips up everyone at least once. When in doubt, write it out: `p_world = T_WC @ p_camera`.

---

## Euler Angles and Gimbal Lock (Why We Don't Always Use Them)

Euler angles represent a rotation as three sequential rotations around the axes — often called roll, pitch, yaw. They are intuitive, compact (just 3 numbers), and used heavily in aviation and robotics interfaces.

```python
def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Extrinsic XYZ Euler angles (roll around X, then pitch around Y, then yaw around Z).
    Returns a 3×3 rotation matrix.
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
```

The problem is **gimbal lock**: when the middle rotation reaches ±90°, two of the three axes align, and you lose a degree of freedom. The robot can no longer rotate in one particular direction without a complicated multi-axis maneuver. This is not a software bug, it is a geometric property of how Euler angles work. It caused problems on Apollo 11. For smooth interpolation and animation, use quaternions.

---

## Quaternions (The Brief Version)

A **quaternion** is a 4-number representation of rotation: `[w, x, y, z]` (or `[x, y, z, w]` depending on which library is annoying you today). They do not suffer from gimbal lock and are much better for interpolating between rotations.

```python
# Using scipy — the practical choice for quaternions in Python
from scipy.spatial.transform import Rotation

# From Euler angles (roll, pitch, yaw in radians)
r = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])

# Convert to different representations
R_matrix = r.as_matrix()    # 3×3 rotation matrix
quat     = r.as_quat()      # [x, y, z, w] convention (scipy uses xyzw)
rotvec   = r.as_rotvec()    # axis-angle: direction = axis, magnitude = angle

# Compose rotations
r1 = Rotation.from_euler("z", np.pi / 2)
r2 = Rotation.from_euler("y", np.pi / 4)
r_combined = r2 * r1         # r1 applied first, then r2

# Interpolate between two rotations (SLERP — spherical linear interpolation)
r_interp = Rotation.concatenate([r1, r2])
times = np.linspace(0, 1, 10)
# Scipy Slerp
from scipy.spatial.transform import Slerp
slerp = Slerp([0, 1], r_interp)
r_mid = slerp(0.5)   # halfway between r1 and r2
```

The practical rule: **store and compute rotations as rotation matrices or quaternions. Convert to Euler only for display or human-readable output.** This is the standard in robotic and aerospace engineering. Writing a navigation software using Euler system is the same thing as writing it with the imperial system, a mistake that caused Nasa to lose its Mars Orbiter.

---

## Eigenvalues and Eigenvectors

An eigenvector of a matrix `M` is a vector that, when transformed by `M`, only gets scaled — not rotated. The scale factor is the corresponding eigenvalue.

$$
M\mathbf{v} = \lambda\mathbf{v}
$$

```python
M = np.array([[3.0, 1.0],
              [1.0, 3.0]])

eigenvalues, eigenvectors = np.linalg.eig(M)
# eigenvalues:   [4.0, 2.0]
# eigenvectors:  columns of the returned matrix

v0 = eigenvectors[:, 0]     # first eigenvector
lam0 = eigenvalues[0]

np.allclose(M @ v0, lam0 * v0)    # True — confirms the definition
```

Where this shows up: **Principal Component Analysis (PCA)** — the principal axes of a point cloud are the eigenvectors of its covariance matrix, and the eigenvalues tell you how spread out the data is along each axis. Very useful for fitting bounding boxes to 3D object point clouds.

```python
def pca_axes(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns principal axes and variances for a point cloud.

    Args:
        pts: (N, 3) array of 3D points.
    Returns:
        axes: (3, 3) — each row is a principal axis (sorted by variance descending)
        variances: (3,) — variance along each axis
    """
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / len(pts)       # 3×3 covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric matrices: faster + stable
    order = np.argsort(eigenvalues)[::-1]             # sort descending
    return eigenvectors[:, order].T, eigenvalues[order]
```

---

## Singular Value Decomposition (SVD)

SVD decomposes any matrix `M` into three matrices:

$$
M = U \Sigma V^\top
$$

Where `U` and `V` are rotation matrices, and `Σ` is a diagonal matrix of non-negative scale factors (the singular values). Every matrix, no matter how weird, can be decomposed this way. It is the Swiss army knife of numerical linear algebra.

```python
M = np.random.randn(100, 10)

U, S, Vt = np.linalg.svd(M, full_matrices=False)
# U:  (100, 10)  — left singular vectors
# S:  (10,)      — singular values (always non-negative, sorted descending)
# Vt: (10, 10)   — right singular vectors (transposed)

# Reconstruct M from its decomposition
M_reconstructed = U @ np.diag(S) @ Vt
np.allclose(M, M_reconstructed)  # True

# Low-rank approximation: keep only the top k singular values
k = 3
M_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]   # shape (100, 10), but rank 3
```

The best rotation between two aligned point clouds (the Kabsch algorithm) uses SVD. The pseudoinverse (`np.linalg.pinv`) uses SVD. Most of what scipy does internally uses SVD.

---

## NumPy Quick Reference for Robotics

```python
import numpy as np
pts = np.random.randn(1000, 3)   # 1000 points, shape (1000, 3)

# --- Norms and distances ---
np.linalg.norm(v)                          # length of a single vector
np.linalg.norm(pts, axis=1)               # length of each row → (1000,)
np.linalg.norm(pts, axis=1, keepdims=True) # same, but (1000,1) for broadcasting

# --- Apply rotations ---
pts_rot = (R @ pts.T).T                   # 3×3 rotation, result (1000, 3)

# --- Apply 4×4 transform ---
pts_h   = np.hstack([pts, np.ones((1000, 1))])
pts_out = (T @ pts_h.T).T[:, :3]          # (1000, 3)

# --- Geometry ---
centroid = pts.mean(axis=0)               # (3,)
cov      = (pts - centroid).T @ (pts - centroid) / len(pts)   # 3×3 covariance

# --- Decompositions ---
np.linalg.inv(T)                          # matrix inverse (use R.T for pure rotations)
np.linalg.eigh(cov)                       # eigendecomposition (use eigh for symmetric)
U, S, Vt = np.linalg.svd(M, full_matrices=False)  # SVD

# --- Useful checks ---
np.allclose(R.T @ R, np.eye(3))           # is R a valid rotation matrix?
np.allclose(np.linalg.det(R), 1.0)        # no reflection?
np.isfinite(pts).all()                    # any NaN or Inf lurking in your data?
```

!!! tip "Where this shows up in the repo"
    - `core/geometry.py` — every function works on `(N, 3)` point arrays
    - `core/io.py` — `load_poses()` returns `(T, 4, 4)` transform matrices
    - `mapping/tracker.py` — centroids computed as `pcd.mean(axis=0)`
    - `models/detection.py` — PCA bounding box alignment uses `np.linalg.eigh`
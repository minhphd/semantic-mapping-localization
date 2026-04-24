#!/usr/bin/env python3
"""
Generate documentation figures from notebook cells and real data.

Run from the repo root:
    conda run -n semantic_mapping python scripts/export_doc_figures.py

Outputs land in docs/assets/figures/. Figures are referenced by the
tutorial markdown pages in docs/tutorials/.
"""

import os, sys, json
import matplotlib
matplotlib.use('Agg')          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR   = os.path.join(ROOT, 'docs', 'assets', 'figures')
IPHONE    = os.path.join(ROOT, 'data', 'iphone', '3578aa5730')
GRAPH_JSON= os.path.join(ROOT, 'data', 'graph', 'graph_dataset', 'graph.json')

os.makedirs(OUT_DIR, exist_ok=True)

DPI = 150   # increase to 200 for sharper exports


def save(name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close('all')
    print(f'  ✓  {name}')


def section(title: str) -> None:
    print(f'\n── {title}')


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def fig_01_pipeline():
    """End-to-end pipeline architecture (no data required)."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis('off')

    stages = [
        ("RGB-D\nInput",        0.5,  "#AED6F1"),
        ("Detection\n(YOLO)",   2.5,  "#A9DFBF"),
        ("Segmentation\n(SAM)", 4.5,  "#A9DFBF"),
        ("3-D Projection",       6.5,  "#FAD7A0"),
        ("Object\nTracker",     8.5,  "#FAD7A0"),
        ("Captioning\n(BLIP-2)",10.5, "#D2B4DE"),
        ("Relations\n(VLM)",   12.5,  "#D2B4DE"),
        ("Scene\nGraph",       14.5,  "#F1948A"),
    ]

    for label, x, colour in stages:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-0.85, 1.2), 1.7, 1.6, boxstyle="round,pad=0.1",
            facecolor=colour, edgecolor='#555', linewidth=1.5))
        ax.text(x, 2.0, label, ha='center', va='center', fontsize=9, fontweight='bold')

    for i in range(len(stages)-1):
        x0 = stages[i][1] + 0.85
        x1 = stages[i+1][1] - 0.85
        ax.annotate('', xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    legend_patches = [
        mpatches.Patch(color='#AED6F1', label='Input'),
        mpatches.Patch(color='#A9DFBF', label='2-D Perception'),
        mpatches.Patch(color='#FAD7A0', label='3-D Mapping'),
        mpatches.Patch(color='#D2B4DE', label='Semantic Enrichment'),
        mpatches.Patch(color='#F1948A', label='Output'),
    ]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.02),
              ncol=5, fontsize=8, framealpha=0.9)
    ax.set_title('Spot Semantic Mapping — End-to-End Pipeline', fontsize=13, pad=20)
    plt.tight_layout()
    save('nb01_pipeline.png')


def _load_poses(csv_path):
    """Load SE(3) poses from odometry CSV without importing open3d."""
    from scipy.spatial.transform import Rotation
    odometry = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    poses = []
    for line in odometry:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(line[5:]).as_matrix()
        T[:3,  3] = line[2:5]
        poses.append(T)
    return poses


def _load_depth(png_path, conf=None, filter_level=0):
    """Load depth map in metres without importing open3d."""
    from PIL import Image
    depth_m = np.array(Image.open(png_path)).astype(np.float32) / 1000.0
    if conf is not None:
        depth_m[conf < filter_level] = 0.0
    return depth_m


def _load_conf(png_path):
    from PIL import Image
    return np.array(Image.open(png_path)) if os.path.exists(png_path) else None


def fig_01_trajectory():
    """Camera trajectory from iPhone odometry CSV."""
    poses     = _load_poses(os.path.join(IPHONE, 'odometry.csv'))
    positions = np.array([T[:3, 3] for T in poses])

    fig = plt.figure(figsize=(14, 5))

    # 3-D trajectory
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.plot(positions[:,0], positions[:,2], positions[:,1], 'b-', lw=0.8, alpha=0.6)
    ax3d.scatter(*positions[0,  [0,2,1]], color='green', s=80, label='Start', zorder=5)
    ax3d.scatter(*positions[-1, [0,2,1]], color='red',   s=80, label='End',   zorder=5)

    step = max(1, len(poses) // 12)
    for T in poses[::step]:
        o = T[:3, 3]
        for col, ci in [('r',0),('g',1),('b',2)]:
            d = T[:3, ci] * 0.05
            ax3d.quiver(o[0], o[2], o[1], d[0], d[2], d[1],
                        color=col, alpha=0.6, length=0.04, normalize=False)

    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Z (m)'); ax3d.set_zlabel('Y (m)')
    ax3d.set_title('3-D Camera Trajectory with Orientations', fontsize=10)
    ax3d.legend(fontsize=8)

    # Top-down view
    ax2d = fig.add_subplot(122)
    ax2d.plot(positions[:,0], positions[:,2], 'b-', lw=1, alpha=0.7)
    ax2d.scatter(positions[0,0],  positions[0,2],  c='green', s=80, zorder=5, label='Start')
    ax2d.scatter(positions[-1,0], positions[-1,2], c='red',   s=80, zorder=5, label='End')
    ax2d.set_xlabel('X (m)'); ax2d.set_ylabel('Z (m)')
    ax2d.set_title('Top-down Trajectory (XZ plane)', fontsize=10)
    ax2d.set_aspect('equal'); ax2d.legend(fontsize=8); ax2d.grid(True, alpha=0.3)

    dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    ax2d.text(0.02, 0.02, f'Path length: {dist:.2f} m  |  {len(poses)} frames',
              transform=ax2d.transAxes, fontsize=8, color='grey')

    plt.tight_layout()
    save('nb01_trajectory.png')


def fig_01_rgbd_frame():
    """RGB + depth + confidence for frame 0 of the iPhone sequence."""
    from PIL import Image

    frame_idx = 0
    conf  = _load_conf(os.path.join(IPHONE, 'confidence', f'{frame_idx:06d}.png'))
    depth = _load_depth(
        os.path.join(IPHONE, 'depth', f'{frame_idx:06d}.png'),
        conf, filter_level=1,
    )
    rgb = np.array(Image.open(os.path.join(IPHONE, 'rgb_frames', f'{frame_idx:06d}.png')))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB  — frame {frame_idx}\n{rgb.shape[1]}×{rgb.shape[0]} px', fontsize=10)
    axes[0].axis('off')

    im = axes[1].imshow(depth, cmap='plasma', vmin=0, vmax=depth[depth > 0].max())
    axes[1].set_title(
        f'Depth (confidence-filtered)\nvalid: {(depth>0).mean()*100:.1f}%', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='metres', shrink=0.8)

    axes[2].imshow(conf, cmap='viridis')
    axes[2].set_title('Confidence map\n(0 = low  ·  2 = high)', fontsize=10)
    axes[2].axis('off')

    plt.suptitle('RGB-D Frame Components', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb01_rgbd_frame.png')


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Detection & Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def fig_02_detection_pipeline():
    """Detection pipeline overview — four conceptual stages."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles       = ['① Raw RGB frame', '② YOLO bounding boxes',
                    '③ SAM2 masks',    '④ Filtered crops']
    colors       = ['#AED6F1', '#A9DFBF', '#FAD7A0', '#F1948A']
    descriptions = [
        'H×W×3 uint8\nCamera output',
        'N × (x1,y1,x2,y2)\n+ class + confidence',
        'N × (H×W bool)\nPixel-level binary masks',
        'M ≤ N accepted\n(area · score · structure)',
    ]

    for ax, title, col, desc in zip(axes, titles, colors, descriptions):
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.25), 0.9, 0.5, boxstyle='round,pad=0.04',
            facecolor=col, edgecolor='#555', linewidth=1.5,
            transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.52, title, ha='center', va='center',
                transform=ax.transAxes, fontsize=9, fontweight='bold')
        ax.text(0.5, 0.33, desc, ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='#333')
        ax.axis('off')

    # Arrows between panels
    for i in range(3):
        axes[i].annotate('', xy=(1.08, 0.5), xytext=(1.02, 0.5),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle='->', lw=2, color='#444'))

    plt.suptitle('Per-Frame Detection & Segmentation Pipeline', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb02_detection_pipeline.png')


def fig_02_yolo_sample():
    """Show a sample iPhone RGB frame with simulated bounding-box annotations."""
    from PIL import Image

    rgb_path = os.path.join(IPHONE, 'rgb_frames', '000050.png')
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(IPHONE, 'rgb_frames', '000000.png')
    rgb = np.array(Image.open(rgb_path))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb)
    ax.set_title('Sample iPhone RGB frame — input to YOLO detector', fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    save('nb02_sample_frame.png')


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Scene Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def fig_03_projection_pipeline():
    """3D projection: depth image → binary mask → point cloud."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Synthetic depth image ────────────────────────────────────────────────
    ax = axes[0]; ax.axis('off')
    ax.set_title('① Depth image D  (256×192)', fontsize=10, fontweight='bold')
    np.random.seed(0)
    D = np.ones((12, 16)) * 2.0
    D[4:9, 5:12] = 1.2
    D += np.random.randn(12, 16) * 0.05
    im = ax.imshow(D, cmap='plasma', vmin=0.5, vmax=3.0, aspect='auto')
    plt.colorbar(im, ax=ax, label='depth (m)', shrink=0.7)

    # ── Binary mask ──────────────────────────────────────────────────────────
    ax = axes[1]; ax.axis('off')
    ax.set_title('② SAM2 mask M  (object pixels only)', fontsize=10, fontweight='bold')
    mask = np.zeros((12, 16), dtype=bool)
    mask[4:9, 5:12] = True
    display = np.zeros((12, 16, 4))
    display[mask]  = [0.2, 0.8, 0.3, 0.8]
    display[~mask] = [0.9, 0.9, 0.9, 0.3]
    ax.imshow(display, aspect='auto')
    ax.text(0.5, -0.08, 'White = masked in · Grey = masked out',
            ha='center', transform=ax.transAxes, fontsize=8, color='#555')

    # ── 3-D point cloud ──────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title('③ 3-D point cloud P  (back-projected)', fontsize=10, fontweight='bold')
    ys, xs = np.where(mask)
    fx, fy, cx, cy = 211.0, 211.0, 128.0, 96.0
    z = D[ys, xs]
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    scatter = ax.scatter(X, -Y, c=z, cmap='plasma', s=20, vmin=1.0, vmax=1.4)
    plt.colorbar(scatter, ax=ax, label='Z (m)')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.suptitle('Depth → Mask → 3-D Point Cloud', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb03_projection_pipeline.png')


def fig_03_tracking():
    """Illustration of geometric + semantic tracking similarity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── Geometric overlap ────────────────────────────────────────────────────
    ax = axes[0]; np.random.seed(0)
    A = np.random.randn(60, 2) * 0.15 + [0.0, 0.0]
    B = np.random.randn(60, 2) * 0.15 + [0.12, 0.08]
    C = np.random.randn(60, 2) * 0.15 + [1.0, 0.5]
    ax.scatter(*A.T, c='steelblue', alpha=0.4, s=15, label='Existing object')
    ax.scatter(*B.T, c='darkorange',alpha=0.4, s=15, label='New det — HIGH overlap')
    ax.scatter(*C.T, c='seagreen',  alpha=0.4, s=15, label='New det — LOW overlap')
    ax.set_title('Geometric Overlap  $s_{geo}$', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.set_aspect('equal')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.grid(True, alpha=0.3)

    # ── Semantic similarity ──────────────────────────────────────────────────
    ax = axes[1]; np.random.seed(1)
    D_feat = 16
    emb_base  = np.random.randn(D_feat); emb_base  /= np.linalg.norm(emb_base)
    emb_close = emb_base + np.random.randn(D_feat)*0.15
    emb_close /= np.linalg.norm(emb_close)
    emb_far   = np.random.randn(D_feat); emb_far /= np.linalg.norm(emb_far)

    cos_close = float(emb_base @ emb_close)
    cos_far   = float(emb_base @ emb_far)

    bar_colors = ['steelblue', 'darkorange', 'seagreen']
    bars = ax.bar(['Existing\nobject', 'New det\n(same class)', 'New det\n(diff class)'],
                  [1.0, cos_close, cos_far], color=bar_colors, alpha=0.8, edgecolor='#444')
    ax.axhline(0.65, color='red', ls='--', lw=1.2, label='match_threshold')
    ax.set_ylim(-0.1, 1.15)
    ax.set_ylabel('Cosine similarity'); ax.legend(fontsize=8)
    ax.set_title('Semantic Similarity  $s_{sem}$  (CLIP)', fontsize=10, fontweight='bold')
    for bar, val in zip(bars, [1.0, cos_close, cos_far]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.2f}',
                ha='center', fontsize=9)

    # ── Combined score ───────────────────────────────────────────────────────
    ax = axes[2]
    w_geo, w_sem = 0.5, 0.5
    match_thr = 0.65

    geo_scores  = [1.0, 0.18, 0.72, 0.05]
    sem_scores  = [1.0, 0.91, 0.22, 0.15]
    labels      = ['Same obj\n(ideal)', 'Moved\nobject', 'Similar\nappear.', 'Different\nobj']
    combined    = [w_geo*g + w_sem*s for g, s in zip(geo_scores, sem_scores)]

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w,   geo_scores, w, label='$s_{geo}$',      color='#AED6F1', edgecolor='#555')
    ax.bar(x,       sem_scores, w, label='$s_{sem}$',      color='#A9DFBF', edgecolor='#555')
    ax.bar(x + w,   combined,   w, label='Combined score', color='#FAD7A0', edgecolor='#555')
    ax.axhline(match_thr, color='red', ls='--', lw=1.2, label=f'threshold={match_thr}')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.set_title('Combined Association Score', fontsize=10, fontweight='bold')

    plt.suptitle('ObjectTracker3D — Association Mechanism', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb03_tracking.png')


def fig_03_scene_graph():
    """Top-down spatial layout + relation summary from pre-built graph."""
    with open(GRAPH_JSON) as f:
        g = json.load(f)

    nodes = g['nodes']
    edges = g['edges']

    # Build a lookup by oid
    node_map = {n['oid']: n for n in nodes}

    # Derive a rough semantic class from the last word of class_name
    def sem_class(n):
        words = n['class_name'].split()
        return words[-1] if words else 'object'

    class_names = sorted(set(sem_class(n) for n in nodes))
    palette     = plt.cm.tab20(np.linspace(0, 1, max(len(class_names), 1)))
    cmap        = {c: palette[i] for i, c in enumerate(class_names)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: top-down layout ─────────────────────────────────────────────────
    ax = axes[0]
    for n in nodes:
        pos  = n['position']
        col  = cmap[sem_class(n)]
        ax.scatter(pos[0], pos[2], c=[col], s=80, zorder=5)
        ax.text(pos[0]+0.04, pos[2]+0.04, n['class_name'][:14],
                fontsize=5.5, alpha=0.75)

    drawn = 0
    for e in edges:
        src = node_map.get(e['src_id'])
        dst = node_map.get(e['dst_id'])
        if src and dst and drawn < 200:
            ax.annotate(
                '', xy=(dst['position'][0], dst['position'][2]),
                xytext=(src['position'][0], src['position'][2]),
                arrowprops=dict(arrowstyle='->', lw=0.6, color='grey', alpha=0.35))
            drawn += 1

    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_title(f'Top-down Scene Layout  ({len(nodes)} objects)', fontsize=11, fontweight='bold')
    legend_patches = [mpatches.Patch(color=cmap[c], label=c)
                      for c in list(class_names)[:12]]
    ax.legend(handles=legend_patches, fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    # ── Right: relation type bar chart ────────────────────────────────────────
    from collections import Counter
    ax = axes[1]
    rel_counts = Counter(e['relation'] for e in edges)
    rels   = [r for r, _ in rel_counts.most_common(10)]
    counts = [rel_counts[r] for r in rels]
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(rels)))
    ax.barh(rels[::-1], counts[::-1], color=colors_bar[::-1], edgecolor='#444', alpha=0.85)
    ax.set_xlabel('Number of edges'); ax.set_title('Relation Types', fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    for i, (r, c) in enumerate(zip(rels[::-1], counts[::-1])):
        ax.text(c + 2, i, str(c), va='center', fontsize=9)

    plt.suptitle('Pre-built Scene Graph — Miller Street Lab', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save('nb03_scene_graph.png')


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 04 — Localization (VPR)
# ─────────────────────────────────────────────────────────────────────────────

def fig_04_vpr_concept():
    """VPR concept: robot trajectory with DB frames and a query."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    np.random.seed(7)

    # ── Robot trajectory + DB frames + query ──────────────────────────────────
    ax = axes[0]
    t      = np.linspace(0, 2*np.pi, 200)
    traj_x = np.cos(t) + 0.05*np.random.randn(200)
    traj_y = np.sin(t)*0.6 + 0.05*np.random.randn(200)
    ax.plot(traj_x, traj_y, 'b-', lw=1.5, alpha=0.5, label='Robot path')

    db_idx = np.arange(0, 200, 10)
    ax.scatter(traj_x[db_idx], traj_y[db_idx], c='steelblue', s=50, zorder=5, label='DB frames')

    q_idx = 37
    ax.scatter(traj_x[q_idx], traj_y[q_idx], c='red', s=120, marker='*', zorder=6, label='Query')
    ax.set_title('Database + Query', fontsize=10, fontweight='bold')
    ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Embedding retrieval ───────────────────────────────────────────────────
    ax = axes[1]
    np.random.seed(42)
    emb_q   = np.random.randn(32); emb_q  /= np.linalg.norm(emb_q)
    db_embs = np.random.randn(20, 32)
    db_embs /= np.linalg.norm(db_embs, axis=1, keepdims=True)
    scores  = db_embs @ emb_q
    top_k   = np.argsort(-scores)[:5]

    ax.bar(range(20), scores, color='#AED6F1', edgecolor='#555', alpha=0.8)
    ax.bar(top_k, scores[top_k], color='darkorange', edgecolor='#555', alpha=0.9, label='Top-5')
    ax.axhline(scores[top_k[-1]], color='red', ls='--', lw=1, label='k=5 threshold')
    ax.set_xlabel('DB frame index'); ax.set_ylabel('Cosine similarity')
    ax.set_title('Retrieval Scores', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    # ── Spatial window → subgraph ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(traj_x, traj_y, 'b-', lw=1.5, alpha=0.3)
    ax.scatter(traj_x[db_idx], traj_y[db_idx], c='steelblue', s=40, alpha=0.5)
    ax.scatter(traj_x[q_idx],  traj_y[q_idx],  c='red', s=120, marker='*', zorder=6, label='Query')

    for k in top_k:
        xi, yi = traj_x[db_idx[k]], traj_y[db_idx[k]]
        ax.scatter(xi, yi, c='darkorange', s=100, zorder=5)
        circle = plt.Circle((xi, yi), 0.25, color='orange', alpha=0.15)
        ax.add_patch(circle)

    ax.set_title('Top-k Matches + Spatial Window', fontsize=10, fontweight='bold')
    ax.set_aspect('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, 'Objects inside window → subgraph nodes',
            transform=ax.transAxes, fontsize=8, color='grey')

    plt.suptitle('Visual Place Recognition — End-to-End', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb04_vpr_concept.png')


def fig_04_vlad():
    """VLAD aggregation concept: patch features → codebook → descriptor."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    np.random.seed(42)
    K = 5

    patches1 = np.vstack([
        np.random.randn(20, 2)*0.25 + [ 0.8,  0.5],
        np.random.randn(15, 2)*0.25 + [-0.5,  0.9],
        np.random.randn(10, 2)*0.25 + [ 0.0, -0.8],
    ])
    patches2 = np.vstack([
        np.random.randn(20, 2)*0.25 + [ 0.7,  0.6],
        np.random.randn(15, 2)*0.25 + [-0.4,  0.8],
        np.random.randn(10, 2)*0.25 + [ 0.1, -0.7],
    ])

    # Simple K-means centres
    from sklearn.cluster import KMeans
    all_pts = np.vstack([patches1, patches2])
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(all_pts)
    centres = km.cluster_centers_
    pal = plt.cm.Set1(np.linspace(0, 1, K))

    # ── Patch features ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(*patches1.T, c='steelblue', alpha=0.5, s=20, label='Image 1 patches')
    ax.scatter(*patches2.T, c='darkorange',alpha=0.5, s=20, label='Image 2 patches')
    ax.scatter(*centres.T,  c=[pal[i] for i in range(K)], s=150, marker='X',
               edgecolors='black', linewidths=0.8, label='Codebook', zorder=6)
    ax.set_title('DINOv2 Patch Features\n+ VLAD Codebook (K=5)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Residual assignment ───────────────────────────────────────────────────
    ax = axes[1]
    labels1 = km.predict(patches1)
    for i in range(K):
        pts = patches1[labels1 == i]
        if len(pts):
            ax.scatter(*pts.T, c=[pal[i]]*len(pts), alpha=0.6, s=20)
            ax.scatter(*centres[i], c=[pal[i]], s=150, marker='X',
                       edgecolors='black', linewidths=0.8, zorder=6)
            for p in pts[:3]:
                ax.annotate('', xy=centres[i], xytext=p,
                            arrowprops=dict(arrowstyle='->', lw=0.6, color=pal[i], alpha=0.5))
    ax.set_title('Residual Vectors\n(patch → nearest centroid)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ── VLAD descriptor comparison ────────────────────────────────────────────
    ax = axes[2]
    def vlad_descriptor(patches, km, K):
        labels = km.predict(patches)
        desc = []
        for k in range(K):
            diff = patches[labels==k] - km.cluster_centers_[k]
            desc.append(diff.sum(axis=0) if len(diff) else np.zeros(2))
        v = np.concatenate(desc)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    v1 = vlad_descriptor(patches1, km, K)
    v2 = vlad_descriptor(patches2, km, K)
    x  = np.arange(len(v1))
    ax.bar(x - 0.2, v1, 0.4, label='Image 1 VLAD', color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, v2, 0.4, label='Image 2 VLAD', color='darkorange',alpha=0.8)
    sim = float(v1 @ v2)
    ax.set_title(f'VLAD Descriptors\ncosine similarity = {sim:.3f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Descriptor dimension'); ax.legend(fontsize=8)
    ax.axhline(0, color='black', lw=0.8)

    plt.suptitle('VLAD Aggregation — DINOv2 Patches → Global Descriptor',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb04_vlad.png')


def fig_04_recall():
    """Simulated Recall@k curve illustrating VPR evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    np.random.seed(0)

    # ── Recall@k ──────────────────────────────────────────────────────────────
    ax   = axes[0]
    ks   = [1, 5, 10, 15, 20, 25]
    r_vlad = [0.62, 0.81, 0.88, 0.91, 0.93, 0.95]
    r_gem  = [0.54, 0.74, 0.82, 0.86, 0.89, 0.92]
    r_gap  = [0.48, 0.68, 0.77, 0.82, 0.85, 0.87]

    ax.plot(ks, r_vlad, 'o-', label='VLAD (ours)',  color='steelblue',  lw=2)
    ax.plot(ks, r_gem,  's--',label='GeM',          color='darkorange', lw=2)
    ax.plot(ks, r_gap,  '^:', label='GAP (baseline)',color='seagreen',   lw=2)
    ax.set_xlabel('k'); ax.set_ylabel('Recall@k')
    ax.set_title('Recall@k — VPR Aggregation Methods', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4); ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)

    # ── Localization error CDF ─────────────────────────────────────────────────
    ax    = axes[1]
    errs  = np.abs(np.random.randn(500) * 0.8 + 0.5)
    errs  = np.clip(errs, 0, 5)
    vals  = np.sort(errs)
    cdf   = np.arange(1, len(vals)+1) / len(vals)
    ax.plot(vals, cdf, lw=2, color='steelblue')
    ax.axvline(np.median(errs), color='red', ls='--', lw=1.5,
               label=f'Median: {np.median(errs):.2f} m')
    ax.set_xlabel('Localisation error (m)'); ax.set_ylabel('CDF')
    ax.set_title('Localisation Error CDF', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    plt.suptitle('VPR Evaluation Metrics', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save('nb04_recall.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

FIGURES = [
    # (function,            description)
    (fig_01_pipeline,          'NB01 — pipeline architecture diagram'),
    (fig_01_trajectory,        'NB01 — camera trajectory from iPhone odometry'),
    (fig_01_rgbd_frame,        'NB01 — RGB + depth + confidence frame'),
    (fig_02_detection_pipeline,'NB02 — detection pipeline overview'),
    (fig_02_yolo_sample,       'NB02 — sample iPhone RGB frame'),
    (fig_03_projection_pipeline,'NB03 — 3D projection steps'),
    (fig_03_tracking,          'NB03 — ObjectTracker3D association'),
    (fig_03_scene_graph,       'NB03 — pre-built scene graph visualization'),
    (fig_04_vpr_concept,       'NB04 — VPR concept end-to-end'),
    (fig_04_vlad,              'NB04 — VLAD aggregation'),
    (fig_04_recall,            'NB04 — Recall@k and error CDF'),
]

if __name__ == '__main__':
    print(f'Exporting {len(FIGURES)} figures → {OUT_DIR}\n')
    failed = []
    for fn, desc in FIGURES:
        section(desc)
        try:
            fn()
        except Exception as e:
            print(f'  ✗  SKIPPED — {e}')
            failed.append((desc, str(e)))

    print(f'\nDone. {len(FIGURES)-len(failed)}/{len(FIGURES)} figures exported.')
    if failed:
        print('\nSkipped:')
        for d, e in failed:
            print(f'  • {d}: {e}')

"""Geometric depth encoding shared by I1 (depth fusion) and I3 (hi-res mask branch).

All images share one fixed intrinsic matrix K (fx=fy=1910.81, cx=cy=511.5; verified across
train/val/test in seg_project/reports/analysis/gt_overlap_camera.json), so a single K is used
everywhere. Produces, from a normalized metric depth map and its validity mask:
  - height above an estimated support-plane (RANSAC on the point cloud; robust to the camera
    distance differences between train/val/test noted in docs/plan.md §2)
  - surface normals (3 channels, from local depth gradients via the pinhole model)
  - depth discontinuity magnitude (relative neighbor depth jump; marks occlusion boundaries)
  - the validity mask itself

Output channels are standardized using seg_project/datasets/blender/annotations/depth_stats.json
(train-only statistics) so their magnitude matches the RGB preprocessing (std ~50), fixing the
~1/60 scale mismatch diagnosed in docs/history/20261020_算法优化计划_maskrcnn.md §1.
"""
import numpy as np

# Fixed camera intrinsics shared by every image in this dataset (see module docstring).
FX = FY = 1910.8100134752654
CX = CY = 511.5


def backproject(depth_m, fx=FX, fy=FY, cx=CX, cy=CY):
    """Depth map (H,W) in meters -> point cloud (H,W,3) in the camera_cv frame (+Z forward)."""
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth_m
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def fit_support_plane(points, valid, max_iters=200, dist_thresh_m=0.01, rng=None):
    """RANSAC plane n.x + d = 0 (unit n) on valid 3D points. Returns (n, d) or None if too few points.

    All `max_iters` trial triplets are drawn and scored in one vectorized pass (no per-iteration
    Python loop): this cuts the per-image cost from ~3.9s to a few ms, which matters because this
    runs inside the training dataloader. See seg_project/tools/tests/test_depth_geometry.py for
    the same correctness checks (flat-table normal recovery, known box height) re-verified against
    this implementation.
    """
    rng = rng or np.random.default_rng(0)
    pts = points[valid]
    if pts.shape[0] < 50:
        return None
    n_pts = pts.shape[0]
    sample_n = min(n_pts, 20000)
    idx = rng.choice(n_pts, size=sample_n, replace=False) if n_pts > sample_n else np.arange(n_pts)
    sample = pts[idx]  # (S, 3)

    triplets = rng.integers(0, sample.shape[0], size=(max_iters, 3))
    p0, p1, p2 = sample[triplets[:, 0]], sample[triplets[:, 1]], sample[triplets[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)  # (max_iters, 3)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    ok = norms[:, 0] > 1e-9  # reject degenerate (near-collinear or repeated-point) triplets
    normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-9)
    d = -np.einsum('ij,ij->i', normals, p0)  # (max_iters,)

    dist = np.abs(sample @ normals.T + d[None, :])  # (S, max_iters)
    inlier_counts = np.where(ok, (dist < dist_thresh_m).sum(axis=0), -1)
    best = int(np.argmax(inlier_counts))
    if inlier_counts[best] < 0:
        return None
    n, d = normals[best], d[best]

    # Refine with least squares on the inlier set for stability.
    inlier_mask = dist[:, best] < dist_thresh_m
    if inlier_mask.sum() >= 3:
        pts_in = sample[inlier_mask]
        centroid = pts_in.mean(axis=0)
        # full_matrices=False: only need the thin (3x3) V; the default full U (len(pts_in) x
        # len(pts_in), tens of thousands squared) costs seconds for no benefit.
        _, _, vt = np.linalg.svd(pts_in - centroid, full_matrices=False)
        n = vt[-1]
        n = n / (np.linalg.norm(n) + 1e-12)
        d = -n @ centroid
    # Orient the normal to point back toward the camera (z<0 side), so "height above plane" is positive.
    if n[2] > 0:
        n, d = -n, -d
    return n, d


def height_above_plane(points, plane):
    """Signed distance of each point to the plane; plane=(n,d) with unit n oriented toward the camera."""
    n, d = plane
    return points @ n + d


def surface_normals(points, valid):
    """Per-pixel normals from local x/y gradients of the point cloud; invalid-adjacent pixels get 0."""
    gx = np.gradient(points, axis=1)
    gy = np.gradient(points, axis=0)
    n = np.cross(gx, gy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-9)
    # Point normals toward the camera (negative z half-space in camera_cv convention).
    flip = n[..., 2] > 0
    n[flip] *= -1
    n[~valid] = 0
    return n.astype(np.float32)


def depth_discontinuity(depth_m, valid):
    """Relative neighbor depth jump (4-connected), 0 at invalid pixels; large near occlusion edges."""
    d = depth_m.copy()
    d[~valid] = 0
    eps = 1e-3
    dx = np.abs(d - np.roll(d, 1, axis=1)) / (np.abs(d) + eps)
    dy = np.abs(d - np.roll(d, 1, axis=0)) / (np.abs(d) + eps)
    disc = np.maximum(dx, dy)
    disc[~valid] = 0
    return disc.astype(np.float32)


def encode(depth_m, valid, plane=None, rng=None):
    """Full encoding: returns dict with 'height', 'normals' (3ch), 'discontinuity', 'valid', 'plane'.

    `plane` may be passed in (e.g. fit once per image and reused by both I1 and I3 at different
    resolutions) or left None to fit here.
    """
    points = backproject(depth_m)
    if plane is None:
        plane = fit_support_plane(points, valid, rng=rng)
    if plane is None:
        height = np.zeros_like(depth_m, dtype=np.float32)
    else:
        height = height_above_plane(points, plane).astype(np.float32)
        height[~valid] = 0
    normals = surface_normals(points, valid)
    disc = depth_discontinuity(depth_m, valid)
    return {'height': height, 'normals': normals, 'discontinuity': disc,
            'valid': valid.astype(np.float32), 'plane': plane}


def standardize(channels_dict, stats, target_std=50.0):
    """Scale each channel to roughly match RGB's preprocessed magnitude (std ~target_std).

    `stats` is the dict produced by tools/analysis fit on train (see fit_channel_stats below):
    {'height': {'mean':..., 'std':...}, 'normals': {'std':...}, 'discontinuity': {'p99':...}}
    """
    height = (channels_dict['height'] - stats['height']['mean']) / max(stats['height']['std'], 1e-6) * target_std
    normals = channels_dict['normals'] / max(stats['normals']['std'], 1e-6) * target_std
    disc = np.clip(channels_dict['discontinuity'] / max(stats['discontinuity']['p99'], 1e-6), 0, 1) * target_std
    valid = channels_dict['valid']
    height = height * valid
    normals = normals * valid[..., None]
    out = np.concatenate([height[..., None], normals, disc[..., None], valid[..., None]], axis=-1)
    return out.astype(np.float32)  # (H, W, 6): height, nx, ny, nz, discontinuity, valid

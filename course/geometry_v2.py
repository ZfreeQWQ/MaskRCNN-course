"""Label-free camera-space geometry, computed before synchronized augmentation.

Channels: metric Z, height above an estimated support plane, XYZ normal,
relative depth discontinuity, valid depth. Invalid geometry is zero. Plane
estimation is deterministic per image and never uses instance annotations.
"""
import numpy as np


CHANNELS = ("z_m", "height_m", "normal_x", "normal_y", "normal_z", "relative_edge", "valid")
GEOMETRY_VERSION = "v2-center-neighbor-normals-f64-stats"


def estimate_plane(points, seed=0, threshold=.0015, trials=96, max_points=2048):
    points = np.asarray(points, np.float32).reshape(-1, 3)
    if len(points) < 64:
        return None, 0.
    rng = np.random.default_rng(int(seed) % (2**32))
    points = points[rng.choice(len(points), min(max_points, len(points)), replace=False)]
    triples = points[rng.integers(0, len(points), size=(trials, 3))]
    normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
    length = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(length[:, None], 1e-12)
    offsets = -(normals * triples[:, 0]).sum(1)
    distance = np.abs(points @ normals.T + offsets)
    support = (distance < threshold).mean(0)
    support[(length < 1e-9) | (np.abs(normals[:, 2]) < .2)] = 0
    best = int(support.argmax())
    fraction = float(support[best])
    if fraction < .2:
        return None, fraction
    inliers = points[distance[:, best] < threshold]
    center = inliers.mean(0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] > 0:
        normal = -normal
    return np.r_[normal, -normal @ center].astype(np.float32), fraction


def encode_geometry(depth_m, K, seed=0):
    depth = np.asarray(depth_m, np.float32)
    K = np.asarray(K, np.float32).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid supplied camera intrinsics")
    valid = np.isfinite(depth) & (depth > 0) & (depth < 2.)
    z = np.where(valid, depth, 0)
    y, x = np.indices(z.shape, dtype=np.float32)
    points = np.stack([(x - K[0, 2]) * z / K[0, 0],
                       (y - K[1, 2]) * z / K[1, 1], z], -1)
    # Use a fixed spatial subsample for the plane; no full-resolution point list.
    sparse_points, sparse_valid = points[::8, ::8], valid[::8, ::8]
    plane, support = estimate_plane(sparse_points[sparse_valid], seed)
    height = np.zeros_like(z) if plane is None else np.clip(points @ plane[:3] + plane[3], -.25, .25)
    height[~valid] = 0
    normal = np.zeros_like(points)
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(dx, dy)
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    n /= np.maximum(length, 1e-10)
    n[n[..., 2] > 0] *= -1
    good = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:] &
            valid[:-2, 1:-1] & valid[2:, 1:-1] & (length[..., 0] > 1e-10))
    # Do not invent normals spanning an occlusion discontinuity.
    center = z[1:-1, 1:-1]
    good &= ((np.abs(z[1:-1, 2:] - center) < .005) &
             (np.abs(z[1:-1, :-2] - center) < .005) &
             (np.abs(z[2:, 1:-1] - center) < .005) &
             (np.abs(z[:-2, 1:-1] - center) < .005))
    normal[1:-1, 1:-1] = n * good[..., None]
    edge = np.zeros_like(z)
    dd = np.abs(z[:, 1:] - z[:, :-1]) * (valid[:, 1:] & valid[:, :-1])
    edge[:, 1:] = np.maximum(edge[:, 1:], dd / np.maximum(z[:, 1:], 1e-6))
    edge[:, :-1] = np.maximum(edge[:, :-1], dd / np.maximum(z[:, :-1], 1e-6))
    dd = np.abs(z[1:] - z[:-1]) * (valid[1:] & valid[:-1])
    edge[1:] = np.maximum(edge[1:], dd / np.maximum(z[1:], 1e-6))
    edge[:-1] = np.maximum(edge[:-1], dd / np.maximum(z[:-1], 1e-6))
    geometry = np.concatenate([z[..., None], height[..., None], normal,
                               np.clip(edge, 0, .25)[..., None], valid[..., None]], -1).astype(np.float32)
    assert np.isfinite(geometry).all()
    return geometry, dict(plane_valid=plane is not None, plane_support=support,
                         valid_fraction=float(valid.mean()))


def standardize_geometry(geometry, mean, std):
    valid = geometry[..., 6:7] > .5
    normalized = (geometry - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    normalized[..., :6] *= valid
    normalized[..., 6] = valid[..., 0]
    return normalized.astype(np.float32)

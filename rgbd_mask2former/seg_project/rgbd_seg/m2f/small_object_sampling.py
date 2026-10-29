"""I2: small-object-aware point sampling helpers for the matcher/criterion (docs/plan.md §4 I2).

Pure tensor math, no detectron2/mask2former dependency, so it's unit-testable on CPU without
run.sh. Coordinates follow detectron2.projects.point_rend.point_features.point_sample's
convention: (x, y) pairs normalized to [0, 1] x [0, 1], (0, 0) = top-left, (1, 1) = bottom-right.

Used by rgbd_seg.m2f.matcher.SmallObjectMatcher (extra box-local points added to the shared
matching point set, bounded total budget) and rgbd_seg.m2f.criterion.SmallObjectCriterion
(per-matched-pair point_coords partially replaced by box-local points, only for small targets;
total point count per pair is unchanged, so there is no inference-time or training-time compute
cost -- see docs/plan.md I2: "只改训练，不改推理结构").
"""
import torch


def masks_to_normalized_bboxes(masks, threshold=0.5):
    """masks: (N, H, W) float/bool tensor -> bboxes (N, 4) as (x0, y0, x1, y1) in [0, 1].

    Returns (bboxes, valid) where `valid` (N,) is False for masks with no foreground pixel
    (bbox entries for those rows are meaningless zeros).
    """
    n, h, w = masks.shape
    fg = masks > threshold if masks.dtype.is_floating_point else masks.bool()
    valid = fg.flatten(1).any(dim=1)
    # Always float32 regardless of the input mask's dtype (masks.new_zeros would silently
    # inherit e.g. bool, corrupting every box coordinate into {0, 1}).
    bboxes = torch.zeros((n, 4), dtype=torch.float32, device=masks.device)
    if not valid.any():
        return bboxes, valid
    rows = fg.any(dim=2)  # (N, H): any foreground in this row
    cols = fg.any(dim=1)  # (N, W): any foreground in this column
    device = masks.device
    y_idx = torch.arange(h, device=device).float()
    x_idx = torch.arange(w, device=device).float()
    big = float(max(h, w) + 1)
    y0 = torch.where(rows, y_idx.unsqueeze(0).expand(n, -1), torch.full_like(rows, big, dtype=torch.float)).min(dim=1).values
    y1 = torch.where(rows, y_idx.unsqueeze(0).expand(n, -1), torch.full_like(rows, -1, dtype=torch.float)).max(dim=1).values + 1
    x0 = torch.where(cols, x_idx.unsqueeze(0).expand(n, -1), torch.full_like(cols, big, dtype=torch.float)).min(dim=1).values
    x1 = torch.where(cols, x_idx.unsqueeze(0).expand(n, -1), torch.full_like(cols, -1, dtype=torch.float)).max(dim=1).values + 1
    bboxes[:, 0] = x0 / w
    bboxes[:, 1] = y0 / h
    bboxes[:, 2] = x1 / w
    bboxes[:, 3] = y1 / h
    bboxes[~valid] = 0
    return bboxes, valid


def expand_bboxes(bboxes, ratio):
    """Expand each (x0,y0,x1,y1) box by `ratio` about its center, clipped to [0, 1]."""
    cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
    half_w = (bboxes[:, 2] - bboxes[:, 0]) / 2 * (1 + ratio)
    half_h = (bboxes[:, 3] - bboxes[:, 1]) / 2 * (1 + ratio)
    out = torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)
    return out.clamp(0.0, 1.0)


def small_mask_flags(masks, area_threshold_px=1024, threshold=0.5):
    """masks: (N, H, W) -> bool (N,) flagging masks whose foreground pixel count is small.

    area_threshold_px follows the COCO "small" convention (area <= 32**2 = 1024 px^2), applied
    at the mask's current (possibly augmented) resolution -- see rgbd_seg/m2f/mapper.py and
    docs/plan.md §4 I2 for why this is computed post-augmentation rather than from the original
    COCO `area` field.
    """
    fg = masks > threshold if masks.dtype.is_floating_point else masks.bool()
    area = fg.flatten(1).sum(dim=1)
    return area <= area_threshold_px


def sample_points_in_bboxes(bboxes, num_points_per_box, generator=None):
    """bboxes: (M, 4) (x0,y0,x1,y1) in [0,1] -> (M, num_points_per_box, 2) uniform (x,y) points.

    Degenerate (zero-area) boxes still produce points (collapsed to a single location), callers
    should pre-filter with masks_to_normalized_bboxes's `valid` flag if that matters.
    """
    m = bboxes.shape[0]
    device = bboxes.device
    if generator is not None:
        u = torch.rand(m, num_points_per_box, 2, generator=generator).to(device)
    else:
        u = torch.rand(m, num_points_per_box, 2, device=device)
    x0, y0, x1, y1 = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3], bboxes[:, 3:4]
    xs = x0 + u[..., 0] * (x1 - x0).clamp(min=1e-6)
    ys = y0 + u[..., 1] * (y1 - y0).clamp(min=1e-6)
    return torch.stack([xs, ys], dim=-1)


def replace_small_target_points(point_coords, target_masks, fraction, box_expand=0.2,
                                 area_threshold_px=1024):
    """point_coords: (N, P, 2); target_masks: (N, H, W) already indexed to matched pairs.

    Returns new point_coords with the last `fraction` of points replaced by points drawn from
    within each small target's own expanded box. Rows whose target is empty or not small are
    returned unchanged. Pure function (no nn.Module state) so it's directly unit-testable; used
    by rgbd_seg.m2f.criterion.SmallObjectCriterion.loss_masks.
    """
    n, p, _ = point_coords.shape
    k = int(round(p * fraction))
    if k <= 0 or n == 0:
        return point_coords
    bboxes, valid = masks_to_normalized_bboxes(target_masks)
    small = small_mask_flags(target_masks, area_threshold_px) & valid
    if not small.any():
        return point_coords
    out = point_coords.clone()
    idx = small.nonzero(as_tuple=True)[0]
    boxes = expand_bboxes(bboxes[idx], box_expand)
    extra = sample_points_in_bboxes(boxes, k).to(point_coords.dtype)  # (len(idx), k, 2)
    out[idx, p - k:, :] = extra
    return out


def area_weights(target_masks, ref=1024.0, power=0.25, max_weight=2.0):
    """target_masks: (N, H, W) -> (N,) weight in [1, max_weight], larger for smaller masks.

    weight = clip((ref / area) ** power, 1, max_weight); docs/plan.md §4 I2 "面积感知权重".
    """
    area = target_masks.flatten(1).sum(dim=1).clamp(min=1.0)
    w = (ref / area).pow(power)
    return w.clamp(1.0, max_weight)


def allocate_point_budget(n_targets, total_points):
    """Split `total_points` as evenly as possible across `n_targets` targets (list of ints, sum == total_points, 0 if n_targets == 0)."""
    if n_targets <= 0 or total_points <= 0:
        return []
    base = total_points // n_targets
    remainder = total_points - base * n_targets
    counts = [base + 1 if i < remainder else base for i in range(n_targets)]
    return counts

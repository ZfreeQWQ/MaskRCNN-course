"""Unregistered ROI extensions for the official Mask2Former matcher/criterion.

Targets use ``masks``, ``labels``, ``small``, ``original_area`` and ``roi_boxes``.
Masks share the padded batch canvas. Boxes are normalized xyxy on that canvas,
ALREADY expanded by the caller (total 20%, i.e. 10% per side), and clipped to
[0, 1]. This module never expands them again. Areas are original annotation
areas, before resize/crop. ROI points are uniform over the box, including both
foreground and background; they are never pooled across different GTs.

ROICriterion deliberately supports a single training process only. Its optional
area-weight normalization uses the mean over ALL local targets, including
unmatched targets. Distributed use needs a global weighted normalizer first.
No architecture/config registration or upstream source changes are performed.
"""

import math

import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size
from mask2former.modeling.criterion import SetCriterion, calculate_uncertainty
from mask2former.modeling.matcher import (
    HungarianMatcher,
    batch_dice_loss_jit,
    batch_sigmoid_ce_loss_jit,
)


def _check_boxes(boxes):
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.is_floating_point():
        raise ValueError("roi_boxes must be a floating tensor of shape [N, 4]")
    if not torch.isfinite(boxes).all():
        raise ValueError("roi_boxes must be finite")
    if ((boxes < 0) | (boxes > 1)).any():
        raise ValueError("roi_boxes must be normalized to the padded canvas [0, 1]")
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("roi_boxes must have positive width and height")


def sample_roi_points(roi_boxes, num_points, *, generator=None):
    """Return [N, K, 2] uniform xy points in already-expanded [N, 4] boxes.

    No mask lookup, expansion, or mixing between boxes occurs. Empty batches
    and K=0 are supported. The optional generator must match the box device.
    """
    _check_boxes(roi_boxes)
    if not isinstance(num_points, int) or num_points < 0:
        raise ValueError("num_points must be a nonnegative integer")
    random = torch.rand(
        (len(roi_boxes), num_points, 2), device=roi_boxes.device,
        dtype=roi_boxes.dtype, generator=generator,
    )
    return roi_boxes[:, None, :2] + random * (
        roi_boxes[:, None, 2:] - roi_boxes[:, None, :2]
    )


def _check_target(target, *, roi=False, area=False):
    labels, masks = target["labels"], target["masks"]
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("labels must be an int64 tensor of shape [N]")
    n = len(labels)
    if masks.ndim != 3 or masks.shape[0] != n or min(masks.shape[-2:]) <= 0:
        raise ValueError("masks must have shape [N, H, W] on a nonempty padded canvas")
    if roi:
        if target["small"].shape != (n,) or target["small"].dtype != torch.bool:
            raise ValueError("small must be a bool tensor of shape [N]")
        if target["roi_boxes"].shape != (n, 4):
            raise ValueError("roi_boxes must align with labels: [N, 4]")
        _check_boxes(target["roi_boxes"])
    if area:
        original = target["original_area"]
        if original.shape != (n,) or not torch.isfinite(original).all() or (original <= 0).any():
            raise ValueError("original_area must have shape [N] and finite positive values")


class ROIHungarianMatcher(HungarianMatcher):
    """Keep official global/class costs; mix ROI mask costs in small GT columns.

    For small GT j: mask_cost[:, j] = (1-w)*global[:, j] + w*roi[:, j],
    separately for BCE and Dice. Class cost is never rescaled. Other columns
    remain official. Global and ROI costs each normalize by their own point
    budgets. GT chunks bound sampled logits to Q*chunk_size*roi_points; full
    GT masks are never expanded across queries or other GTs.
    """

    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1, num_points=0,
                 *, roi_points=256, roi_weight=.3, roi_chunk_size=16):
        super().__init__(cost_class, cost_mask, cost_dice, num_points)
        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError("num_points must be a positive global point budget")
        if not math.isfinite(roi_weight) or not 0 <= roi_weight <= 1:
            raise ValueError("roi_weight must lie in [0, 1]")
        if not isinstance(roi_points, int) or roi_points <= 0:
            raise ValueError("roi_points must be a positive integer")
        if not isinstance(roi_chunk_size, int) or roi_chunk_size <= 0:
            raise ValueError("roi_chunk_size must be a positive integer")
        self.roi_points = roi_points
        self.roi_weight = roi_weight
        self.roi_chunk_size = roi_chunk_size

    def _roi_costs(self, out_masks, gt_masks, boxes, small_indices):
        """Return Q x S BCE/Dice, using each small GT's own points and mask."""
        q = len(out_masks)
        mask_costs, dice_costs = [], []
        for ids in small_indices.split(self.roi_chunk_size):
            coords = sample_roi_points(boxes[ids], self.roi_points)
            # One query mask serves every GT in this chunk: only the small
            # coordinate/sample tensors gain a GT dimension, never full masks.
            query_coords = coords.reshape(1, -1, 2).expand(q, -1, -1)
            logits = point_sample(out_masks, query_coords, align_corners=False)
            logits = logits[:, 0].reshape(q, len(ids), self.roi_points).float()
            labels = point_sample(gt_masks[ids], coords, align_corners=False)[:, 0].float()
            with torch.autocast(device_type=logits.device.type, enabled=False):
                bce = F.binary_cross_entropy_with_logits(
                    logits, labels[None].expand_as(logits), reduction="none",
                ).mean(-1)
                prob = logits.sigmoid()
                dice = 1 - (2 * (prob * labels[None]).sum(-1) + 1) / (
                    prob.sum(-1) + labels.sum(-1)[None] + 1
                )
            mask_costs.append(bce)
            dice_costs.append(dice)
        return torch.cat(mask_costs, dim=1), torch.cat(dice_costs, dim=1)

    @torch.no_grad()
    def _cost_matrix(self, logits, masks, target):
        """Build a nonempty image's Q x T matrix (also useful for CPU audits)."""
        class_cost = -logits.softmax(-1)[:, target["labels"]]
        out_masks = masks[:, None]
        gt_masks = target["masks"].to(out_masks)[:, None]
        # Preserve the official single shared UNIFORM global point set.
        coords = torch.rand(1, self.num_points, 2, device=masks.device)
        global_gt = point_sample(
            gt_masks, coords.repeat(len(gt_masks), 1, 1), align_corners=False,
        ).squeeze(1)
        global_pred = point_sample(
            out_masks, coords.repeat(len(out_masks), 1, 1), align_corners=False,
        ).squeeze(1)
        with torch.autocast(device_type=masks.device.type, enabled=False):
            mask_cost = batch_sigmoid_ce_loss_jit(global_pred.float(), global_gt.float())
            dice_cost = batch_dice_loss_jit(global_pred.float(), global_gt.float())
        if self.roi_weight:
            ids = target["small"].to(device=masks.device).nonzero().flatten()
            if len(ids):
                boxes = target["roi_boxes"].to(device=masks.device, dtype=torch.float32)
                roi_mask, roi_dice = self._roi_costs(out_masks, gt_masks, boxes, ids)
                w = self.roi_weight
                mask_cost[:, ids] = (1 - w) * mask_cost[:, ids] + w * roi_mask
                dice_cost[:, ids] = (1 - w) * dice_cost[:, ids] + w * roi_dice
        cost = self.cost_mask * mask_cost + self.cost_class * class_cost + self.cost_dice * dice_cost
        if not torch.isfinite(cost).all():
            raise ValueError("Nonfinite Hungarian cost (check logits, masks and targets)")
        return cost

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        logits, masks = outputs["pred_logits"], outputs["pred_masks"]
        if (logits.ndim != 3 or masks.ndim != 4 or logits.shape[:2] != masks.shape[:2]
                or logits.shape[0] != len(targets) or logits.shape[-1] < 2
                or min(masks.shape[-2:]) <= 0):
            raise ValueError("Expected pred_logits [B,Q,C+1], pred_masks [B,Q,H,W], B targets")
        for target in targets:
            _check_target(target, roi=bool(self.roi_weight))
            if ((target["labels"] < 0) | (target["labels"] >= logits.shape[-1] - 1)).any():
                raise ValueError("Target labels must exclude the no-object class")
        # Exact official execution/RNG when disabled, for nonempty assignments.
        if not self.roi_weight and logits.shape[1] and all(len(t["labels"]) for t in targets):
            if not torch.isfinite(logits).all() or not torch.isfinite(masks).all():
                raise ValueError("Nonfinite matcher predictions")
            return super().memory_efficient_forward(outputs, targets)
        indices = []
        for i, target in enumerate(targets):
            if not masks.shape[1] or not len(target["labels"]):
                if not self.roi_weight and masks.shape[1]:
                    # Upstream still draws its global points for an empty GT
                    # image. Keep later images' RNG/assignments identical when
                    # disabled, without relying on zero-batch grid_sample.
                    torch.rand(1, self.num_points, 2, device=masks.device)
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty.clone()))
                continue
            cost = self._cost_matrix(logits[i], masks[i], target)
            src, dst = linear_sum_assignment(cost.cpu())
            indices.append((torch.as_tensor(src, dtype=torch.int64), torch.as_tensor(dst, dtype=torch.int64)))
        return indices


class ROICriterion(SetCriterion):
    """Replace floor(roi_fraction*P) official points for matched small GTs only.

    Start with the official P-point uncertainty/random sample for EVERY match.
    Replacement slots are spread across that sample, retaining global uncertain
    and random context; all non-small matches keep their exact official points.
    Inherit classification, auxiliary rematching and loss names from SetCriterion.
    Optional area weights are clip((1024/original_area)**.25, 1, 2), disabled by
    default. Normalize by num_masks * mean(weight over ALL targets).
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 roi_fraction=.3, area_weight=False):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses,
                         num_points, oversample_ratio, importance_sample_ratio)
        if not isinstance(num_points, int) or num_points <= 0:
            raise ValueError("num_points must be a positive integer")
        if not math.isfinite(roi_fraction) or not 0 <= roi_fraction <= 1:
            raise ValueError("roi_fraction must lie in [0, 1]")
        self.roi_fraction = roi_fraction
        self.area_weight = area_weight

    @staticmethod
    def _single_process():
        if get_world_size() != 1:
            raise RuntimeError("ROICriterion is single-process only; distributed weighted normalization is not implemented")

    def forward(self, outputs, targets):
        # Guard before the inherited forward reaches distributed collectives.
        self._single_process()
        return super().forward(outputs, targets)

    def loss_masks(self, outputs, targets, indices, num_masks):
        self._single_process()
        predictions = outputs["pred_masks"]
        if predictions.ndim != 4 or len(targets) != predictions.shape[0] or len(indices) != len(targets):
            raise ValueError("Expected pred_masks [B,Q,H,W] with B targets and index pairs")
        if not math.isfinite(float(num_masks)) or float(num_masks) <= 0:
            raise ValueError("num_masks must be finite and positive (official empty-target clamp is 1)")
        if not indices or not any(len(src) for src, _ in indices):
            zero = predictions.sum() * 0
            return {"loss_mask": zero, "loss_dice": zero}
        k = int(self.num_points * self.roi_fraction)
        for target in targets:
            _check_target(target, roi=bool(k), area=self.area_weight)
        if not k and not self.area_weight:
            return super().loss_masks(outputs, targets, indices, num_masks)
        src = predictions[self._get_src_permutation_idx(indices)][:, None]
        # Select each matched GT once; never repeat full masks across queries.
        matched_masks = [t["masks"][j] for t, (_, j) in zip(targets, indices)]
        if len({tuple(t["masks"].shape[-2:]) for t in targets}) != 1:
            raise ValueError("All target masks/roi_boxes must use the same padded batch canvas")
        target_masks = torch.cat([m.to(src) for m in matched_masks])[:, None]
        with torch.no_grad():
            coords = get_uncertain_point_coords_with_randomness(
                src, calculate_uncertainty, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio,
            )
            if k:
                small = torch.cat([t["small"][j].to(src.device) for t, (_, j) in zip(targets, indices)])
                rows = small.nonzero().flatten()
                if len(rows):
                    boxes = torch.cat([t["roi_boxes"][j].to(device=src.device, dtype=torch.float32)
                                       for t, (_, j) in zip(targets, indices)])
                    slots = torch.div(torch.arange(k, device=src.device) * self.num_points,
                                      k, rounding_mode="floor")
                    coords[rows[:, None], slots[None]] = sample_roi_points(boxes[rows], k)
            labels = point_sample(target_masks, coords, align_corners=False).squeeze(1)
        logits = point_sample(src, coords, align_corners=False).squeeze(1)
        with torch.autocast(device_type=src.device.type, enabled=False):
            logits, labels = logits.float(), labels.float()
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
            prob = logits.sigmoid()
            dice = 1 - (2 * (prob * labels).sum(1) + 1) / (prob.sum(1) + labels.sum(1) + 1)
            weights = torch.ones_like(bce)
            mean_weight = bce.new_tensor(1.)
            if self.area_weight:
                all_weights = [(1024. / t["original_area"].to(device=src.device, dtype=torch.float32)).pow(.25).clamp(1, 2)
                               for t in targets]
                weights = torch.cat([w[j] for w, (_, j) in zip(all_weights, indices)])
                mean_weight = torch.cat(all_weights).mean()
            denominator = num_masks * mean_weight
            result = {"loss_mask": (bce * weights).sum() / denominator,
                      "loss_dice": (dice * weights).sum() / denominator}
        if not all(torch.isfinite(value) for value in result.values()):
            raise ValueError("Nonfinite ROI mask loss")
        return result

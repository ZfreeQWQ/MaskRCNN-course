"""I2 matcher: adds a bounded budget of small-object-box-local points to the shared matching
point set (docs/plan.md §4 I2, "匹配" paragraph).

The official HungarianMatcher.memory_efficient_forward draws ONE shared set of random points
per image (`point_coords`, shape [1, num_points, 2]) and uses it for every (query, target) pair
in that image's cost matrix -- this is what makes the cost computation batchable. We keep that
structure exactly, only changing WHICH points are in the shared set: concatenate the official
uniform-random points with up to PROJECT.SMALL_OBJECT_POINTS.NUM_POINTS extra points, drawn from
within each small target's expanded bounding box (split evenly across however many small targets
exist that image, see small_object_sampling.allocate_point_budget). Extra points for small
targets also slightly sharpen the cost signal for OTHER (non-small) targets at those same
locations -- this is harmless: it's the same thing the existing uniform points already do
everywhere (most points are irrelevant background for most pairs).
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from mask2former.modeling.matcher import (
    HungarianMatcher,
    batch_dice_loss_jit,
    batch_sigmoid_ce_loss_jit,
)

from .small_object_sampling import (
    allocate_point_budget,
    expand_bboxes,
    masks_to_normalized_bboxes,
    sample_points_in_bboxes,
    small_mask_flags,
)


class SmallObjectMatcher(HungarianMatcher):
    def __init__(self, *args, extra_points=256, box_expand=0.2, area_threshold_px=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_points = extra_points
        self.box_expand = box_expand
        self.area_threshold_px = area_threshold_px

    def _build_point_coords(self, tgt_mask, device):
        """tgt_mask: (M, H, W) target masks for this image -> (1, num_points + extra, 2)."""
        base = torch.rand(1, self.num_points, 2, device=device)
        if self.extra_points <= 0 or tgt_mask.shape[0] == 0:
            return base
        bboxes, valid = masks_to_normalized_bboxes(tgt_mask)
        small = small_mask_flags(tgt_mask, self.area_threshold_px) & valid
        n_small = int(small.sum().item())
        if n_small == 0:
            return base
        counts = allocate_point_budget(n_small, self.extra_points)
        small_boxes = expand_bboxes(bboxes[small], self.box_expand)
        extra_chunks = [sample_points_in_bboxes(small_boxes[i:i + 1], k)
                        for i, k in enumerate(counts) if k > 0]
        if not extra_chunks:
            return base
        extra = torch.cat(extra_chunks, dim=1)  # (1, sum(counts), 2)
        return torch.cat([base, extra], dim=1)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets[b]["labels"]
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask_4d = tgt_mask[:, None]
            point_coords = self._build_point_coords(tgt_mask, out_mask.device)

            tgt_mask_sampled = point_sample(
                tgt_mask_4d, point_coords.repeat(tgt_mask_4d.shape[0], 1, 1), align_corners=False,
            ).squeeze(1)
            out_mask_sampled = point_sample(
                out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask_sampled = out_mask_sampled.float()
                tgt_mask_sampled = tgt_mask_sampled.float()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask_sampled, tgt_mask_sampled)
                cost_dice = batch_dice_loss_jit(out_mask_sampled, tgt_mask_sampled)

            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]

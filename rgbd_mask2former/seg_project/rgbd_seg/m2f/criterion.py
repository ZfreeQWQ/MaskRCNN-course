"""I2/I3 criterion: small-object point sampling + optional depth-guided hi-res mask refinement
(docs/plan.md §4 I2 "损失"/"面积感知权重" and I3).

`SmallObjectCriterion` overrides only the point-sampling step of `loss_masks`. For each matched
(query, target) pair, the official SetCriterion samples `num_points` points via PointRend-style
predicted-uncertainty sampling (`get_uncertain_point_coords_with_randomness`) -- already biased
toward ambiguous/boundary regions, but over the WHOLE image, so a ~40px target gets a vanishingly
small share. We keep the total point count per pair unchanged (so there is no training-time
compute cost), but for small targets, replace a `LOSS_POINT_FRACTION` tail of those points with
points drawn uniformly from within the target's own expanded box -- concentrating supervision
where the small object actually is.

`HiResCriterion` (extends `SmallObjectCriterion`) additionally runs `rgbd_seg.m2f.hires_mask.
HiResMaskHead` on matched pairs' coarse masks + the model's own input image tensor, adding
`loss_mask_hires`/`loss_dice_hires` terms evaluated at the SAME normalized point coordinates as
the base losses (point_sample works in [0,1] coords regardless of the hires head's actual output
resolution, so this reuses docs/plan.md's point-sampling infrastructure directly). This needs the
model's input tensor, which plain `SetCriterion.forward(outputs, targets)` never sees; see
rgbd_seg.m2f.model.RGBDMaskFormer.forward's minimal override that passes `images=`.

IMPORTANT (found via a GPU OOM crash in tools/train_m2f.py's "dsh" smoke test, not caught by
any unit/integration test because those used tiny synthetic batches with only 1-2 instances):
`HiResMaskHead.refine` operates at the hi-res feature map's FULL resolution (H/2 x W/2, e.g.
512x512 for a 1024 input) and allocates one (hidden+1)-channel activation PER matched pair. A
real training batch with COCO density (~30 instances/image, docs/plan.md §2) can have 50-90
matched pairs after Hungarian matching, so running every pair through the hires head at 512x512
costs tens of GB and reliably OOMs even at batch 2. docs/plan.md's I3 was always meant to target
*small* objects specifically ("对每个 ROI...从原分辨率输入..."), not every instance uniformly, so
`HiResCriterion` restricts the hires branch to the SMALL matched pairs only (via
`small_mask_flags`), additionally capped at `hires_max_pairs` per loss_masks call as a hard
memory ceiling regardless of how many small targets a given image contains. This is both more
correct (matches the plan's actual intent) and fixes the OOM; it is not merely a memory
workaround bolted onto an otherwise-uncapped design.

A second, larger factor behind the same OOM: `SetCriterion.forward` calls `loss_masks` once for
the main decoder output AND once per auxiliary (deep-supervision) decoder layer -- 10 calls
total for the default 9-layer decoder (see third_party/Mask2Former/mask2former/modeling/
criterion.py's `forward`). Capping pairs per call to 24 still meant 10 * 24 = 240 pair-images
worth of activations accumulating in the *same* backward graph, which is what actually exceeded
32GB (the per-call estimate alone, ~24 pairs, is only a few GB). The hires branch therefore only
runs on the MAIN decoder output, detected via `"aux_outputs" in outputs` (present only on the
top-level call -- see `SetCriterion.forward`'s code: the main call passes the full `outputs`
dict including its `aux_outputs` key, while each aux call passes that aux layer's own dict,
which has no nested `aux_outputs` key of its own). This both fixes the OOM and is a reasonable
design choice on its own merits: unlike the coarse mask/dice losses (where deep supervision
genuinely helps gradient flow through early decoder layers), the hires branch is a post-hoc
refinement of whatever mask the decoder already produced, so supervising it once (on the layer
whose output is actually used at inference) is sufficient.
"""
import torch
import torch.nn.functional as F
from mask2former.modeling.criterion import (
    SetCriterion,
    calculate_uncertainty,
    dice_loss_jit,
    sigmoid_ce_loss_jit,
)
from mask2former.utils.misc import nested_tensor_from_tensor_list
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from . import small_object_sampling as sos


class SmallObjectCriterion(SetCriterion):
    def __init__(self, *args, loss_point_fraction=0.3, box_expand=0.2, area_threshold_px=1024,
                 area_weight_enabled=False, area_weight_ref=1024.0, area_weight_power=0.25,
                 area_weight_max=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_point_fraction = loss_point_fraction
        self.box_expand = box_expand
        self.area_threshold_px = area_threshold_px
        self.area_weight_enabled = area_weight_enabled
        self.area_weight_ref = area_weight_ref
        self.area_weight_power = area_weight_power
        self.area_weight_max = area_weight_max

    def _sample_points(self, outputs, targets, indices):
        """Shared by loss_masks (and HiResCriterion's extension): matches+samples once.

        Returns (point_coords, src_masks, target_masks, point_logits, point_labels), where
        src_masks/target_masks are (N, h, w) -- one row per matched (query, target) pair -- and
        point_coords/point_logits/point_labels are (N, num_points[, 2]).
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, _valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks_4d = src_masks[:, None]
        target_masks_4d = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks_4d, lambda logits: calculate_uncertainty(logits),
                self.num_points, self.oversample_ratio, self.importance_sample_ratio,
            )
            if self.loss_point_fraction > 0:
                point_coords = sos.replace_small_target_points(
                    point_coords, target_masks, self.loss_point_fraction,
                    self.box_expand, self.area_threshold_px)
            point_labels = point_sample(target_masks_4d, point_coords, align_corners=False).squeeze(1)

        point_logits = point_sample(src_masks_4d, point_coords, align_corners=False).squeeze(1)
        return point_coords, src_masks, target_masks, point_logits, point_labels

    def _base_mask_losses(self, point_logits, point_labels, target_masks, num_masks):
        if self.area_weight_enabled and target_masks.shape[0] > 0:
            w = sos.area_weights(target_masks, self.area_weight_ref, self.area_weight_power,
                                  self.area_weight_max)
            return {
                "loss_mask": _weighted_sigmoid_ce(point_logits, point_labels, w) / num_masks,
                "loss_dice": _weighted_dice(point_logits, point_labels, w) / num_masks,
            }
        return {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        point_coords, src_masks, target_masks, point_logits, point_labels = self._sample_points(
            outputs, targets, indices)
        losses = self._base_mask_losses(point_logits, point_labels, target_masks, num_masks)
        del src_masks, target_masks
        return losses


class HiResCriterion(SmallObjectCriterion):
    def __init__(self, *args, hires_head=None, hires_max_pairs=24, **kwargs):
        super().__init__(*args, **kwargs)
        self.hires_head = hires_head  # nn.Module or None; auto-registered as a submodule if set
        self.hires_max_pairs = hires_max_pairs
        self._current_img_feats = None

    def forward(self, outputs, targets, images=None):
        # Compute the hi-res image features ONCE per forward() call, not once per decoder layer:
        # SetCriterion.forward (called below via super()) invokes loss_masks once for the main
        # output and again for each of the 9 aux (deep-supervision) layers, all against the SAME
        # input image -- recomputing image_features() in every call would mean 10x redundant
        # conv passes through the hires stem per training step.
        if self.hires_head is not None and images is not None:
            self._current_img_feats = self.hires_head.image_features(images)
        try:
            return super().forward(outputs, targets)
        finally:
            self._current_img_feats = None

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        point_coords, src_masks, target_masks, point_logits, point_labels = self._sample_points(
            outputs, targets, indices)
        losses = self._base_mask_losses(point_logits, point_labels, target_masks, num_masks)

        is_main_call = "aux_outputs" in outputs  # see HiResCriterion's module docstring
        if self.hires_head is not None and self._current_img_feats is not None and is_main_call:
            losses["loss_mask_hires"], losses["loss_dice_hires"] = self._hires_losses(
                src_masks, target_masks, point_coords, indices, num_masks)

        del src_masks, target_masks
        return losses

    def _hires_losses(self, src_masks, target_masks, point_coords, indices, num_masks):
        """Always returns a (loss_mask_hires, loss_dice_hires) pair of two DISTINCT tensors.

        FOUND VIA TWO SEPARATE REAL 4-GPU CRASHES (docs/plan.md's "dsh" screening run,
        2026-10-23/25) from an earlier version that only set these dict keys when small targets
        existed to refine:
          1. A DDP deadlock: when one rank's local batch had zero small targets this iteration
             while another rank's had some, the two ranks' backward passes touched different
             parameter sets (hires_head's, in particular), and without
             find_unused_parameters=True (now set defensively in Trainer.__init__, but the real
             fix belongs here) DDP's gradient-bucket reducer hung waiting for a bucket some rank
             never produced.
          2. After fixing (1), a KeyError crash in detectron2's SimpleTrainer.write_metrics,
             which indexes every rank's loss dict by the FIRST rank's key set -- if one rank's
             dict lacks a key the others have, it crashes (this is a logging-only call, but
             unhandled, so it still kills the whole run).
        Both are instances of the same bug: the keys/parameter-usage must be identical on every
        rank, every iteration. When there is nothing real to refine (no small targets, or --
        extremely unlikely given FILTER_EMPTY_ANNOTATIONS and this dataset's >=4-instances/image
        floor -- no matched pairs at all for this rank's local batch), we still route an exact
        zero through every hires_head parameter, which is a correctness/stability no-op (zero
        gradient, zero numerical contribution), not an approximation of the real loss.
        """
        n = target_masks.shape[0]
        if n > 0:
            small = sos.small_mask_flags(target_masks, self.area_threshold_px)
            idx = small.nonzero(as_tuple=True)[0]
        else:
            idx = target_masks.new_zeros(0, dtype=torch.long)
        if idx.numel() > self.hires_max_pairs:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.hires_max_pairs]]

        if idx.numel() > 0:
            # _get_src_permutation_idx's tensors come straight from the matcher's
            # torch.as_tensor(...) with no device argument, i.e. always CPU, regardless of
            # where target_masks/idx live (CUDA during real training) -- index on CPU, then
            # move the result to match img_feats' device for the embedding-style lookup below.
            batch_idx_all = self._get_src_permutation_idx(indices)[0]
            batch_idx = batch_idx_all[idx.cpu()].to(self._current_img_feats.device)
            refined = self.hires_head.refine(src_masks[idx], self._current_img_feats, batch_idx)
            refined_points = point_sample(refined[:, None], point_coords[idx], align_corners=False).squeeze(1)
            target_points = point_sample(target_masks[idx][:, None], point_coords[idx], align_corners=False).squeeze(1)
            return (sigmoid_ce_loss_jit(refined_points, target_points, num_masks),
                    dice_loss_jit(refined_points, target_points, num_masks))

        zero = sum(p.sum() for p in self.hires_head.parameters()) * 0.0
        return zero, zero.clone()


def _weighted_sigmoid_ce(inputs, targets, weight):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (loss.mean(1) * weight).sum()


def _weighted_dice(inputs, targets, weight):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return (loss * weight).sum()

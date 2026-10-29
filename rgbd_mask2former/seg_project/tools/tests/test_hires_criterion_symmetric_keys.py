#!/usr/bin/env python3
"""GPU regression check: HiResCriterion.loss_masks always returns loss_mask_hires/loss_dice_hires
with a nonzero gradient path to hires_head's parameters, EVEN WHEN there are no small targets to
refine (or no targets at all). Guards against the two crashes found in real 4-GPU training (see
criterion.py's HiResCriterion._hires_losses docstring): a DDP deadlock from asymmetric parameter
usage across ranks, and a KeyError in detectron2's metrics logging from asymmetric dict keys.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.hires_mask import HiResMaskHead
from rgbd_seg.m2f.criterion import HiResCriterion
from mask2former.modeling.matcher import HungarianMatcher


def make_criterion():
    matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0, num_points=64)
    hires_head = HiResMaskHead(in_channels=3, hidden=8).cuda()
    return HiResCriterion(
        num_classes=1, matcher=matcher, weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        eos_coef=0.1, losses=["labels", "masks"], num_points=64, oversample_ratio=3.0,
        importance_sample_ratio=0.75, hires_head=hires_head, hires_max_pairs=24,
    ).cuda()


def run_case(name, n_pairs, make_target_masks):
    """n_pairs: number of matched (query, target) pairs this call simulates."""
    crit = make_criterion()
    src_masks = torch.randn(n_pairs, 32, 32, device='cuda', requires_grad=True)
    target_masks = make_target_masks(n_pairs)
    point_coords = torch.rand(n_pairs, 16, 2, device='cuda')
    # indices as _get_src_permutation_idx would build them: (batch_idx, src_idx), both CPU.
    indices = [(torch.arange(n_pairs), torch.arange(n_pairs))] if n_pairs > 0 else [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    crit._current_img_feats = crit.hires_head.image_features(torch.randn(1, 3, 64, 64, device='cuda'))
    loss_mask_hires, loss_dice_hires = crit._hires_losses(src_masks, target_masks, point_coords, indices, num_masks=max(n_pairs, 1))
    assert torch.isfinite(loss_mask_hires) and torch.isfinite(loss_dice_hires), name
    assert loss_mask_hires is not loss_dice_hires, f"{name}: must be distinct tensor objects (shared object would double-scale under in-place *=)"
    total = loss_mask_hires + loss_dice_hires
    total.backward()
    n_grad = sum(1 for p in crit.hires_head.parameters() if p.grad is not None)
    n_total = sum(1 for _ in crit.hires_head.parameters())
    print(f"{name}: loss_mask_hires={loss_mask_hires.item():.4f} loss_dice_hires={loss_dice_hires.item():.4f} "
          f"hires_head grad coverage={n_grad}/{n_total}")
    assert n_grad == n_total, f"{name}: not every hires_head parameter received a gradient"
    return total.item()


def main():
    # Case 1: normal -- some pairs, all "small" (tiny mask -> guaranteed small_mask_flags=True).
    def small_masks(n):
        m = torch.zeros(n, 32, 32, device='cuda')
        m[:, :3, :3] = 1.0
        return m
    loss_small = run_case("all_small", 4, small_masks)

    # Case 2: the bug trigger -- pairs exist, but NONE are small (large masks only). Use a mask
    # genuinely larger than the COCO "small" threshold (area <= 32**2 = 1024, inclusive); a
    # bare 32x32 full mask is exactly 1024 and would itself count as "small".
    def large_masks(n):
        return torch.ones(n, 64, 64, device='cuda')
    loss_none_small = run_case("no_small_targets", 4, large_masks)
    assert loss_none_small == 0.0, "no-small-targets case should be an exact zero loss"

    # Case 3: extreme edge case -- zero matched pairs at all for this rank's local batch.
    def empty_masks(n):
        return torch.zeros(0, 32, 32, device='cuda')
    loss_empty = run_case("zero_pairs", 0, empty_masks)
    assert loss_empty == 0.0, "zero-pairs case should be an exact zero loss"

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

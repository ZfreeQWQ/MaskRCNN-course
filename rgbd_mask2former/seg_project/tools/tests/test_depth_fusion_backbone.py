#!/usr/bin/env python3
"""GPU integration check for rgbd_seg.m2f.depth_fusion.GatedRGBDResNetBackbone.

Verifies the key design property from docs/plan.md I1: at init (zero-initialized fusion
projections), the 9-channel RGBD backbone's output is numerically IDENTICAL to running the
plain RGB-only backbone on just the first 3 channels -- i.e. the model starts out exactly
equivalent to the loaded RGB-only COCO checkpoint, regardless of what the depth branch (which
is randomly initialized, not loaded from any checkpoint) produces.

The gradient-unlock property (proj.weight=0 blocks gradient into depth_stem/depth_stages/gate at
step 0; one optimizer step unlocks it) is instead checked in test_depth_fusion_training_step.py,
using the real loaded COCO checkpoint and real mapped RGB-D data. An earlier version of this
script tried to check it here too, with a from-scratch (non-pretrained) ResNet + FrozenBN fed
std-50 synthetic noise; that combination is numerically degenerate on its own (FrozenBN's fixed
running stats don't match an uncalibrated random backbone), which intermittently produced
Inf/NaN unrelated to depth_fusion.py's correctness and made that check flaky. Don't re-add a
gradient check here without pretrained (or at least BN-calibrated) weights.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.depth_fusion import build_gated_rgbd_resnet_backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import build_resnet_backbone


def main():
    torch.manual_seed(0)
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.PROJECT.USE_DEPTH = True
    cfg.freeze()

    device = 'cuda'
    rgbd_backbone = build_gated_rgbd_resnet_backbone(cfg, ShapeSpec(channels=9)).to(device)
    rgb_backbone = build_resnet_backbone(cfg, ShapeSpec(channels=3)).to(device)
    rgbd_backbone.rgb_backbone.load_state_dict(rgb_backbone.state_dict())

    rgbd_backbone.eval()  # disable depth dropout for this exact-equivalence check
    rgb_backbone.eval()

    bs = 2
    rgb = torch.randn(bs, 3, 256, 256, device=device) * 50
    depth = torch.randn(bs, 6, 256, 256, device=device) * 50  # nonzero, "informative" depth
    x = torch.cat([rgb, depth], dim=1)

    with torch.no_grad():
        fused_feats = rgbd_backbone(x)
        rgb_feats = rgb_backbone(rgb)

    for name in rgb_feats:
        diff = (fused_feats[name] - rgb_feats[name]).abs().max().item()
        print(f"{name}: max abs diff = {diff:.3e}, shape={tuple(fused_feats[name].shape)}")
        assert diff < 1e-5, f"{name} differs at init: zero-init fusion should be an exact no-op"

    print("ALL CHECKS PASSED (see test_depth_fusion_training_step.py for the gradient-unlock check)")


if __name__ == "__main__":
    main()

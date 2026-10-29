#!/usr/bin/env python3
"""GPU integration check: RGBDMaskFormer with I3 (hires_mask) enabled, alone and combined with
I1 (depth fusion) + I2 (small-object points) -- the plan's "DSH"/full-combo configuration.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.checkpoint import RGBDBackboneCheckpointer
import rgbd_seg.m2f.model  # noqa: F401 -- registers RGBDMaskFormer
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone
from rgbd_seg.m2f.criterion import HiResCriterion
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, Boxes


def make_cfg(use_depth, small_object, hires, hires_use_depth=True):
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.MODEL.META_ARCHITECTURE = "RGBDMaskFormer"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'
    cfg.PROJECT.USE_DEPTH = use_depth
    if use_depth:
        cfg.MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone"
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED = small_object
    cfg.PROJECT.SMALL_OBJECT_POINTS.NUM_POINTS = 64
    cfg.PROJECT.HIRES_MASK.ENABLED = hires
    cfg.PROJECT.HIRES_MASK.USE_DEPTH = hires_use_depth
    cfg.freeze()
    return cfg


def make_batch(use_depth):
    torch.manual_seed(0)
    channels = 9 if use_depth else 3
    img = torch.rand(channels, 256, 256, device='cuda') * 50
    img[:3] = img[:3] * 5  # keep RGB-like channels roughly in [0,255) range
    gt = Instances((256, 256))
    boxes = torch.tensor([[10.0, 10.0, 14.0, 14.0], [20.0, 20.0, 220.0, 220.0]])
    gt.gt_boxes = Boxes(boxes)
    gt.gt_classes = torch.zeros(2, dtype=torch.int64)
    masks = torch.zeros(2, 256, 256, dtype=torch.float32)
    masks[0, 10:14, 10:14] = 1.0
    masks[1, 20:220, 20:220] = 1.0
    gt.gt_masks = masks
    return [{'image': img, 'height': 256, 'width': 256, 'instances': gt}]


def run(cfg, tag):
    model = build_model(cfg)
    if cfg.PROJECT.USE_DEPTH:
        RGBDBackboneCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        stem = model.backbone.rgb_backbone.stem.conv1.weight
    else:
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        stem = model.backbone.stem.conv1.weight
    assert stem.detach().abs().mean().item() > 0.01, f"[{tag}] backbone looks uninitialized"

    model.train()
    batched = make_batch(cfg.PROJECT.USE_DEPTH)
    losses = model(batched)
    total = sum(losses.values())
    assert torch.isfinite(total), f"[{tag}] loss not finite: {losses}"
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    hires_keys = [k for k in losses if "hires" in k]
    print(f"[{tag}] total={total.item():.3f}, params_with_grad={n_grad}/{n_total}, "
          f"hires_loss_keys={len(hires_keys)}")
    return total.item(), losses, model


def main():
    print("### I3 alone (RGB only) ###")
    cfg = make_cfg(use_depth=False, small_object=False, hires=True)
    loss_i3, losses_i3, model = run(cfg, "I3-alone")
    assert isinstance(model.criterion, HiResCriterion)
    assert any("hires" in k for k in losses_i3), "expected loss_mask_hires/loss_dice_hires to appear"
    n_grad = sum(1 for p in model.criterion.hires_head.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.criterion.hires_head.parameters())
    print(f"  hires_head params with grad: {n_grad}/{n_total}")
    assert n_grad == n_total

    print("### No hires, for comparison ###")
    cfg_off = make_cfg(use_depth=False, small_object=False, hires=False)
    loss_off, losses_off, _ = run(cfg_off, "no-hires")
    assert not any("hires" in k for k in losses_off)

    print("### Full combo: I1 + I2 + I3 (DSH) ###")
    cfg_full = make_cfg(use_depth=True, small_object=True, hires=True, hires_use_depth=True)
    loss_full, losses_full, model_full = run(cfg_full, "DSH")
    assert any("hires" in k for k in losses_full)
    assert model_full.criterion.hires_head.in_channels == 9

    print("### I3 with hires_use_depth=False (SH-style: hires branch sees RGB only) ###")
    cfg_sh = make_cfg(use_depth=True, small_object=True, hires=True, hires_use_depth=False)
    _, _, model_sh = run(cfg_sh, "SH")
    assert model_sh.criterion.hires_head.in_channels == 3

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

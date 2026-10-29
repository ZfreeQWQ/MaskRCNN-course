#!/usr/bin/env python3
"""GPU integration check: RGBDMaskFormer (I2 small-object matcher/criterion) end to end.

Builds a real model with cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED, feeds a synthetic batch
containing both a tiny (16px) and a large instance, and checks: no crash, finite loss, every
parameter gets a gradient, and that the small-object code path actually executes (by comparing
losses with the feature on vs off -- they must differ, proving the point-replacement logic ran
rather than being silently skipped).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
import rgbd_seg.m2f.model  # noqa: F401 -- registers RGBDMaskFormer
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone (unused here; I1 is independent of I2, see model.py)
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances, Boxes


def make_cfg(small_object_enabled):
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.MODEL.META_ARCHITECTURE = "RGBDMaskFormer"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'
    cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED = small_object_enabled
    cfg.PROJECT.SMALL_OBJECT_POINTS.NUM_POINTS = 64
    cfg.PROJECT.SMALL_OBJECT_POINTS.LOSS_POINT_FRACTION = 0.3
    cfg.freeze()
    return cfg


def make_batch(seed):
    torch.manual_seed(seed)
    img = torch.rand(3, 256, 256, device='cuda') * 255
    gt = Instances((256, 256))
    boxes = torch.tensor([[10.0, 10.0, 14.0, 14.0],      # 4x4 = 16 px, "small"
                          [20.0, 20.0, 220.0, 220.0]])    # large
    gt.gt_boxes = Boxes(boxes)
    gt.gt_classes = torch.zeros(2, dtype=torch.int64)
    masks = torch.zeros(2, 256, 256, dtype=torch.float32)
    masks[0, 10:14, 10:14] = 1.0
    masks[1, 20:220, 20:220] = 1.0
    gt.gt_masks = masks
    return [{'image': img, 'height': 256, 'width': 256, 'instances': gt}]


def total_loss(cfg):
    # Plain (non-gated) backbone here -- I2 is independent of I1, so the checkpoint's
    # backbone.* keys match this model's backbone.* directly; no key remapping needed
    # (contrast with test_depth_fusion_training_step.py, which enables the I1 backbone and
    # therefore needs RGBDBackboneCheckpointer's backbone.*->backbone.rgb_backbone.* remap).
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    assert model.backbone.stem.conv1.weight.detach().abs().mean().item() > 0.01, (
        "backbone looks uninitialized/random -- checkpoint did not load (see "
        "test_checkpoint_metadata.py for the kind of silent-failure this guards against)")
    model.train()
    batched = make_batch(seed=0)
    losses = model(batched)
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f"{k} is not finite: {v}"
    total = sum(losses.values())
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grad == n_total, f"only {n_grad}/{n_total} params got gradients"
    return total.item(), losses


def main():
    print("### SMALL_OBJECT_POINTS disabled (should match plain MaskFormer behavior) ###")
    cfg_off = make_cfg(small_object_enabled=False)
    loss_off, _ = total_loss(cfg_off)
    print(f"  total loss = {loss_off:.4f}")

    print("### SMALL_OBJECT_POINTS enabled ###")
    cfg_on = make_cfg(small_object_enabled=True)
    loss_on, losses_on = total_loss(cfg_on)
    print(f"  total loss = {loss_on:.4f}")
    print(f"  loss keys: {sorted(losses_on.keys())[:6]} ... ({len(losses_on)} total, 3 per decoder layer x 10)")

    assert abs(loss_on - loss_off) > 1e-6, (
        "enabling SMALL_OBJECT_POINTS produced an identical loss -- the small-object code path "
        "likely did not execute")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

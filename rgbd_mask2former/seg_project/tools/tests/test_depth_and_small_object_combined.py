#!/usr/bin/env python3
"""GPU integration check: I1 (depth fusion) + I2 (small-object matcher/criterion) together
(docs/plan.md's "DS" experiment). Confirms the two independent features compose without
conflict: RGBDMaskFormer meta-arch + build_gated_rgbd_resnet_backbone + 9-channel pixel mean/std.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.checkpoint import RGBDBackboneCheckpointer
from rgbd_seg.m2f.mapper import RGBDInstanceDatasetMapper
from rgbd_seg.data import register_datasets
import rgbd_seg.m2f.model  # noqa: F401 -- registers RGBDMaskFormer
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone
from detectron2.data import DatasetCatalog
from detectron2.modeling import build_model


def main():
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.DATASETS.TRAIN = ("blender_train",)
    cfg.DATASETS.TEST = ("blender_val",)
    cfg.MODEL.META_ARCHITECTURE = "RGBDMaskFormer"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'
    cfg.MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.PROJECT.USE_DEPTH = True
    cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED = True
    cfg.PROJECT.SMALL_OBJECT_POINTS.NUM_POINTS = 64
    cfg.freeze()

    register_datasets(cfg)
    records = DatasetCatalog.get("blender_train")[:3]
    mapper = RGBDInstanceDatasetMapper(cfg, is_train=True)
    mapped = [mapper(r) for r in records]
    assert all(m["image"].shape[0] == 9 for m in mapped)

    model = build_model(cfg)
    RGBDBackboneCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    assert type(model.criterion).__name__ == "SmallObjectCriterion"
    assert type(model.criterion.matcher).__name__ == "SmallObjectMatcher"
    assert model.backbone.rgb_backbone.stem.conv1.weight.detach().abs().mean().item() > 0.01

    model.train()
    batched = [{"image": m["image"].cuda(), "height": cfg.INPUT.IMAGE_SIZE,
                "width": cfg.INPUT.IMAGE_SIZE, "instances": m["instances"].to("cuda")}
               for m in mapped]
    losses = model(batched)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"total loss={total.item():.3f}, params with grad={n_grad}/{n_total}")
    assert n_grad == n_total, (
        f"only {n_grad}/{n_total} got gradients -- note depth branch is gradient-blocked until "
        "proj.weight leaves its zero-init (see test_depth_fusion_training_step.py); step the "
        "optimizer once if this fails only for depth_stem/depth_stages/.gate. params")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU integration check: RGBDInstanceDatasetMapper -> real Mask2Former model fwd+bwd.

Not a pure-CPU unit test (needs detectron2+mask2former via run.sh, and the GPU for the model
pass), so it's a standalone script rather than part of tools/tests' unittest suite.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.data import register_datasets
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.mapper import RGBDInstanceDatasetMapper
from rgbd_seg.m2f.checkpoint import RGBDBackboneCheckpointer
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone
from detectron2.data import DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def make_cfg(use_depth):
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.DATASETS.TRAIN = ("blender_train",)
    cfg.DATASETS.TEST = ("blender_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'
    cfg.MODEL.DEVICE = 'cuda'
    cfg.PROJECT.USE_DEPTH = use_depth
    cfg.INPUT.ROTATE_90 = True
    cfg.INPUT.COLOR_AUG_SSD = True
    if use_depth:
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.freeze()
    return cfg


def main():
    cfg = make_cfg(use_depth=False)
    register_datasets(cfg)
    train_records = DatasetCatalog.get("blender_train")
    val_records = DatasetCatalog.get("blender_val")
    print(f"train records: {len(train_records)}, val records: {len(val_records)}")

    for use_depth in (False, True):
        cfg = make_cfg(use_depth=use_depth)
        mapper_train = RGBDInstanceDatasetMapper(cfg, is_train=True)
        mapper_test = RGBDInstanceDatasetMapper(cfg, is_train=False)

        t0 = time.time()
        n = 6
        mapped = [mapper_train(r) for r in train_records[:n]]
        dt = time.time() - t0
        print(f"[use_depth={use_depth}] mapped {n} train records in {dt:.2f}s ({dt/n:.3f}s/img)")
        expected_channels = 9 if use_depth else 3
        for m in mapped:
            # Depth geometry is concatenated onto "image" as extra channels (not a separate
            # "depth_geometry" key) so MaskFormer.forward()'s stock preprocessing needs no
            # changes; see rgbd_seg/m2f/mapper.py's module docstring.
            assert m["image"].shape[0] == expected_channels, m["image"].shape
            assert m["image"].shape[1:] == (cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
            assert "depth_geometry" not in m
            inst = m["instances"]
            assert isinstance(inst.gt_masks, torch.Tensor), type(inst.gt_masks)
            assert inst.gt_masks.shape[1:] == (cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
            assert inst.gt_masks.shape[0] == inst.gt_classes.shape[0] == len(inst.gt_boxes)
            assert (inst.gt_classes == 0).all()
        print(f"  train instance counts: {[len(m['instances']) for m in mapped]}")

        test_mapped = mapper_test(val_records[0])
        assert "instances" not in test_mapped
        assert test_mapped["image"].shape[0] == expected_channels
        print(f"  test-mode image shape: {tuple(test_mapped['image'].shape)}")

        # End-to-end: real model forward+backward on mapped batches. When use_depth, this uses
        # the I1 gated RGB-D backbone (verified for checkpoint/zero-init correctness separately
        # in test_depth_fusion_backbone.py / test_depth_fusion_training_step.py); here we only
        # need *some* model that accepts `expected_channels`-channel input end to end.
        if use_depth:
            cfg.defrost()
            cfg.MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone"
            cfg.freeze()
            model = build_model(cfg)
            RGBDBackboneCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        else:
            model = build_model(cfg)
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.train()
        batched = [{"image": m["image"].cuda(), "height": cfg.INPUT.IMAGE_SIZE,
                    "width": cfg.INPUT.IMAGE_SIZE, "instances": m["instances"].to("cuda")}
                   for m in mapped[:2]]
        losses = model(batched)
        total = sum(losses.values())
        total.backward()
        print(f"  model fwd+bwd OK, total loss={total.item():.3f}, "
              f"grads nonzero={sum(1 for p in model.parameters() if p.grad is not None)}/{sum(1 for _ in model.parameters())}")
        del model
        torch.cuda.empty_cache()

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

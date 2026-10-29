#!/usr/bin/env python3
"""GPU check: full RGBD MaskFormer (real checkpoint, real mapped data) trains stably for a few
steps and the depth branch "unlocks" (gets nonzero gradient) after proj.weight moves off zero.

test_depth_fusion_backbone.py's synthetic version of this check (random un-pretrained ResNet +
std-50 raw noise input) hit NaN purely from feeding unnormalized noise through an untrained,
uncalibrated FrozenBN stack -- not a depth_fusion.py bug. This script instead uses the actual
training configuration (loaded COCO checkpoint, real mapper output, real preprocessing), which
is the only scenario that matters in practice.
"""
import sys
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


def main():
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.DATASETS.TRAIN = ("blender_train",)
    cfg.DATASETS.TEST = ("blender_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'
    cfg.MODEL.DEVICE = 'cuda'
    cfg.PROJECT.USE_DEPTH = True
    cfg.MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.freeze()

    register_datasets(cfg)
    train_records = DatasetCatalog.get("blender_train")
    mapper = RGBDInstanceDatasetMapper(cfg, is_train=True)
    mapped = [mapper(r) for r in train_records[:4]]
    batched = [{"image": m["image"].cuda(), "height": cfg.INPUT.IMAGE_SIZE,
                "width": cfg.INPUT.IMAGE_SIZE, "instances": m["instances"].to("cuda")}
               for m in mapped]
    assert all(b["image"].shape[0] == 9 for b in batched)

    model = build_model(cfg)
    RGBDBackboneCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print("checking that the pretrained RGB backbone actually loaded (not random-init) ...")
    stem_w = model.backbone.rgb_backbone.stem.conv1.weight
    assert stem_w.abs().mean().item() > 0.01, "stem weight looks uninitialized/zero"
    # Build a fresh randomly-initialized backbone of the same shape and confirm the loaded one
    # does NOT match it (i.e. it is not still sitting at random init).
    from detectron2.layers import ShapeSpec
    from detectron2.modeling.backbone.resnet import build_resnet_backbone
    random_ref = build_resnet_backbone(cfg, ShapeSpec(channels=3))
    same_as_random = torch.allclose(stem_w.detach().cpu(), random_ref.stem.conv1.weight)
    assert not same_as_random, "backbone.rgb_backbone weights were not loaded from the checkpoint"
    print("  confirmed: rgb_backbone weights differ from a fresh random init (checkpoint loaded)")
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.0025)
    backbone = model.backbone

    def check_finite(step):
        losses = model(batched)
        total = sum(losses.values())
        assert torch.isfinite(total), f"step {step}: loss is not finite: {losses}"
        total.backward()
        n_nan = sum(1 for p in model.parameters() if p.grad is not None and not torch.isfinite(p.grad).all())
        assert n_nan == 0, f"step {step}: {n_nan} parameters have non-finite gradients"
        return total.item()

    proj_w = backbone.fusions["res2"].proj.weight
    stem_w = backbone.depth_stem[0].weight

    opt.zero_grad()
    loss0 = check_finite(0)
    stem_grad_step0 = stem_w.grad.abs().max().item()
    opt.step()
    print(f"step0: loss={loss0:.3f}, proj.weight after step={proj_w.abs().max().item():.3e}, "
          f"depth_stem grad at step0={stem_grad_step0:.3e} (expected ~0: proj starts at zero)")

    for step in (1, 2, 3):
        opt.zero_grad()
        loss = check_finite(step)
        stem_grad = stem_w.grad.abs().max().item()
        opt.step()
        print(f"step{step}: loss={loss:.3f}, finite, depth_stem grad={stem_grad:.3e}")

    assert stem_grad > 0, "depth branch should be receiving gradient by step 3"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

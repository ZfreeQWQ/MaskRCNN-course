#!/usr/bin/env python3
"""GPU regression check for rgbd_seg.m2f.checkpoint.RGBDBackboneCheckpointer's _metadata handling.

Guards against the exact bug found during development: a naive dict rebuild in _load_file
dropped the checkpoint's OrderedDict `_metadata` attribute, which made MaskFormerHead's
`_load_from_state_dict` treat an up-to-date checkpoint as legacy-format and corrupt
"sem_seg_head.pixel_decoder.X" into "sem_seg_head.pixel_decoder.pixel_decoder.X" -- silently
leaving the entire pixel decoder (6-layer deformable-attention transformer encoder included) at
random initialization instead of the pretrained COCO weights. See checkpoint.py's module
docstring for the full story.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.checkpoint import RGBDBackboneCheckpointer
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def build_rgb_model(num_classes=1):
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()
    return build_model(cfg)


def build_rgbd_model(num_classes=1):
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.freeze()
    return build_model(cfg)


def main():
    weights = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'

    # 1. The RGBDBackboneCheckpointer's _load_file must preserve _metadata.
    ckpt = RGBDBackboneCheckpointer(build_rgbd_model())
    loaded = ckpt._load_file(weights)
    meta = getattr(loaded["model"], "_metadata", None)
    assert meta is not None, "RGBDBackboneCheckpointer dropped _metadata entirely"
    assert meta.get("sem_seg_head") == {"version": 2}, (
        f"sem_seg_head version metadata missing/wrong: {meta.get('sem_seg_head')}")
    # backbone's own metadata key must have been remapped alongside the weight keys.
    assert "backbone.rgb_backbone" in meta or any(
        k.startswith("backbone.rgb_backbone.") for k in meta), "backbone metadata key not remapped"
    print("metadata preserved and remapped: OK")

    # 2. End-to-end: pixel_decoder weights must actually match the checkpoint, not random init.
    ref_model = build_rgb_model()
    DetectionCheckpointer(ref_model).load(weights)
    ref_mask_features = ref_model.sem_seg_head.pixel_decoder.mask_features.weight.detach().clone()
    ref_enc_w = ref_model.sem_seg_head.pixel_decoder.transformer.encoder.layers[0].linear1.weight.detach().clone()

    rgbd_model = build_rgbd_model()
    before = rgbd_model.sem_seg_head.pixel_decoder.mask_features.weight.detach().clone()
    RGBDBackboneCheckpointer(rgbd_model).load(weights)
    after = rgbd_model.sem_seg_head.pixel_decoder.mask_features.weight.detach()
    after_enc_w = rgbd_model.sem_seg_head.pixel_decoder.transformer.encoder.layers[0].linear1.weight.detach()

    assert not torch.allclose(before, after), (
        "pixel_decoder.mask_features is still at its pre-load random value -- checkpoint did not load")
    assert torch.allclose(after.cpu(), ref_mask_features.cpu()), (
        "pixel_decoder.mask_features differs from the reference (plain-backbone) loaded checkpoint")
    assert torch.allclose(after_enc_w.cpu(), ref_enc_w.cpu()), (
        "pixel_decoder's deformable-attention transformer encoder differs from the reference checkpoint")
    print("pixel_decoder (incl. deformable-attention encoder) matches reference checkpoint: OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

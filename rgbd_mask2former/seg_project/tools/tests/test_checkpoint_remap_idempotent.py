#!/usr/bin/env python3
"""GPU regression check for rgbd_seg.m2f.checkpoint.M2FCheckpointer's backbone remap.

Guards against the exact bug found during a real test-inference run (not static review): the
original _remap_backbone unconditionally remapped EVERY `backbone.*` key, which is correct when
loading the original plain-ResNet COCO .pkl but WRONG when loading one of our OWN already-trained
checkpoints (which already has `backbone.depth_stem.*` / `backbone.depth_stages.*` /
`backbone.fusions.*` at the top level, not nested under `rgb_backbone`) -- the bug silently
discarded I1's entire trained depth-fusion contribution on every resume/inference load, with no
error, just garbage depth features. See checkpoint.py's _remap_backbone docstring for the full
story.
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.config import get_cfg
from rgbd_seg.m2f.checkpoint import M2FCheckpointer
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone
from detectron2.modeling import build_model


def build_rgbd_model():
    cfg = get_cfg()
    cfg.merge_from_file('seg_project/third_party/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml')
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    cfg.freeze()
    return build_model(cfg)


def main():
    weights = 'seg_project/weights/maskformer2_R50_bs16_50ep.pkl'

    # Step 1: build a model, load the ORIGINAL COCO checkpoint (the already-covered, already-
    # correct case), then give its depth branch and the RGB backbone's own stem known,
    # non-trivial, non-default values so a later silent-discard would be detectable.
    model_a = build_rgbd_model()
    M2FCheckpointer(model_a).load(weights)
    with torch.no_grad():
        for p in model_a.backbone.depth_stem.parameters():
            p.add_(1.0)
        for p in model_a.backbone.depth_stages.parameters():
            p.add_(2.0)
        for p in model_a.backbone.fusions.parameters():
            p.add_(3.0)
        model_a.backbone.rgb_backbone.stem.conv1.weight.add_(5.0)
    saved_depth_stem = {k: v.clone() for k, v in model_a.backbone.depth_stem.state_dict().items()}
    saved_depth_stages = {k: v.clone() for k, v in model_a.backbone.depth_stages.state_dict().items()}
    saved_fusions = {k: v.clone() for k, v in model_a.backbone.fusions.state_dict().items()}
    saved_rgb_stem = model_a.backbone.rgb_backbone.stem.conv1.weight.clone()

    # Save it as OUR OWN checkpoint (this is the scenario the bug broke: loading a checkpoint
    # that already has the correct multi-branch layout).
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = str(Path(tmp) / 'own_checkpoint.pth')
        torch.save({'model': model_a.state_dict()}, ckpt_path)

        # Step 2: a FRESH model (different random init for the depth branch -- COCO loading
        # never touches it), load OUR OWN checkpoint via M2FCheckpointer.
        model_b = build_rgbd_model()
        before_depth_stem = model_b.backbone.depth_stem.state_dict()['0.weight'].clone()
        M2FCheckpointer(model_b).load(ckpt_path)

        after = model_b.backbone.depth_stem.state_dict()
        for k, v in saved_depth_stem.items():
            torch.testing.assert_close(after[k].cpu(), v.cpu(),
                msg=f"depth_stem.{k} was NOT preserved when reloading our own checkpoint "
                    f"(bug: silently discarded and left at a different random init)")
        assert not torch.allclose(before_depth_stem.cpu(), after['0.weight'].cpu()), (
            "sanity check failed: depth_stem didn't actually change between model_b's own "
            "random init and the reloaded value -- the test fixture itself is broken")

        after_stages = model_b.backbone.depth_stages.state_dict()
        for k, v in saved_depth_stages.items():
            torch.testing.assert_close(after_stages[k].cpu(), v.cpu(), msg=f"depth_stages.{k} not preserved")

        after_fusions = model_b.backbone.fusions.state_dict()
        for k, v in saved_fusions.items():
            torch.testing.assert_close(after_fusions[k].cpu(), v.cpu(), msg=f"fusions.{k} not preserved")

        torch.testing.assert_close(
            model_b.backbone.rgb_backbone.stem.conv1.weight.cpu(), saved_rgb_stem.cpu(),
            msg="rgb_backbone.stem not preserved when reloading our own checkpoint")

    print("depth_stem, depth_stages, fusions, and rgb_backbone.stem all correctly preserved "
          "when reloading our own already-trained checkpoint: OK")

    # Step 3: confirm the ORIGINAL-COCO-checkpoint remap path (the non-bug, already-tested-
    # elsewhere case) still works after the fix -- plain ResNet keys must still get remapped.
    model_c = build_rgbd_model()
    M2FCheckpointer(model_c).load(weights)
    assert model_c.backbone.rgb_backbone.stem.conv1.weight.detach().abs().mean().item() > 0.01, (
        "regression: loading the original COCO checkpoint no longer populates rgb_backbone.stem")
    print("original COCO checkpoint still correctly remapped into rgb_backbone.*: OK")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Numerical tests for additive loss, pixel coordinates and aligned RGB-D crops."""
import argparse
import copy
import json
from pathlib import Path

import train_m2f as entry
import numpy as np
import torch
from detectron2.data import DatasetCatalog
from detectron2.projects.point_rend.point_features import point_sample
from mask2former.modeling.matcher import HungarianMatcher
import rgbd_seg.m2f.frequency_fusion
from rgbd_seg.data import register_datasets
from rgbd_seg.m2f.mpformer import MPCriterion
from rgbd_seg.m2f.generalization import (LocalMPCriterion, FocusCropMapper, local_coordinates,
                                         containing_crop, setup_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {}
    rng = np.random.RandomState(92)
    for _ in range(200):
        box = [100, 70, 250, 280]
        x, y, side = containing_crop(box, 1024, 1024, [640, 768, 896], rng)
        assert 0 <= x <= box[0] and x + side >= box[2] and x + side <= 1024
        assert 0 <= y <= box[1] and y + side >= box[3] and y + side <= 1024
    assert containing_crop([0, 0, 1024, 1024], 1024, 1024, [640]) is None
    results["complete_focus_instance_retained"] = True
    mask = torch.zeros(32, 32)
    mask[2, 2] = 1
    mask[10:15, 11:16] = 1
    for dtype in (torch.bool, torch.uint8, torch.float32):
        coords = local_coordinates(mask.to(dtype=dtype))
        labels = point_sample(mask[None, None], coords[None], align_corners=False)[0, 0]
        assert coords.shape == (256, 2) and ((coords >= 0) & (coords <= 1)).all()
        assert torch.equal(labels[:96], torch.ones(96))
        assert (labels[96:] < .5).any()
    results["foreground_centers_and_local_background"] = True

    weights = {"loss_ce": 2., "loss_mask": 5., "loss_dice": 5.}
    matcher = HungarianMatcher(cost_class=2., cost_mask=5., cost_dice=5., num_points=32)
    kwargs = dict(num_classes=1, matcher=matcher, weight_dict=weights, eos_coef=.1,
                  losses=["labels", "masks"], num_points=32, oversample_ratio=3., importance_sample_ratio=.75)
    base = MPCriterion(**kwargs)
    criterion = LocalMPCriterion(**kwargs, max_pairs=2)
    masks = torch.stack([mask, torch.roll(mask, 4, 0)]).bool()
    targets = [{"labels": torch.zeros(2, dtype=torch.long), "masks": masks},
               {"labels": torch.zeros(0, dtype=torch.long), "masks": torch.zeros(0, 32, 32)}]
    outputs = {"pred_logits": torch.randn(2, 4, 2, requires_grad=True),
               "pred_masks": torch.randn(2, 4, 8, 8, requires_grad=True),
               "aux_outputs": [{"pred_logits": torch.randn(2, 4, 2, requires_grad=True),
                                "pred_masks": torch.randn(2, 4, 8, 8, requires_grad=True)}]}
    outputs["pilot_outputs"] = {
        "pred_logits": torch.randn(2, 2, 2, requires_grad=True),
        "pred_masks": torch.randn(2, 2, 8, 8, requires_grad=True),
        "aux_outputs": [{"pred_logits": torch.randn(2, 2, 2, requires_grad=True),
                         "pred_masks": torch.randn(2, 2, 8, 8, requires_grad=True)}]}
    torch.manual_seed(42)
    expected = base(outputs, targets)
    torch.manual_seed(42)
    actual = criterion(outputs, targets)
    assert all(torch.equal(value, actual[name]) for name, value in expected.items())
    assert set(actual) - set(expected) == {"loss_local_bce", "loss_local_dice"}
    local = actual["loss_local_bce"] + actual["loss_local_dice"]
    local.backward()
    assert outputs["pred_masks"].grad.abs().sum() > 0
    assert outputs["pilot_outputs"]["pred_masks"].grad is None
    assert outputs["aux_outputs"][0]["pred_masks"].grad is None
    results["base_and_pilot_losses_bitwise_unchanged"] = True
    results["auxiliary_gradients_only_final_ordinary_masks"] = True
    empty = [{"labels": torch.zeros(0, dtype=torch.long), "masks": torch.zeros(0, 32, 32)}] * 2
    actual = criterion(outputs, empty)
    assert actual["loss_local_bce"] == 0 and actual["loss_local_dice"] == 0
    assert all(torch.isfinite(v) for v in actual.values())
    results["empty_targets_finite_and_same_loss_keys"] = True

    cfg = setup_cfg(args.config)
    register_datasets(cfg)
    mapper = FocusCropMapper(cfg)
    records = DatasetCatalog.get("blender_train")
    picked = next(r for r in records if mapper._focus(r) is not None)
    mapper.focus_probability = 1.
    mapped = mapper(picked)
    assert mapped["focus_crop"] and mapped["image"].shape == (9, 1024, 1024)
    assert len(mapped["instances"]) > 0
    assert mapped["instances"].gt_masks.shape[-2:] == (1024, 1024)
    assert torch.isfinite(mapped["image"]).all()
    # Check every transformed GT box against its visible mask, including crop truncation.
    from detectron2.structures import BitMasks
    boxes = BitMasks(mapped["instances"].gt_masks).get_bounding_boxes().tensor
    assert torch.equal(boxes, mapped["instances"].gt_boxes.tensor)
    results["real_train_rgbd_mapper"] = {"image_id": picked["image_id"],
                                         "instances_after_crop": len(mapped["instances"])}
    # RGB and 6 geometric channels must follow exactly the same sampled transforms.
    original_depth = mapper._depth_geometry_native
    original_read = __import__("rgbd_seg.m2f.generalization", fromlist=["utils"]).utils.read_image
    try:
        height, width = picked["height"], picked["width"]
        marker = np.indices((height, width)).sum(0).astype(np.float32) % 251
        mapper._depth_geometry_native = lambda record: np.repeat(marker[..., None], 6, axis=2)
        utils = __import__("rgbd_seg.m2f.generalization", fromlist=["utils"]).utils
        utils.read_image = lambda *a, **kw: np.repeat(marker[..., None], 3, axis=2).astype(np.uint8)
        # Remove color/resize operations; flips and crop must match pixel-exactly.
        from detectron2.data import transforms as T
        mapper.focus_augs = T.AugmentationList([T.RandomFlip(prob=1., horizontal=True)])
        synthetic = mapper(picked)
        assert torch.equal(synthetic["image"][0], synthetic["image"][3])
    finally:
        mapper._depth_geometry_native = original_depth
        utils.read_image = original_read
    results["native_depth_and_rgb_crop_flip_alignment"] = True
    report = {"valid": True, "tests": results, "test_annotations_accessed": False}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

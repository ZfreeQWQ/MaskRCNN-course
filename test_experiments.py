"""CPU correctness checks; GPU end-to-end smoke training is run separately."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher

from course.augmentation import ExperimentMapper, paste_instance
from course.common import configuration
from course.experiments import mapper
from course.modeling import DepthP2Guide, SmallInstanceCriterion


def criterion(kind, weight=1., fraction=0.):
    obj = kind(1, HungarianMatcher(num_points=32), {"loss_mask": 5., "loss_dice": 5.},
               .1, ["masks"], 32, 3., .75)
    if isinstance(obj, SmallInstanceCriterion):
        obj.small_weight, obj.foreground_fraction = weight, fraction
    return obj


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.records = json.loads(Path("data_cache/index.json").read_text())
        cls.first = cls.records["train"][0]

    def test_baseline_defaults_use_original_mapper(self):
        cfg = configuration("configs/blender_rgbd_r50.yaml", ["MODEL.DEVICE", "cpu"])
        self.assertEqual(mapper(cfg, True).__class__.__name__, "RGBDMapper")
        self.assertEqual(mapper(cfg, False).__class__.__name__, "RGBDMapper")

    def test_configs_are_single_factor(self):
        expected = {
            "e01_small_resample": {"RESAMPLE_SMALL"}, "e02_small_crop": {"CROP_PROB"},
            "e03_rgbd_copy_paste": {"COPY_PASTE_PROB"}, "e04_small_loss": {"SMALL_LOSS_WEIGHT"},
            "e05_foreground_points": {"FOREGROUND_POINT_FRACTION"},
            "e06_depth_p2_gate": {"DEPTH_GATE"}, "e07_resolution1280": {"TEST_SIZE"}}
        base = configuration("configs/blender_rgbd_r50.yaml")
        for path in sorted(Path("configs/experiments").glob("*.yaml")):
            cfg = configuration(str(path))
            changed = {key for key in cfg.COURSE if key != "NAME" and cfg.COURSE[key] != base.COURSE[key]}
            self.assertEqual(changed, expected[path.stem])
            self.assertEqual(cfg.SEED, 42)
            self.assertEqual(cfg.SOLVER.MAX_ITER, 30000)
            self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, 2)

    def test_paste_updates_pixels_depth_and_visible_masks(self):
        rgb, depth = np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 100., np.float32)
        masks = np.zeros((1, 8, 8), bool)
        masks[0, 1:7, 1:7] = True
        patch = np.ones((2, 2), bool)
        rgb, depth, masks, flags, accepted = paste_instance(
            rgb, depth, masks, np.array([False]), np.full((2, 2, 3), 77, np.uint8),
            np.full((2, 2), 90., np.float32), patch, 3, 3)
        self.assertTrue(accepted)
        self.assertTrue((rgb[3:5, 3:5] == 77).all())
        self.assertTrue((depth[3:5, 3:5] == 90).all())
        self.assertFalse(masks[0, 3:5, 3:5].any())
        self.assertEqual(masks[1].sum(), 4)
        self.assertEqual(flags.tolist(), [False, True])

    def test_paste_rejects_behind_or_capacity(self):
        rgb, depth = np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 100., np.float32)
        args = (rgb, depth, np.zeros((0, 8, 8), bool), np.zeros(0, bool),
                np.ones((2, 2, 3), np.uint8), np.full((2, 2), 110., np.float32), np.ones((2, 2), bool), 1, 1)
        self.assertFalse(paste_instance(*args)[-1])
        self.assertFalse(paste_instance(*args, max_instances=0)[-1])

    def test_augmented_mapper_matches_baseline_geometry(self):
        cfg = configuration("configs/experiments/e04_small_loss.yaml")
        original, enhanced = mapper(configuration("configs/blender_rgbd_r50.yaml"), True), mapper(cfg, True)
        for obj in (original, enhanced):
            obj.augmentations = T.AugmentationList([T.HFlipTransform(1024)])
        a, b = original(self.first), enhanced(self.first)
        torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
        torch.testing.assert_close(a["instances"].gt_masks, b["instances"].gt_masks, rtol=0, atol=0)
        torch.testing.assert_close(a["instances"].gt_boxes.tensor, b["instances"].gt_boxes.tensor)
        self.assertEqual(len(b["instances"].gt_small), len(b["instances"]))

    def test_val_augmentation_is_disabled_and_output_geometry_retained(self):
        for path in Path("configs/experiments").glob("*.yaml"):
            cfg = configuration(str(path))
            mapped = mapper(cfg, False)(self.records["val"][0])
            self.assertNotIn("instances", mapped)
            self.assertEqual(tuple(mapped["image"].shape), (4, cfg.COURSE.TEST_SIZE, cfg.COURSE.TEST_SIZE))
            self.assertEqual((mapped["height"], mapped["width"]), (1024, 1024))

    def test_loss_default_is_exactly_official(self):
        outputs = {"pred_masks": torch.randn(1, 2, 8, 8, requires_grad=True)}
        masks = torch.zeros(2, 16, 16)
        masks[0, 2:5, 2:5] = 1
        masks[1, 4:14, 4:14] = 1
        targets = [{"masks": masks, "small": torch.tensor([True, False])}]
        indices = [(torch.arange(2), torch.arange(2))]
        torch.manual_seed(9)
        a = criterion(SetCriterion).loss_masks(outputs, targets, indices, 2)
        torch.manual_seed(9)
        b = criterion(SmallInstanceCriterion).loss_masks(outputs, targets, indices, 2)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for weight, fraction in [(2., 0.), (1., .1)]:
            targets[0]["foreground_coords"] = torch.full((2, 3, 2), 3.5/16)
            loss = criterion(SmallInstanceCriterion, weight, fraction).loss_masks(outputs, targets, indices, 2)
            self.assertTrue(all(torch.isfinite(v) for v in loss.values()))
            sum(loss.values()).backward()
            self.assertTrue(torch.isfinite(outputs["pred_masks"].grad).all())

    def test_depth_gate_initial_identity_and_gradient(self):
        gate = DepthP2Guide(16, 32., 64.)
        feature = torch.randn(2, 16, 16, 16, requires_grad=True)
        depth = torch.randn(2, 1, 64, 64)
        actual = gate(feature, depth)
        torch.testing.assert_close(feature, actual, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertGreater(float(gate.gate[-1].weight.grad.abs().sum()), 0)

    def test_depth_backbone_keeps_pretrained_keys(self):
        torch.manual_seed(42)
        baseline = build_model(configuration("configs/blender_rgbd_r50.yaml", ["MODEL.DEVICE", "cpu"]))
        expected_classifier = baseline.sem_seg_head.predictor.class_embed.weight.detach().clone()
        del baseline
        torch.manual_seed(42)
        cfg = configuration("configs/experiments/e06_depth_p2_gate.yaml", ["MODEL.DEVICE", "cpu"])
        model = build_model(cfg)
        state = model.state_dict()
        self.assertEqual(tuple(state["backbone.stem.conv1.weight"].shape), (64, 4, 7, 7))
        self.assertIn("backbone.res2.0.conv1.weight", state)
        self.assertIn("backbone.depth_guide.gate.3.weight", state)
        weights = torch.load("weights/mask2former_r50_coco_rgbd.pth", map_location="cpu", weights_only=False)["model"]
        self.assertTrue(all(k in state and state[k].shape == v.shape for k, v in weights.items()))
        torch.testing.assert_close(model.sem_seg_head.predictor.class_embed.weight, expected_classifier, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

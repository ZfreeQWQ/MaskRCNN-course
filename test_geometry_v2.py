"""Synthetic geometry and exact baseline compatibility checks; no new data."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from course.common import configuration
from course.experiments import mapper
from course.geometry_v2 import encode_geometry, standardize_geometry
from course.modeling_v2 import GeometryFusion


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_plane_height_normals_and_invalid(self):
        depth = np.ones((96, 96), np.float32) * .5
        depth[35:55, 35:55] = .49
        depth[:4, :4] = 0
        K = [[160, 0, 47.5], [0, 160, 47.5], [0, 0, 1]]
        geometry, audit = encode_geometry(depth, K, 42)
        self.assertTrue(audit["plane_valid"])
        self.assertAlmostEqual(float(geometry[45, 45, 1]), .01, places=5)
        np.testing.assert_allclose(geometry[20, 20, 2:5], [0, 0, -1], atol=1e-5)
        self.assertTrue((geometry[:4, :4, :2] == 0).all())
        self.assertTrue((geometry[:4, :4, 6] == 0).all())
        normalized = standardize_geometry(geometry, np.ones(7), np.ones(7))
        self.assertTrue((normalized[:4, :4] == 0).all())
        second, _ = encode_geometry(depth, K, 42)
        np.testing.assert_array_equal(geometry, second)

    def test_empty_depth_is_finite_and_plane_fallback(self):
        encoded, audit = encode_geometry(np.zeros((32, 32), np.float32), [[100, 0, 16], [0, 100, 16], [0, 0, 1]])
        self.assertFalse(audit["plane_valid"])
        self.assertTrue(np.isfinite(encoded).all())
        self.assertEqual(float(np.abs(encoded).sum()), 0)

    def test_isolated_depth_center_does_not_inherit_neighbor_normal(self):
        depth = np.full((64, 64), .5, np.float32)
        depth[32, 32] = .45
        geometry, _ = encode_geometry(depth, [[100, 0, 32], [0, 100, 32], [0, 0, 1]])
        np.testing.assert_array_equal(geometry[32, 32, 2:5], [0, 0, 0])

    def test_four_scale_initial_identity_and_later_gradients(self):
        fusion = GeometryFusion([32, 64, 128, 256], dropout=0)
        geometry = torch.randn(2, 7, 64, 64)
        features = {n: torch.randn(2, c, 64//s, 64//s) for n, c, s in
                    zip(("res2", "res3", "res4", "res5"), (32, 64, 128, 256), (4, 8, 16, 32))}
        result = fusion(features.copy(), geometry)
        for n in features:
            torch.testing.assert_close(result[n], features[n], rtol=0, atol=0)
        optimizer = torch.optim.SGD(fusion.parameters(), lr=.1)
        for step in range(2):
            optimizer.zero_grad()
            sum(v.square().mean() for v in fusion(features.copy(), geometry).values()).backward()
            if step == 1:
                self.assertGreater(float(fusion.stages[0][0][0].weight.grad.abs().sum()), 0)
                self.assertGreater(float(fusion.spatial_gates[0][0].weight.grad.abs().sum()), 0)
            optimizer.step()
        fusion.enabled = False
        for n, v in fusion(features.copy(), geometry).items():
            torch.testing.assert_close(v, features[n], rtol=0, atol=0)

    def test_mapping_same_rgb_depth_masks_and_original_area(self):
        records = json.loads(Path("data_cache/index.json").read_text())
        base = mapper(configuration("configs/blender_rgbd_r50.yaml"), True)
        candidate = mapper(configuration("configs/v2/d01_geometry_multiscale.yaml"), True)
        for obj in (base, candidate):
            obj.augmentations = T.AugmentationList([T.HFlipTransform(1024)])
        a, b = base(records["train"][0]), candidate(records["train"][0])
        torch.testing.assert_close(a["image"], b["image"][:4], rtol=0, atol=0)
        torch.testing.assert_close(a["instances"].gt_masks, b["instances"].gt_masks, rtol=0, atol=0)
        self.assertEqual(b["image"].shape[0], 11)
        self.assertTrue(torch.equal(b["instances"].gt_small, b["instances"].gt_original_area <= 1024))
        val = mapper(configuration("configs/v2/d01_geometry_multiscale.yaml"), False)(records["val"][0])
        self.assertNotIn("instances", val)
        self.assertEqual(tuple(val["image"].shape), (11, 1024, 1024))

    def test_backbone_retains_weights_classifier_and_exact_initial_features(self):
        torch.manual_seed(42)
        base = build_model(configuration("configs/blender_rgbd_r50.yaml", ["MODEL.DEVICE", "cpu"])).eval()
        torch.manual_seed(42)
        candidate = build_model(configuration("configs/v2/d01_geometry_multiscale.yaml", ["MODEL.DEVICE", "cpu"])).eval()
        torch.testing.assert_close(base.sem_seg_head.predictor.class_embed.weight,
                                   candidate.sem_seg_head.predictor.class_embed.weight, rtol=0, atol=0)
        for model in (base, candidate):
            DetectionCheckpointer(model).load("weights/mask2former_r50_coco_rgbd.pth")
        x = torch.randn(1, 11, 64, 64)
        with torch.no_grad():
            a, b = base.backbone(x[:, :4]), candidate.backbone(x)
        for n in a:
            torch.testing.assert_close(a[n], b[n], rtol=0, atol=0)
        self.assertEqual(candidate.backbone.stem.conv1.weight.shape[1], 4)

    def test_new_module_lr_is_not_backbone_rate(self):
        from train_rgbd import Trainer
        cfg = configuration("configs/v2/d01_geometry_multiscale.yaml", ["MODEL.DEVICE", "cpu"])
        model = build_model(cfg)
        optimizer = Trainer.build_optimizer(cfg, model)
        names = {id(p): n for n, p in model.named_parameters()}
        checked = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                name = names[id(p)]
                if "geometry_fusion" in name:
                    self.assertEqual(group["lr"], 1e-4)
                    checked += 1
                elif name.startswith("backbone."):
                    self.assertEqual(group["lr"], 1e-5)
        self.assertGreater(checked, 0)

    def test_roi_target_boxes_use_padded_canvas_and_original_area(self):
        from detectron2.structures import ImageList
        cfg = configuration("configs/v2/d02_roi_supervision.yaml", ["MODEL.DEVICE", "cpu"])
        record = json.loads(Path("data_cache/index.json").read_text())["train"][0]
        obj = mapper(cfg, True)
        obj.augmentations = T.AugmentationList([T.HFlipTransform(1024)])
        mapped = obj(record)
        instance = mapped["instances"]
        model = build_model(cfg)
        canvas = ImageList.from_tensors([mapped["image"]], size_divisibility=64)
        targets = model.prepare_targets([instance], canvas)
        boxes = instance.gt_boxes.tensor
        size = boxes[:, 2:] - boxes[:, :2]
        expected = torch.cat([boxes[:, :2] - .1*size, boxes[:, 2:] + .1*size], 1).clamp(0, 1024) / 1024
        torch.testing.assert_close(targets[0]["roi_boxes"], expected)
        torch.testing.assert_close(targets[0]["original_area"], instance.gt_original_area)
        self.assertEqual(tuple(targets[0]["masks"].shape[-2:]), (1024, 1024))


if __name__ == "__main__":
    unittest.main()

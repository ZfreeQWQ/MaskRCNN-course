"""Exact input equivalence and real resized inference geometry before V3 runs."""
import json
from pathlib import Path
import unittest
import torch
from detectron2.data import transforms as T
from course.common import configuration
from course.experiments import mapper


class V3DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.records = json.loads(Path("data_cache/index.json").read_text())

    def test_single_read_mapper_is_bitwise_equal(self):
        for training, split in ((True, "train"), (False, "val")):
            old = mapper(configuration("configs/v2/d01_geometry_multiscale.yaml"), training)
            new = mapper(configuration("configs/v3/g00_fast_control.yaml"), training)
            if training:
                for obj in (old, new):
                    obj.augmentations = T.AugmentationList([T.HFlipTransform(1024)])
            for index in (0, len(self.records[split])//2, len(self.records[split])-1):
                a, b = old(self.records[split][index]), new(self.records[split][index])
                torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
                if training:
                    torch.testing.assert_close(a["instances"].gt_masks, b["instances"].gt_masks, rtol=0, atol=0)

    def test_1280_is_actual_input_not_just_report_field(self):
        cfg = configuration("configs/v3/g01_scale1280.yaml")
        result = mapper(cfg, False)(self.records["val"][0])
        self.assertEqual(tuple(result["image"].shape), (11, 1280, 1280))
        self.assertEqual((result["height"], result["width"]), (1024, 1024))
        self.assertNotIn("instances", result)
        self.assertTrue(torch.isfinite(result["image"]).all())
        self.assertTrue(set(result["image"][10].unique().tolist()) <= {0., 1.})


if __name__ == "__main__":
    unittest.main()

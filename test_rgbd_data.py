import json
from pathlib import Path
import unittest

import numpy as np
import torch
from detectron2.data import transforms as T

from course.common import configuration
from course.data import AnnotationReader, RGBDMapper, read_rgbd


class RGBDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(Path("data_cache/index.json").read_text())["train"][0]
        cls.cfg = configuration("configs/blender_rgbd_r50.yaml")

    def test_depth_range(self):
        rgb, depth = read_rgbd(self.record)
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertTrue(np.isfinite(depth).all())
        self.assertTrue((depth >= 0).all() and (depth <= 255).all())

    def test_horizontal_geometry_is_synchronized(self):
        mapper = RGBDMapper(self.cfg)
        mapper.augmentations = T.AugmentationList([T.HFlipTransform(1024)])
        mapped = mapper(self.record)
        rgb, depth = read_rgbd(self.record)
        np.testing.assert_array_equal(mapped["image"][:3].permute(1, 2, 0).numpy(), rgb[:, ::-1])
        np.testing.assert_array_equal(mapped["image"][3].numpy(), depth[:, ::-1])
        self.assertEqual(len(mapped["instances"]), len(AnnotationReader().get(self.record)))

    def test_padding_is_not_foreground_or_fake_depth(self):
        mapper = RGBDMapper(self.cfg)
        mapper.augmentations = T.AugmentationList([
            T.ResizeScale(0.5, 0.5, 1024, 1024),
            T.FixedSizeCrop((1024, 1024), seg_pad_value=0)])
        mapped = mapper(self.record)
        masks = mapped["instances"].gt_masks
        self.assertEqual(tuple(mapped["image"].shape), (4, 1024, 1024))
        self.assertFalse(bool(masks[:, 512:, :].any()))
        self.assertFalse(bool(masks[:, :, 512:].any()))
        self.assertEqual(int(torch.count_nonzero(mapped["image"][3, 512:, :])), 0)

    def test_train_val_disjoint(self):
        index = json.loads(Path("data_cache/index.json").read_text())
        self.assertFalse({item["image_id"] for item in index["train"]} &
                         {item["image_id"] for item in index["val"]})


if __name__ == "__main__":
    unittest.main()

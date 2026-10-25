"""Check every input value at all planned scales before any accuracy run."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from detectron2.data import transforms as T
from course.common import configuration
from course.experiments import mapper
from course.data_v4 import nearest_indices


class CompactInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.records = json.loads(Path("data_cache/index.json").read_text())["val"]

    def test_nearest_map_matches_installed_transform_for_all_pixels(self):
        rng = np.random.default_rng(42)
        source = rng.normal(size=(32, 32, 7)).astype(np.float32)
        for size in (32, 40, 48):
            rows, cols = nearest_indices(32, 32, size)
            expected = T.ResizeTransform(32, 32, size, size).apply_segmentation(source)
            np.testing.assert_array_equal(source[rows[:, None], cols[None, :]], expected)

    def test_actual_rgb_depth_and_geometry_bitwise_equal_at_all_scales(self):
        for size in (1024, 1280, 1536):
            old_cfg = configuration("configs/v3/g00_fast_control.yaml", ["COURSE.TEST_SIZE", str(size)])
            new_cfg = configuration("configs/v4/h00_fast1280.yaml", ["COURSE.TEST_SIZE", str(size)])
            old, new = mapper(old_cfg, False), mapper(new_cfg, False)
            for i in (0, len(self.records)//2, len(self.records)-1):
                before, after = old(self.records[i]), new(self.records[i])
                torch.testing.assert_close(before["image"], after["image"], rtol=0, atol=0)
                self.assertEqual(tuple(after["image"].shape), (11, size, size))
                self.assertEqual((after["height"], after["width"]), (1024, 1024))
                self.assertNotIn("instances", after)


if __name__ == "__main__":
    unittest.main()

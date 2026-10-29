#!/usr/bin/env python3
"""CPU unit checks for rgbd_seg.m2f.geo_transforms.Rot90Transform (no GPU, no dataset)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.geo_transforms import Rot90Transform, RandomRot90


class Rot90TransformTests(unittest.TestCase):
    def test_apply_image_matches_np_rot90_for_each_k(self):
        img = np.arange(3 * 5).reshape(3, 5).astype(np.float32)
        for k in range(4):
            t = Rot90Transform(k, h=3, w=5)
            np.testing.assert_array_equal(t.apply_image(img), np.rot90(img, k, axes=(0, 1)))

    def test_apply_image_multichannel(self):
        # 6 channels, like the depth-geometry stack (height, nx, ny, nz, discontinuity, valid).
        img = np.random.default_rng(0).standard_normal((4, 6, 6)).astype(np.float32)
        t = Rot90Transform(1, h=4, w=6)
        out = t.apply_image(img)
        self.assertEqual(out.shape, (6, 4, 6))
        np.testing.assert_array_equal(out, np.rot90(img, 1, axes=(0, 1)))

    def test_apply_segmentation_matches_apply_image(self):
        img = np.random.default_rng(0).standard_normal((4, 6, 3)).astype(np.float32)
        t = Rot90Transform(3, h=4, w=6)
        np.testing.assert_array_equal(t.apply_segmentation(img), t.apply_image(img))

    def test_coords_corners_k1(self):
        # k=1: a point at the original top-right corner (x=w, y=0) must land at the new
        # top-right corner (x=new_w, y=0) of the (now w-tall, h-wide) rotated image.
        h, w = 3, 5
        t = Rot90Transform(1, h, w)
        coords = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        out = t.apply_coords(coords.copy())
        # new_w, new_h = h, w
        expected = np.array([[0, w], [0, 0], [h, w], [h, 0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected)

    def test_coords_pixel_value_matches_array_rotation(self):
        # Build a small labeled grid, verify a pixel's new array position (via apply_image)
        # agrees with where apply_coords says its (continuous, pixel-center) coordinate moves.
        h, w = 4, 6
        img = np.arange(h * w).reshape(h, w).astype(np.float32)
        for k in range(4):
            t = Rot90Transform(k, h, w)
            rotated = t.apply_image(img)
            for (row, col) in [(0, 0), (1, 4), (3, 5), (2, 2)]:
                value = img[row, col]
                # Pixel-center coordinate (x, y) = (col + 0.5, row + 0.5).
                new_xy = t.apply_coords(np.array([[col + 0.5, row + 0.5]], dtype=np.float32))[0]
                new_col, new_row = int(np.floor(new_xy[0])), int(np.floor(new_xy[1]))
                self.assertEqual(rotated[new_row, new_col], value,
                                  f"k={k} orig=({row},{col}) -> ({new_row},{new_col})")

    def test_inverse_round_trips_coords(self):
        h, w = 3, 5
        coords = np.array([[1.5, 2.0], [4.0, 0.5], [0.0, 0.0], [5.0, 3.0]], dtype=np.float32)
        for k in range(4):
            t = Rot90Transform(k, h, w)
            inv = t.inverse()
            back = inv.apply_coords(t.apply_coords(coords.copy()))
            np.testing.assert_allclose(back, coords, atol=1e-4)

    def test_inverse_shapes_swap_for_odd_k(self):
        t = Rot90Transform(1, h=3, w=5)
        inv = t.inverse()
        self.assertEqual((inv.h, inv.w), (5, 3))
        t2 = Rot90Transform(2, h=3, w=5)
        inv2 = t2.inverse()
        self.assertEqual((inv2.h, inv2.w), (3, 5))

    def test_k0_is_identity(self):
        img = np.random.default_rng(1).standard_normal((4, 6, 2)).astype(np.float32)
        t = Rot90Transform(0, h=4, w=6)
        np.testing.assert_array_equal(t.apply_image(img), img)
        coords = np.array([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(t.apply_coords(coords.copy()), coords)

    def test_random_rot90_only_emits_registered_choices(self):
        aug = RandomRot90(choices=(0, 90))
        img = np.zeros((4, 6, 3), dtype=np.float32)
        seen = {aug.get_transform(img).k for _ in range(50)}
        self.assertTrue(seen <= {0, 1})


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""CPU unit checks for rgbd_seg.depth_geometry using a synthetic scene (no GPU, no dataset)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg import depth_geometry as dg


def synthetic_scene(h=256, w=256, table_z=0.8, box_h=0.05, box_region=(80, 176, 80, 176)):
    """Flat table at z=table_z with a raised box of height box_h in the given pixel region."""
    depth = np.full((h, w), table_z, dtype=np.float32)
    y0, y1, x0, x1 = box_region
    depth[y0:y1, x0:x1] = table_z - box_h  # closer to camera => raised above table
    valid = np.ones((h, w), dtype=bool)
    return depth, valid, box_region, box_h


class DepthGeometryTests(unittest.TestCase):
    def test_backproject_shape_and_pinhole_formula(self):
        depth = np.full((64, 64), 1.0, dtype=np.float32)
        pts = dg.backproject(depth)
        self.assertEqual(pts.shape, (64, 64, 3))
        # Check the pinhole formula directly at an in-bounds pixel (no need to hit the
        # principal point, which lies outside this deliberately small test image).
        px, py = 10, 20
        expected_x = (px - dg.CX) * 1.0 / dg.FX
        expected_y = (py - dg.CY) * 1.0 / dg.FY
        np.testing.assert_allclose(pts[py, px, :2], [expected_x, expected_y], atol=1e-6)
        np.testing.assert_allclose(pts[:, :, 2], 1.0)

    def test_plane_fit_recovers_flat_table(self):
        depth, valid, _, _ = synthetic_scene()
        points = dg.backproject(depth)
        plane = dg.fit_support_plane(points, valid, rng=np.random.default_rng(0))
        self.assertIsNotNone(plane)
        n, d = plane
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-5)
        # Table is frontoparallel (z=const): normal should be ~(0,0,-1) given the camera-facing flip.
        self.assertLess(n[2], 0)
        self.assertGreater(abs(n[2]), 0.99)

    def test_plane_fit_returns_none_with_too_few_points(self):
        depth = np.full((4, 4), 1.0, dtype=np.float32)
        valid = np.zeros((4, 4), dtype=bool)
        valid[0, 0] = True
        points = dg.backproject(depth)
        self.assertIsNone(dg.fit_support_plane(points, valid))

    def test_height_above_plane_matches_known_box_height(self):
        depth, valid, (y0, y1, x0, x1), box_h = synthetic_scene()
        points = dg.backproject(depth)
        plane = dg.fit_support_plane(points, valid, rng=np.random.default_rng(0))
        height = dg.height_above_plane(points, plane)
        table_height = height[10:30, 10:30].mean()
        box_height = height[(y0 + y1) // 2, (x0 + x1) // 2]
        self.assertAlmostEqual(table_height, 0.0, delta=0.01)
        self.assertAlmostEqual(box_height, box_h, delta=0.01)

    def test_discontinuity_peaks_at_box_edge_not_interior(self):
        depth, valid, (y0, y1, x0, x1), _ = synthetic_scene()
        disc = dg.depth_discontinuity(depth, valid)
        edge = disc[y0, (x0 + x1) // 2]
        interior = disc[(y0 + y1) // 2, (x0 + x1) // 2]
        flat_region = disc[10:30, 10:30].mean()
        self.assertGreater(edge, interior)
        self.assertGreater(edge, flat_region)
        self.assertAlmostEqual(interior, 0.0, delta=1e-6)
        self.assertAlmostEqual(flat_region, 0.0, delta=1e-6)

    def test_discontinuity_zero_at_invalid_pixels(self):
        depth, valid, _, _ = synthetic_scene()
        valid = valid.copy()
        valid[0, 0] = False
        disc = dg.depth_discontinuity(depth, valid)
        self.assertEqual(disc[0, 0], 0.0)

    def test_normals_point_toward_camera_on_flat_table(self):
        depth, valid, _, _ = synthetic_scene()
        points = dg.backproject(depth)
        normals = dg.surface_normals(points, valid)
        flat = normals[10:30, 10:30].reshape(-1, 3)
        # Flat frontoparallel table: normals should be close to (0, 0, -1).
        np.testing.assert_allclose(flat[:, 2], -1.0, atol=0.05)
        np.testing.assert_allclose(np.linalg.norm(flat, axis=-1), 1.0, atol=0.05)

    def test_normals_zero_at_invalid_pixels(self):
        depth, valid, _, _ = synthetic_scene()
        valid = valid.copy()
        valid[5, 5] = False
        points = dg.backproject(depth)
        normals = dg.surface_normals(points, valid)
        np.testing.assert_array_equal(normals[5, 5], 0.0)

    def test_encode_end_to_end_shapes_and_invalid_zeroing(self):
        depth, valid, _, _ = synthetic_scene()
        valid = valid.copy()
        valid[0, :10] = False
        out = dg.encode(depth, valid, rng=np.random.default_rng(0))
        self.assertEqual(out['height'].shape, depth.shape)
        self.assertEqual(out['normals'].shape, depth.shape + (3,))
        self.assertEqual(out['discontinuity'].shape, depth.shape)
        self.assertTrue(np.all(out['height'][0, :10] == 0))
        self.assertTrue(np.all(out['normals'][0, :10] == 0))

    def test_standardize_matches_rgb_like_scale(self):
        depth, valid, _, box_h = synthetic_scene()
        out = dg.encode(depth, valid, rng=np.random.default_rng(0))
        # Fake train-derived stats; std=1 heights distributed over [0, box_h].
        stats = {'height': {'mean': 0.0, 'std': box_h / 2},
                  'normals': {'std': 0.3},
                  'discontinuity': {'p99': out['discontinuity'].max() + 1e-6}}
        enc = dg.standardize(out, stats, target_std=50.0)
        self.assertEqual(enc.shape, depth.shape + (6,))
        # Magnitude should land in the same ballpark as RGB after mean-subtraction (std ~ 50).
        box_pixel_height_channel = enc[128, 128, 0]
        self.assertGreater(abs(box_pixel_height_channel), 10.0)
        self.assertLess(abs(box_pixel_height_channel), 500.0)
        # Validity channel (index 5) is a binary indicator, kept at {0, 1} (not rescaled like
        # the continuous geometric channels), since every pixel in this synthetic scene is valid.
        self.assertTrue(set(np.unique(enc[..., 5]).round(3).tolist()) <= {0.0, 1.0})


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""CPU unit checks for rgbd_seg.m2f.small_object_sampling (no GPU, no detectron2 needed)."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.small_object_sampling import (
    allocate_point_budget,
    area_weights,
    expand_bboxes,
    masks_to_normalized_bboxes,
    replace_small_target_points,
    sample_points_in_bboxes,
    small_mask_flags,
)


class MasksToBboxesTests(unittest.TestCase):
    def test_single_square_mask_exact_box(self):
        m = torch.zeros(1, 10, 10)
        m[0, 2:5, 3:7] = 1  # rows 2..4, cols 3..6 (inclusive) -> y in [2,5), x in [3,7)
        boxes, valid = masks_to_normalized_bboxes(m)
        self.assertTrue(valid[0].item())
        x0, y0, x1, y1 = boxes[0].tolist()
        self.assertAlmostEqual(x0, 3 / 10)
        self.assertAlmostEqual(y0, 2 / 10)
        self.assertAlmostEqual(x1, 7 / 10)
        self.assertAlmostEqual(y1, 5 / 10)

    def test_empty_mask_marked_invalid(self):
        m = torch.zeros(2, 8, 8)
        m[1, 3, 3] = 1
        boxes, valid = masks_to_normalized_bboxes(m)
        self.assertFalse(valid[0].item())
        self.assertTrue(valid[1].item())
        torch.testing.assert_close(boxes[0], torch.zeros(4))

    def test_full_mask_covers_whole_image(self):
        m = torch.ones(1, 5, 5)
        boxes, valid = masks_to_normalized_bboxes(m)
        self.assertTrue(valid[0].item())
        torch.testing.assert_close(boxes[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_bool_dtype_input(self):
        m = torch.zeros(1, 10, 10, dtype=torch.bool)
        m[0, 0:2, 0:2] = True
        boxes, valid = masks_to_normalized_bboxes(m)
        self.assertTrue(valid[0].item())
        x0, y0, x1, y1 = boxes[0].tolist()
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(x1, 0.2)
        self.assertAlmostEqual(y1, 0.2)


class ExpandBboxesTests(unittest.TestCase):
    def test_expand_by_20_percent_doubles_each_side_growth(self):
        boxes = torch.tensor([[0.4, 0.4, 0.6, 0.6]])  # width=height=0.2, centered at 0.5,0.5
        out = expand_bboxes(boxes, 0.2)
        # new half-width = 0.1 * 1.2 = 0.12 -> box = [0.38, 0.38, 0.62, 0.62]
        torch.testing.assert_close(out[0], torch.tensor([0.38, 0.38, 0.62, 0.62]), atol=1e-6, rtol=0)

    def test_expand_clips_to_unit_square(self):
        boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        out = expand_bboxes(boxes, 5.0)  # huge expansion
        self.assertTrue((out >= 0).all() and (out <= 1).all())
        x0, y0, x1, y1 = out[0].tolist()
        self.assertEqual(x0, 0.0)
        self.assertEqual(y0, 0.0)

    def test_zero_ratio_is_identity(self):
        boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        out = expand_bboxes(boxes, 0.0)
        torch.testing.assert_close(out, boxes)


class SmallMaskFlagsTests(unittest.TestCase):
    def test_flags_small_and_large_correctly(self):
        m = torch.zeros(2, 64, 64)
        m[0, :10, :10] = 1  # 100 px, small
        m[1, :40, :40] = 1  # 1600 px, not small (threshold 1024)
        flags = small_mask_flags(m, area_threshold_px=1024)
        self.assertTrue(flags[0].item())
        self.assertFalse(flags[1].item())

    def test_boundary_area_equal_to_threshold_is_small(self):
        m = torch.zeros(1, 64, 64)
        m[0, :32, :1] = 1  # exactly 32 px
        flags = small_mask_flags(m, area_threshold_px=32)
        self.assertTrue(flags[0].item())


class SamplePointsInBboxesTests(unittest.TestCase):
    def test_points_land_inside_their_box(self):
        boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.6, 0.9, 0.8]])
        pts = sample_points_in_bboxes(boxes, 50)
        self.assertEqual(pts.shape, (2, 50, 2))
        for i in range(2):
            x0, y0, x1, y1 = boxes[i]
            self.assertTrue((pts[i, :, 0] >= x0 - 1e-6).all() and (pts[i, :, 0] <= x1 + 1e-6).all())
            self.assertTrue((pts[i, :, 1] >= y0 - 1e-6).all() and (pts[i, :, 1] <= y1 + 1e-6).all())

    def test_degenerate_zero_area_box_collapses_without_crashing(self):
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        pts = sample_points_in_bboxes(boxes, 10)
        torch.testing.assert_close(pts, torch.full((1, 10, 2), 0.5))

    def test_generator_gives_reproducible_points(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        pts1 = sample_points_in_bboxes(boxes, 20, generator=g1)
        pts2 = sample_points_in_bboxes(boxes, 20, generator=g2)
        torch.testing.assert_close(pts1, pts2)


class AllocatePointBudgetTests(unittest.TestCase):
    def test_even_split(self):
        self.assertEqual(allocate_point_budget(4, 256), [64, 64, 64, 64])

    def test_remainder_distributed_to_first_targets(self):
        counts = allocate_point_budget(3, 10)
        self.assertEqual(sum(counts), 10)
        self.assertEqual(counts, [4, 3, 3])

    def test_zero_targets_returns_empty(self):
        self.assertEqual(allocate_point_budget(0, 256), [])

    def test_zero_budget_returns_empty(self):
        self.assertEqual(allocate_point_budget(5, 0), [])

    def test_more_targets_than_points_still_sums_correctly(self):
        counts = allocate_point_budget(10, 3)
        self.assertEqual(sum(counts), 3)
        self.assertEqual(len(counts), 10)
        self.assertEqual(sorted(counts), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])



class ReplaceSmallTargetPointsTests(unittest.TestCase):
    def test_small_target_points_land_inside_its_box(self):
        masks = torch.zeros(2, 64, 64)
        masks[0, 10:14, 10:14] = 1  # 16 px, small
        masks[1, :, :] = 1  # full image, not small
        point_coords = torch.rand(2, 100, 2)
        out = replace_small_target_points(point_coords, masks, fraction=0.3, area_threshold_px=1024)
        k = int(round(100 * 0.3))
        # Row 1 (large target) must be completely unchanged.
        torch.testing.assert_close(out[1], point_coords[1])
        # Row 0's untouched head must also be unchanged.
        torch.testing.assert_close(out[0, : 100 - k], point_coords[0, : 100 - k])
        # Row 0's replaced tail must land inside the (expanded) box of its small target.
        bboxes, _ = masks_to_normalized_bboxes(masks[0:1])
        box = expand_bboxes(bboxes, 0.2)[0]
        tail = out[0, 100 - k :]
        self.assertTrue((tail[:, 0] >= box[0] - 1e-6).all() and (tail[:, 0] <= box[2] + 1e-6).all())
        self.assertTrue((tail[:, 1] >= box[1] - 1e-6).all() and (tail[:, 1] <= box[3] + 1e-6).all())

    def test_zero_fraction_is_identity(self):
        masks = torch.zeros(1, 32, 32)
        masks[0, :5, :5] = 1
        point_coords = torch.rand(1, 50, 2)
        out = replace_small_target_points(point_coords, masks, fraction=0.0)
        torch.testing.assert_close(out, point_coords)

    def test_empty_target_batch_returns_input_unchanged(self):
        masks = torch.zeros(0, 32, 32)
        point_coords = torch.zeros(0, 50, 2)
        out = replace_small_target_points(point_coords, masks, fraction=0.3)
        self.assertEqual(out.shape, point_coords.shape)

    def test_no_small_targets_returns_input_unchanged(self):
        masks = torch.ones(2, 32, 32)  # both full-image, neither small
        point_coords = torch.rand(2, 50, 2)
        out = replace_small_target_points(point_coords, masks, fraction=0.3, area_threshold_px=4)
        torch.testing.assert_close(out, point_coords)

    def test_empty_mask_row_is_not_touched(self):
        masks = torch.zeros(1, 32, 32)  # fully empty -> invalid, not "small"
        point_coords = torch.rand(1, 50, 2)
        out = replace_small_target_points(point_coords, masks, fraction=0.3)
        torch.testing.assert_close(out, point_coords)


class AreaWeightsTests(unittest.TestCase):
    def test_small_mask_gets_higher_weight_than_large(self):
        masks = torch.zeros(2, 64, 64)
        masks[0, :4, :4] = 1    # 16 px
        masks[1, :64, :64] = 1  # 4096 px
        w = area_weights(masks, ref=1024.0, power=0.25, max_weight=2.0)
        self.assertGreater(w[0].item(), w[1].item())

    def test_weight_is_clamped_to_bounds(self):
        masks = torch.zeros(2, 64, 64)
        masks[0, 0, 0] = 1       # tiny -> would exceed max_weight without clamping
        masks[1, :, :] = 1       # huge -> ratio < 1, must clamp up to 1.0 not below
        w = area_weights(masks, ref=1024.0, power=0.25, max_weight=2.0)
        self.assertLessEqual(w[0].item(), 2.0)
        self.assertGreaterEqual(w[1].item(), 1.0)

    def test_large_mask_weight_is_exactly_one(self):
        masks = torch.ones(1, 64, 64)  # area 4096 >> ref 1024 -> (ref/area) < 1 -> clamp to 1.0
        w = area_weights(masks, ref=1024.0, power=0.25, max_weight=2.0)
        self.assertAlmostEqual(w[0].item(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()

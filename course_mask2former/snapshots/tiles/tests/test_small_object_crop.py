"""CPU tests requiring the pinned MMDetection/MMCV environment.

Run: python -m unittest discover -s /workspace/scripts/tests \
    -p test_small_object_crop.py -v
No dataset, checkpoint or GPU is required.
"""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from mmdet.datasets.transforms import RandomCrop
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizations.small_object_crop import SmallObjectAwareCrop  # noqa: E402


def make_results(boxes=((78, 60, 83, 65), (0, 0, 40, 40), (20, 50, 23, 53)),
                 shape=(80, 100), scale_factor=(1.0, 1.0), ignored=None):
    height, width = shape
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    masks = np.zeros((len(boxes), height, width), dtype=np.uint8)
    for mask, box in zip(masks, boxes):
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = 1
    image = np.arange(height * width * 3, dtype=np.int32).reshape(height, width, 3)
    return dict(img=image, img_shape=shape, ori_shape=shape,
                scale_factor=scale_factor,
                gt_bboxes=HorizontalBoxes(boxes),
                gt_masks=BitmapMasks(masks, height, width),
                gt_bboxes_labels=np.arange(len(boxes), dtype=np.int64) + 10,
                gt_ignore_flags=np.zeros(len(boxes), dtype=bool) if ignored is None
                else np.asarray(ignored, dtype=bool),
                gt_instances_ids=np.arange(len(boxes), dtype=np.int64) + 100,
                gt_seg_map=np.arange(height * width, dtype=np.int32).reshape(shape))


class TestSmallObjectAwareCrop(unittest.TestCase):
    def assertResultsEqual(self, left, right):
        if left is None or right is None:
            self.assertIs(left, right)
            return
        self.assertEqual(set(left), set(right))
        for key in left:
            a, b = left[key], right[key]
            if isinstance(a, BaseBoxes):
                np.testing.assert_array_equal(a.numpy(), b.numpy(), err_msg=key)
            elif isinstance(a, BitmapMasks):
                self.assertEqual((a.height, a.width), (b.height, b.width))
                np.testing.assert_array_equal(a.masks, b.masks, err_msg=key)
            elif isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b, err_msg=key)
            else:
                self.assertEqual(a, b, key)

    def assertMatchesUpstream(self, results, prob=0, seed=123, **kwargs):
        common = dict(crop_size=(32, 32), allow_negative_crop=True,
                      recompute_bbox=True)
        common.update(kwargs)
        upstream = RandomCrop(**common)
        candidate = SmallObjectAwareCrop(prob=prob, small_area_thr=64, **common)
        np.random.seed(seed)
        expected = upstream(copy.deepcopy(results))
        next_expected = np.random.random()
        np.random.seed(seed)
        actual = candidate(copy.deepcopy(results))
        next_actual = np.random.random()
        self.assertResultsEqual(expected, actual)
        self.assertEqual(next_expected, next_actual, 'fallback consumed extra RNG')
        return actual

    def test_small_mask_preserved_and_annotations_synchronized(self):
        source = make_results(ignored=[False, False, True])
        crop = SmallObjectAwareCrop(crop_size=(32, 32), prob=1,
                                    small_area_thr=64, recompute_bbox=True,
                                    allow_negative_crop=False)
        for seed in range(20):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                result = crop(copy.deepcopy(source))
                self.assertIsNotNone(result)
                self.assertEqual(result['img_shape'], (32, 32))
                chosen = np.flatnonzero(result['gt_bboxes_labels'] == 10)
                self.assertEqual(len(chosen), 1)
                self.assertEqual(result['gt_masks'].areas[chosen[0]], 25)
                x = int(-result['homography_matrix'][0, 2])
                y = int(-result['homography_matrix'][1, 2])
                np.testing.assert_array_equal(result['img'], source['img'][y:y + 32, x:x + 32])
                np.testing.assert_array_equal(result['gt_seg_map'], source['gt_seg_map'][y:y + 32, x:x + 32])
                np.testing.assert_array_equal(result['gt_bboxes'].numpy(),
                result['gt_masks'].get_bboxes('hbox').numpy())
                for row, label in enumerate(result['gt_bboxes_labels']):
                    original = int(label - 10)
                    self.assertEqual(result['gt_instances_ids'][row], original + 100)
                    self.assertEqual(result['gt_ignore_flags'][row], source['gt_ignore_flags'][original])
                    np.testing.assert_array_equal(result['gt_masks'].masks[row],
                                                  source['gt_masks'].masks[original, y:y + 32, x:x + 32])
                self.assertIsNone(crop._guided_offset)

    def test_probability_zero_is_exact_upstream_for_every_crop_type(self):
        variants = [('absolute', (32, 24)), ('absolute_range', (20, 35)),
                    ('relative', (0.4, 0.3)), ('relative_range', (0.3, 0.4))]
        for crop_type, crop_size in variants:
            for seed in (0, 4, 999):
                with self.subTest(crop_type=crop_type, seed=seed):
                    self.assertMatchesUpstream(make_results(), crop_type=crop_type,
                                               crop_size=crop_size, seed=seed)

    def test_no_small_instance_exactly_falls_back(self):
        self.assertMatchesUpstream(make_results(boxes=((0, 0, 40, 40),)), prob=1)

    def test_empty_gt_with_and_without_negative_crops(self):
        for allow_negative in (True, False):
            with self.subTest(allow_negative=allow_negative):
                actual = self.assertMatchesUpstream(make_results(boxes=()), prob=1,
                                                   allow_negative_crop=allow_negative)
                if allow_negative:
                    self.assertEqual(len(actual['gt_masks']), 0)
                else:
                    self.assertIsNone(actual)

    def test_ignored_small_instances_do_not_guide(self):
        self.assertMatchesUpstream(make_results(boxes=((1, 1, 5, 5),), ignored=[True]), prob=1)

    def test_negative_crop_preserves_upstream_contract(self):
        source = make_results(boxes=((0, 0, 4, 4),))
        for allow_negative in (True, False):
            with self.subTest(allow_negative=allow_negative):
                with patch.object(RandomCrop, '_rand_offset', return_value=(48, 68)):
                    actual = self.assertMatchesUpstream(source, prob=0,
                                                       allow_negative_crop=allow_negative)
                if allow_negative:
                    self.assertEqual(len(actual['gt_bboxes']), 0)
                    self.assertEqual(actual['gt_masks'].masks.shape, (0, 32, 32))
                    self.assertEqual(len(actual['gt_bboxes_labels']), 0)
                else:
                    self.assertIsNone(actual)

    def test_original_area_is_backprojected_through_resize(self):
        source = make_results(boxes=((60, 40, 80, 60),), scale_factor=(2.0, 2.0))
        original = SmallObjectAwareCrop(crop_size=(32, 32), prob=1,
                                        small_area_thr=100, area_mode='original')
        current = SmallObjectAwareCrop(crop_size=(32, 32), prob=1,
                                       small_area_thr=100, area_mode='current')
        self.assertEqual(len(original._candidate_ranges(source, (32, 32))), 1)
        self.assertEqual(len(current._candidate_ranges(source, (32, 32))), 0)
        result = original(copy.deepcopy(source))
        self.assertEqual(result['gt_masks'].areas[0], 400)
        # Sampling does not rewrite scale_factor or any evaluator annotations.
        self.assertEqual(result['scale_factor'], (2.0, 2.0))

    def test_uses_mask_area_not_larger_bbox_area(self):
        source = make_results(boxes=((60, 40, 80, 60),))
        source['gt_masks'].masks[0] = 0
        source['gt_masks'].masks[0, 45:49, 65:70] = 1
        crop = SmallObjectAwareCrop(crop_size=(32, 32), prob=1, small_area_thr=32,
                                    recompute_bbox=True)
        self.assertEqual(len(crop._candidate_ranges(source, (32, 32))), 1)
        result = crop(copy.deepcopy(source))
        self.assertEqual(result['gt_masks'].areas[0], 20)
        self.assertEqual(result['gt_bboxes'].areas.numpy()[0], 20)

    def test_bbox_fallback_without_masks(self):
        source = make_results(boxes=((78, 60, 83, 65),))
        del source['gt_masks']
        crop = SmallObjectAwareCrop(crop_size=(32, 32), prob=1, small_area_thr=32)
        result = crop(source)
        self.assertEqual(result['gt_bboxes'].areas.numpy()[0], 25)

    def test_small_area_but_too_long_to_fit_falls_back(self):
        source = make_results(boxes=((0, 0, 50, 1),))
        self.assertMatchesUpstream(source, prob=1)

    def test_numpy_boxes_are_supported_by_inherited_autocast(self):
        source = make_results(boxes=((78, 60, 83, 65),))
        source['gt_bboxes'] = source['gt_bboxes'].numpy()
        crop = SmallObjectAwareCrop(crop_size=(32, 32), prob=1, small_area_thr=32,
                                    recompute_bbox=True)
        result = crop(source)
        self.assertIsInstance(result['gt_bboxes'], np.ndarray)
        self.assertEqual(result['gt_masks'].areas[0], 25)

    def test_image_smaller_than_crop(self):
        source = make_results(boxes=((1, 1, 4, 4),), shape=(10, 15))
        crop = SmallObjectAwareCrop(crop_size=(32, 32), prob=1, small_area_thr=32)
        result = crop(copy.deepcopy(source))
        np.testing.assert_array_equal(result['img'], source['img'])
        self.assertEqual(result['img_shape'], (10, 15))
        self.assertEqual(result['gt_masks'].areas[0], 9)

    def test_registration_and_repr(self):
        self.assertIs(TRANSFORMS.get('SmallObjectAwareCrop'), SmallObjectAwareCrop)
        text = repr(SmallObjectAwareCrop(crop_size=(32, 32), prob=0.5))
        self.assertIn('prob=0.5', text)
        self.assertIn("area_mode='original'", text)

    def test_invalid_parameters_and_scale_are_rejected(self):
        for kwargs in (dict(prob=-0.1), dict(prob=1.1), dict(prob=float('nan')),
                       dict(small_area_thr=0), dict(small_area_thr=float('inf')),
                       dict(area_mode='unknown')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SmallObjectAwareCrop(crop_size=(32, 32), **kwargs)
        crop = SmallObjectAwareCrop(crop_size=(32, 32), prob=1)
        for scale in ((0, 1), (float('nan'), 1), (1, 2, 3)):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                crop(make_results(scale_factor=scale))


if __name__ == '__main__':
    unittest.main()

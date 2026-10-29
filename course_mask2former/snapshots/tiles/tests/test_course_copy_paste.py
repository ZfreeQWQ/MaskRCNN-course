"""Train-only RGB/depth/mask consistency and bounded donor selection tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mmcv
import numpy as np
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks
from pycocotools import mask as mask_utils

from rgbd.course_copy_paste import CourseRGBDSmallCopyPaste, _new_activity
from rgbd.data import PackRGBDDetInputs


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


class CourseCopyPasteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ('annotations', 'rgb/train', 'depth/train'):
            (self.root / directory).mkdir(parents=True)
        self.shape = (8, 10)
        self.donor_mask = np.zeros(self.shape, dtype=np.uint8)
        self.donor_mask[2:4, 3:5] = 1
        self.annotation = dict(
            categories=[dict(id=1, name='component', supercategory='component')],
            images=[dict(id=i, file_name=f'rgb/train/{i}.png',
                         depth_file_name=f'depth/train/{i}.png', width=10, height=8,
                         depth_scale_m=.001, depth_invalid_code=0, split='train')
                    for i in (1, 2)],
            annotations=[dict(id=7, image_id=2, category_id=1, area=4,
                              bbox=[3, 2, 2, 2], iscrowd=0,
                              segmentation=encode(self.donor_mask))])
        self.write_index()
        for i in (1, 2):
            rgb = np.full((*self.shape, 3), 20 if i == 1 else 180, dtype=np.uint8)
            depth = np.full(self.shape, 5000 if i == 1 else 2000, dtype=np.uint16)
            mmcv.imwrite(rgb, str(self.root / f'rgb/train/{i}.png'))
            mmcv.imwrite(depth, str(self.root / f'depth/train/{i}.png'))

    def write_index(self):
        (self.root / 'annotations/instances_train.json').write_text(json.dumps(self.annotation))

    def transform(self, **kwargs):
        options = dict(enabled=True, data_root=str(self.root),
                       ann_file='annotations/instances_train.json', prob=1, max_paste=1)
        options.update(kwargs)
        return CourseRGBDSmallCopyPaste(**options)

    def sample(self, masks=None):
        masks = self.donor_mask[None] if masks is None else np.asarray(masks, dtype=np.uint8)
        bitmap = BitmapMasks(masks.copy(), *self.shape)
        return dict(img=np.full((*self.shape, 3), 20, dtype=np.float32),
                    img_shape=self.shape, ori_shape=self.shape, img_id=1,
                    img_path=str(self.root / 'rgb/train/1.png'),
                    depth_file=str(self.root / 'depth/train/1.png'), depth_scale_m=.001,
                    depth_invalid_code=0, rgbd_depth=np.full(self.shape, 5., dtype=np.float32),
                    rgbd_raw_valid=np.ones(self.shape, dtype=bool),
                    rgbd_label_valid=np.ones(self.shape, dtype=bool),
                    gt_masks=bitmap, gt_bboxes=bitmap.get_bboxes(HorizontalBoxes),
                    gt_bboxes_labels=np.zeros(len(bitmap), dtype=np.int64),
                    gt_ignore_flags=np.zeros(len(bitmap), dtype=bool))

    def direct_paste(self, source, y=2, x=3):
        transform = self.transform()
        transform._load_index()
        donor = transform._load_object(self.annotation['annotations'][0])
        activity = _new_activity()
        result = transform._paste(source, donor, y, x, activity, self.annotation['annotations'][0])
        return result, source, activity

    def test_disabled_exact_identity_no_rng_no_io_no_metadata(self):
        source = self.sample()
        np.random.seed(92)
        expected = np.random.random()
        np.random.seed(92)
        with patch.object(Path, 'resolve', side_effect=AssertionError('filesystem access')):
            with patch.object(Path, 'read_text', side_effect=AssertionError('index read')):
                result = CourseRGBDSmallCopyPaste()(source)
        self.assertIs(result, source)
        self.assertEqual(np.random.random(), expected)
        self.assertNotIn('course_copy_paste_activity', result)

    def test_registered_default_off(self):
        from mmdet.registry import TRANSFORMS
        transform = TRANSFORMS.build(dict(type='CourseRGBDSmallCopyPaste'))
        self.assertFalse(transform.enabled)

    def test_enabled_requires_literal_boolean(self):
        for value in ('false', 'true', 0, 1, None, []):
            with self.subTest(value=value), self.assertRaises(TypeError):
                CourseRGBDSmallCopyPaste(enabled=value)

    def test_rgb_depth_validity_and_mask_are_exactly_aligned(self):
        np.random.seed(5)
        result = self.transform()(self.sample())
        activity = result['course_copy_paste_activity']
        self.assertEqual(activity['pasted_instances'], 1)
        pasted = result['gt_masks'].masks[-1].astype(bool)
        self.assertEqual(pasted.sum(), 4)
        self.assertTrue(np.all(result['img'][pasted] == 180))
        self.assertTrue(np.all(result['rgbd_depth'][pasted] == 2))
        self.assertTrue(result['rgbd_raw_valid'].all())
        self.assertTrue(result['rgbd_label_valid'].all())
        self.assertEqual(result['img'].dtype, np.float32)
        self.assertEqual(result['rgbd_depth'].dtype, np.float32)
        self.assertEqual(result['rgbd_raw_valid'].dtype, bool)
        self.assertEqual(activity['donor_image_ids'], [2])
        self.assertEqual(activity['donor_annotation_ids'], [7])

    def test_fully_occluded_destination_is_removed(self):
        success, result, activity = self.direct_paste(self.sample())
        self.assertTrue(success)
        self.assertEqual(len(result['gt_masks']), 1)
        self.assertEqual(activity['removed_instances'], 1)
        np.testing.assert_array_equal(result['gt_masks'].masks[0], self.donor_mask)

    def test_partial_occlusion_recomputes_truthful_visible_mask_and_box(self):
        mask = np.zeros(self.shape, dtype=np.uint8)
        mask[2:4, 3:7] = 1
        success, result, activity = self.direct_paste(self.sample([mask]))
        self.assertTrue(success)
        self.assertEqual(activity['removed_instances'], 0)
        self.assertEqual(result['gt_masks'].areas.tolist(), [4, 4])
        np.testing.assert_array_equal(result['gt_bboxes'].tensor.numpy(), [[5, 2, 7, 4], [3, 2, 5, 4]])
        self.assertFalse((result['gt_masks'].masks[0] & result['gt_masks'].masks[1]).any())

    def test_empty_destination_can_receive_annotated_object(self):
        result = self.transform()(self.sample(np.empty((0, *self.shape), dtype=np.uint8)))
        self.assertEqual(len(result['gt_masks']), 1)
        self.assertEqual(result['gt_bboxes_labels'].tolist(), [0])

    def test_instance_cap_is_checked_before_pixels_change(self):
        source = self.sample()
        transform = self.transform(max_instances=1)
        transform._load_index()
        donor = transform._load_object(self.annotation['annotations'][0])
        activity = _new_activity()
        before = copy.deepcopy(source)
        self.assertFalse(transform._paste(source, donor, 0, 0, activity, self.annotation['annotations'][0]))
        np.testing.assert_array_equal(source['img'], before['img'])
        self.assertEqual(activity['cap_skipped'], 1)
        # Replacing a fully hidden instance does not exceed the cap.
        self.assertTrue(transform._paste(source, donor, 2, 3, activity, self.annotation['annotations'][0]))
        self.assertEqual(len(source['gt_masks']), 1)

    def test_optional_ids_are_unique_and_follow_dropped_masks(self):
        source = self.sample()
        source['gt_instances_ids'] = np.array([20], dtype=np.int64)
        _, result, _ = self.direct_paste(source)
        self.assertEqual(result['gt_instances_ids'].tolist(), [-1])
        _, result, _ = self.direct_paste(result, 0, 0)
        self.assertEqual(result['gt_instances_ids'].tolist(), [-1, -2])
        self.assertEqual(len(result['gt_instances_ids']), len(result['gt_masks']))

    def test_ndarray_bbox_and_instance_dtypes_preserved(self):
        source = self.sample()
        source['gt_bboxes'] = source['gt_bboxes'].tensor.numpy().astype(np.float64)
        source['gt_bboxes_labels'] = source['gt_bboxes_labels'].astype(np.int32)
        _, result, _ = self.direct_paste(source)
        self.assertEqual(result['gt_bboxes'].dtype, np.float64)
        self.assertEqual(result['gt_bboxes_labels'].dtype, np.int32)
        self.assertEqual(result['gt_ignore_flags'].dtype, bool)
        self.assertEqual(len(result['gt_bboxes']), len(result['gt_ignore_flags']))

    def test_invalid_or_missing_donor_depth_rejects_whole_object(self):
        raw = np.full(self.shape, 2000, dtype=np.uint16)
        raw[2, 3] = 0
        mmcv.imwrite(raw, str(self.root / 'depth/train/2.png'))
        source = self.sample()
        result = self.transform()(source)
        self.assertEqual(result['course_copy_paste_activity']['pasted_instances'], 0)
        self.assertEqual(result['course_copy_paste_activity']['invalid_depth_objects'], 1)
        self.assertTrue(np.all(result['img'] == 20))

    def test_invalid_destination_depth_is_not_guessed(self):
        for value in (0, -1, float('nan'), float('inf')):
            source = self.sample()
            source['rgbd_depth'][:] = value
            result = self.transform()(source)
            self.assertEqual(result['course_copy_paste_activity']['pasted_instances'], 0)
            self.assertGreater(result['course_copy_paste_activity']['depth_rejected_placements'], 0)

    def test_unmeasured_destination_depth_is_rejected(self):
        source = self.sample()
        source['rgbd_raw_valid'][:] = False
        result = self.transform()(source)
        self.assertEqual(result['course_copy_paste_activity']['pasted_instances'], 0)

    def test_donor_behind_destination_is_rejected_without_depth_translation(self):
        source = self.sample()
        source['rgbd_depth'][:] = 1.
        result = self.transform()(source)
        self.assertEqual(result['course_copy_paste_activity']['pasted_instances'], 0)
        self.assertTrue(np.all(result['rgbd_depth'] == 1))

    def test_robust_and_pixelwise_front_checks_both_apply(self):
        transform = self.transform()
        transform._load_index()
        donor = transform._load_object(self.annotation['annotations'][0])
        source = self.sample()
        self.assertTrue(transform._placement_is_front(source, donor, 0, 0))
        source['rgbd_depth'][0, 0] = 1.
        self.assertFalse(transform._placement_is_front(source, donor, 0, 0))

    def test_no_candidates_preserves_negative_frame(self):
        self.annotation['annotations'] = []
        self.write_index()
        result = self.transform()(self.sample(np.empty((0, *self.shape), dtype=np.uint8)))
        self.assertEqual(len(result['gt_masks']), 0)
        self.assertEqual(result['course_copy_paste_activity']['donor_attempts'], 0)

    def test_same_source_image_is_not_used_as_donor(self):
        self.annotation['annotations'][0]['image_id'] = 1
        self.write_index()
        result = self.transform()(self.sample())
        self.assertEqual(result['course_copy_paste_activity']['pasted_instances'], 0)
        self.assertEqual(result['course_copy_paste_activity']['donor_attempts'], 12)

    def test_probability_zero_no_draw_but_has_activity(self):
        np.random.seed(49)
        expected = np.random.random()
        np.random.seed(49)
        result = self.transform(prob=0)(self.sample())
        self.assertEqual(np.random.random(), expected)
        self.assertEqual(result['course_copy_paste_activity']['probability_skipped'], 1)

    def test_deterministic_seed_produces_same_pixels_masks_activity_and_rng(self):
        np.random.seed(39)
        a = self.transform()(self.sample())
        after_a = np.random.random()
        np.random.seed(39)
        b = self.transform()(self.sample())
        self.assertEqual(np.random.random(), after_a)
        for key in ('img', 'rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a['gt_masks'].masks, b['gt_masks'].masks)
        self.assertEqual(a['course_copy_paste_activity'], b['course_copy_paste_activity'])

    def test_repeated_candidate_is_never_pasted_twice(self):
        result = self.transform(max_paste=3)(self.sample())
        activity = result['course_copy_paste_activity']
        self.assertEqual(activity['pasted_instances'], 1)
        self.assertLessEqual(activity['placement_attempts'], 8)
        self.assertLessEqual(activity['donor_attempts'], 12)

    def test_three_paste_limit_and_annotation_cardinality(self):
        self.annotation['annotations'] = [dict(self.annotation['annotations'][0], id=i) for i in range(7, 17)]
        self.write_index()
        result = self.transform(max_paste=3)(self.sample())
        activity = result['course_copy_paste_activity']
        self.assertLessEqual(activity['pasted_instances'], 3)
        self.assertGreater(activity['pasted_instances'], 0)
        n = len(result['gt_masks'])
        self.assertLessEqual(n, 100)
        self.assertEqual(n, len(result['gt_bboxes']))
        self.assertEqual(n, len(result['gt_bboxes_labels']))
        self.assertEqual(n, len(result['gt_ignore_flags']))

    def test_bad_probability_and_unbounded_settings_rejected(self):
        for options in (dict(prob=-.1), dict(prob=float('nan')), dict(max_paste=4),
                        dict(max_instances=101), dict(max_donor_attempts=100),
                        dict(max_position_attempts=100), dict(area_threshold=1025),
                        dict(front_margin_m=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.transform(**options)

    def test_val_test_arbitrary_or_escaping_annotation_rejected_before_read(self):
        for name in ('annotations/instances_val.json', 'annotations/instances_test.json',
                     'annotations/anything.json', '../instances_train.json', '/tmp/instances_train.json'):
            with self.subTest(name=name), patch.object(Path, 'read_text', side_effect=AssertionError('read')):
                with self.assertRaises(ValueError):
                    self.transform(ann_file=name)(self.sample())

    def test_cross_split_rgb_depth_and_traversal_rejected_before_image_read(self):
        for field, bad in (('file_name', 'rgb/val/2.png'), ('depth_file_name', 'depth/test/2.png'),
                           ('file_name', '../outside.png'), ('file_name', '/tmp/outside.png')):
            original = self.annotation['images'][1][field]
            self.annotation['images'][1][field] = bad
            self.write_index()
            with self.subTest(field=field, bad=bad), patch.object(mmcv, 'imread', side_effect=AssertionError('read')):
                with self.assertRaises(ValueError):
                    self.transform()(self.sample())
            self.annotation['images'][1][field] = original

    def test_symlink_escape_rejected(self):
        (self.root / 'rgb/train/escape').symlink_to(self.root / 'depth')
        self.annotation['images'][1]['file_name'] = 'rgb/train/escape/2.png'
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'train root'):
            self.transform()(self.sample())

    def test_destination_id_paths_or_geometry_mismatch_rejected(self):
        cases = [dict(img_id=9), dict(img_path=str(self.root/'rgb/val/1.png')),
                 dict(depth_file=str(self.root/'depth/val/1.png')),
                 dict(depth_scale_m=1.), dict(depth_invalid_code=65535),
                 dict(flip=True), dict(scale_factor=(2., 2.)), dict(homography_matrix=np.eye(3))]
        for changes in cases:
            source = self.sample()
            source.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.transform()(source)

    def test_full_label_supervision_is_required_and_not_sensor_masked(self):
        source = self.sample()
        source['rgbd_label_valid'][0, 0] = False
        with self.assertRaisesRegex(ValueError, 'label-valid'):
            self.transform()(source)

    def test_bad_category_or_duplicate_index_is_rejected(self):
        self.annotation['annotations'][0]['category_id'] = 2
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'category'):
            self.transform()(self.sample())
        self.annotation['annotations'][0]['category_id'] = 1
        self.annotation['images'].append(self.annotation['images'][0])
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.transform()(self.sample())

    def test_corrupt_decoded_area_is_not_silently_used(self):
        self.annotation['annotations'][0]['area'] = 3
        self.write_index()
        with self.assertRaisesRegex(ValueError, 'area'):
            self.transform()(self.sample())

    def test_destination_instance_cardinality_is_checked(self):
        source = self.sample()
        source['gt_bboxes_labels'] = np.zeros(2, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'cardinality'):
            self.transform()(source)

    def test_pack_preserves_activity_for_main_process_aggregation(self):
        source = self.transform()(self.sample())
        packed = PackRGBDDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape',
                                              'img_shape', 'course_copy_paste_activity'))(source)
        self.assertEqual(packed['data_samples'].metainfo['course_copy_paste_activity'],
                         source['course_copy_paste_activity'])
        self.assertTrue(packed['data_samples'].label_valid.data.all())


if __name__ == '__main__':
    unittest.main(verbosity=2)

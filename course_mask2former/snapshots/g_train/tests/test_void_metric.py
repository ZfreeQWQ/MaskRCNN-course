"""Synthetic-only tests of the explicitly nonofficial void-aware AP protocol."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbd.metric import VoidAwareCocoMetric, clip_masks_to_valid  # noqa: E402


class VoidMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.valid = np.ones((8, 10), dtype=bool)
        self.valid[:, 6:] = False
        self.target = np.zeros((8, 10), dtype=np.uint8)
        self.target[2:6, 2:5] = 1
        np.save(self.root / 'valid.npy', self.valid)
        rle = mask_utils.encode(np.asfortranarray(self.target))
        rle['counts'] = rle['counts'].decode('ascii')
        self.annotation = dict(
            info=dict(description='Synthetic test only'),
            images=[dict(id=7, width=10, height=8, file_name='unused.png', label_valid_file='valid.npy')],
            categories=[dict(id=1, name='thing')],
            annotations=[dict(id=1, image_id=7, category_id=1, iscrowd=0,
                              bbox=[2, 2, 3, 4], area=12, segmentation=rle)])
        self.ann_file = self.root / 'annotations.json'
        self.write_annotations()

    def write_annotations(self):
        self.ann_file.write_text(json.dumps(self.annotation), encoding='utf-8')

    def metric(self, **kwargs):
        metric = VoidAwareCocoMetric(ann_file=str(self.ann_file), data_root=str(self.root), **kwargs)
        metric.dataset_meta = dict(classes=('thing',))
        return metric

    def sample(self, numpy=False, empty=False):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0] = self.target.astype(bool)
        masks[:, :, 6:] = True  # Second prediction is wholly unlabelled.
        pred = dict(masks=masks, bboxes=np.array([[0, 0, 10, 8], [6, 0, 10, 8]], dtype=np.float32),
                    scores=np.array([0.9, 0.99], dtype=np.float32), labels=np.array([0, 0], dtype=np.int64),
                    mask_scores=np.array([0.8, 0.98], dtype=np.float32))
        if empty:
            pred = {key: value[:0] for key, value in pred.items()}
        if not numpy:
            pred = {key: torch.from_numpy(value.copy()) for key, value in pred.items()}
        return dict(img_id=7, ori_shape=(8, 10), pred_instances=pred)

    def test_numpy_and_torch_masks_clip_without_input_mutation(self):
        for numpy in (True, False):
            with self.subTest(numpy=numpy):
                sample = self.sample(numpy=numpy)
                original = np.asarray(sample['pred_instances']['masks']).copy()
                metric = self.metric()
                metric.process({}, [sample])
                result = metric.results[0][1]
                self.assertEqual(len(result['masks']), 1)
                self.assertTrue(np.array_equal(mask_utils.decode(result['masks'][0]), self.target))
                self.assertTrue(np.array_equal(result['scores'], np.array([0.9], dtype=np.float32)))
                self.assertTrue(np.array_equal(result['mask_scores'], np.array([0.8], dtype=np.float32)))
                self.assertTrue(np.array_equal(result['labels'], np.array([0], dtype=np.int64)))
                self.assertTrue(np.array_equal(np.asarray(sample['pred_instances']['masks']), original))

    def test_clipped_prediction_has_perfect_development_ap(self):
        metric = self.metric()
        metric.process({}, [self.sample()])
        result = metric.compute_metrics(metric.results)
        self.assertEqual(result['segm_mAP'], 1.0)
        self.assertEqual(result['segm_mAP_s'], 1.0)
        self.assertEqual(result['segm_mAP_m'], -1.0)
        self.assertEqual(metric.prefix, 'nyuv2_void')

    def test_all_predictions_in_void_produce_zero_ap_not_missing_metrics(self):
        sample = self.sample()
        sample['pred_instances']['masks'][:, :, :6] = False
        metric = self.metric()
        metric.process({}, [sample])
        self.assertEqual(len(metric.results[0][1]['masks']), 0)
        result = metric.compute_metrics(metric.results)
        self.assertEqual(result['segm_mAP'], 0)
        self.assertEqual(result['segm_mAP_s'], 0)
        self.assertEqual(result['segm_mAP_m'], -1)

    def test_originally_empty_predictions_supported(self):
        metric = self.metric()
        metric.process({}, [self.sample(empty=True)])
        self.assertEqual(metric.compute_metrics(metric.results)['segm_mAP'], 0)

    def test_gt_in_void_or_incorrect_area_fails(self):
        for wrong_area in (False, True):
            with self.subTest(wrong_area=wrong_area):
                if wrong_area:
                    self.annotation['annotations'][0]['area'] = 80
                    np.save(self.root / 'valid.npy', self.valid)
                else:
                    invalid = self.valid.copy()
                    invalid[2, 2] = False
                    np.save(self.root / 'valid.npy', invalid)
                self.write_annotations()
                with self.assertRaises(ValueError):
                    self.metric().process({}, [self.sample()])

    def test_resized_predictions_or_wrong_ori_shape_fail(self):
        sample = self.sample()
        sample['pred_instances']['masks'] = torch.zeros((2, 4, 5), dtype=torch.bool)
        with self.assertRaises(ValueError):
            self.metric().process({}, [sample])
        sample = self.sample()
        sample['ori_shape'] = (4, 5)
        with self.assertRaises(ValueError):
            self.metric().process({}, [sample])

    def test_missing_valid_path_or_escaping_path_fail(self):
        for path in (None, '../valid.npy', str(self.root / 'valid.npy')):
            with self.subTest(path=path):
                if path is None:
                    self.annotation['images'][0].pop('label_valid_file', None)
                else:
                    self.annotation['images'][0]['label_valid_file'] = path
                self.write_annotations()
                with self.assertRaises(ValueError):
                    self.metric().process({}, [self.sample()])

    def test_probability_masks_are_not_silently_thresholded(self):
        masks = np.full((1, 8, 10), 0.5, dtype=np.float32)
        with self.assertRaises(ValueError):
            clip_masks_to_valid(masks, self.valid)

    def test_wrong_valid_shape_and_nonfinite_data_fail(self):
        for invalid in (np.ones((4, 5)), np.full((8, 10), float('nan'))):
            np.save(self.root / 'valid.npy', invalid)
            with self.assertRaises(ValueError):
                self.metric().process({}, [self.sample()])

    def test_bbox_metric_cannot_accidentally_mix_protocols(self):
        with self.assertRaises(ValueError):
            self.metric(metric='bbox')


if __name__ == '__main__':
    unittest.main(verbosity=2)

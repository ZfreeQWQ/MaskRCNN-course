"""CPU RGB-D alignment, normalization, disabled/zero-init equivalence tests.

Run in the pinned environment: python -m unittest discover -s scripts/tests
    -p test_rgbd_pipeline.py -v
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mmcv
import numpy as np
import torch
from torch import nn
from mmengine.structures import InstanceData, PixelData
from mmdet.datasets.transforms import RandomCrop, Resize
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.detectors import Mask2Former
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbd.data import (LoadRGBDImageFromFile, PackRGBDDetInputs, RGBDCocoDataset,
                       RGBDRandomCrop, RGBDRandomFlip, RGBDResize,
                       RGBDSmallObjectAwareCrop)
from rgbd.model import RGBDDataPreprocessor, RGBDMask2Former


def setUpModule():
    register_all_modules(init_default_scope=True)


def paired_sample(shape=(8, 10)):
    height, width = shape
    depth = (np.arange(height * width, dtype=np.float32).reshape(shape) + 1) / 10
    mask = np.zeros((1, height, width), dtype=np.uint8)
    mask[0, 2:4, 3:5] = 1
    return dict(img=np.repeat(depth[..., None], 3, axis=2), img_shape=shape,
                ori_shape=shape, img_id=1, img_path='fixture.png',
                scale_factor=(1.0, 1.0),
                rgbd_depth=depth, rgbd_raw_valid=(depth * 10).astype(int) % 2 == 0,
                rgbd_label_valid=(depth * 10).astype(int) % 3 == 0,
                gt_bboxes=HorizontalBoxes([[3, 2, 5, 4]]),
                gt_bboxes_labels=np.array([0], dtype=np.int64),
                gt_ignore_flags=np.array([False]),
                gt_masks=BitmapMasks(mask, height, width))


def input_sample(height=3, width=5):
    sample = DetDataSample(metainfo=dict(img_shape=(height, width), ori_shape=(height, width)))
    sample.gt_instances = InstanceData(bboxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.long))
    sample.label_valid = PixelData(data=torch.ones((1, height, width), dtype=torch.bool))
    data = torch.arange(5 * height * width, dtype=torch.float32).reshape(5, height, width)
    data[3] = 2.5
    data[4] = 1
    return data, sample


class TestRGBDGeometry(unittest.TestCase):
    def test_every_flip_direction_exactly_aligned(self):
        source = paired_sample()
        for direction in ('horizontal', 'vertical', 'diagonal'):
            with self.subTest(direction=direction):
                result = RGBDRandomFlip(prob=1.0, direction=direction)(copy.deepcopy(source))
                for key in ('img', 'rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
                    np.testing.assert_array_equal(result[key], mmcv.imflip(source[key], direction))

    def test_flip_off_preserves_arrays(self):
        source = paired_sample()
        result = RGBDRandomFlip(prob=0.0)(copy.deepcopy(source))
        for key in ('img', 'rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
            np.testing.assert_array_equal(result[key], source[key])

    def test_resize_rgb_matches_upstream_and_validity_remains_binary(self):
        source = paired_sample()
        for keep_ratio in (False, True):
            with self.subTest(keep_ratio=keep_ratio):
                result = RGBDResize(scale=(13, 9), keep_ratio=keep_ratio)(copy.deepcopy(source))
                reference = Resize(scale=(13, 9), keep_ratio=keep_ratio)(copy.deepcopy(source))
                np.testing.assert_array_equal(result['img'], reference['img'])
                height, width = result['img_shape']
                expected_depth = mmcv.imresize(source['rgbd_depth'], (width, height), interpolation='bilinear')
                np.testing.assert_allclose(result['rgbd_depth'], expected_depth)
                for key in ('rgbd_raw_valid', 'rgbd_label_valid'):
                    expected = mmcv.imresize(source[key].astype(np.uint8), (width, height), interpolation='nearest')
                    np.testing.assert_array_equal(result[key], expected.astype(bool))
                    self.assertEqual(result[key].dtype, bool)

    def test_resize_invalid_depth_does_not_bleed_zero_or_nan(self):
        source = paired_sample((2, 2))
        source['rgbd_depth'] = np.array([[2, np.nan], [2, 0]], dtype=np.float32)
        source['rgbd_raw_valid'][:] = True
        result = RGBDResize(scale=(8, 8))(source)
        self.assertTrue(np.isfinite(result['rgbd_depth']).all())
        self.assertTrue(set(np.unique(result['rgbd_depth'])).issubset({0.0, 2.0}))
        self.assertFalse(result['rgbd_raw_valid'][result['rgbd_depth'] == 0].any())

    def test_standard_crop_exactly_matches_upstream_rng(self):
        source = paired_sample()
        np.random.seed(7)
        expected = RandomCrop(crop_size=(5, 6), allow_negative_crop=True)(copy.deepcopy(source))
        next_expected = np.random.random()
        np.random.seed(7)
        actual = RGBDRandomCrop(crop_size=(5, 6), allow_negative_crop=True)(copy.deepcopy(source))
        self.assertEqual(np.random.random(), next_expected)
        np.testing.assert_array_equal(actual['img'], expected['img'])
        x = int(-actual['homography_matrix'][0, 2])
        y = int(-actual['homography_matrix'][1, 2])
        # MMDetection absolute crop_size is (width, height), whereas array
        # slices and the protected _crop_data argument are (height, width).
        self.assertEqual(actual['img_shape'], (6, 5))
        for key in ('rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
            np.testing.assert_array_equal(actual[key], source[key][y:y + 6, x:x + 5])

    def test_small_crop_on_keeps_instance_and_paired_planes(self):
        source = paired_sample()
        for seed in range(5):
            np.random.seed(seed)
            actual = RGBDSmallObjectAwareCrop(crop_size=(4, 4), prob=1,
                                              small_area_thr=8, recompute_bbox=True)(copy.deepcopy(source))
            self.assertEqual(actual['gt_masks'].areas.tolist(), [4])
            x = int(-actual['homography_matrix'][0, 2])
            y = int(-actual['homography_matrix'][1, 2])
            for key in ('rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
                np.testing.assert_array_equal(actual[key], source[key][y:y + 4, x:x + 4])

    def test_small_crop_off_matches_paired_upstream(self):
        source = paired_sample()
        np.random.seed(31)
        expected = RGBDRandomCrop(crop_size=(4, 4), allow_negative_crop=True)(copy.deepcopy(source))
        state = np.random.random()
        np.random.seed(31)
        actual = RGBDSmallObjectAwareCrop(crop_size=(4, 4), prob=0,
                                          allow_negative_crop=True)(copy.deepcopy(source))
        self.assertEqual(np.random.random(), state)
        for key in ('img', 'rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_rejected_crop_restores_three_channel_contract(self):
        source = paired_sample()
        with patch.object(RandomCrop, '_rand_offset', return_value=(6, 8)):
            result = RGBDRandomCrop(crop_size=(2, 2), allow_negative_crop=False)(source)
        self.assertIsNone(result)
        self.assertEqual(source['img'].shape[-1], 3)
        self.assertEqual(source['rgbd_depth'].shape, source['img'].shape[:2])

    def test_composed_geometry_keeps_depth_equal_to_rgb_and_masks(self):
        source = paired_sample()
        for transform in (RGBDRandomFlip(prob=1.0), RGBDResize(scale=(20, 16)),
                          RGBDRandomCrop(crop_size=(10, 12), allow_negative_crop=True)):
            source = transform(source)
            np.testing.assert_allclose(source['img'][..., 0], source['rgbd_depth'])
            self.assertEqual(source['gt_masks'].masks.shape[-2:], source['rgbd_label_valid'].shape)
        packed = PackRGBDDetInputs()(source)
        self.assertEqual(packed['inputs'].shape[0], 5)
        np.testing.assert_array_equal(packed['data_samples'].label_valid.data.numpy()[0],
                                      source['rgbd_label_valid'])

    def test_misalignment_is_rejected(self):
        source = paired_sample()
        source['rgbd_depth'] = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            PackRGBDDetInputs()(source)

    def test_random_resize_builds_registered_paired_resize(self):
        transform = TRANSFORMS.build(dict(type='RandomResize', scale=(20, 16),
                                           ratio_range=(1.0, 1.0),
                                           resize_type='RGBDResize', keep_ratio=True))
        result = transform(paired_sample())
        np.testing.assert_allclose(result['img'][..., 0], result['rgbd_depth'])
        self.assertEqual(result['img_shape'], (16, 20))


class TestRGBDLoading(unittest.TestCase):
    def test_loader_cleans_invalid_depth_but_preserves_inpainting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mmcv.imwrite(np.zeros((2, 3, 3), dtype=np.uint8), str(root / 'rgb.png'))
            np.save(root / 'depth.npy', np.array([[np.nan, np.inf, -1], [0, 2, 4]], dtype=np.float32))
            np.save(root / 'raw.npy', np.array([[1, 1, 1], [1, 1, 0]], dtype=np.uint8))
            np.save(root / 'label.npy', np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8))
            result = LoadRGBDImageFromFile()(dict(img_path=str(root / 'rgb.png'),
                depth_file=str(root / 'depth.npy'), raw_valid_file=str(root / 'raw.npy'),
                label_valid_file=str(root / 'label.npy')))
            np.testing.assert_array_equal(result['rgbd_depth'], [[0, 0, 0], [0, 2, 4]])
            np.testing.assert_array_equal(result['rgbd_raw_valid'], [[False, False, False], [False, True, False]])
            self.assertEqual(result['img'].dtype, np.float32)

    def test_dataset_keeps_paths_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = dict(images=[dict(id=1, file_name='rgb.png', height=8, width=10,
                                          depth_file='depth/a.npy', raw_valid_file='raw/a.npy',
                                          label_valid_file='label/a.npy', scene='room_a')],
                              annotations=[], categories=[dict(id=1, name='chair')])
            (root / 'ann.json').write_text(json.dumps(annotation))
            dataset = RGBDCocoDataset(data_root=str(root), ann_file='ann.json',
                                      data_prefix=dict(img='rgb/'), pipeline=[],
                                      metainfo=dict(classes=('chair',)), test_mode=True)
            item = dataset.get_data_info(0)
            self.assertEqual(item['depth_file'], str(root / 'depth/a.npy'))
            self.assertEqual(item['img_path'], str(root / 'rgb/rgb.png'))
            self.assertEqual(item['scene'], 'room_a')
            annotation['images'][0]['depth_file'] = '../outside.npy'
            (root / 'ann.json').write_text(json.dumps(annotation))
            with self.assertRaisesRegex(ValueError, 'relative'):
                RGBDCocoDataset(data_root=str(root), ann_file='ann.json',
                                data_prefix=dict(img='rgb/'), pipeline=[],
                                metainfo=dict(classes=('chair',)), test_mode=True)


class TestRGBDPreprocessor(unittest.TestCase):
    def test_rgb_exact_upstream_and_extras_padded(self):
        items, samples = zip(input_sample(3, 5), input_sample(2, 4))
        options = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
                       bgr_to_rgb=True, pad_size_divisor=4)
        reference = DetDataPreprocessor(**options)(
            dict(inputs=[item[:3].clone() for item in items], data_samples=copy.deepcopy(samples)))
        actual = RGBDDataPreprocessor(**options)(
            dict(inputs=list(items), data_samples=copy.deepcopy(samples)))
        torch.testing.assert_close(actual['inputs'][:, :3], reference['inputs'], rtol=0, atol=0)
        self.assertEqual(actual['inputs'].shape, (2, 5, 4, 8))
        for index, item in enumerate(items):
            h, w = item.shape[-2:]
            self.assertTrue(torch.all(actual['inputs'][index, 3, :h, :w] == 0.25))
            self.assertTrue(torch.all(actual['inputs'][index, 4, :h, :w] == 1))
            self.assertEqual(actual['inputs'][index, 3:, h:, :].count_nonzero(), 0)
            self.assertEqual(actual['inputs'][index, 3:, :, w:].count_nonzero(), 0)
            valid = actual['data_samples'][index].label_valid.data
            self.assertEqual(valid.dtype, torch.bool)
            self.assertEqual(valid.count_nonzero(), h * w)
            self.assertEqual(valid.shape, (1, 4, 8))

    def test_fixed_batch_padding_aligns_label_valid(self):
        item, sample = input_sample()
        processor = RGBDDataPreprocessor(batch_augments=[dict(type='BatchFixedSizePad', size=(8, 8))])
        actual = processor(dict(inputs=[item], data_samples=[sample]), training=True)
        self.assertEqual(actual['inputs'].shape, (1, 5, 8, 8))
        self.assertEqual(actual['data_samples'][0].label_valid.data.shape, (1, 8, 8))
        self.assertEqual(actual['data_samples'][0].label_valid.data.count_nonzero(), 15)

    def test_batched_inputs_supported(self):
        item, sample = input_sample()
        actual = RGBDDataPreprocessor()(dict(inputs=item[None], data_samples=[sample]))
        self.assertEqual(actual['inputs'].shape, (1, 5, 3, 5))

    def test_nan_inf_negative_and_clipped_depth(self):
        item, sample = input_sample(1, 5)
        item[3] = torch.tensor([float('nan'), float('inf'), -1, 0, 20])
        actual = RGBDDataPreprocessor(depth_max=10)(dict(inputs=[item], data_samples=[sample]))
        self.assertTrue(torch.isfinite(actual['inputs']).all())
        torch.testing.assert_close(actual['inputs'][0, 3], torch.tensor([[0., 0., 0., 0., 1.]]))
        torch.testing.assert_close(actual['inputs'][0, 4], torch.tensor([[0., 0., 0., 0., 1.]]))

    def test_unsupported_geometry_and_missing_validity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'BatchFixedSizePad'):
            RGBDDataPreprocessor(batch_augments=[dict(type='BatchSyncRandomResize')])
        item, sample = input_sample()
        del sample.label_valid
        with self.assertRaisesRegex(ValueError, 'label_valid'):
            RGBDDataPreprocessor()(dict(inputs=[item], data_samples=[sample]))
        for maximum in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RGBDDataPreprocessor(depth_max=maximum)


def tiny_parent_init(instance, *args, **kwargs):
    nn.Module.__init__(instance)
    instance.backbone = nn.Conv2d(3, 4, kernel_size=1, bias=False)
    nn.init.ones_(instance.backbone.weight)
    instance.neck = None


class TestDepthAdapter(unittest.TestCase):
    def tiny_model(self, enabled):
        with patch.object(Mask2Former, '__init__', tiny_parent_init):
            return RGBDMask2Former(depth_fusion=dict(enabled=enabled))

    def test_zero_init_matches_rgb_and_adapter_receives_gradient(self):
        model = self.tiny_model(True)
        self.assertEqual(model.depth_adapter.weight.count_nonzero(), 0)
        inputs = torch.ones((2, 5, 4, 6))
        actual = model.extract_feat(inputs)
        expected = model.backbone(inputs[:, :3])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(model.depth_adapter.weight.grad).all())
        self.assertGreater(model.depth_adapter.weight.grad.abs().sum().item(), 0)

    def test_off_ignores_depth_even_after_adapter_learns(self):
        model = self.tiny_model(False)
        nn.init.ones_(model.depth_adapter.weight)
        inputs = torch.randn((2, 5, 4, 6))
        inputs[:, 3:] = float('nan')
        actual = model.extract_feat(inputs)
        expected = model.backbone(inputs[:, :3])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertIsNone(model.depth_adapter.weight.grad)

    def test_toggle_changes_only_branch_not_parameter_set(self):
        on = self.tiny_model(True)
        off = self.tiny_model(False)
        self.assertEqual(set(on.state_dict()), set(off.state_dict()))
        self.assertEqual(on.depth_adapter.weight.numel(), 6)
        off.load_state_dict(on.state_dict())
        inputs = torch.randn((2, 5, 4, 6))
        torch.testing.assert_close(on.extract_feat(inputs), off.extract_feat(inputs), rtol=0, atol=0)
        nn.init.ones_(on.depth_adapter.weight)
        self.assertFalse(torch.equal(on.extract_feat(inputs), off.extract_feat(inputs)))


if __name__ == '__main__':
    unittest.main()

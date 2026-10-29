"""CPU tests for semantic transfer, fair switches, loading, and NYUv2 recipes.

Run in the pinned environment. Config integration tests use /workspace by
default; AI_CLASS_TEST_WORKSPACE can point to another root containing the
MMDetection checkout. No NYUv2 downloads or full segmentation models needed.
"""

import argparse
import ast
import contextlib
import copy
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmdet.datasets.transforms import FilterAnnotations
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_nyu_experiment as subject


def categories():
    rows = json.loads((Path(subject.__file__).with_name('nyuv2_things_v1.json')).read_text())['categories']
    return [dict(row, id=index + 1, nyuv2_source_id=row['id']) for index, row in enumerate(rows)]


def arguments(workspace='/workspace', **updates):
    values = dict(mode='train', workspace=str(workspace), data_root='/tmp/nyuv2_test_data',
                  output='/tmp/nyuv2_test_output', checkpoint='/tmp/coco_original.pth',
                  trained_checkpoint=False, iters=100, batch_size=2, image_size=512,
                  workers=0, lr=1e-5, seed=20260923, depth_fusion=False,
                  small_object_crop=False, small_object_loss=False, boundary_loss=False,
                  small_area_threshold=1024.0, small_crop_prob=0.5,
                  small_loss_max_weight=2.0, boundary_loss_weight=0.1)
    values.update(updates)
    return argparse.Namespace(**values)


def original_state():
    return {'panoptic_head.cls_embed.weight': torch.arange(81 * 4, dtype=torch.float32).reshape(81, 4),
            'panoptic_head.cls_embed.bias': torch.arange(81, dtype=torch.float32),
            'backbone.fixture': torch.tensor([12.0, 23.0])}


class TestCOCOTransfer(unittest.TestCase):
    def test_semantic_rows_and_no_object_are_exact(self):
        state = original_state()
        output = subject.map_coco_state(state, categories())
        # The fixed taxonomy is ordered NYU book,bottle,chair,bowl,cup,vase,
        # sofa,bed, not COCO numerical index order. Final row is no-object.
        expected = [73, 39, 56, 45, 41, 75, 57, 59, 80]
        for suffix in ('weight', 'bias'):
            key = f'panoptic_head.cls_embed.{suffix}'
            torch.testing.assert_close(output[key], state[key][expected], rtol=0, atol=0)
            self.assertNotEqual(output[key].data_ptr(), state[key].data_ptr())
        self.assertEqual(output['depth_adapter.weight'].shape, (3, 2, 1, 1))
        self.assertEqual(output['depth_adapter.weight'].count_nonzero(), 0)
        torch.testing.assert_close(output['backbone.fixture'], state['backbone.fixture'], rtol=0, atol=0)

    def test_mapping_does_not_mutate_original_classifier(self):
        state = original_state()
        old = copy.deepcopy(state)
        mapped = subject.map_coco_state(state, categories())
        mapped['panoptic_head.cls_embed.weight'].fill_(-99)
        mapped['panoptic_head.cls_embed.bias'].fill_(-99)
        self.assertEqual(set(state), set(old))
        for key in state:
            torch.testing.assert_close(state[key], old[key], rtol=0, atol=0)

    def test_duplicate_noobject_negative_and_out_of_range_mappings_rejected(self):
        for invalid in (39, 80, -1, 81):
            rows = categories()
            rows[0]['coco80_class_index'] = invalid
            with self.subTest(index=invalid), self.assertRaisesRegex(ValueError, 'mapping'):
                subject.map_coco_state(original_state(), rows)

    def test_adapted_or_wrong_size_source_rejected(self):
        for suffix in ('weight', 'bias'):
            state = original_state()
            state[f'panoptic_head.cls_embed.{suffix}'] = state[f'panoptic_head.cls_embed.{suffix}'][:9]
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, '80-class'):
                subject.map_coco_state(state, categories())
        state = original_state()
        state['depth_adapter.weight'] = torch.ones((3, 2, 1, 1))
        with self.assertRaisesRegex(ValueError, 'already adapted'):
            subject.map_coco_state(state, categories())

    def test_category_files_must_match_with_contiguous_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'annotations').mkdir()
            for split in ('train', 'val'):
                (root / f'annotations/instances_{split}.json').write_text(json.dumps({'categories': categories()}))
            self.assertEqual(subject.categories_from_export(root), categories())
            for replacement in (list(reversed(categories())), [dict(row, id=row['id'] + 1) for row in categories()]):
                (root / 'annotations/instances_val.json').write_text(json.dumps({'categories': replacement}))
                with self.assertRaisesRegex(ValueError, 'categories'):
                    subject.categories_from_export(root)


class TestNYUConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(os.environ.get('AI_CLASS_TEST_WORKSPACE', '/workspace'))
        if not (cls.workspace / 'mmdetection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py').is_file():
            raise unittest.SkipTest('Set AI_CLASS_TEST_WORKSPACE to the prepared checkout workspace')

    def build(self, **updates):
        return subject.build_config(arguments(self.workspace, **updates), categories())

    def test_all_sixteen_switch_combinations_are_independent_and_fair(self):
        baseline = self.build()
        for depth, crop, small, boundary in itertools.product((False, True), repeat=4):
            with self.subTest(depth=depth, crop=crop, small=small, boundary=boundary):
                cfg = self.build(depth_fusion=depth, small_object_crop=crop,
                                 small_object_loss=small, boundary_loss=boundary)
                head = cfg.model.panoptic_head
                self.assertEqual(cfg.model.depth_fusion.enabled, depth)
                self.assertEqual(head.small_object_loss.enabled, small)
                self.assertEqual(head.boundary_loss.enabled, boundary)
                self.assertTrue(head.ignore_void)
                actual_crop = cfg.train_dataloader.dataset.pipeline[4]
                self.assertEqual(actual_crop['type'], 'RGBDSmallObjectAwareCrop' if crop else 'RGBDRandomCrop')
                self.assertEqual(cfg.val_dataloader.to_dict(), baseline.val_dataloader.to_dict())
                self.assertEqual(cfg.test_dataloader.to_dict(), baseline.test_dataloader.to_dict())
                for key in ('optim_wrapper', 'randomness', 'train_cfg', 'val_evaluator', 'test_evaluator'):
                    self.assertEqual(cfg[key], baseline[key], key)
                self.assertEqual(cfg.param_scheduler, baseline.param_scheduler)
                self.assertEqual(cfg.train_dataloader.batch_size, baseline.train_dataloader.batch_size)
                self.assertEqual(cfg.model.data_preprocessor, baseline.model.data_preprocessor)

    def test_class_counts_classifier_weights_and_strict_load_configuration(self):
        cfg = self.build()
        self.assertEqual(cfg.model.type, 'RGBDMask2Former')
        self.assertEqual(cfg.model.panoptic_head.num_things_classes, 8)
        self.assertEqual(cfg.model.panoptic_head.num_stuff_classes, 0)
        self.assertEqual(cfg.model.panoptic_fusion_head.num_things_classes, 8)
        self.assertEqual(cfg.model.panoptic_head.loss_cls.class_weight, [1.0] * 8 + [0.1])
        self.assertEqual(cfg.train_dataloader.dataset.metainfo.classes, tuple(row['name'] for row in categories()))
        self.assertIsNone(cfg.load_from)
        self.assertFalse(cfg.resume)
        self.assertIsNone(cfg.model.backbone.init_cfg)
        self.assertFalse(cfg.model.depth_fusion.enabled)
        self.assertFalse(cfg.model.panoptic_head.small_object_loss.enabled)
        self.assertFalse(cfg.model.panoptic_head.boundary_loss.enabled)

    def test_dataset_paths_empty_frames_and_training_geometry(self):
        cfg = self.build()
        dataset = cfg.train_dataloader.dataset
        self.assertEqual(dataset.ann_file, 'annotations/instances_train.json')
        self.assertEqual(dataset.data_prefix.img, '')
        self.assertIsNone(dataset.filter_cfg)
        self.assertFalse(dataset.test_mode)
        self.assertEqual(dataset.pipeline[0]['type'], 'LoadRGBDImageFromFile')
        self.assertEqual(dataset.pipeline[2]['type'], 'RGBDRandomFlip')
        self.assertEqual(dataset.pipeline[3]['resize_type'], 'RGBDResize')
        self.assertTrue(dataset.pipeline[4]['allow_negative_crop'])
        self.assertFalse(dataset.pipeline[5]['keep_empty'])
        self.assertEqual(dataset.pipeline[-1]['type'], 'PackRGBDDetInputs')
        filter_args = dict(dataset.pipeline[5])
        filter_args.pop('type')
        transform = FilterAnnotations(**filter_args)
        for count in (0, 1):
            # One degenerate instance becomes empty after filtering; an
            # originally empty frame must also remain a usable negative frame.
            sample = dict(gt_bboxes=HorizontalBoxes(torch.zeros((count, 4))),
                          gt_bboxes_labels=np.zeros(count, dtype=np.int64),
                          gt_ignore_flags=np.zeros(count, dtype=bool),
                          gt_masks=BitmapMasks(np.zeros((count, 8, 8), dtype=np.uint8), 8, 8))
            result = transform(sample)
            self.assertIsNotNone(result)
            self.assertEqual(len(result['gt_bboxes']), 0)

    def test_eval_config_has_no_training_and_original_coordinate_metadata(self):
        cfg = self.build(mode='eval', depth_fusion=True)
        for key in ('train_cfg', 'train_dataloader', 'optim_wrapper', 'param_scheduler',
                    'val_cfg', 'val_dataloader', 'val_evaluator'):
            self.assertIsNone(cfg[key], key)
        self.assertEqual(cfg.test_cfg.type, 'TestLoop')
        pipeline = cfg.test_dataloader.dataset.pipeline
        self.assertEqual([row['type'] for row in pipeline],
                         ['LoadRGBDImageFromFile', 'RGBDResize', 'LoadAnnotations', 'PackRGBDDetInputs'])
        self.assertTrue(set(('img_id', 'ori_shape', 'img_shape', 'scale_factor')).issubset(pipeline[-1]['meta_keys']))
        self.assertEqual(cfg.test_evaluator.type, 'VoidAwareCocoMetric')
        self.assertEqual(cfg.test_evaluator.prefix, 'nyuv2_void')
        self.assertEqual(cfg.test_evaluator.metric, 'segm')
        self.assertEqual(cfg.test_dataloader.dataset.ann_file, 'annotations/instances_val.json')
        self.assertFalse(cfg.test_dataloader.sampler.shuffle)
        self.assertTrue(cfg.test_dataloader.dataset.test_mode)

    def test_learning_rate_batch_budget_and_disabled_autoscaling(self):
        cfg = self.build(lr=3e-5, batch_size=3, iters=7)
        self.assertEqual(cfg.optim_wrapper.optimizer.lr, 3e-5)
        self.assertEqual(cfg.optim_wrapper.optimizer.type, 'AdamW')
        self.assertEqual(cfg.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult, 0.1)
        self.assertNotIn('depth_adapter', cfg.optim_wrapper.paramwise_cfg.custom_keys)
        self.assertEqual(cfg.train_dataloader.batch_size, 3)
        self.assertEqual(cfg.train_cfg.max_iters, 7)
        self.assertEqual(cfg.train_cfg.val_interval, 7)
        self.assertFalse(cfg.auto_scale_lr.enable)
        self.assertEqual(cfg.param_scheduler, [])
        self.assertEqual(cfg.optim_wrapper.clip_grad.max_norm, 0.01)
        self.assertTrue(cfg.optim_wrapper.clip_grad.error_if_nonfinite)

    def test_build_config_never_mutates_cli_namespace(self):
        args = arguments(self.workspace, depth_fusion=True, small_object_crop=True,
                         small_object_loss=True, boundary_loss=True)
        before = copy.deepcopy(vars(args))
        subject.build_config(args, categories())
        self.assertEqual(vars(args), before)


class _GuardReached(Exception):
    pass


class TestEvaluationGuards(unittest.TestCase):
    def call_to_config_guard(self, prior_depth=False, requested_depth=False,
                             prior_manifest='expected-manifest'):
        prior = Config(dict(model=dict(depth_fusion=dict(enabled=prior_depth)),
                            experiment_provenance=dict(dataset_manifest_sha256=prior_manifest)))
        saved = dict(state_dict=subject.map_coco_state(original_state(), categories()),
                     meta=dict(cfg=prior.pretty_text))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            argv = ['run_nyu_experiment.py', 'eval', '--trained-checkpoint',
                    '--output', str(Path(directory) / 'output')]
            if requested_depth:
                argv.append('--depth-fusion')
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(subject, 'verify_export', return_value={}))
            stack.enter_context(patch.object(subject, 'categories_from_export', return_value=categories()))
            stack.enter_context(patch.object(subject.torch, 'load', return_value=saved))
            stack.enter_context(patch.object(subject, 'sha256', return_value='expected-manifest'))
            stack.enter_context(patch.object(subject, 'register_all_modules'))
            build = stack.enter_context(patch.object(subject, 'build_config', side_effect=_GuardReached))
            try:
                subject.main()
            finally:
                self.assertFalse((Path(directory) / 'output').exists())
            return build

    def test_depth_switch_mismatch_rejected_before_constructing_model(self):
        for prior, requested in ((False, True), (True, False)):
            with self.subTest(prior=prior), self.assertRaisesRegex(ValueError, 'depth-fusion switch'):
                self.call_to_config_guard(prior_depth=prior, requested_depth=requested)

    def test_different_dataset_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different dataset manifest'):
            self.call_to_config_guard(prior_manifest='different')

    def test_matching_checkpoint_metadata_reaches_config_construction(self):
        for depth in (False, True):
            with self.subTest(depth=depth), self.assertRaises(_GuardReached):
                self.call_to_config_guard(prior_depth=depth, requested_depth=depth)

    def test_invalid_cli_hyperparameters_rejected_before_dataset_io(self):
        options = ['--lr=0', '--lr=nan', '--lr=inf', '--iters=0', '--image-size=31',
                   '--image-size=0', '--workers=-1', '--small-crop-prob=nan',
                   '--small-area-threshold=inf', '--small-loss-max-weight=0.5',
                   '--boundary-loss-weight=nan']
        for option in options:
            with self.subTest(option=option), patch.object(sys, 'argv',
                    ['run_nyu_experiment.py', 'train', '--output', '/tmp/unused_nyu_test', option]), \
                    patch.object(subject, 'verify_export') as verification, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                try:
                    subject.main()
                finally:
                    verification.assert_not_called()

    def test_adapted_checkpoint_training_not_silently_treated_as_resume(self):
        with patch.object(sys, 'argv', ['run_nyu_experiment.py', 'train', '--trained-checkpoint',
                                       '--output', '/tmp/unused_nyu_test']), \
                patch.object(subject, 'verify_export') as verification, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            try:
                subject.main()
            finally:
                verification.assert_not_called()


class _TinyInitializedModel(BaseModel):
    def __init__(self):
        super().__init__(init_cfg=dict(type='Constant', layer='Linear', val=0.5))
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        prediction = self.linear(inputs)
        return dict(loss=prediction.square().mean()) if mode == 'loss' else prediction


class _CaptureBeforeStep(Hook):
    def __init__(self):
        self.weight = None

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.weight = runner.model.linear.weight.detach().clone()


class TestStrictLoadingAndInitialization(unittest.TestCase):
    def test_entrypoint_initializes_then_loads_strict_before_train(self):
        tree = ast.parse(Path(subject.__file__).read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        initializes = [node for node in calls if node.func.attr == 'init_weights']
        loads = [node for node in calls if node.func.attr == 'load_state_dict']
        trains = [node for node in calls if node.func.attr == 'train']
        self.assertEqual(len(initializes), 1)
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(trains), 1)
        self.assertLess(initializes[0].lineno, loads[0].lineno)
        self.assertLess(loads[0].lineno, trains[0].lineno)
        strict = next(item.value for item in loads[0].keywords if item.arg == 'strict')
        self.assertIsInstance(strict, ast.Constant)
        self.assertIs(strict.value, True)

    def test_real_runner_second_initialization_keeps_loaded_weight_and_updates_it(self):
        # A real one-step CPU MMEngine train loop, not a mock of init_weights.
        model = _TinyInitializedModel()
        loader = DataLoader([dict(inputs=torch.ones(1))], batch_size=1)
        capture = _CaptureBeforeStep()
        with tempfile.TemporaryDirectory() as directory, \
                patch('mmengine.runner.runner.get_device', return_value='cpu'):
            runner = Runner(model=model, work_dir=directory, train_dataloader=loader,
                            train_cfg=dict(type='IterBasedTrainLoop', max_iters=1),
                            optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)),
                            default_scope='mmengine', custom_hooks=[capture],
                            default_hooks=dict(logger=None, checkpoint=None, param_scheduler=None),
                            visualizer=dict(type='Visualizer', vis_backends=[]),
                            randomness=dict(seed=101))
            try:
                runner.model.init_weights()
                loaded = {'linear.weight': torch.tensor([[3.0]], device=next(runner.model.parameters()).device)}
                runner.model.load_state_dict(loaded, strict=True)
                runner.train()
                torch.testing.assert_close(capture.weight.cpu(), torch.tensor([[3.0]]), rtol=0, atol=0)
                torch.testing.assert_close(runner.model.linear.weight.detach().cpu(),
                                           torch.tensor([[2.94]]), rtol=1e-6, atol=1e-6)
            finally:
                # MMLogger remains in a process-global manager after Runner is
                # released. Close its file descriptors before removing the
                # fixture: NFS otherwise retains open logs as .nfs* files and
                # TemporaryDirectory fails with "Directory not empty".
                for handler in list(runner.logger.handlers):
                    try:
                        handler.flush()
                    finally:
                        handler.close()
                        runner.logger.removeHandler(handler)
                runner.visualizer.close()
                del runner

    def test_strict_state_loading_rejects_missing_extra_and_wrong_shapes(self):
        model = _TinyInitializedModel()
        for state in ({}, {'linear.weight': torch.ones((2, 1))},
                      {'linear.weight': torch.ones((1, 1)), 'unexpected': torch.zeros(1)}):
            with self.subTest(keys=list(state)), self.assertRaises(RuntimeError):
                model.load_state_dict(state, strict=True)


if __name__ == '__main__':
    unittest.main()

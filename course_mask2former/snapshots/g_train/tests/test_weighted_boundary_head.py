"""Loss-only tests using actual MMDetection 3.3.0 assignment/loss functions.

Run in the prepared container (no dataset or weights required):
  AI_CLASS_TEST_DEVICE=cuda python -m unittest discover -s /workspace/scripts/tests \
      -p test_weighted_boundary_head.py -v
Use AI_CLASS_TEST_DEVICE=cpu for CPU. Dependencies: torch, mmcv, mmdet, scipy.
The lightweight heads omit unused forward modules, but use real Hungarian
matching, MaskPseudoSampler, point_sample, classification, BCE and Dice losses.
The application should additionally smoke-test a complete configured model.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import register_all_modules

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizations.weighted_boundary_head import (  # noqa: E402
    WeightedBoundaryMask2FormerHead, _checked_options)


class WeightedBoundaryHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.device = torch.device(os.environ.get(
            'AI_CLASS_TEST_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))

    def head(self, optimized=True, small=False, boundary=False, max_weight=2.0):
        head_type = WeightedBoundaryMask2FormerHead if optimized else Mask2FormerHead
        head = head_type.__new__(head_type)
        torch.nn.Module.__init__(head)
        head.num_classes, head.num_queries, head.num_points = 2, 4, 32
        head.oversample_ratio, head.importance_sample_ratio = 2.0, 0.5
        head.class_weight = [1.0, 1.0, 0.1]
        head.loss_cls = MODELS.build(dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0,
            reduction='mean', class_weight=head.class_weight))
        head.loss_mask = MODELS.build(dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0,
            reduction='mean'))
        head.loss_dice = MODELS.build(dict(
            type='DiceLoss', use_sigmoid=True, activate=True, naive_dice=True,
            eps=1.0, loss_weight=5.0, reduction='mean'))
        head.assigner = TASK_UTILS.build(dict(
            type='HungarianAssigner', match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)]))
        head.sampler = TASK_UTILS.build(dict(type='MaskPseudoSampler'))
        if optimized:
            head.small_object_loss, head.boundary_loss = _checked_options(
                dict(enabled=small, area_threshold=64.0, max_weight=max_weight),
                dict(enabled=boundary, loss_weight=0.1, radius=1))
        return head.to(self.device)

    def inputs(self, counts=(2, 1), layers=3):
        torch.manual_seed(571)
        batch = len(counts)
        classes = [torch.randn(batch, 4, 3, device=self.device, requires_grad=True)
                   for _ in range(layers)]
        masks = [torch.randn(batch, 4, 8, 8, device=self.device, requires_grad=True)
                 for _ in range(layers)]
        instances, metas = [], []
        for count in counts:
            target = torch.zeros((count, 32, 32), dtype=torch.long, device=self.device)
            if count >= 1:
                target[0, 4:8, 4:8] = 1  # 16 px: boosted by 2.
            if count >= 2:
                target[1, 12:28, 12:28] = 1  # 256 px: weight 1.
            instances.append(InstanceData(
                labels=torch.arange(count, device=self.device, dtype=torch.long) % 2,
                masks=target))
            metas.append(dict(img_shape=(32, 32), batch_input_shape=(32, 32)))
        return classes, masks, instances, metas

    def losses(self, head, inputs):
        torch.manual_seed(20260922)  # Both assignment and uncertain-point sampling.
        return head.loss_by_feat(*inputs)

    def assert_base_equal(self, actual, expected):
        for key, value in expected.items():
            self.assertTrue(torch.equal(actual[key], value), key)

    def assert_finite_backward(self, losses, inputs):
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), key)
        sum(value.sum() for value in losses.values()).backward()
        for value in inputs[0] + inputs[1]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_both_disabled_are_bitwise_equal_to_upstream(self):
        for counts in ((2, 1), (0, 0), (0, 2)):
            with self.subTest(counts=counts):
                inputs = self.inputs(counts)
                reference = self.losses(self.head(optimized=False), inputs)
                actual = self.losses(self.head(), inputs)
                self.assertEqual(actual.keys(), reference.keys())
                self.assert_base_equal(actual, reference)

    def test_small_only_leaves_classification_unchanged(self):
        inputs = self.inputs()
        reference = self.losses(self.head(optimized=False), inputs)
        actual = self.losses(self.head(small=True), inputs)
        self.assertEqual(actual.keys(), reference.keys())
        for key in reference:
            if key.endswith('loss_cls'):
                self.assertTrue(torch.equal(actual[key], reference[key]), key)
            else:
                self.assertGreaterEqual(actual[key].item(), reference[key].item())
        self.assertGreater(actual['loss_mask'].item(), reference['loss_mask'].item())
        self.assertGreater(actual['loss_dice'].item(), reference['loss_dice'].item())
        self.assert_finite_backward(actual, inputs)

    def test_unit_small_weight_is_neutral(self):
        inputs = self.inputs()
        reference = self.losses(self.head(optimized=False), inputs)
        actual = self.losses(self.head(small=True, max_weight=1.0), inputs)
        self.assert_base_equal(actual, reference)

    def test_boundary_only_preserves_all_upstream_losses(self):
        inputs = self.inputs()
        reference = self.losses(self.head(optimized=False), inputs)
        actual = self.losses(self.head(boundary=True), inputs)
        self.assertEqual(set(actual) - set(reference), {'loss_boundary'})
        self.assert_base_equal(actual, reference)
        self.assertGreater(actual['loss_boundary'].item(), 0)
        self.assert_finite_backward(actual, inputs)

    def test_both_enabled_boundary_independent_of_small_weights(self):
        inputs = self.inputs()
        boundary = self.losses(self.head(boundary=True), inputs)
        small = self.losses(self.head(small=True), inputs)
        both = self.losses(self.head(small=True, boundary=True), inputs)
        self.assert_base_equal(both, small)
        self.assertTrue(torch.equal(both['loss_boundary'], boundary['loss_boundary']))
        self.assert_finite_backward(both, inputs)

    def test_boundary_does_not_repeat_matching_or_auxiliary_layers(self):
        head = self.head(small=True, boundary=True)
        inputs = self.inputs(layers=3)
        with patch.object(head, 'get_targets', wraps=head.get_targets) as targets, \
                patch.object(head, '_boundary_region_loss',
                             wraps=head._boundary_region_loss) as boundary:
            actual = self.losses(head, inputs)
        self.assertEqual(targets.call_count, 3)
        self.assertEqual(boundary.call_count, 1)
        self.assertEqual([key for key in actual if 'boundary' in key], ['loss_boundary'])

    def test_empty_and_mixed_gt_have_finite_gradients(self):
        for counts in ((0, 0), (0, 2)):
            for small, boundary in ((True, False), (False, True), (True, True)):
                with self.subTest(counts=counts, small=small, boundary=boundary):
                    inputs = self.inputs(counts)
                    losses = self.losses(self.head(small=small, boundary=boundary), inputs)
                    self.assert_finite_backward(losses, inputs)
                    if counts == (0, 0):
                        for key, value in losses.items():
                            if not key.endswith('loss_cls'):
                                self.assertEqual(value.item(), 0)

    def test_area_weights_are_bounded_and_zero_area_is_neutral(self):
        head = self.head(small=True)
        masks = torch.zeros((4, 16, 16), device=self.device)
        masks[1, :2, :2] = 1
        masks[2, :8, :8] = 1
        masks[3] = 1
        actual = head._instance_weights(masks)
        self.assertTrue(torch.equal(actual, actual.new_tensor([1, 2, 1, 1])))

    def test_boundary_empty_constant_tiny_and_extreme_logits(self):
        head = self.head(boundary=True)
        targets = torch.zeros((4, 32, 32), device=self.device)
        targets[1] = 1  # Constant mask has no internal boundary band.
        targets[2, 7, 7] = 1  # Downsampled peak 1/16: auxiliary loss must skip.
        targets[3, 8:24, 8:24] = 1  # Resolvable object still receives boundary loss.
        logits = torch.full((4, 8, 8), 1000.0, device=self.device, requires_grad=True)
        loss = head._boundary_region_loss(logits, targets, 4)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(loss.item(), 0)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[:3].abs().sum().item(), 0)
        self.assertGreater(logits.grad[3].abs().sum().item(), 0)

    def test_subpixel_target_is_skipped_with_connected_zero_gradient(self):
        head = self.head(boundary=True)
        target = torch.zeros((1, 32, 32), device=self.device)
        target[0, 7, 7] = 1  # Exactly 1/16 peak after 4x area downsampling.
        logits = torch.zeros((1, 8, 8), device=self.device, requires_grad=True)
        loss = head._boundary_region_loss(logits, target, 1)
        self.assertEqual(loss.item(), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0)

        # The threshold is genuinely configurable. Lowering it disables this
        # protection for the fixture, exposing the background-pushing gradient
        # that the conservative default deliberately avoids.
        head.boundary_loss['min_target_peak'] = 0.01
        logits = torch.zeros((1, 8, 8), device=self.device, requires_grad=True)
        loss = head._boundary_region_loss(logits, target, 1)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(logits.grad[0, 1, 1].item(), 0)

    def test_resolvable_boundary_gradient_has_correct_foreground_direction(self):
        head = self.head(boundary=True)
        target = torch.zeros((1, 32, 32), device=self.device)
        target[0, 8:24, 8:24] = 1
        logits = torch.zeros((1, 8, 8), device=self.device, requires_grad=True)
        loss = head._boundary_region_loss(logits, target, 1)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 2, 2].item(), 0)  # Descent raises foreground.
        self.assertGreater(logits.grad[0, 1, 2].item(), 0)  # Descent lowers background.
        self.assertEqual(logits.grad[0, 3, 3].item(), 0)  # Interior is outside the band.

    def test_options_default_off_and_invalid_options_rejected(self):
        small, boundary = _checked_options()
        self.assertFalse(small['enabled'])
        self.assertFalse(boundary['enabled'])
        self.assertEqual(boundary['min_target_peak'], 0.5)
        for values in (dict(area_threshold=0), dict(max_weight=0.9),
                       dict(max_weight=float('nan')), dict(unrecognized=True)):
            with self.subTest(small=values), self.assertRaises(ValueError):
                _checked_options(values)
        for values in (dict(loss_weight=-1), dict(radius=0), dict(radius=1.5),
                       dict(min_target_peak=0), dict(min_target_peak=1.1),
                       dict(min_target_peak=float('nan'))):
            with self.subTest(boundary=values), self.assertRaises(ValueError):
                _checked_options(boundary_loss=values)


if __name__ == '__main__':
    unittest.main(verbosity=2)

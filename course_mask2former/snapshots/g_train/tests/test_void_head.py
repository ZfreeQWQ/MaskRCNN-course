"""Real MMDetection loss tests for explicit NYUv2 void supervision.

Run in the prepared container:
  AI_CLASS_TEST_DEVICE=cuda python -m unittest discover -s /workspace/scripts/tests \
      -p test_void_head.py -v
These test actual assignment/BCE/Dice and independently switchable additions.
"""
import os
from pathlib import Path
import sys
import unittest

import torch
from mmdet.utils import register_all_modules

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_weighted_boundary_head as base_tests  # noqa: E402
from rgbd.void_head import VoidAwareMask2FormerHead, valid_pixel_points  # noqa: E402


class VoidHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cls.device = torch.device(os.environ.get(
            'AI_CLASS_TEST_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))

    def head(self, ignore_void=True, small=False, boundary=False, reference=False):
        head = base_tests.WeightedBoundaryHeadTests.head(
            self, optimized=True, small=small, boundary=boundary)
        if not reference:
            # Real losses/assigner; omit expensive forward modules, as the
            # existing head tests do. A complete model smoke is also required.
            head.__class__ = VoidAwareMask2FormerHead
            head.ignore_void = ignore_void
        return head

    def inputs(self, counts=(2, 1), layers=2, policy='partial'):
        values = base_tests.WeightedBoundaryHeadTests.inputs(self, counts, layers)
        for meta in values[3]:
            valid = torch.ones((32, 32), dtype=torch.bool, device=self.device)
            if policy == 'partial':
                valid[:4] = False
                valid[:, 24:] = False
            elif policy == 'void':
                valid[:] = False
            elif policy != 'full':
                raise ValueError(policy)
            meta['label_valid'] = valid
        return values

    def losses(self, head, inputs):
        torch.manual_seed(20260924)
        return head.loss_by_feat(*inputs)

    def assert_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def assert_backward(self, losses, inputs, all_zero=False):
        for key, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), key)
            if all_zero:
                self.assertEqual(value.item(), 0, key)
        sum(losses.values()).backward()
        for value in inputs[0] + inputs[1]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            if all_zero:
                self.assertEqual(value.grad.abs().sum().item(), 0)

    def test_all_valid_is_bitwise_equal_for_all_loss_switches(self):
        for small in (False, True):
            for boundary in (False, True):
                for counts in ((2, 1), (0, 0), (0, 2)):
                    with self.subTest(small=small, boundary=boundary, counts=counts):
                        inputs = self.inputs(counts, policy='full')
                        actual = self.losses(self.head(small=small, boundary=boundary), inputs)
                        expected = self.losses(self.head(small=small, boundary=boundary, reference=True), inputs)
                        self.assert_equal(actual, expected)

    def test_ignore_disabled_restores_previous_policy_without_map(self):
        inputs = self.inputs()
        for meta in inputs[3]:
            del meta['label_valid']
        actual = self.losses(self.head(ignore_void=False, small=True, boundary=True), inputs)
        expected = self.losses(self.head(small=True, boundary=True, reference=True), inputs)
        self.assert_equal(actual, expected)

    def test_partial_valid_and_empty_gt_have_finite_gradients(self):
        for counts in ((2, 1), (0, 0), (0, 2)):
            for small in (False, True):
                for boundary in (False, True):
                    with self.subTest(counts=counts, small=small, boundary=boundary):
                        inputs = self.inputs(counts)
                        self.assert_backward(self.losses(self.head(small=small, boundary=boundary), inputs), inputs)

    def test_all_void_has_zero_connected_losses_even_with_gt(self):
        for counts in ((2, 1), (0, 0)):
            inputs = self.inputs(counts, policy='void')
            losses = self.losses(self.head(small=True, boundary=True), inputs)
            self.assert_backward(losses, inputs, all_zero=True)

    def test_void_image_has_zero_gradient_in_mixed_batch(self):
        inputs = self.inputs()
        inputs[3][0]['label_valid'][:] = False
        losses = self.losses(self.head(small=True, boundary=True), inputs)
        self.assert_backward(losses, inputs)
        for value in inputs[0] + inputs[1]:
            self.assertEqual(value.grad[0].abs().sum().item(), 0)
        self.assertGreater(inputs[0][-1].grad[1].abs().sum().item(), 0)

    def test_void_image_does_not_dilute_valid_image_mask_losses(self):
        inputs = self.inputs()
        inputs[3][0]['label_valid'][:] = False
        single = ([value[1:] for value in inputs[0]],
                  [value[1:] for value in inputs[1]], inputs[2][1:], inputs[3][1:])
        head = self.head(small=True, boundary=True)
        mixed_loss, single_loss = self.losses(head, inputs), self.losses(head, single)
        for key in mixed_loss:
            if not key.endswith('loss_cls'):
                self.assertTrue(torch.equal(mixed_loss[key], single_loss[key]), key)

    def test_modifying_unknown_gt_never_changes_any_loss(self):
        for small in (False, True):
            for boundary in (False, True):
                with self.subTest(small=small, boundary=boundary):
                    inputs = self.inputs()
                    head = self.head(small=small, boundary=boundary)
                    before = self.losses(head, inputs)
                    for gt, meta in zip(inputs[2], inputs[3]):
                        gt.masks[:, ~meta['label_valid']] = 1 - gt.masks[:, ~meta['label_valid']]
                    after = self.losses(head, inputs)
                    self.assert_equal(before, after)

    def test_random_and_uncertain_candidates_are_valid_pixel_centres(self):
        valid = torch.zeros((31, 37), dtype=torch.bool, device=self.device)
        valid[2, 3] = valid[5, 9] = valid[29, 35] = True
        head = self.head()
        predictions = torch.randn((4, 8, 10), device=self.device)
        for indices, coords in (valid_pixel_points(valid, 4, 250),
                                head._uncertain_valid_points(predictions, valid)):
            self.assertTrue(valid.flatten()[indices].all())
            x = coords[..., 0] * valid.shape[1] - 0.5
            y = coords[..., 1] * valid.shape[0] - 0.5
            self.assertTrue(torch.allclose(x, (indices % 37).float(), atol=1e-5))
            self.assertTrue(torch.allclose(y, (indices // 37).float(), atol=1e-5))

    def test_single_valid_pixel_and_tiny_target_are_not_discarded(self):
        inputs = self.inputs(counts=(1,))
        valid = inputs[3][0]['label_valid']
        valid[:] = False
        valid[7, 7] = True
        inputs[2][0].masks[:] = 0
        inputs[2][0].masks[0, 7, 7] = 1
        head = self.head(small=True, boundary=True)
        result = head._get_targets_single(inputs[0][0][0], inputs[1][0][0], inputs[2][0], inputs[3][0])
        self.assertEqual(result[2].shape[0], 1)
        self.assertEqual(result[2].sum().item(), 1)
        losses = self.losses(head, inputs)
        self.assertGreater(losses['loss_mask'].item(), 0)
        self.assert_backward(losses, inputs)

    def test_boundary_excludes_void_and_its_artificial_edge(self):
        head = self.head(boundary=True)
        valid = torch.ones((32, 32), dtype=torch.bool, device=self.device)
        valid[:, 16:] = False
        target = torch.zeros((1, 32, 32), device=self.device)
        target[:, 8:24, 8:16] = 1
        logits = torch.zeros((1, 8, 8), device=self.device, requires_grad=True)
        loss = head._void_boundary_loss(logits, target, [valid], 1)
        loss.backward()
        # Columns >=4 are void; column3 borders void. Neither is supervised.
        self.assertEqual(logits.grad[:, :, 3:].abs().sum().item(), 0)
        self.assertGreater(logits.grad[:, :, :3].abs().sum().item(), 0)

    def test_missing_bad_dtype_or_misaligned_map_fails_loudly(self):
        for bad in (None, torch.ones((32, 32)), torch.ones((31, 32), dtype=torch.bool)):
            inputs = self.inputs()
            if bad is None:
                del inputs[3][0]['label_valid']
            else:
                inputs[3][0]['label_valid'] = bad.to(self.device)
            with self.assertRaises(ValueError):
                self.losses(self.head(), inputs)

    def test_sampler_rejects_all_void_instead_of_using_background(self):
        with self.assertRaises(ValueError):
            valid_pixel_points(torch.zeros((4, 4), dtype=torch.bool, device=self.device), 1, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Actual MMDetection loss tests for the full-supervision point quota."""
import unittest
import torch

import test_weighted_boundary_head as base
from rgbd.course_head import CoursePointQuotaMask2FormerHead


class CourseHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.WeightedBoundaryHeadTests.setUpClass()
        cls.fixture = base.WeightedBoundaryHeadTests()

    def head(self, enabled=False, small=False, boundary=False):
        head = self.fixture.head(small=small, boundary=boundary)
        head.__class__ = CoursePointQuotaMask2FormerHead
        head.ignore_void = False
        head.foreground_sampling = dict(enabled=False)
        head.course_point_sampling = dict(enabled=enabled, quota=8, area_threshold=64.)
        head.course_point_counters = dict(calls=0, small_targets=0, replaced_points=0)
        return head

    def test_off_exact_losses_and_rng_all_existing_switches(self):
        for small, boundary in ((False, False), (True, False), (False, True), (True, True)):
            inputs = self.fixture.inputs()
            reference = self.fixture.losses(self.fixture.head(small=small, boundary=boundary), inputs)
            state = torch.random.get_rng_state()
            actual = self.fixture.losses(self.head(False, small, boundary), inputs)
            self.fixture.assert_base_equal(actual, reference)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_quota_centres_prefix_large_and_empty_preserved(self):
        head = self.head(True)
        targets = torch.zeros(3, 32, 32, device=self.fixture.device)
        targets[0, 5, 7] = 1
        targets[1, 8:24, 8:24] = 1
        preds = torch.randn(3, 8, 8, device=self.fixture.device)
        torch.manual_seed(7)
        old = self.head(False)._mask_loss_point_coords(preds, targets)
        torch.manual_seed(7)
        new = head._mask_loss_point_coords(preds, targets)
        self.assertEqual(new.shape, old.shape)
        self.assertTrue(torch.equal(new[:, :-8], old[:, :-8]))
        self.assertTrue(torch.equal(new[1:], old[1:]))
        expected = new.new_tensor([7.5 / 32, 5.5 / 32]).expand(8, 2)
        self.assertTrue(torch.equal(new[0, -8:], expected))
        self.assertEqual(head.course_point_counters['replaced_points'], 8)

    def test_enabled_loss_backward_empty_mixed_and_combined(self):
        for counts in ((2, 1), (0, 0), (0, 2)):
            for small, boundary in ((False, False), (True, True)):
                inputs = self.fixture.inputs(counts=counts)
                head = self.head(True, small, boundary)
                losses = self.fixture.losses(head, inputs)
                self.fixture.assert_finite_backward(losses, inputs)
                if sum(counts):
                    self.assertGreater(head.course_point_counters['replaced_points'], 0)
                else:
                    self.assertEqual(head.course_point_counters['replaced_points'], 0)

    def test_quota_does_not_modify_matching_for_fixed_rng(self):
        inputs = self.fixture.inputs(layers=1)
        scores, masks, gt, meta = inputs
        torch.manual_seed(52)
        old = self.head(False)._get_targets_single(scores[0][0], masks[0][0], gt[0], meta[0])
        torch.manual_seed(52)
        new = self.head(True)._get_targets_single(scores[0][0], masks[0][0], gt[0], meta[0])
        for a, b in zip(old[:6], new[:6]):
            self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main(verbosity=2)

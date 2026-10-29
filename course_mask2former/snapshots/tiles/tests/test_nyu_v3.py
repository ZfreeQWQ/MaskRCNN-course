"""v3 off-path equivalence, geometry, regularization and real loss tests."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from mmdet.models import Mask2Former

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_nyu_v2 as v2tests
import test_nyu_runner as runner_tests
from rgbd.advanced_model import AdvancedRGBDMask2Former, relative_log_depth
from rgbd.matching_head import SmallAwareMatchingHead, balanced_small_points
from rgbd.void_head import valid_pixel_points
from run_nyu_v3 import parser, build_config, validate_args
from nyu_v3_plan import experiment_jobs, FLAGS


class AdvancedModelTests(unittest.TestCase):
    def model(self, relative=False, dropout=False, rgb=False):
        with patch.object(Mask2Former, '__init__', v2tests.tiny_init):
            return AdvancedRGBDMask2Former(multiscale_depth=dict(enabled=True, reliability_gate=True,
                source='rgb' if rgb else 'depth'),
                advanced_depth=dict(relative_depth=relative, modality_dropout=dropout, dropout_prob=.2))

    def batch(self):
        data = torch.randn(2, 5, 64, 64)
        data[:, 3] = torch.rand(2, 64, 64) + .1
        data[:, 4] = 1
        return data

    def test_relative_scale_invariance_and_no_cross_image_leak(self):
        depth = torch.rand(2, 1, 16, 16) + .1
        a = relative_log_depth(depth)
        b = relative_log_depth(depth * 2)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        depth[1] *= 100
        torch.testing.assert_close(a[0], relative_log_depth(depth)[0], rtol=0, atol=0)

    def test_relative_invalid_padding_constant_and_one_pixel(self):
        depth = torch.tensor([[[[0., float('nan'), float('inf'), -1.], [3., 3., 3., 3.]]]])
        result = relative_log_depth(depth)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.count_nonzero().item(), 0)
        for values in (torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)):
            self.assertEqual(relative_log_depth(values).count_nonzero().item(), 0)
        depth = torch.zeros(1, 1, 8, 8)
        depth[0, 0, 3, 4] = 2
        self.assertEqual(relative_log_depth(depth).count_nonzero().item(), 0)

    def test_disabled_equals_v2_bitwise_and_rng(self):
        torch.manual_seed(8)
        advanced = self.model()
        torch.manual_seed(8)
        reference = v2tests.FusionTests().model(True, True)
        data = self.batch()
        # Same parameter shape/construction and values, including zero heads.
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(advanced.state_dict()[key], value))
        for p in advanced.spatial_fusion.projections.parameters():
            torch.nn.init.normal_(p)
        reference.load_state_dict(advanced.state_dict())
        torch.manual_seed(7)
        actual = advanced.extract_feat(data)
        rng = torch.get_rng_state()
        torch.manual_seed(7)
        expected = reference.extract_feat(data)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def test_dropped_image_is_exact_rgb_all_scales_others_scaled(self):
        model = self.model(dropout=True)
        for p in model.spatial_fusion.projections.parameters():
            torch.nn.init.normal_(p, std=.01)
        data = self.batch()
        model.eval()
        full = model.extract_feat(data)
        rgb = model.backbone(data[:, :3])
        model.train()
        with patch('rgbd.advanced_model.torch.rand', return_value=torch.tensor([0., .9]).reshape(2, 1, 1, 1)):
            out = model.extract_feat(data)
        for a, r, f in zip(out, rgb, full):
            self.assertTrue(torch.equal(a[0], r[0]))
            torch.testing.assert_close(a[1], r[1] + (f[1] - r[1]) / .8)
        self.assertEqual(model.advanced_counters['dropped_samples'], 1)

    def test_eval_dropout_off_and_relative_counts_training_only(self):
        model = self.model(relative=True, dropout=True)
        data = self.batch()
        model.eval()
        with patch('rgbd.advanced_model.torch.rand', side_effect=AssertionError('No eval dropout')):
            out = model.extract_feat(data)
        self.assertFalse(any(model.advanced_counters.values()))
        model.train()
        sum(x.mean() for x in model.extract_feat(data)).backward()
        self.assertEqual(model.advanced_counters['relative_training_samples'], 2)
        self.assertTrue(all(torch.isfinite(x).all() for x in out))

    def test_rgb_control_ignores_all_depth_under_new_switches(self):
        model = self.model(relative=True, dropout=True, rgb=True).eval()
        for p in model.spatial_fusion.projections.parameters():
            torch.nn.init.normal_(p, std=.01)
        data = self.batch()
        other = data.clone()
        other[:, 3:] = float('nan')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.extract_feat(data), model.extract_feat(other))))

    def test_new_flags_add_no_parameters(self):
        a, b = self.model(), self.model(True, True)
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v2tests.ForegroundTests.setUpClass()
        cls.fixture = v2tests.ForegroundTests()
        cls.device = cls.fixture.device

    def head(self, enabled=True, foreground=True, small=False, boundary=False):
        head = self.fixture.head(foreground, small, boundary)
        head.__class__ = SmallAwareMatchingHead
        head.small_matching = dict(enabled=enabled, fraction=.25, area_threshold=64.)
        head.matching_counters = dict(calls=0, augmented_calls=0, foreground_points=0)
        return head

    def test_disabled_exact_v2_all_policies_and_loss_flags(self):
        f = self.fixture.fixture
        for policy in ('full', 'partial', 'void'):
            for foreground in (False, True):
                for small, boundary in ((False, False), (True, True)):
                    data = f.inputs(policy=policy)
                    expected = f.losses(self.fixture.head(foreground, small, boundary), data)
                    actual = f.losses(self.head(False, foreground, small, boundary), data)
                    f.assert_equal(actual, expected)

    def test_all_combinations_real_assignment_loss_backward(self):
        f = self.fixture.fixture
        for policy in ('full', 'partial', 'void'):
            for foreground in (False, True):
                for counts in ((2, 1), (0, 0), (0, 2)):
                    data = f.inputs(counts=counts, policy=policy)
                    head = self.head(True, foreground, True, True)
                    f.assert_backward(f.losses(head, data), data, all_zero=policy == 'void')

    def test_shared_grid_equal_quota_tiny_targets_valid_only(self):
        valid = torch.ones(32, 32, dtype=torch.bool, device=self.device)
        valid[:4] = False
        masks = torch.zeros(4, 32, 32, device=self.device)
        masks[0, 5, 2] = masks[1, 7, 9] = 1
        masks[2, 8:24, 8:24] = 1  # too large
        masks[3, :4] = 1  # unknown region, must not contribute
        head = self.head()
        indices, coords = head._matching_points(valid, masks)
        self.assertEqual(indices.shape, (1, 32))
        self.assertTrue(valid.flatten()[indices].all())
        tail = indices[0, -8:]
        self.assertEqual((tail == 5 * 32 + 2).sum().item(), 4)
        self.assertEqual((tail == 7 * 32 + 9).sum().item(), 4)
        self.assertEqual(head.matching_counters['foreground_points'], 8)
        torch.testing.assert_close(coords[0, :, 0] * 32 - .5, (indices[0] % 32).float())

    def test_empty_large_and_void_small_targets_do_not_change_uniform_points(self):
        valid = torch.ones(16, 16, dtype=torch.bool, device=self.device)
        valid[:4] = False
        for masks in (torch.zeros(0, 16, 16, device=self.device),
                      torch.ones(1, 16, 16, device=self.device)):
            torch.manual_seed(7)
            expected = valid_pixel_points(valid, 1, 32)
            torch.manual_seed(7)
            actual = self.head()._matching_points(valid, masks)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def test_less_quota_than_targets_is_bounded(self):
        valid = torch.ones(8, 8, dtype=torch.bool, device=self.device)
        masks = torch.zeros(6, 8, 8, device=self.device)
        for i in range(6):
            masks[i, i, i] = 1
        indices, coords = valid_pixel_points(valid, 1, 32)
        _, _, count = balanced_small_points(valid, masks, indices, coords, 2, 64)
        self.assertEqual(count, 2)
        self.assertEqual(indices[0, -2:].unique().numel(), 2)

    def test_unknown_gt_modifications_leave_losses_equal(self):
        f = self.fixture.fixture
        data = f.inputs()
        head = self.head()
        before = f.losses(head, data)
        for gt, meta in zip(data[2], data[3]):
            gt.masks[:, ~meta['label_valid']] = 1 - gt.masks[:, ~meta['label_valid']]
        f.assert_equal(before, f.losses(head, data))


class ConfigTests(unittest.TestCase):
    def test_defaults_dependencies_and_ranges(self):
        args = parser().parse_args(['train', '--output', '/tmp/v3'])
        validate_args(args)
        self.assertFalse(any(getattr(args, f) for f in FLAGS))
        for changes in (dict(relative_depth=True), dict(modality_dropout=True), dict(dropout_prob=1),
                        dict(dropout_prob=float('nan')), dict(matching_fraction=.8)):
            current = copy.copy(args)
            vars(current).update(changes)
            with self.assertRaises(ValueError):
                validate_args(current)

    def test_all_ten_configs_share_training_eval_and_optimizer(self):
        if not Path('/workspace/mmdetection').exists():
            self.skipTest('Prepared environment required')
        args = parser().parse_args(['train', '--output', '/tmp/v3'])
        reference = build_config(args, runner_tests.categories())
        for job in experiment_jobs([1]):
            current = copy.copy(args)
            vars(current).update(job['flags'])
            cfg = build_config(current, runner_tests.categories())
            self.assertEqual(cfg.val_dataloader, reference.val_dataloader)
            self.assertEqual(cfg.train_dataloader, reference.train_dataloader)
            self.assertEqual(cfg.optim_wrapper, reference.optim_wrapper)
            self.assertEqual(cfg.model.advanced_depth.relative_depth, current.relative_depth)
            self.assertEqual(cfg.model.panoptic_head.small_matching.enabled, current.small_matching)


if __name__ == '__main__':
    unittest.main()

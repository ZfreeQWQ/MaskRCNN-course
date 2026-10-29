"""CPU/CUDA regression tests; real model smoke is additionally required."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models import Mask2Former

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbd.multiscale import SpatialFusion, MultiScaleRGBDMask2Former
from rgbd.foreground_head import ForegroundQuotaMask2FormerHead
from rgbd.void_head import VoidAwareMask2FormerHead
import test_void_head as old
import test_nyu_runner as runner_tests
from run_nyu_v2 import parser, build_config, validate_args, load_initial
from nyu_v2_plan import experiment_jobs, FLAGS
import summarize_nyu_v2 as summary
from launch_nyu_sweep import docker_command


class FourLevels(nn.Module):
    def forward(self, rgb):
        base = rgb.mean(1, keepdim=True)
        return tuple(F.avg_pool2d(base, 2 ** (i + 2)).repeat(1, ch, 1, 1)
                     for i, ch in enumerate((256, 512, 1024, 2048)))


def tiny_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    self.backbone = FourLevels()
    self.neck = None


class FusionTests(unittest.TestCase):
    def model(self, enabled=False, gate=False, rgb=False, early=False):
        with patch.object(Mask2Former, '__init__', tiny_init):
            return MultiScaleRGBDMask2Former(depth_fusion=dict(enabled=early),
                multiscale_depth=dict(enabled=enabled, reliability_gate=gate, source='rgb' if rgb else 'depth'))

    def test_zero_residual_exact_at_all_scales(self):
        batch = torch.randn(2, 5, 64, 64)
        model = self.model(True, True)
        for actual, expected in zip(model.extract_feat(batch), model.backbone(batch[:, :3])):
            self.assertTrue(torch.equal(actual, expected))

    def test_off_ignores_depth_and_has_no_branch_gradients(self):
        batch = torch.randn(2, 5, 64, 64, requires_grad=True)
        model = self.model()
        sum(x.sum() for x in model.extract_feat(batch)).backward()
        self.assertEqual(batch.grad[:, 3:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.spatial_fusion.parameters()))

    def test_rgb_control_never_reads_depth_even_after_training(self):
        model = self.model(True, True, True)
        for p in model.spatial_fusion.projections.parameters():
            nn.init.normal_(p)
        batch = torch.randn(2, 5, 64, 64)
        changed = batch.clone()
        changed[:, 3:] += 500
        for a, b in zip(model.extract_feat(batch), model.extract_feat(changed)):
            self.assertTrue(torch.equal(a, b))

    def test_depth_input_effect_and_training_path_gate_on_off(self):
        for gate in (False, True):
            model = self.model(True, gate)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            initial = {k: v.detach().clone() for k, v in model.spatial_fusion.named_parameters()}
            batch = torch.randn(2, 5, 64, 64)
            for _ in range(3):
                optimizer.zero_grad()
                sum(x.square().mean() for x in model.extract_feat(batch)).backward()
                optimizer.step()
            updates = {k: not torch.equal(v, initial[k]) for k, v in model.spatial_fusion.named_parameters()}
            self.assertTrue(any(v for k, v in updates.items() if k.startswith('projections')))
            self.assertTrue(any(v for k, v in updates.items() if k.startswith('stem')))
            self.assertEqual(any(v for k, v in updates.items() if k.startswith('gates')), gate)
            other = batch.clone()
            other[:, 3] += 1
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(model.extract_feat(batch), model.extract_feat(other))))

    def test_parameter_accounting(self):
        off, on, gated, control = self.model(), self.model(True), self.model(True, True), self.model(True, True, True)
        self.assertEqual(sum(p.numel() for p in off.parameters()), sum(p.numel() for p in on.parameters()))
        self.assertLess(off.active_parameter_count(), on.active_parameter_count())
        self.assertLess(on.active_parameter_count(), gated.active_parameter_count())
        self.assertEqual(gated.active_parameter_count(), control.active_parameter_count())

    def test_illegal_combinations_rejected(self):
        for kwargs in (dict(gate=True), dict(rgb=True), dict(enabled=True, early=True)):
            with self.assertRaises(ValueError):
                self.model(**kwargs)

    def test_strict_loading_does_not_hide_missing_core_weights(self):
        model = self.model(True, True)
        state = {k: v for k, v in model.state_dict().items() if not k.startswith('spatial_fusion.')}
        initial = load_initial(model, state)
        self.assertIn('spatial_fusion.projections.0.weight', initial)
        with self.assertRaises(ValueError):
            load_initial(model, {})


class ForegroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old.VoidHeadTests.setUpClass()
        cls.fixture = old.VoidHeadTests()
        cls.device = cls.fixture.device

    def head(self, enabled, small=False, boundary=False):
        head = self.fixture.head(small=small, boundary=boundary)
        head.__class__ = ForegroundQuotaMask2FormerHead
        head.foreground_sampling = dict(enabled=enabled, quota=8, area_threshold=64.)
        return head

    def test_off_is_bitwise_equal_to_v1_all_loss_combinations(self):
        for policy in ('full', 'partial', 'void'):
            for small in (False, True):
                for boundary in (False, True):
                    inputs = self.fixture.inputs(policy=policy)
                    a = self.fixture.losses(self.head(False, small, boundary), inputs)
                    b = self.fixture.losses(self.fixture.head(small=small, boundary=boundary), inputs)
                    self.fixture.assert_equal(a, b)

    def test_enabled_real_losses_finite_and_backward(self):
        for policy in ('full', 'partial', 'void'):
            for counts in ((2, 1), (0, 0), (0, 2)):
                for small, boundary in ((False, False), (True, True)):
                    inputs = self.fixture.inputs(counts=counts, policy=policy)
                    losses = self.fixture.losses(self.head(True, small, boundary), inputs)
                    self.fixture.assert_backward(losses, inputs, all_zero=policy == 'void')

    def test_quota_only_replaces_random_tail_and_samples_valid_foreground(self):
        head = self.head(True)
        valid = torch.ones(32, 32, dtype=torch.bool, device=self.device)
        valid[:4] = False
        targets = torch.zeros(3, 32, 32, device=self.device)
        targets[0, 5, 7] = 1  # one-pixel target; sampling must support replacement
        targets[1, 8:24, 8:24] = 1  # too large, unchanged
        predictions = torch.randn(3, 8, 8, device=self.device)
        torch.manual_seed(7)
        indices, coords = head._sample_loss_points(predictions, valid, targets)
        torch.manual_seed(7)
        old_indices, old_coords = head._uncertain_valid_points(predictions, valid)
        self.assertEqual(indices.shape, (3, head.num_points))
        self.assertTrue(valid.flatten()[indices].all())
        self.assertTrue((indices[0, -8:] == 5 * 32 + 7).all())
        self.assertTrue(torch.equal(indices[:, :-8], old_indices[:, :-8]))
        self.assertTrue(torch.equal(indices[1:], old_indices[1:]))
        self.assertTrue(torch.equal(coords[1:], old_coords[1:]))

    def test_unknown_labels_do_not_affect_quota_or_loss(self):
        inputs = self.fixture.inputs()
        head = self.head(True, True, True)
        before = self.fixture.losses(head, inputs)
        for gt, meta in zip(inputs[2], inputs[3]):
            gt.masks[:, ~meta['label_valid']] = 1 - gt.masks[:, ~meta['label_valid']]
        self.fixture.assert_equal(before, self.fixture.losses(head, inputs))

    def test_assignment_implementation_is_unchanged(self):
        self.assertIs(ForegroundQuotaMask2FormerHead._get_targets_single,
                      VoidAwareMask2FormerHead._get_targets_single)


class PlanTests(unittest.TestCase):
    def test_fixed_jobs_switches_and_independent_defaults(self):
        jobs = experiment_jobs([1, 2, 3])
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({j['id'] for j in jobs}), 24)
        self.assertFalse(any(jobs[0]['flags'].values()))
        args = parser().parse_args(['train', '--output', '/tmp/test_v2'])
        validate_args(args)
        self.assertFalse(any(getattr(args, flag) for flag in FLAGS))
        for changes in (dict(reliability_gate=True), dict(rgb_control=True), dict(iters=1001),
                        dict(multiscale_depth=True, depth_fusion=True)):
            bad = copy.copy(args)
            vars(bad).update(changes)
            with self.assertRaises(ValueError):
                validate_args(bad)

    def test_config_budget_and_matched_control(self):
        if not Path('/workspace/mmdetection').exists():
            self.skipTest('Prepared container required')
        args = parser().parse_args(['train', '--output', '/tmp/test_v2'])
        baseline = build_config(args, runner_tests.categories())
        for job in experiment_jobs([1]):
            current = copy.copy(args)
            vars(current).update(job['flags'])
            cfg = build_config(current, runner_tests.categories())
            self.assertEqual(cfg.model.type, 'MultiScaleRGBDMask2Former')
            self.assertEqual(cfg.train_cfg.max_iters, 5000)
            self.assertEqual(cfg.train_cfg.val_interval, 1000)
            self.assertEqual(cfg.default_hooks.checkpoint.max_keep_ckpts, 2)
            self.assertEqual(cfg.optim_wrapper, baseline.optim_wrapper)
            self.assertEqual(cfg.val_dataloader, baseline.val_dataloader)
            self.assertEqual(cfg.model.panoptic_head.foreground_sampling.enabled, current.foreground_points)


if __name__ == '__main__':
    unittest.main()

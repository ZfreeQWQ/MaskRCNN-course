"""CPU model-path tests; separate real GPU smoke is still required."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from mmdet.models import Mask2Former

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbd.multiscale import MultiScaleRGBDMask2Former
from rgbd.course_feature_models import (CourseFeatureRGBDMask2Former,
                                       Stride4DetailResidual,
                                       checked_feature_options,
                                       sensor_reliability)


class FourLevels(nn.Module):
    def forward(self, rgb):
        base = rgb.mean(1, keepdim=True)
        return tuple(F.avg_pool2d(base, 2 ** (index + 2)).repeat(1, channels, 1, 1)
                     for index, channels in enumerate((256, 512, 1024, 2048)))


def tiny_init(self, *args, **kwargs):
    nn.Module.__init__(self)
    self.backbone = FourLevels()
    self.neck = None


class CourseFeatureTests(unittest.TestCase):
    def model(self, detail=False, reliability=False, parent=False,
              multiscale=True, source='depth', gate=True, early=False):
        cls = MultiScaleRGBDMask2Former if parent else CourseFeatureRGBDMask2Former
        options = {} if parent else dict(course_detail=dict(enabled=detail),
            course_depth_reliability=dict(enabled=reliability))
        with patch.object(Mask2Former, '__init__', tiny_init):
            return cls(depth_fusion=dict(enabled=early), multiscale_depth=dict(
                enabled=multiscale, reliability_gate=gate, source=source), **options)

    def batch(self, height=64, width=64):
        data = torch.randn(2, 5, height, width)
        data[:, 3] = torch.rand(2, height, width) + .1
        data[:, 4] = 1
        return data

    def learned_fusion(self, model):
        for param in model.spatial_fusion.projections.parameters():
            nn.init.normal_(param, std=.01)
        for param in model.spatial_fusion.gates.parameters():
            nn.init.normal_(param, std=.01)

    def test_disabled_exact_state_constructor_rng_outputs_and_forward_rng(self):
        torch.manual_seed(234)
        actual = self.model()
        state = torch.get_rng_state()
        torch.manual_seed(234)
        reference = self.model(parent=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(set(actual.state_dict()), set(reference.state_dict()))
        for key in actual.state_dict():
            self.assertTrue(torch.equal(actual.state_dict()[key], reference.state_dict()[key]))
        self.learned_fusion(reference)
        actual.load_state_dict(reference.state_dict(), strict=True)
        data = self.batch()
        for train in (True, False):
            actual.train(train); reference.train(train)
            rng = torch.get_rng_state()
            a, b = actual.extract_feat(data), reference.extract_feat(data)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertFalse(any(actual.feature_activity.values()))

    def test_enabled_detail_construction_preserves_parent_rng_and_base_weights(self):
        torch.manual_seed(235)
        model = self.model(detail=True)
        state = torch.get_rng_state()
        torch.manual_seed(235)
        parent = self.model(parent=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        self.assertEqual(model.course_detail_adapter.weight.count_nonzero().item(), 0)

    def test_zero_init_detail_exact_parent_at_all_shapes(self):
        for height, width in ((64, 64), (80, 96)):
            model = self.model(detail=True)
            self.learned_fusion(model)
            parent = self.model(parent=True)
            parent.load_state_dict({k: v for k, v in model.state_dict().items()
                                    if not k.startswith('course_detail_adapter.')})
            data = self.batch(height, width)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                model.extract_feat(data), parent.extract_feat(data))))

    def test_detail_only_changes_stride_four_after_update(self):
        model = self.model(detail=True)
        parent = self.model(parent=True)
        parent.load_state_dict({k: v for k, v in model.state_dict().items()
                                if not k.startswith('course_detail_adapter.')})
        data = self.batch()
        with torch.no_grad():
            model.course_detail_adapter.weight.fill_(.1)
        actual, reference = model.extract_feat(data), parent.extract_feat(data)
        self.assertFalse(torch.equal(actual[0], reference[0]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual[1:], reference[1:])))

    def test_detail_constant_feature_has_zero_response(self):
        adapter = Stride4DetailResidual(3)
        with torch.no_grad():
            adapter.weight.fill_(1.)
        self.assertEqual(adapter(torch.ones(2, 3, 7, 11)).count_nonzero().item(), 0)

    def test_sensor_reliability_bounds_holes_and_detached(self):
        depth = torch.ones(1, 1, 7, 9, requires_grad=True)
        measured = torch.ones_like(depth)
        measured[..., 3, 4] = 0
        reliability = sensor_reliability(depth, measured)
        self.assertFalse(reliability.requires_grad)
        self.assertEqual(reliability[..., 3, 4].item(), 0)
        self.assertAlmostEqual(reliability[..., 3, 3].item(), 8 / 9, places=6)
        self.assertEqual(reliability[..., 0, 0].item(), 1)
        self.assertTrue(((reliability >= 0) & (reliability <= 1)).all())

    def test_sensor_reliability_nonfinite_and_nonpositive_zero(self):
        depth = torch.tensor([[[[float('nan'), float('inf'), 0., -1., 1.]]]])
        measured = torch.ones_like(depth)
        result = sensor_reliability(depth, measured)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[..., :4].count_nonzero().item(), 0)
        measured[..., 4] = float('nan')
        self.assertEqual(sensor_reliability(depth, measured).count_nonzero().item(), 0)

    def test_reliability_all_valid_constant_exact_parent_and_invalid_exact_rgb(self):
        for gate in (False, True):
            model = self.model(reliability=True, gate=gate)
            self.learned_fusion(model)
            parent = self.model(parent=True, gate=gate)
            parent.load_state_dict(model.state_dict(), strict=True)
            data = self.batch()
            data[:, 3] = .5
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                model.extract_feat(data), parent.extract_feat(data))))
            data[:, 4] = 0
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                model.extract_feat(data), model.backbone(data[:, :3]))))

    def test_reliability_relative_consistency_suppresses_depth_step_not_flat_regions(self):
        depth = torch.ones(1, 1, 8, 12)
        depth[..., 6:] = 2
        measured = torch.ones_like(depth)
        reliability = sensor_reliability(depth, measured)
        self.assertTrue((reliability[..., 5:7] < 1).all())
        self.assertTrue((reliability[..., :5] == 1).all())
        self.assertTrue((reliability[..., 7:] == 1).all())
        for factor in (.1, 3.):
            torch.testing.assert_close(reliability, sensor_reliability(depth * factor, measured),
                                       atol=1e-6, rtol=1e-6)

    def test_reliability_partial_valid_modifies_only_depth_residual(self):
        model = self.model(reliability=True)
        self.learned_fusion(model)
        parent = self.model(parent=True)
        parent.load_state_dict(model.state_dict(), strict=True)
        data = self.batch()
        data[:, 4, :, :32] = 0
        reliability = sensor_reliability(data[:, 3:4], data[:, 4:5])
        actual, full, rgb = model.extract_feat(data), parent.extract_feat(data), model.backbone(data[:, :3])
        for result, original, feature in zip(actual, full, rgb):
            scale = F.interpolate(reliability, feature.shape[-2:], mode='area')
            torch.testing.assert_close(result, feature + (original - feature) * scale,
                                       atol=1e-6, rtol=1e-5)
            no_sensor = scale.expand_as(result) == 0
            self.assertTrue(torch.equal(result[no_sensor], feature[no_sensor]))

    def test_all_switch_combinations_finite_backward_and_parameter_updates(self):
        for detail, reliability in ((True, False), (False, True), (True, True)):
            model = self.model(detail=detail, reliability=reliability)
            data = self.batch().requires_grad_()
            initial = {name: value.detach().clone() for name, value in model.named_parameters()}
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            for _ in range(3):
                optimizer.zero_grad()
                loss = sum(level.square().mean() for level in model.extract_feat(data))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                optimizer.step()
            self.assertTrue(any(not torch.equal(parameter, initial[name])
                for name, parameter in model.named_parameters()
                if name.startswith('spatial_fusion.projections.')))
            if detail:
                self.assertFalse(torch.equal(model.course_detail_adapter.weight,
                                              initial['course_detail_adapter.weight']))
            self.assertTrue(torch.isfinite(data.grad).all())
            self.assertGreater(data.grad[:, :3].abs().sum().item(), 0)

    def test_parameter_accounting_new_module_only_when_enabled(self):
        base = self.model(parent=True)
        for detail, reliability in ((False, False), (True, False), (False, True), (True, True)):
            model = self.model(detail=detail, reliability=reliability)
            self.assertEqual(model.active_parameter_count() - base.active_parameter_count(),
                             256 * 9 if detail else 0)
            self.assertEqual(hasattr(model, 'course_detail_adapter'), detail)

    def test_detail_rgb_control_ignores_depth_and_keeps_single_backbone_call(self):
        model = self.model(detail=True, source='rgb').eval()
        self.learned_fusion(model)
        data = self.batch()
        other = data.clone(); other[:, 3:] = float('nan')
        with patch.object(model.backbone, 'forward', wraps=model.backbone.forward) as backbone:
            actual = model.extract_feat(data)
            self.assertEqual(backbone.call_count, 1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, model.extract_feat(other))))

    def test_activity_metadata_train_and_eval(self):
        model = self.model(detail=True, reliability=True)
        model.extract_feat(self.batch())
        model.eval().extract_feat(self.batch())
        self.assertEqual(model.feature_activity, dict(train_calls=1, eval_calls=1,
                                                     detail_samples=4, reliability_samples=4,
                                                     detail_train_samples=2, reliability_train_samples=2))
        self.assertTrue(all(type(value) is int for value in model.feature_activity.values()))
        snapshot = model.feature_activity_snapshot()
        self.assertEqual(snapshot['training_sensor_valid_pixels'], 2 * 64 * 64)
        self.assertGreater(snapshot['training_reliability_suppressed_pixels'], 0)
        self.assertGreater(snapshot['training_mean_reliability_on_valid'], 0)
        self.assertLess(snapshot['training_mean_reliability_on_valid'], 1)
        self.assertFalse(model._feature_reliability_totals.requires_grad)

    def test_activity_valid_pixel_statistics_exclude_invalid_padding_and_eval(self):
        model = self.model(reliability=True)
        data = self.batch()
        data[:, 3] = .5
        data[:, 4, 32:] = 0
        model.extract_feat(data)
        before = model.feature_activity_snapshot()
        self.assertEqual(before['training_sensor_valid_pixels'], 2 * 32 * 64)
        model.eval().extract_feat(data)
        after = model.feature_activity_snapshot()
        for key in ('training_sensor_valid_pixels', 'training_reliability_suppressed_pixels',
                    'training_reliability_sum', 'training_mean_reliability_on_valid'):
            self.assertEqual(before[key], after[key])

    def test_validation_rejects_unknown_nonboolean_and_incompatible(self):
        for options in (dict(enabled=1), dict(enabled='yes'), dict(extra=True)):
            with self.assertRaises(ValueError):
                checked_feature_options(options)
        for options in (dict(reliability=True, multiscale=False, gate=False),
                        dict(reliability=True, source='rgb'),
                        dict(detail=True, multiscale=False, gate=False, early=True)):
            with self.assertRaises(ValueError):
                self.model(**options)
        with self.assertRaises(ValueError):
            sensor_reliability(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4))
        with self.assertRaises(ValueError):
            Stride4DetailResidual(0)
        with self.assertRaises(ValueError):
            Stride4DetailResidual(2)(torch.zeros(1, 3, 4, 4))


if __name__ == '__main__':
    unittest.main(verbosity=2)

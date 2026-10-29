"""Two independent, default-off course feature interventions.

``course_detail`` learns a zero-initialized depthwise 3x3 residual on the
high-pass RGB stride-4 feature.  It is a cheap local-detail hypothesis, NOT a
PointRend implementation, an increased-resolution feature pyramid, or evidence
of an accuracy gain.  The existing four-scale backbone/fusion stays intact.

``course_depth_reliability`` multiplies each depth residual by a sensor-derived
reliability map before addition to RGB.  At input resolution R = measured_valid
* local_3x3_valid_fraction / (1 + abs(depth - local_mean) / local_mean), where the
mean uses only valid measurements.  Area pooling gives the per-feature-cell
multiplier. R is in [0, 1]. All-valid CONSTANT depth preserves parent fusion;
all-invalid input removes its entire depth residual. Filled depth still enters
the encoder. This is a fixed local-consistency heuristic, not calibrated sensor
confidence: true object boundaries can also be suppressed and performance can
fall. No ground truth or label-valid plane enters either path.

Metadata counters and three detached device-side training totals are collected.
There is no per-forward host/device synchronization or persistent computation
graph. JSON conversion synchronizes only in feature_activity_snapshot(), which
the reporting hook should call at its existing sparse interval, not every step.
"""

import torch
import torch.nn.functional as F
from torch import nn
from mmdet.registry import MODELS

from rgbd.model import RGBDMask2Former
from rgbd.multiscale import MultiScaleRGBDMask2Former


def checked_feature_options(options=None, name='course feature'):
    supplied = dict(options or {})
    if set(supplied) - {'enabled'}:
        raise ValueError(f'Unknown {name} options: {sorted(set(supplied) - {"enabled"})}')
    enabled = supplied.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError(f'{name}.enabled must be bool')
    return dict(enabled=enabled)


def _sensor_reliability_and_valid(depth, measured):
    """Detached local sensor validity/consistency, never annotation validity.

    Both inputs are aligned N1HW planes after RGBDDataPreprocessor.  Numerical
    validity is checked defensively; missing measurements and zero padding
    remain zero.  Pooling excludes outside-image padding so a fully valid image
    with constant depth retains reliability one at its edges and interior.
    Positive local means divide exactly, rather than adding a scale-dependent
    epsilon; zero-support locations use denominator one and are masked out.
    """
    if depth.ndim != 4 or depth.shape[1] != 1 or measured.shape != depth.shape:
        raise ValueError('Reliability requires aligned N1HW depth/measurement planes')
    with torch.no_grad():
        valid_bool = (torch.isfinite(depth) & (depth > 0) &
                      torch.isfinite(measured) & (measured > .5))
        valid = valid_bool.to(depth.dtype)
        support = F.avg_pool2d(valid, 3, stride=1, padding=1,
                               count_include_pad=False)
        safe_depth = torch.where(valid_bool, depth, torch.zeros_like(depth))
        local_sum = F.avg_pool2d(safe_depth, 3, stride=1, padding=1,
                                 count_include_pad=False)
        local_mean = local_sum / torch.where(support > 0, support, torch.ones_like(support))
        denominator = torch.where(local_mean > 0, local_mean, torch.ones_like(local_mean))
        relative_deviation = (safe_depth - local_mean).abs() / denominator
        reliability = valid * support / (1 + relative_deviation)
        return reliability, valid_bool


def sensor_reliability(depth, measured):
    """Return a detached N1HW reliability map in [0, 1]."""
    return _sensor_reliability_and_valid(depth, measured)[0]


class Stride4DetailResidual(nn.Module):
    """Zero-init per-channel local filter, without random initialization draws."""

    def __init__(self, channels):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError('Detail channels must be a positive integer')
        self.channels = channels
        # nn.Conv2d would consume RNG before zeroing.  Direct Parameter creation
        # also preserves parent initialization/RNG when this switch is enabled.
        self.weight = nn.Parameter(torch.zeros(channels, 1, 3, 3))

    def forward(self, rgb_feature):
        if rgb_feature.ndim != 4 or rgb_feature.shape[1] != self.channels:
            raise ValueError('Detail feature channels disagree with the backbone')
        smooth = F.avg_pool2d(rgb_feature, 3, stride=1, padding=1,
                              count_include_pad=False)
        high_pass = rgb_feature - smooth
        return F.conv2d(high_pass, self.weight, padding=1, groups=self.channels)


@MODELS.register_module()
class CourseFeatureRGBDMask2Former(MultiScaleRGBDMask2Former):
    """Matched-base feature ablations; disabled is the exact parent path."""

    def __init__(self, *args, course_detail=None,
                 course_depth_reliability=None, **kwargs):
        detail = checked_feature_options(course_detail, 'course_detail')
        reliability = checked_feature_options(course_depth_reliability,
                                              'course_depth_reliability')
        super().__init__(*args, **kwargs)
        if reliability['enabled'] and (
                not self.multiscale_options['enabled'] or
                self.multiscale_options['source'] != 'depth'):
            raise ValueError('Course depth reliability requires multiscale depth source')
        if detail['enabled'] and self.depth_fusion_enabled:
            raise ValueError('Course RGB detail requires an unmodified RGB backbone input')
        self.course_detail = detail
        self.course_depth_reliability = reliability
        if detail['enabled']:
            self.course_detail_adapter = Stride4DetailResidual(
                self.spatial_fusion.projections[0].out_channels)
        self.feature_activity = dict(train_calls=0, eval_calls=0,
                                     detail_samples=0, reliability_samples=0,
                                     detail_train_samples=0, reliability_train_samples=0)
        self._feature_reliability_totals = None

    def feature_activity_snapshot(self):
        """JSON-safe audit; host sync only here, never during feature extraction.

        Pixel totals are from training only and exclude all sensor-invalid
        pixels, including padded zeros. They describe augmented input pixels,
        not unique dataset pixels or ground-truth instances.
        """
        snapshot = dict(self.feature_activity)
        totals = self._feature_reliability_totals
        valid, suppressed, reliability_sum = (0., 0., 0.) if totals is None else totals.cpu().tolist()
        snapshot.update(training_sensor_valid_pixels=int(valid),
                        training_reliability_suppressed_pixels=int(suppressed),
                        training_reliability_sum=reliability_sum,
                        training_mean_reliability_on_valid=(reliability_sum / valid if valid else None))
        return snapshot

    def _reliable_fusion(self, rgb_features, batch_inputs):
        """Parent fusion arithmetic with one bounded residual multiplier.

        Multiply BEFORE addition, not via ``fused - rgb``: subtracting an
        already-rounded sum would lose small residuals and needlessly change
        the all-valid control.  Modules/parameters are the parent's originals.
        """
        fusion = self.spatial_fusion
        if len(rgb_features) != 4:
            raise ValueError('Course reliability requires four backbone levels')
        planes, cue = batch_inputs[:, 3:5], batch_inputs[:, 4:5]
        reliability, sensor_valid = _sensor_reliability_and_valid(planes[:, :1], cue)
        if self.training:
            # FP64 scalar accumulation prevents loss of integer pixel counts
            # after many megapixel batches. These tensors are detached and are
            # not model parameters/buffers or part of checkpoint weights.
            totals = torch.stack((sensor_valid.sum(dtype=torch.int64).to(torch.float64),
                (sensor_valid & (reliability < 1 - 1e-6)).sum(dtype=torch.int64).to(torch.float64),
                reliability.sum(dtype=torch.float64)))
            if self._feature_reliability_totals is None:
                self._feature_reliability_totals = totals.detach()
            else:
                self._feature_reliability_totals.add_(totals.detach())
        level = fusion.stem(planes)
        depth_levels = [level]
        for stage in fusion.stages:
            level = stage(level)
            depth_levels.append(level)
        fused = []
        for index, (rgb, depth) in enumerate(zip(rgb_features, depth_levels)):
            if rgb.shape[1] != fusion.projections[index].out_channels:
                raise ValueError('Backbone channels disagree with depth projections')
            size = rgb.shape[-2:]
            depth = F.interpolate(depth, size, mode='bilinear', align_corners=False)
            residual = fusion.projections[index](depth)
            if self.multiscale_options['reliability_gate']:
                measured = F.interpolate(cue, size, mode='area')
                rgb_context = torch.tanh(rgb.mean(1, keepdim=True))
                gate = 2 * torch.sigmoid(fusion.gates[index](
                    torch.cat([depth, rgb_context, measured], 1)))
                residual = residual * gate
            residual = residual * F.interpolate(reliability, size, mode='area')
            fused.append(rgb + residual)
        return tuple(fused)

    def extract_feat(self, batch_inputs):
        detail = self.course_detail['enabled']
        reliability = self.course_depth_reliability['enabled']
        if not detail and not reliability:
            return super().extract_feat(batch_inputs)

        # Exactly one RGB backbone call.  Detail is based on RGB features, not
        # the already fused signal; the switches remain independently usable.
        rgb_features = RGBDMask2Former.extract_feat(self, batch_inputs)
        if reliability:
            fused = self._reliable_fusion(rgb_features, batch_inputs)
        elif self.multiscale_options['enabled']:
            if self.multiscale_options['source'] == 'depth':
                planes, cue = batch_inputs[:, 3:5], batch_inputs[:, 4:5]
            else:
                gray = batch_inputs[:, :3].mean(1, keepdim=True)
                cue = torch.ones_like(gray)
                planes = torch.cat([gray, cue], 1)
            fused = self.spatial_fusion(rgb_features, planes, cue,
                                       self.multiscale_options['reliability_gate'])
        else:
            fused = rgb_features
        if detail:
            fused = (fused[0] + self.course_detail_adapter(rgb_features[0]), *fused[1:])

        self.feature_activity['train_calls' if self.training else 'eval_calls'] += 1
        self.feature_activity['detail_samples'] += batch_inputs.shape[0] if detail else 0
        self.feature_activity['reliability_samples'] += batch_inputs.shape[0] if reliability else 0
        if self.training:
            self.feature_activity['detail_train_samples'] += batch_inputs.shape[0] if detail else 0
            self.feature_activity['reliability_train_samples'] += batch_inputs.shape[0] if reliability else 0
        return fused

    # Inherited active_parameter_count sums self.parameters(), including the
    # enabled detail adapter, then removes only the parent's inactive branches.
    # Reliability introduces zero parameters.  Disabled detail has no module.

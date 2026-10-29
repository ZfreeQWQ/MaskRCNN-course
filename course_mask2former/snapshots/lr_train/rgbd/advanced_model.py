"""Parameter-free relative log-depth and training-only residual branch dropout.

These are hypotheses, not HHA or a reproduction of a published RGB-D model.
Relative normalization uses positive finite *filled depth*, never labels.
Raw measurement validity is preserved as the second plane and gating cue.
"""
import torch
from mmdet.registry import MODELS
from rgbd.model import RGBDMask2Former
from rgbd.multiscale import MultiScaleRGBDMask2Former


def relative_log_depth(depth):
    valid = torch.isfinite(depth) & (depth > 0)
    safe = torch.where(valid, depth, torch.ones_like(depth))
    log = safe.clamp_min(1e-6).log()
    count = valid.sum((-2, -1), keepdim=True).clamp_min(1)
    mean = (log * valid).sum((-2, -1), keepdim=True) / count
    centred = torch.where(valid, log - mean, torch.zeros_like(log))
    std = ((centred.square().sum((-2, -1), keepdim=True) / count).clamp_min(.01)).sqrt()
    return (centred / (3 * std)).clamp(-1, 1)


@MODELS.register_module()
class AdvancedRGBDMask2Former(MultiScaleRGBDMask2Former):
    def __init__(self, *args, advanced_depth=None, **kwargs):
        options = dict(relative_depth=False, modality_dropout=False, dropout_prob=.2)
        supplied = dict(advanced_depth or {})
        if set(supplied) - set(options):
            raise ValueError('Unknown advanced depth option')
        options.update(supplied)
        if any(type(options[k]) is not bool for k in ('relative_depth', 'modality_dropout')):
            raise TypeError('Advanced switches must be boolean')
        if not 0 < options['dropout_prob'] < 1:
            raise ValueError('dropout_prob must be finite and strictly between 0 and 1')
        super().__init__(*args, **kwargs)
        if (options['relative_depth'] or options['modality_dropout']) and not self.multiscale_options['enabled']:
            raise ValueError('Relative depth/dropout require multiscale fusion')
        self.advanced_options = options
        self.advanced_counters = dict(relative_training_samples=0, dropout_training_samples=0, dropped_samples=0)

    def extract_feat(self, batch_inputs):
        relative, dropout = (self.advanced_options[k] for k in ('relative_depth', 'modality_dropout'))
        if not relative and not dropout:
            return super().extract_feat(batch_inputs)  # exact v2 path and RNG when disabled
        rgb_features = RGBDMask2Former.extract_feat(self, batch_inputs)
        if self.multiscale_options['source'] == 'rgb':
            gray = batch_inputs[:, :3].mean(1, keepdim=True)
            cue = torch.ones_like(gray)
            # Positive pseudo-intensity for the relative transform. No depth or
            # raw-valid reads, including in this matched-capacity control.
            first = gray.sigmoid() if relative else gray
        else:
            first, cue = batch_inputs[:, 3:4], batch_inputs[:, 4:5]
        if relative:
            first = relative_log_depth(first)
            if self.training:
                self.advanced_counters['relative_training_samples'] += batch_inputs.shape[0]
        planes = torch.cat([first, cue], 1)
        fused = self.spatial_fusion(rgb_features, planes, cue, self.multiscale_options['reliability_gate'])
        if dropout and self.training:
            probability = self.advanced_options['dropout_prob']
            keep = torch.rand((batch_inputs.shape[0], 1, 1, 1), device=batch_inputs.device) >= probability
            self.advanced_counters['dropout_training_samples'] += keep.numel()
            self.advanced_counters['dropped_samples'] += int((~keep).sum())
            # One Bernoulli decision per image shared by all four scales.
            # Drop after encoding: even learned normalization biases cannot
            # inject a residual for a dropped image. Inverted dropout scaling.
            fused = tuple(rgb + (out - rgb) * keep.to(out.dtype) / (1 - probability)
                          for rgb, out in zip(rgb_features, fused))
        return fused

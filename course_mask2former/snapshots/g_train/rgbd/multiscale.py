"""Opt-in spatial RGB-D fusion; a matched-capacity RGB-only control is included.

No labels enter extract_feat. The four zero-initialized residual projections
initially preserve the pretrained RGB features exactly. Spatial gates start at
one (2*sigmoid(0)), so the gate switch does not halve the initial update scale.
Raw measurement validity is an input cue, not a guaranteed confidence score.
"""
import torch
import torch.nn.functional as F
from torch import nn
from mmdet.registry import MODELS
from rgbd.model import RGBDMask2Former


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU(inplace=False))


class SpatialFusion(nn.Module):
    widths = (32, 48, 64, 96)

    def __init__(self, out_channels=(256, 512, 1024, 2048)):
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError('Four backbone scales are required')
        self.stem = nn.Sequential(block(2, 16, 2), block(16, self.widths[0], 2))
        self.stages = nn.ModuleList([block(a, b, 2) for a, b in zip(self.widths, self.widths[1:])])
        self.projections = nn.ModuleList([nn.Conv2d(a, b, 1, bias=False)
                                          for a, b in zip(self.widths, out_channels)])
        self.gates = nn.ModuleList([nn.Conv2d(a + 2, 1, 1) for a in self.widths])
        for module in [*self.projections, *self.gates]:
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, rgb_features, planes, cue, use_gate):
        if len(rgb_features) != 4 or planes.shape[1] != 2 or cue.shape[1] != 1:
            raise ValueError('Expected four feature levels, two input planes and one validity cue')
        x = self.stem(planes)
        levels = [x]
        for stage in self.stages:
            x = stage(x)
            levels.append(x)
        fused = []
        for index, (rgb, depth) in enumerate(zip(rgb_features, levels)):
            if rgb.shape[1] != self.projections[index].out_channels:
                raise ValueError('Backbone channels disagree with depth projections')
            depth = F.interpolate(depth, rgb.shape[-2:], mode='bilinear', align_corners=False)
            residual = self.projections[index](depth)
            if use_gate:
                # All three inputs are features/sensor data, never label_valid.
                measured = F.interpolate(cue, rgb.shape[-2:], mode='area')
                rgb_context = torch.tanh(rgb.mean(1, keepdim=True))
                gate = 2 * torch.sigmoid(self.gates[index](torch.cat([depth, rgb_context, measured], 1)))
                residual = residual * gate
            fused.append(rgb + residual)
        return tuple(fused)


@MODELS.register_module()
class MultiScaleRGBDMask2Former(RGBDMask2Former):
    def __init__(self, *args, multiscale_depth=None, **kwargs):
        options = dict(enabled=False, reliability_gate=False, source='depth')
        supplied = dict(multiscale_depth or {})
        if set(supplied) - set(options):
            raise ValueError('Unknown multiscale fusion options')
        options.update(supplied)
        if any(type(options[k]) is not bool for k in ('enabled', 'reliability_gate')):
            raise TypeError('Fusion switches must be bool')
        if options['source'] not in ('depth', 'rgb'):
            raise ValueError('Fusion source must be depth or rgb')
        if not options['enabled'] and (options['reliability_gate'] or options['source'] == 'rgb'):
            raise ValueError('Gate/RGB control requires multiscale fusion')
        super().__init__(*args, **kwargs)
        if options['enabled'] and self.depth_fusion_enabled:
            raise ValueError('Early and multiscale fusion are alternative modes in this experiment')
        self.multiscale_options = options
        # Same constructor and parameter initialization for every ablation.
        self.spatial_fusion = SpatialFusion()

    def extract_feat(self, batch_inputs):
        rgb_features = super().extract_feat(batch_inputs)
        if not self.multiscale_options['enabled']:
            return rgb_features
        if self.multiscale_options['source'] == 'depth':
            planes = batch_inputs[:, 3:5]
            cue = batch_inputs[:, 4:5]
        else:
            # Capacity control deliberately reads neither depth nor raw_valid.
            gray = batch_inputs[:, :3].mean(1, keepdim=True)
            cue = torch.ones_like(gray)
            planes = torch.cat([gray, cue], 1)
        return self.spatial_fusion(rgb_features, planes, cue,
                                   self.multiscale_options['reliability_gate'])

    def active_parameter_count(self):
        count = sum(p.numel() for p in self.parameters())
        if not self.depth_fusion_enabled:
            count -= self.depth_adapter.weight.numel()
        if not self.multiscale_options['enabled']:
            count -= sum(p.numel() for p in self.spatial_fusion.parameters())
        elif not self.multiscale_options['reliability_gate']:
            count -= sum(p.numel() for p in self.spatial_fusion.gates.parameters())
        return count

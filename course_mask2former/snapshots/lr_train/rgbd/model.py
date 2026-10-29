"""RGB-preserving preprocessing and independently switchable depth fusion."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from mmengine.structures import PixelData
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.detectors import Mask2Former
from mmdet.registry import MODELS


@MODELS.register_module()
class RGBDDataPreprocessor(DetDataPreprocessor):
    """Run the upstream RGB path unchanged, then attach normalized depth.

Input channels are BGR/RGB(3), metres(1), raw-valid(1). RGB uses the configured
pretrained mean/std/channel order. Finite positive depth is clipped at
``depth_max`` metres and divided by depth_max. The sensor-valid channel remains
binary. Right/bottom padding for depth/validity/label_valid is always zero.

Only BatchFixedSizePad batch augmentation is supported, because arbitrary
post-pack RGB geometry could silently misalign the separate depth planes.
Label-valid PixelData is required for every sample: the associated void-aware
head uses it to avoid treating unknown labels as supervised background.
    """

    def __init__(self, depth_max=10.0, **kwargs):
        if not math.isfinite(depth_max) or depth_max <= 0:
            raise ValueError('depth_max must be finite and positive')
        for augment in kwargs.get('batch_augments') or []:
            if augment.get('type') != 'BatchFixedSizePad':
                raise ValueError('RGBDDataPreprocessor only supports BatchFixedSizePad batch augmentation')
            if augment.get('img_pad_value', 0) != 0:
                raise ValueError('RGB-D batch padding must be zero')
        if kwargs.get('pad_value', 0) != 0:
            raise ValueError('RGB-D padding must be zero')
        super().__init__(**kwargs)
        self.depth_max = float(depth_max)

    def forward(self, data, training=False):
        data = self.cast_data(data)
        incoming = data['inputs']
        is_batched = isinstance(incoming, torch.Tensor)
        if is_batched and incoming.ndim != 4:
            raise ValueError('Batched RGB-D inputs must be NCHW')
        items = list(incoming) if is_batched else incoming
        if not items or any(x.ndim != 3 or x.shape[0] != 5 for x in items):
            raise ValueError('RGB-D inputs must contain exactly five channels')
        samples = data.get('data_samples')
        if samples is None or len(samples) != len(items):
            raise ValueError('RGB-D inputs require one data_sample per image')
        for item, sample in zip(items, samples):
            if 'label_valid' not in sample or sample.label_valid.data.shape != (1, *item.shape[-2:]):
                raise ValueError('Each sample needs an aligned label_valid.data [1,H,W]')

        rgb = incoming[:, :3] if is_batched else [item[:3] for item in items]
        rgb_data = dict(data, inputs=rgb)
        output = super().forward(rgb_data, training=training)
        height, width = output['inputs'].shape[-2:]
        extra = []
        for item, sample in zip(items, output['data_samples']):
            depth = item[3:4].float()
            numerical_valid = torch.isfinite(depth) & (depth > 0)
            depth = torch.where(numerical_valid, depth, 0).clamp(max=self.depth_max) / self.depth_max
            measured = (torch.isfinite(item[4:5]) & (item[4:5] > 0.5) & numerical_valid).to(depth.dtype)
            pad_h, pad_w = height - item.shape[-2], width - item.shape[-1]
            if min(pad_h, pad_w) < 0:
                raise ValueError('Unexpected geometry-changing RGB batch augmentation')
            padding = (0, pad_w, 0, pad_h)
            extra.append(F.pad(torch.cat([depth, measured], dim=0), padding, value=0))
            sample.label_valid = PixelData(
                data=F.pad(sample.label_valid.data.bool(), padding, value=False))
        output['inputs'] = torch.cat([output['inputs'], torch.stack(extra)], dim=1)
        return output


@MODELS.register_module()
class RGBDMask2Former(Mask2Former):
    """Zero-initialized depth residual at the pretrained RGB backbone input.

``depth_fusion=dict(enabled=False)`` disables the entire depth branch. On/off
models contain the same six adapter parameters; when off they receive no
gradient or optimizer update. At zero initialization, on/off features are
identical for the same RGB/backbone parameters. This is an exploratory early
fusion baseline, not a claim that depth improves accuracy.
    """

    def __init__(self, *args, depth_fusion=None, **kwargs):
        options = dict(depth_fusion or {})
        if set(options) - {'enabled'}:
            raise ValueError(f'Unknown depth_fusion options: {set(options) - {"enabled"}}')
        enabled = options.get('enabled', False)
        if not isinstance(enabled, bool):
            raise TypeError('depth_fusion.enabled must be bool')
        super().__init__(*args, **kwargs)
        self.depth_fusion_enabled = enabled
        self.depth_adapter = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        nn.init.zeros_(self.depth_adapter.weight)

    def extract_feat(self, batch_inputs):
        if batch_inputs.ndim != 4 or batch_inputs.shape[1] != 5:
            raise ValueError('RGBDMask2Former expects NCHW input with five channels')
        rgb = batch_inputs[:, :3]
        if self.depth_fusion_enabled:
            rgb = rgb + self.depth_adapter(batch_inputs[:, 3:5])
        return super().extract_feat(rgb)

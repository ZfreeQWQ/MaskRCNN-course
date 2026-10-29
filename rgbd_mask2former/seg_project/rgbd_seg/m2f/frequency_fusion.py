"""Frequency-aware final FPN fusion, inspired by FreqFusion (arXiv:2408.12879).

Project adaptation, not a reproduction of the complete published module: joint
compressed features generate Hamming-normalized 5x5 low-pass and 3x3 high-pass
kernels. No learned offset sampler or recursive compressed-feature refinement.
Explicit PyTorch reassembly avoids assuming a working vendor CARAFE kernel.
Only stride 8 -> 4 changes; a zero-initialized residual gate preserves D weights.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder


def _reassemble_chunk(x, weights, kernel, up):
    n, c, h, w = x.shape
    padded = F.pad(x, [kernel // 2] * 4, mode="replicate")
    patches = F.unfold(padded, kernel_size=kernel).reshape(n, c * kernel * kernel, h, w)
    if up != 1:
        patches = F.interpolate(patches, scale_factor=up, mode="nearest")
    patches = patches.reshape(n, c, kernel * kernel, h * up, w * up)
    return (patches * weights[:, None]).sum(2)


def reassemble(x, weights, kernel, up=1, chunk=8):
    """Channel-chunked, recomputed CARAFE-style weighted neighborhood sum."""
    if tuple(weights.shape) != (x.shape[0], kernel * kernel, x.shape[2] * up, x.shape[3] * up):
        raise ValueError("Reassembly weights do not match input geometry")
    outputs = []
    for part in x.split(chunk, dim=1):
        if torch.is_grad_enabled() and (part.requires_grad or weights.requires_grad):
            outputs.append(checkpoint(_reassemble_chunk, part, weights, kernel, up, use_reentrant=False))
        else:
            outputs.append(_reassemble_chunk(part, weights, kernel, up))
    return torch.cat(outputs, dim=1)


class FrequencyResidualFusion(nn.Module):
    def __init__(self, channels, compressed=32):
        super().__init__()
        self.hr_compress = nn.Conv2d(channels, compressed, 1)
        self.lr_compress = nn.Conv2d(channels, compressed, 1)
        self.low_encoder = nn.Conv2d(compressed, 25, 3, padding=1)
        self.high_encoder = nn.Conv2d(compressed, 9, 3, padding=1)
        self.alpha = nn.Parameter(torch.zeros(()))
        for name, kernel in (("low_window", 5), ("high_window", 3)):
            window = torch.hamming_window(kernel, periodic=False)
            self.register_buffer(name, torch.outer(window, window).reshape(1, kernel * kernel, 1, 1))
        for encoder in (self.low_encoder, self.high_encoder):
            nn.init.normal_(encoder.weight, std=0.001)
            nn.init.zeros_(encoder.bias)

    @staticmethod
    def normalize(logits, window):
        weights = logits.softmax(1) * window
        return weights / weights.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, high, low):
        if high.shape[-2] != low.shape[-2] * 2 or high.shape[-1] != low.shape[-1] * 2:
            raise ValueError("This adaptation requires the final 2x FPN stage")
        up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        original = high + up
        # Eval parity is exact, with no unnecessary reassembly at initialization.
        if not self.training and float(self.alpha.detach()) == 0:
            return original
        joint = self.hr_compress(high) + F.interpolate(self.lr_compress(low), size=high.shape[-2:],
                                                     mode="bilinear", align_corners=False)
        low_weights = self.normalize(self.low_encoder(joint), self.low_window)
        high_weights = self.normalize(self.high_encoder(joint), self.high_window)
        low_filtered = reassemble(low, low_weights, 5, up=2)
        high_detail = high - reassemble(high, high_weights, 3)
        return original + self.alpha.tanh() * (low_filtered - up + high_detail)


@SEM_SEG_HEADS_REGISTRY.register()
class FrequencyPixelDecoder(MSDeformAttnPixelDecoder):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.num_fpn_levels != 1 or self.common_stride != 4 or min(self.transformer_feature_strides) != 8:
            raise ValueError("FrequencyPixelDecoder supports the verified stride 8 -> 4 layout only")
        self.frequency_fusion = FrequencyResidualFusion(kwargs["conv_dim"])

    @torch.cuda.amp.autocast(enabled=False)
    def forward_features(self, features):
        # Same encoder and output order as the pinned third-party pixel decoder.
        srcs, pos = [], []
        for idx, feature in enumerate(self.transformer_in_features[::-1]):
            x = features[feature].float()
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        y, spatial_shapes, starts = self.transformer(srcs, pos)
        sections = [starts[i + 1] - starts[i] if i < self.transformer_num_feature_levels - 1
                    else y.shape[1] - starts[i] for i in range(self.transformer_num_feature_levels)]
        out = [z.transpose(1, 2).view(y.shape[0], -1, spatial_shapes[i][0], spatial_shapes[i][1])
               for i, z in enumerate(torch.split(y, sections, dim=1))]
        for idx, feature in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            high = self.lateral_convs[idx](features[feature].float())
            out.append(self.output_convs[idx](self.frequency_fusion(high, out[-1])))
        return self.mask_features(out[-1]), out[0], out[:self.maskformer_num_feature_levels]

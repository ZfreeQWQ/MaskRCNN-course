"""I1: geometric depth encoding + zero-initialized gated fusion backbone (docs/plan.md §4 I1).

Registered as MODEL.BACKBONE.NAME = "build_gated_rgbd_resnet_backbone". The model-level input is
a single 9-channel tensor (3 RGB + 6 depth-geometry channels from rgbd_seg.depth_geometry /
rgbd_seg.m2f.mapper), produced by MaskFormer.forward's ordinary `(x - pixel_mean) / pixel_std`
preprocessing -- PIXEL_MEAN/STD for the 6 depth channels should be 0/1 since
rgbd_seg.depth_geometry.standardize() already scales them to RGB-like magnitude, so that
preprocessing step is a no-op for them. This backbone:

  1. Splits the 9 channels back into RGB (0:3) and depth-geometry (3:9) inside forward().
  2. Runs the ordinary (checkpoint-compatible, untouched 3-channel-stem) ResNet on RGB.
  3. Runs a small from-scratch depth encoder (4 stages aligned to res2-res5 strides) on depth.
  4. Fuses at every stage: F_fused = F_rgb + sigmoid(gate) * zero_init_conv1x1(F_depth).
     The projection conv's weight AND bias are zero-initialized, so at step 0 F_fused == F_rgb
     exactly, regardless of the gate -- the model starts out byte-for-byte equivalent to the
     loaded RGB-only COCO checkpoint (docs/plan.md: "训练起点与 B 完全一致").
  5. Randomly zeroes the whole depth input for a fraction of samples during training (depth
     dropout) so the model cannot come to depend on depth always being informative.

Output is the standard {res2, res3, res4, res5} feature dict with the RGB ResNet's channel
counts, so the pixel decoder / FPN downstream need no changes at all.
"""
import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.resnet import build_resnet_backbone

_STAGE_NAMES = ("res2", "res3", "res4", "res5")


class DepthStage(nn.Module):
    """One depth-encoder stage: 3x3 conv (stride 1 or 2) + GroupNorm + ReLU."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class GatedFusion(nn.Module):
    """F_rgb + sigmoid(gate([F_rgb, F_depth])) * zero_init_conv1x1(F_depth)."""

    def __init__(self, rgb_ch, depth_ch):
        super().__init__()
        self.proj = nn.Conv2d(depth_ch, rgb_ch, kernel_size=1)
        self.gate = nn.Conv2d(rgb_ch + depth_ch, 1, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)  # sigmoid(0) = 0.5, a near-neutral starting gate

    def forward(self, f_rgb, f_depth):
        gate = torch.sigmoid(self.gate(torch.cat([f_rgb, f_depth], dim=1)))
        return f_rgb + gate * self.proj(f_depth)


class GatedRGBDResNetBackbone(Backbone):
    def __init__(self, cfg, rgb_channels=3, depth_channels=6, depth_widths=(32, 64, 128, 256),
                 depth_dropout_p=0.2):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.depth_dropout_p = depth_dropout_p

        rgb_shape = ShapeSpec(channels=rgb_channels)
        self.rgb_backbone = build_resnet_backbone(cfg, rgb_shape)
        rgb_out_shapes = self.rgb_backbone.output_shape()
        self._out_features = list(rgb_out_shapes.keys())
        self._out_feature_strides = {k: v.stride for k, v in rgb_out_shapes.items()}
        self._out_feature_channels = {k: v.channels for k, v in rgb_out_shapes.items()}

        # A single stride-4 conv would need a large kernel to avoid aliasing; use a small
        # ResNet-style stem (conv/stride2 + maxpool/stride2) to reach stride 4 instead.
        self.depth_stem = nn.Sequential(
            nn.Conv2d(depth_channels, depth_widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(32, depth_widths[0]), depth_widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # stride 4 total, matches res2

        stages = []
        prev_w = depth_widths[0]
        for i, w in enumerate(depth_widths):
            stride = 1 if i == 0 else 2  # stage 0 (res2) stays at stem's stride-4; rest halve
            stages.append(DepthStage(prev_w, w, stride))
            prev_w = w
        self.depth_stages = nn.ModuleList(stages)

        self.fusions = nn.ModuleDict({
            name: GatedFusion(self._out_feature_channels[name], w)
            for name, w in zip(_STAGE_NAMES, depth_widths) if name in self._out_features
        })

    def forward(self, x):
        rgb = x[:, :self.rgb_channels]
        depth = x[:, self.rgb_channels:self.rgb_channels + self.depth_channels]
        if self.training and self.depth_dropout_p > 0:
            keep = (torch.rand(depth.shape[0], 1, 1, 1, device=depth.device) >= self.depth_dropout_p).float()
            depth = depth * keep

        rgb_feats = self.rgb_backbone(rgb)

        depth_feats = {}
        d = self.depth_stem(depth)
        for name, stage in zip(_STAGE_NAMES, self.depth_stages):
            d = stage(d)
            depth_feats[name] = d

        return {name: self.fusions[name](rgb_feats[name], depth_feats[name])
                if name in self.fusions else rgb_feats[name]
                for name in rgb_feats}

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name],
                                stride=self._out_feature_strides[name])
                for name in self._out_features}


@BACKBONE_REGISTRY.register()
def build_gated_rgbd_resnet_backbone(cfg, input_shape):
    depth_channels = input_shape.channels - 3
    return GatedRGBDResNetBackbone(cfg, rgb_channels=3, depth_channels=depth_channels,
                                   depth_widths=tuple(cfg.PROJECT.DEPTH_WIDTHS),
                                   depth_dropout_p=cfg.PROJECT.DEPTH_DROPOUT_P)

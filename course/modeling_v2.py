"""V2 opt-in extensions; all official Mask2Former/Detectron2 sources unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from detectron2.modeling.backbone import ResNet, build_resnet_backbone
from mask2former.maskformer_model import MaskFormer


class ConvGN(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class GeometryFusion(nn.Module):
    """Independent four-scale geometry encoder plus joint channel/spatial gates.

    Zero-initialized projections make the initial residual exactly zero.
    The gate/encoder receive gradients only once a projection becomes nonzero.
    This is inspired by multi-modal gated fusion, not a reproduction of CMX.
    """
    def __init__(self, rgb_channels, dropout=.2):
        super().__init__()
        self.dropout = dropout
        self.stem = ConvGN(7, 16, 2)
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.rgb_reducers = nn.ModuleList()
        self.spatial_gates = nn.ModuleList()
        self.channel_gates = nn.ModuleList()
        cin = 16
        for depth_channels, rgb in zip((32, 64, 128, 256), rgb_channels):
            self.stages.append(nn.Sequential(ConvGN(cin, depth_channels, 2), ConvGN(depth_channels, depth_channels)))
            projection = nn.Conv2d(depth_channels, rgb, 1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.projections.append(projection)
            self.rgb_reducers.append(nn.Conv2d(rgb, 16, 1))
            self.spatial_gates.append(nn.Sequential(nn.Conv2d(16 + depth_channels, 16, 1), nn.ReLU(),
                                                     nn.Conv2d(16, 1, 1)))
            self.channel_gates.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(16 + depth_channels, 32, 1),
                                                     nn.ReLU(), nn.Conv2d(32, rgb, 1)))
            cin = depth_channels
        self.last_diagnostics = {}
        self.last_keep = None  # detached mask for opt-in branches; no RNG/output change
        self.enabled = True  # explicit residual-off diagnostic; not a learned tensor

    def forward(self, features, geometry):
        if not self.enabled:
            return features
        x = self.stem(geometry)
        batch = x.shape[0]
        keep = ((torch.rand(batch, 1, 1, 1, device=x.device) >= self.dropout).to(x)
                if self.training and self.dropout else x.new_ones(batch, 1, 1, 1))
        self.last_keep = keep.detach()
        audit = {}
        for i, name in enumerate(("res2", "res3", "res4", "res5")):
            x = self.stages[i](x)
            rgb = features[name]
            aligned = F.interpolate(x, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            joint = torch.cat([self.rgb_reducers[i](rgb), aligned], 1)
            gate = self.spatial_gates[i](joint).sigmoid() * self.channel_gates[i](joint).sigmoid()
            residual = gate * self.projections[i](aligned) * keep
            features[name] = rgb + residual
            # Retain only scalar detached tensors; no autograd graph references.
            audit[name] = dict(gate_mean=gate.detach().mean(), rgb_rms=rgb.detach().square().mean().sqrt(),
                               residual_rms=residual.detach().square().mean().sqrt())
        self.last_diagnostics = audit
        return features


class GeometryResNet(ResNet):
    def forward(self, x):
        assert x.shape[1] == 11
        features = super().forward(x[:, :4])
        return self.geometry_fusion(features, x[:, 4:])


@BACKBONE_REGISTRY.register()
def build_geometry_resnet_backbone(cfg, input_shape):
    base = build_resnet_backbone(cfg, ShapeSpec(channels=4))
    model = GeometryResNet(stem=base.stem,
                           stages=[list(getattr(base, n).children()) for n in base.stage_names],
                           out_features=cfg.MODEL.RESNETS.OUT_FEATURES,
                           freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    # Preserve RNG state for identical shared classifier initialization.
    with torch.random.fork_rng(devices=[]):
        model.geometry_fusion = GeometryFusion([model.output_shape()[n].channels for n in
                                              ("res2", "res3", "res4", "res5")], cfg.COURSE.V2.DEPTH_DROPOUT)
    return model


@META_ARCH_REGISTRY.register()
class ROIMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls, cfg):
        from course.roi_v2 import ROICriterion, ROIHungarianMatcher
        args = super().from_config(cfg)
        old = args["criterion"]
        matching = old.matcher
        matcher = ROIHungarianMatcher(cost_class=matching.cost_class, cost_mask=matching.cost_mask,
                                      cost_dice=matching.cost_dice, num_points=matching.num_points,
                                      roi_points=cfg.COURSE.V2.ROI_POINTS,
                                      roi_weight=cfg.COURSE.V2.ROI_MATCH_WEIGHT)
        args["criterion"] = ROICriterion(old.num_classes, matcher, old.weight_dict, old.eos_coef,
                                         old.losses, old.num_points, old.oversample_ratio,
                                         old.importance_sample_ratio,
                                         roi_fraction=cfg.COURSE.V2.ROI_LOSS_FRACTION,
                                         area_weight=cfg.COURSE.V2.AREA_WEIGHT)
        return args

    def prepare_targets(self, instances, images):
        targets = super().prepare_targets(instances, images)
        h, w = images.tensor.shape[-2:]
        for instance, target in zip(instances, targets):
            target["small"] = instance.gt_small
            target["original_area"] = instance.gt_original_area
            boxes = instance.gt_boxes.tensor.clone()
            size = boxes[:, 2:] - boxes[:, :2]
            # Expand each side by 10%: overall box width/height increase 20%.
            boxes[:, :2] -= size * .1
            boxes[:, 2:] += size * .1
            ih, iw = instance.image_size
            boxes[:, 0::2].clamp_(0, iw)
            boxes[:, 1::2].clamp_(0, ih)
            boxes /= boxes.new_tensor([w, h, w, h])
            target["roi_boxes"] = boxes
        return targets

"""Final-only geometry detail residual; official coarse decoder stays intact.

Training matches on original coarse masks and keeps all auxiliary losses.
Only the final mask loss samples coarse + detail logits at identical points.
Inference upsamples the two terms separately, preserving exact initial logits.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import CNNBlockBase, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY, SEM_SEG_HEADS_REGISTRY
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness, point_sample
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.criterion import SetCriterion, calculate_uncertainty, sigmoid_ce_loss_jit, dice_loss_jit
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from course.modeling_v2 import GeometryResNet, build_geometry_resnet_backbone


class DetailStem(CNNBlockBase):
    """Expose the real stride2 activation BEFORE standard stride2 max pooling."""
    def __init__(self, stem):
        super().__init__(stem.in_channels, stem.out_channels, stride=4)
        self.conv1 = stem.conv1
        self.half_feature = None

    def forward(self, x):
        x = F.relu_(self.conv1(x))
        self.half_feature = x
        return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)


class DetailGeometryResNet(GeometryResNet):
    def forward(self, x):
        features = super().forward(x)
        half = self.stem.half_feature
        self.stem.half_feature = None
        features["detail_rgb"] = half
        features["detail_geometry"] = F.interpolate(x[:, 4:], size=half.shape[-2:], mode="nearest")
        return features

    def output_shape(self):
        shape = super().output_shape()
        shape["detail_rgb"] = ShapeSpec(channels=self.stem.out_channels, stride=2)
        shape["detail_geometry"] = ShapeSpec(channels=7, stride=2)
        return shape


@BACKBONE_REGISTRY.register()
def build_detail_geometry_backbone(cfg, input_shape):
    base = build_geometry_resnet_backbone(cfg, input_shape)
    model = DetailGeometryResNet(stem=DetailStem(base.stem),
                                stages=[list(getattr(base, n).children()) for n in base.stage_names],
                                out_features=cfg.MODEL.RESNETS.OUT_FEATURES,
                                freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    model.geometry_fusion = base.geometry_fusion
    return model


@TRANSFORMER_DECODER_REGISTRY.register()
class FinalEmbeddingDecoder(MultiScaleMaskedTransformerDecoder):
    def forward(self, x, mask_features, mask=None):
        self._prediction_calls = 0
        self._final_embedding = None
        try:
            result = super().forward(x, mask_features, mask)
            assert self._final_embedding is not None
            result["final_mask_embedding"] = self._final_embedding
            return result
        finally:
            self._final_embedding = None

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        result = super().forward_prediction_heads(output, mask_features, attn_mask_target_size)
        self._prediction_calls += 1
        if self._prediction_calls == self.num_layers + 1:
            # Deterministic MLP, no dropout/RNG. All original masks, attention
            # masks and state-dict keys are produced by the upstream method.
            self._final_embedding = self.mask_embed(self.decoder_norm(output).transpose(0, 1))
        return result


class GeometryDetailRefiner(nn.Module):
    def __init__(self, mask_channels=256, rgb_channels=64, channels=32):
        super().__init__()
        assert channels % 8 == 0
        self.low = nn.Conv2d(mask_channels, channels, 1)
        self.rgb = nn.Sequential(nn.Conv2d(rgb_channels, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU())
        self.geometry = nn.Sequential(nn.Conv2d(7, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU())
        cin = channels + 32
        self.features = nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
                                      nn.Conv2d(cin, channels, 1, bias=False), nn.GroupNorm(8, channels), nn.ReLU(),
                                      nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                      nn.Conv2d(channels, channels, 1, bias=False), nn.GroupNorm(8, channels), nn.ReLU())
        self.query = nn.Linear(mask_channels, channels, bias=False)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.scale = math.sqrt(channels)

    def forward(self, low, rgb, geometry, embedding):
        low = F.interpolate(self.low(low), size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        features = self.features(torch.cat([low, self.rgb(rgb), self.geometry(geometry)], 1))
        return torch.einsum("bqc,bchw->bqhw", self.query(embedding), self.output(features)) / self.scale


@SEM_SEG_HEADS_REGISTRY.register()
class GeometryRefinementHead(MaskFormerHead):
    @configurable
    def __init__(self, *, refine_channels, detail_rgb_channels, **kwargs):
        super().__init__(**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.refiner = GeometryDetailRefiner(self.predictor.mask_embed.layers[-1].out_features,
                                               detail_rgb_channels, refine_channels)

    @classmethod
    def from_config(cls, cfg, input_shape):
        args = super().from_config(cfg, input_shape)
        args.update(refine_channels=cfg.COURSE.V3.REFINE_CHANNELS,
                    detail_rgb_channels=input_shape["detail_rgb"].channels)
        return args

    def layers(self, features, mask=None):
        assert self.transformer_in_feature == "multi_scale_pixel_decoder"
        low, _, multi = self.pixel_decoder.forward_features(features)
        result = self.predictor(multi, low, mask)
        embedding = result.pop("final_mask_embedding")
        detail = self.refiner(low, features["detail_rgb"], features["detail_geometry"], embedding)
        if self.training:
            result["detail_residual"] = detail
        else:
            # Do NOT combine at stride2 and then upsample: successive bilinear
            # resizes are not identical to the original stride4->input resize.
            full_size = tuple(2*s for s in detail.shape[-2:])
            coarse = F.interpolate(result["pred_masks"], size=full_size, mode="bilinear", align_corners=False)
            result["pred_masks"] = coarse + F.interpolate(detail, size=full_size, mode="bilinear", align_corners=False)
        return result


class DetailCriterion(SetCriterion):
    def loss_masks(self, outputs, targets, indices, num_masks):
        if "detail_residual" not in outputs:
            return super().loss_masks(outputs, targets, indices, num_masks)
        src_idx = self._get_src_permutation_idx(indices)
        coarse = outputs["pred_masks"][src_idx][:, None]
        detail = outputs["detail_residual"][src_idx][:, None]
        if not len(coarse):
            zero = (outputs["pred_masks"].sum() + outputs["detail_residual"].sum()) * 0
            return {"loss_mask": zero, "loss_dice": zero}
        labels = torch.cat([target["masks"][j] for target, (_, j) in zip(targets, indices)]).to(coarse)[:, None]
        with torch.no_grad():
            # Keep the baseline point budget/distribution; no GT-box or
            # foreground-only replacement and no extra boundary loss weight.
            coords = get_uncertain_point_coords_with_randomness(coarse, calculate_uncertainty,
                        self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            truth = point_sample(labels, coords, align_corners=False).squeeze(1)
        logits = (point_sample(coarse, coords, align_corners=False) +
                  point_sample(detail, coords, align_corners=False)).squeeze(1)
        return {"loss_mask": sigmoid_ce_loss_jit(logits, truth, num_masks),
                "loss_dice": dice_loss_jit(logits, truth, num_masks)}


@META_ARCH_REGISTRY.register()
class DetailMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls, cfg):
        args = super().from_config(cfg)
        old = args["criterion"]
        args["criterion"] = DetailCriterion(old.num_classes, old.matcher, old.weight_dict,
                                            old.eos_coef, old.losses, old.num_points,
                                            old.oversample_ratio, old.importance_sample_ratio)
        return args

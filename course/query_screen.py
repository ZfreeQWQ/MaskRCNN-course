"""P2 / explicit-geometry image-conditioned query residuals, opt-in only.

No GT locations, top-k pseudo-labels, extra matching or extra supervision.
Soft spatial pooling is differentiable. Zero output projection preserves the
trained model exactly at initialization; original learned queries stay intact.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, SEM_SEG_HEADS_REGISTRY
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from course.modeling_v2 import GeometryResNet, build_geometry_resnet_backbone


class SpatialQueryInitializer(nn.Module):
    def __init__(self, mask_dim, hidden_dim, channels=32, sigma=.15, gain=.1, geometry=False):
        super().__init__()
        if channels % 8 or not 0 < sigma <= 1 or not 0 < gain <= 1:
            raise ValueError("Invalid query pooling dimensions, sigma or gain")
        self.image = nn.Sequential(nn.Conv2d(mask_dim, channels, 1, bias=False),
                                   nn.GroupNorm(8, channels), nn.GELU())
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.query = nn.Linear(hidden_dim, channels, bias=False)
        self.output = nn.Linear(channels, hidden_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.geometry = (nn.Sequential(nn.Conv2d(7, channels, 1, bias=False),
                                       nn.GroupNorm(8, channels), nn.GELU()) if geometry else None)
        self.sigma, self.gain = sigma, gain
        self.last_diagnostics = {}

    def spatial_prior(self, queries, h, w, device, dtype):
        # Stratified anchors encourage diversity but do not replace learned
        # query positions. Pooling weights remain image/geometry dependent.
        side = math.ceil(math.sqrt(queries))
        positions = torch.arange(queries, device=device)
        centers = torch.stack(((positions % side + .5) / side,
                               (positions // side + .5) / side), -1).to(dtype)
        yy, xx = torch.meshgrid((torch.arange(h, device=device, dtype=dtype) + .5) / h,
                                (torch.arange(w, device=device, dtype=dtype) + .5) / w, indexing="ij")
        pixels = torch.stack((xx.flatten(), yy.flatten()), -1)
        return -(centers[:, None] - pixels[None]).square().sum(-1) / (2 * self.sigma ** 2)

    def forward(self, mask_features, learned_queries, geometry=None):
        context = self.image(mask_features)
        if self.geometry is not None:
            if geometry is None or geometry.shape[0] != context.shape[0] or geometry.shape[1] != 7:
                raise ValueError("Explicit query geometry requires aligned seven-channel input")
            geometry = F.interpolate(geometry, size=context.shape[-2:], mode="nearest")
            context = context + self.geometry(geometry)
        batch, channels, h, w = context.shape
        keys = self.key(context).flatten(2)
        values = self.value(context).flatten(2)
        queries = self.query(learned_queries)
        logits = torch.einsum("qc,bcn->bqn", queries, keys) / math.sqrt(channels)
        logits = logits + self.spatial_prior(len(queries), h, w, context.device, context.dtype)[None]
        attention = logits.softmax(-1)
        pooled = torch.einsum("bqn,bcn->bqc", attention, values)
        pooled = F.layer_norm(pooled, (channels,))
        residual = self.gain * self.output(pooled).tanh()
        self.last_diagnostics = dict(residual_rms=residual.detach().square().mean().sqrt(),
            attention_peak=attention.detach().amax(-1).mean())
        return residual.transpose(0, 1)  # Q,B,C: same shape as original learned queries


class QueryGeometryResNet(GeometryResNet):
    def forward(self, x):
        features = super().forward(x)
        keep = (self.geometry_fusion.last_keep if self.geometry_fusion.enabled else
                x.new_zeros(x.shape[0], 1, 1, 1))
        # Reuse the existing whole-depth-branch dropout draw; no additional RNG
        # consumption and no bypass of depth dropout through the new branch.
        features["query_geometry"] = x[:, 4:] * keep
        return features

    def output_shape(self):
        shape = super().output_shape()
        shape["query_geometry"] = ShapeSpec(channels=7, stride=1)
        return shape


@BACKBONE_REGISTRY.register()
def build_query_geometry_backbone(cfg, input_shape):
    base = build_geometry_resnet_backbone(cfg, input_shape)
    model = QueryGeometryResNet(stem=base.stem,
        stages=[list(getattr(base, n).children()) for n in base.stage_names],
        out_features=cfg.MODEL.RESNETS.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    model.geometry_fusion = base.geometry_fusion
    return model


@SEM_SEG_HEADS_REGISTRY.register()
class SeededMaskFormerHead(MaskFormerHead):
    def layers(self, features, mask=None):
        assert self.transformer_in_feature == "multi_scale_pixel_decoder"
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        return self.predictor(multi_scale_features, mask_features, mask,
                              query_geometry=features.get("query_geometry"))


@TRANSFORMER_DECODER_REGISTRY.register()
class SeededQueryDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(self, in_channels, mask_classification=True, *, query_mode="off", query_channels=32,
                 spatial_sigma=.15, residual_gain=.1, **kwargs):
        super().__init__(in_channels, mask_classification, **kwargs)
        if query_mode not in ("off", "p2", "p2_geometry"):
            raise ValueError(query_mode)
        self.query_mode = query_mode
        self.query_initializer = None
        if query_mode != "off":
            with torch.random.fork_rng(devices=[]):
                self.query_initializer = SpatialQueryInitializer(kwargs["mask_dim"], kwargs["hidden_dim"],
                    query_channels, spatial_sigma, residual_gain, query_mode == "p2_geometry")

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        values = super().from_config(cfg, in_channels, mask_classification)
        values.update(query_mode=cfg.COURSE.QUERY.MODE, query_channels=cfg.COURSE.QUERY.CHANNELS,
                      spatial_sigma=cfg.COURSE.QUERY.SPATIAL_SIGMA,
                      residual_gain=cfg.COURSE.QUERY.RESIDUAL_GAIN)
        return values

    def forward(self, x, mask_features, mask=None, query_geometry=None):
        # Pinned upstream forward, with exactly one optional addition to output.
        # Original attention, predictions, learned positions and auxiliary heads
        # are unchanged. CPU/GPU zero-residual parity tests guard this adapter.
        assert len(x) == self.num_feature_levels
        src, pos, size_list = [], [], []
        del mask
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        _, bs, _ = src[0].shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        if self.query_initializer is not None:
            output = output + self.query_initializer(mask_features, self.query_feat.weight, query_geometry)
        predictions_class, predictions_mask = [], []
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, size_list[0])
        predictions_class.append(outputs_class); predictions_mask.append(outputs_mask)
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            output = self.transformer_cross_attention_layers[i](output, src[level_index], memory_mask=attn_mask,
                memory_key_padding_mask=None, pos=pos[level_index], query_pos=query_embed)
            output = self.transformer_self_attention_layers[i](output, tgt_mask=None, tgt_key_padding_mask=None,
                query_pos=query_embed)
            output = self.transformer_ffn_layers[i](output)
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
                                                                                  size_list[(i+1)%self.num_feature_levels])
            predictions_class.append(outputs_class); predictions_mask.append(outputs_mask)
        assert len(predictions_class) == self.num_layers + 1
        return dict(pred_logits=predictions_class[-1], pred_masks=predictions_mask[-1],
                    aux_outputs=self._set_aux_loss(predictions_class, predictions_mask))

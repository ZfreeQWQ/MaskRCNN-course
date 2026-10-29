"""MP-Former training adaptation for the existing RGB-D checkpoint.

Based on the all-layer, one-group, zero-mask-noise recipe in the official
IDEA-Research/MP-Former implementation (CVPR 2023), commit
a71f504a467d7079cf02726a05ff23bbb933607d. Preserve this project's learned
ordinary-query positional embeddings; the official DN decoder omits them.
Single-class label replacement is an identity, so label noise is explicitly 0.
The normal eval decoder is inherited unchanged, with no GT/pilot computation.
"""
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils import comm
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher

from .model import RGBDMaskFormer


def pilot_self_attention_mask(pad_size, regular_queries, device):
    mask = torch.zeros((pad_size + regular_queries, pad_size + regular_queries),
                       dtype=torch.bool, device=device)
    # Ordinary queries cannot read GT-piloted queries at any decoder layer.
    # Like the official one-group recipe, pilots may read ordinary queries.
    mask[pad_size:, :pad_size] = True
    return mask


@TRANSFORMER_DECODER_REGISTRY.register()
class MaskPilotedDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_enc = nn.Embedding(self.class_embed.out_features - 1,
                                      self.query_feat.embedding_dim)
        self.last_pilot_stats = None

    def forward(self, x, mask_features, mask=None, targets=None):
        # Deliberately ignore even accidentally supplied targets during eval.
        if not self.training or targets is None:
            self.last_pilot_stats = {"enabled": False, "pilot_queries": 0}
            result = super().forward(x, mask_features, mask)
            if self.training:
                result["pred_logits"] = result["pred_logits"] + self.label_enc.weight.sum() * 0
            return result
        counts = [len(t["labels"]) for t in targets]
        pad = max(counts, default=0)
        if pad == 0:
            result = super().forward(x, mask_features, mask)
            result["pred_logits"] = result["pred_logits"] + self.label_enc.weight.sum() * 0
            self.last_pilot_stats = {"enabled": True, "pilot_queries": 0, "targets": 0}
            return result
        if len(x) != self.num_feature_levels or len(targets) != mask_features.shape[0]:
            raise ValueError("Invalid multi-scale features or target batch")
        bs, device = mask_features.shape[0], mask_features.device
        sizes, src, pos = [], [], []
        for level in range(self.num_feature_levels):
            sizes.append(x[level].shape[-2:])
            pos.append(self.pe_layer(x[level], None).flatten(2).permute(2, 0, 1))
            memory = self.input_proj[level](x[level]).flatten(2)
            memory = memory + self.level_embed.weight[level][None, :, None]
            src.append(memory.permute(2, 0, 1))
        pilot_features = mask_features.new_zeros((bs, pad, self.query_feat.embedding_dim))
        for batch, target in enumerate(targets):
            if counts[batch]:
                pilot_features[batch, :counts[batch]] = self.label_enc(target["labels"])
        ordinary = self.query_feat.weight[:, None].repeat(1, bs, 1)
        output = torch.cat((pilot_features.transpose(0, 1), ordinary), dim=0)
        query_pos = torch.cat((torch.zeros_like(pilot_features.transpose(0, 1)),
                               self.query_embed.weight[:, None].repeat(1, bs, 1)), dim=0)
        self_mask = pilot_self_attention_mask(pad, self.num_queries, device)
        gt_attention = []
        vanished = []
        for size in sizes:
            blocked = torch.ones((bs, pad, size[0] * size[1]), dtype=torch.bool, device=device)
            missing = 0
            for batch, target in enumerate(targets):
                if counts[batch]:
                    # Official 'points' mode uses area pooling and >1e-8 support.
                    support = F.interpolate(target["masks"][:, None].float(), size=size,
                                            mode="area")[:, 0].flatten(1) > 1e-8
                    missing += int((~support.any(1)).sum())
                    blocked[batch, :counts[batch]] = ~support
            gt_attention.append(blocked[:, None].expand(-1, self.num_heads, -1, -1)
                                .reshape(bs * self.num_heads, pad, -1))
            vanished.append(missing)
        classes, masks = [], []
        cls, seg, attention = self.forward_prediction_heads(output, mask_features, sizes[0])
        classes.append(cls)
        masks.append(seg)
        for layer in range(self.num_layers):
            level = layer % self.num_feature_levels
            attention[:, :pad] = gt_attention[level]
            # Match upstream all-masked-row fallback, including unused pad slots.
            attention[attention.all(-1)] = False
            output = self.transformer_cross_attention_layers[layer](
                output, src[level], memory_mask=attention, memory_key_padding_mask=None,
                pos=pos[level], query_pos=query_pos)
            output = self.transformer_self_attention_layers[layer](
                output, tgt_mask=self_mask, tgt_key_padding_mask=None, query_pos=query_pos)
            output = self.transformer_ffn_layers[layer](output)
            cls, seg, attention = self.forward_prediction_heads(
                output, mask_features, sizes[(layer + 1) % self.num_feature_levels])
            classes.append(cls)
            masks.append(seg)
        regular_classes, regular_masks = [c[:, pad:] for c in classes], [m[:, pad:] for m in masks]
        pilot_classes, pilot_masks = [c[:, :pad] for c in classes], [m[:, :pad] for m in masks]
        self.last_pilot_stats = {"enabled": True, "pilot_queries": pad, "targets": sum(counts),
                                 "targets_per_image": counts, "vanished_targets_by_scale": vanished,
                                 "all_decoder_layers": True, "groups": 1, "mask_noise": 0.0,
                                 "label_noise": 0.0}
        return {"pred_logits": regular_classes[-1], "pred_masks": regular_masks[-1],
                "aux_outputs": self._set_aux_loss(regular_classes, regular_masks),
                "pilot_outputs": {"pred_logits": pilot_classes[-1], "pred_masks": pilot_masks[-1],
                                  "aux_outputs": self._set_aux_loss(pilot_classes, pilot_masks)}}


class EmptySafeMatcher(HungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets):
        indices = []
        for batch, target in enumerate(targets):
            if not len(target["labels"]):
                empty = torch.empty(0, dtype=torch.long)
                indices.append((empty, empty.clone()))
            else:
                single = {key: outputs[key][batch:batch + 1] for key in ("pred_logits", "pred_masks")}
                indices.append(super().forward(single, [target])[0])
        return indices


class MPCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        safe_matcher = EmptySafeMatcher(cost_class=matcher.cost_class, cost_mask=matcher.cost_mask,
                                        cost_dice=matcher.cost_dice, num_points=matcher.num_points)
        super().__init__(num_classes, safe_matcher, weight_dict, eos_coef, losses,
                         num_points, oversample_ratio, importance_sample_ratio)

    def loss_masks(self, outputs, targets, indices, num_masks):
        if not any(len(source) for source, _ in indices):
            # CoreX's scripted pairwise mask loss cannot process zero-size axes.
            zero = outputs["pred_masks"].sum() * 0
            return {"loss_mask": zero, "loss_dice": zero.clone()}
        return super().loss_masks(outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        regular = {k: outputs[k] for k in ("pred_logits", "pred_masks", "aux_outputs")}
        losses = super().forward(regular, targets)
        # Always execute this collective, even if one rank has no valid targets.
        count = outputs["pred_logits"].new_tensor([sum(len(t["labels"]) for t in targets)])
        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(count)
        normalizer = max(float(count / comm.get_world_size()), 1.0)
        pilots = outputs.get("pilot_outputs")
        indices = [(torch.arange(len(t["labels"])), torch.arange(len(t["labels"]))) for t in targets]
        layers = [pilots] + pilots["aux_outputs"] if pilots is not None else [None] * (1 + len(regular["aux_outputs"]))
        zero = outputs["pred_logits"].sum() * 0
        for layer, pilot in enumerate(layers):
            suffix = "_mp" if layer == 0 else f"_mp_{layer - 1}"
            if pilot is None:
                extra = {"loss_ce": zero, "loss_mask": zero.clone(), "loss_dice": zero.clone()}
            else:
                extra = {}
                for loss in self.losses:
                    extra.update(self.get_loss(loss, pilot, targets, indices, normalizer))
            losses.update({key + suffix: value for key, value in extra.items()})
        return losses


@META_ARCH_REGISTRY.register()
class RGBDMPFormer(RGBDMaskFormer):
    @classmethod
    def from_config(cls, cfg):
        if cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED or cfg.PROJECT.HIRES_MASK.ENABLED:
            raise ValueError("This algorithm run uses D+MP only, with S and H disabled")
        ret = super().from_config(cfg)
        old = ret["criterion"]
        weights = dict(old.weight_dict)
        for name in ("loss_ce", "loss_mask", "loss_dice"):
            weights[name + "_mp"] = weights[name]
            for layer in range(cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1):
                weights[f"{name}_mp_{layer}"] = weights[name]
        ret["criterion"] = MPCriterion(old.num_classes, old.matcher, weights, old.eos_coef,
                                        old.losses, old.num_points, old.oversample_ratio,
                                        old.importance_sample_ratio)
        return ret

    def forward(self, batched_inputs):
        if not self.training:
            return super().forward(batched_inputs)
        images = [record["image"].to(self.device) for record in batched_inputs]
        images = ImageList.from_tensors([(x - self.pixel_mean) / self.pixel_std for x in images],
                                       self.size_divisibility)
        targets = self.prepare_targets([r["instances"].to(self.device) for r in batched_inputs], images)
        features = self.backbone(images.tensor)
        if self.sem_seg_head.transformer_in_feature != "multi_scale_pixel_decoder":
            raise ValueError("MP adapter requires the existing multi-scale pixel decoder")
        mask_features, _, multiscale = self.sem_seg_head.pixel_decoder.forward_features(features)
        outputs = self.sem_seg_head.predictor(multiscale, mask_features, targets=targets)
        losses = self.criterion(outputs, targets)
        return {name: value * self.criterion.weight_dict[name]
                for name, value in losses.items() if name in self.criterion.weight_dict}

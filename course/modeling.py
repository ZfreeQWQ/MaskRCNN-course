"""Opt-in registered subclasses: upstream Mask2Former sources stay unchanged."""
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import BACKBONE_REGISTRY, META_ARCH_REGISTRY
from detectron2.modeling.backbone import ResNet, build_resnet_backbone
from detectron2.projects.point_rend.point_features import get_uncertain_point_coords_with_randomness, point_sample
from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.criterion import SetCriterion, calculate_uncertainty
from mask2former.utils.misc import nested_tensor_from_tensor_list


class SmallInstanceCriterion(SetCriterion):
    def loss_masks(self, outputs, targets, indices, num_masks):
        if self.small_weight == 1 and self.foreground_fraction == 0:
            return super().loss_masks(outputs, targets, indices, num_masks)
        si, ti = self._get_src_permutation_idx(indices), self._get_tgt_permutation_idx(indices)
        src = outputs["pred_masks"][si][:, None]
        if len(src) == 0:
            zero = outputs["pred_masks"].sum() * 0
            return {"loss_mask": zero, "loss_dice": zero}
        padded, _ = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target = padded.to(src)[ti][:, None]
        small = torch.cat([t["small"][j] for t, (_, j) in zip(targets, indices)])
        with torch.no_grad():
            coords = get_uncertain_point_coords_with_randomness(
                src, calculate_uncertainty, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            if self.foreground_fraction:
                fg = torch.cat([t["foreground_coords"][j] for t, (_, j) in zip(targets, indices)])
                k = fg.shape[1]
                coords[small, -k:] = fg[small]
            labels = point_sample(target, coords, align_corners=False).squeeze(1)
        logits = point_sample(src, coords, align_corners=False).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
        probabilities = logits.sigmoid()
        dice = 1 - (2 * (probabilities * labels).sum(1) + 1) / (probabilities.sum(1) + labels.sum(1) + 1)
        weights = torch.where(small, self.small_weight, 1.0).to(bce)
        # Preserve the original per-target normalization when all weights are 1.
        # Experiments are single-GPU; no hidden distributed normalization change.
        all_small = torch.cat([t["small"] for t in targets])
        mean_weight = 1 + (self.small_weight - 1) * all_small.float().mean() if len(all_small) else 1
        denominator = num_masks * mean_weight
        if self.small_weight != 1:
            self.last_weight_diagnostics = dict(reweighted_targets=all_small.detach().sum(),
                matched_targets=weights.new_tensor(len(weights)), matched_small=small.detach().sum(),
                min_matched_weight=weights.detach().min(), max_matched_weight=weights.detach().max(),
                mean_all_target_weight=torch.as_tensor(mean_weight).detach())
        return {"loss_mask": (bce * weights).sum() / denominator,
                "loss_dice": (dice * weights).sum() / denominator}


@META_ARCH_REGISTRY.register()
class CourseMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls, cfg):
        args = super().from_config(cfg)
        old = args["criterion"]
        criterion = SmallInstanceCriterion(
            old.num_classes, matcher=old.matcher, weight_dict=old.weight_dict,
            eos_coef=old.eos_coef, losses=old.losses, num_points=old.num_points,
            oversample_ratio=old.oversample_ratio, importance_sample_ratio=old.importance_sample_ratio)
        criterion.small_weight = cfg.COURSE.SMALL_LOSS_WEIGHT
        criterion.foreground_fraction = cfg.COURSE.FOREGROUND_POINT_FRACTION
        args["criterion"] = criterion
        return args

    def prepare_targets(self, instances, images):
        targets = super().prepare_targets(instances, images)
        for instance, target in zip(instances, targets):
            assert instance.has("gt_small"), "Experiment mapper must retain original-area flags"
            target["small"] = instance.gt_small
            fraction = self.criterion.foreground_fraction
            if fraction:
                k = max(1, int(self.criterion.num_points * fraction))
                h, w = target["masks"].shape[-2:]
                coords = torch.zeros((len(instance), k, 2), device=self.device)
                for i in torch.where(instance.gt_small)[0]:
                    locations = target["masks"][i].nonzero()
                    if not len(locations):
                        continue
                    points = locations[torch.randint(len(locations), (k,), device=self.device)]
                    coords[i, :, 0] = (points[:, 1].float() + .5) / w
                    coords[i, :, 1] = (points[:, 0].float() + .5) / h
                target["foreground_coords"] = coords
        return targets


class DepthP2Guide(nn.Module):
    """Bounded residual gate from supplied depth and its finite-difference edges."""
    def __init__(self, channels, depth_mean, depth_std):
        super().__init__()
        self.depth_mean, self.depth_std = depth_mean, depth_std
        self.gate = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GroupNorm(4, 16),
                                  nn.ReLU(), nn.Conv2d(16, channels, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, feature, normalized_depth):
        depth = (normalized_depth * self.depth_std + self.depth_mean).clamp(0, 255) / 255
        dx = F.pad((depth[..., 1:] - depth[..., :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((depth[..., 1:, :] - depth[..., :-1, :]).abs(), (0, 0, 0, 1))
        geometry = F.interpolate(torch.cat([depth, dx + dy], 1), size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return feature * (1 + .5 * self.gate(geometry).tanh())


class DepthGuidedResNet(ResNet):
    def forward(self, x):
        features = super().forward(x)
        features["res2"] = self.depth_guide(features["res2"], x[:, 3:4])
        return features


@BACKBONE_REGISTRY.register()
def build_depth_guided_resnet_backbone(cfg, input_shape):
    base = build_resnet_backbone(cfg, input_shape)
    model = DepthGuidedResNet(
        stem=base.stem, stages=[list(getattr(base, name).children()) for name in base.stage_names],
        out_features=cfg.MODEL.RESNETS.OUT_FEATURES, freeze_at=cfg.MODEL.BACKBONE.FREEZE_AT)
    # Adding a module must not change the random initialization of the shared
    # single-class classifier created after the backbone.
    with torch.random.fork_rng(devices=[]):
        model.depth_guide = DepthP2Guide(model.output_shape()["res2"].channels,
                                       cfg.MODEL.PIXEL_MEAN[3], cfg.MODEL.PIXEL_STD[3])
    return model

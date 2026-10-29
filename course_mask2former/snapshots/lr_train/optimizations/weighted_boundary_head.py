# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: optional small-instance weights and final-layer boundary loss.
# SPDX-License-Identifier: Apache-2.0
"""Two independently switchable training-only losses for MMDetection 3.3.0.

The base loss implementation below follows OpenMMLab's Apache-2.0-licensed
Mask2FormerHead._loss_by_feat_single in mmdetection v3.3.0. Assignment,
classification loss, random point sampling and the original normalization are
unchanged. Both switches default to False; that path delegates to upstream.

``small_object_loss`` weights matched-instance BCE and Dice by
``min(max_weight, max(1, sqrt(area_threshold / visible_area)))``. The area is
the *augmented/cropped GT mask's* pixel count, NOT the original COCO annotation
area used for APs, nor the original-image area used by an optional crop sampler.
Weights are intentionally not renormalized: they increase small-mask influence
relative to classification. A zero-area target receives weight 1.

``boundary_loss`` is an auxiliary boundary-region mask BCE, not a distance or
edge-prediction loss. Only the last decoder output uses it. GT masks are area-
downsampled to the mask-logit grid; a morphological gradient identifies the
boundary band on that grid. Instances whose downsampled target peak is below
``min_target_peak`` (default 0.5) contribute zero auxiliary loss. This guard
avoids pushing unresolved subpixel objects toward background when all of their
soft target values are below 0.5. It is configurable, but does not change the
upstream point-sampled mask/Dice supervision. Loss is averaged over band pixels
per instance and uses upstream's instance divisor (including skipped targets).
It shares the last layer's assignment and adds no duplicate Hungarian solve.
There are no new learned parameters and no inference-path modifications.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from mmcv.ops import point_sample

from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS
from mmdet.utils import reduce_mean


def _checked_options(small_object_loss=None, boundary_loss=None):
    small = dict(enabled=False, area_threshold=1024.0, max_weight=2.0)
    boundary = dict(enabled=False, loss_weight=0.1, radius=1, min_target_peak=0.5)
    for supplied, defaults, name in (
            (small_object_loss, small, 'small_object_loss'),
            (boundary_loss, boundary, 'boundary_loss')):
        if supplied is not None:
            if not isinstance(supplied, dict):
                raise TypeError(f'{name} must be a dict or None')
            unknown = set(supplied) - set(defaults)
            if unknown:
                raise ValueError(f'Unknown {name} options: {sorted(unknown)}')
            defaults.update(supplied)
        if not isinstance(defaults['enabled'], bool):
            raise TypeError(f'{name}.enabled must be bool')
    for name, value, minimum, strict in (
            ('area_threshold', small['area_threshold'], 0, True),
            ('max_weight', small['max_weight'], 1, False),
            ('loss_weight', boundary['loss_weight'], 0, False)):
        if (not isinstance(value, (int, float)) or not math.isfinite(value)
                or (value <= minimum if strict else value < minimum)):
            raise ValueError(f'Invalid {name}: {value}')
    if (not isinstance(boundary['radius'], int)
            or isinstance(boundary['radius'], bool) or boundary['radius'] < 1):
        raise ValueError('boundary_loss.radius must be a positive integer')
    peak = boundary['min_target_peak']
    if (not isinstance(peak, (int, float)) or isinstance(peak, bool)
            or not math.isfinite(peak) or not 0 < peak <= 1):
        raise ValueError('boundary_loss.min_target_peak must be finite and in (0, 1]')
    return small, boundary


@MODELS.register_module()
class WeightedBoundaryMask2FormerHead(Mask2FormerHead):
    """Upstream-compatible head with independent small-mask/boundary switches."""

    def __init__(self, *args, small_object_loss=None, boundary_loss=None, **kwargs):
        small, boundary = _checked_options(small_object_loss, boundary_loss)
        super().__init__(*args, **kwargs)
        self.small_object_loss = small
        self.boundary_loss = boundary

    def _instance_weights(self, mask_targets):
        with torch.no_grad():
            # Targets are already transformed and padded by MMDetection.
            areas = mask_targets.float().flatten(1).sum(1)
            weights = (self.small_object_loss['area_threshold']
                       / areas.clamp_min(1)).sqrt().clamp(
                           min=1, max=self.small_object_loss['max_weight'])
            return torch.where(areas > 0, weights, torch.ones_like(weights))

    def _mask_loss_point_coords(self, mask_preds, mask_targets):
        """Extension point; default preserves upstream sampling and RNG order."""
        return get_uncertain_point_coords_with_randomness(
            mask_preds.unsqueeze(1), None, self.num_points,
            self.oversample_ratio, self.importance_sample_ratio)

    def _boundary_region_loss(self, mask_preds, mask_targets, num_total_masks):
        if mask_preds.shape[0] == 0:
            return mask_preds.sum() * 0
        with torch.no_grad():
            targets = F.interpolate(
                mask_targets.unsqueeze(1).float(), size=mask_preds.shape[-2:],
                mode='area')
            radius = self.boundary_loss['radius']
            kernel = 2 * radius + 1
            dilated = F.max_pool2d(targets, kernel, stride=1, padding=radius)
            eroded = -F.max_pool2d(-targets, kernel, stride=1, padding=radius)
            band = ((dilated - eroded) > 1e-6).squeeze(1).float()
            targets = targets.squeeze(1)
            # If area downsampling spreads a subpixel object into labels all
            # below 0.5, BCE would reduce its logits even at probability 0.5.
            # Suppress only this new auxiliary term for unresolved instances;
            # multiplying the band by zero preserves a connected zero gradient.
            resolvable = targets.flatten(1).amax(1) >= self.boundary_loss['min_target_peak']
            band = band * resolvable[:, None, None]
        # Float32 BCEWithLogits stays stable with AMP and extreme mask logits.
        per_pixel = F.binary_cross_entropy_with_logits(
            mask_preds.float(), targets, reduction='none')
        per_instance = (per_pixel * band).flatten(1).sum(1) / band.flatten(1).sum(1).clamp_min(1)
        return (per_instance.sum() / num_total_masks
                * self.boundary_loss['loss_weight'])

    def _loss_with_additions(self, cls_scores, mask_preds, batch_gt_instances,
                             batch_img_metas, include_boundary=False):
        # Keep the order of upstream random sampling and loss operations intact.
        num_imgs = cls_scores.size(0)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(
             [cls_scores[i] for i in range(num_imgs)],
             [mask_preds[i] for i in range(num_imgs)], batch_gt_instances,
             batch_img_metas)
        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights,
            avg_factor=class_weight[labels].sum())
        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            zero = mask_preds.sum()
            boundary = (self._boundary_region_loss(mask_preds, mask_targets, num_total_masks)
                        if include_boundary else None)
            return loss_cls, zero, zero, boundary

        with torch.no_grad():
            points_coords = self._mask_loss_point_coords(mask_preds, mask_targets)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        instance_weights = (self._instance_weights(mask_targets)
                            if self.small_object_loss['enabled'] else None)
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, weight=instance_weights,
            avg_factor=num_total_masks)
        point_weights = (instance_weights[:, None].expand(-1, self.num_points).reshape(-1)
                         if instance_weights is not None else None)
        loss_mask = self.loss_mask(
            mask_point_preds.reshape(-1), mask_point_targets.reshape(-1),
            weight=point_weights, avg_factor=num_total_masks * self.num_points)
        boundary = (self._boundary_region_loss(mask_preds, mask_targets, num_total_masks)
                    if include_boundary else None)
        return loss_cls, loss_mask, loss_dice, boundary

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances,
                             batch_img_metas):
        if not self.small_object_loss['enabled']:
            return super()._loss_by_feat_single(
                cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        return self._loss_with_additions(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)[:3]

    def _has_boundary_loss(self):
        return self.boundary_loss['enabled']

    def _boundary_loss_key(self):
        return 'loss_boundary'

    def loss_by_feat(self, all_cls_scores, all_mask_preds, batch_gt_instances,
                     batch_img_metas):
        if not self._has_boundary_loss():
            # With both flags off, every calculation is the upstream method.
            return super().loss_by_feat(
                all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas)
        if not len(all_cls_scores) or len(all_cls_scores) != len(all_mask_preds):
            raise ValueError('Expected equally sized, nonempty decoder output sequences')
        layer_losses = []
        for index, (cls_scores, mask_preds) in enumerate(zip(all_cls_scores, all_mask_preds)):
            if index == len(all_cls_scores) - 1:
                final = self._loss_with_additions(
                    cls_scores, mask_preds, batch_gt_instances, batch_img_metas,
                    include_boundary=True)
                layer_losses.append(final[:3])
                boundary = final[3]
            else:
                layer_losses.append(self._loss_by_feat_single(
                    cls_scores, mask_preds, batch_gt_instances, batch_img_metas))
        names = ('loss_cls', 'loss_mask', 'loss_dice')
        losses = dict(zip(names, layer_losses[-1]))
        for index, layer in enumerate(layer_losses[:-1]):
            losses.update({f'd{index}.{name}': value for name, value in zip(names, layer)})
        losses[self._boundary_loss_key()] = boundary
        return losses

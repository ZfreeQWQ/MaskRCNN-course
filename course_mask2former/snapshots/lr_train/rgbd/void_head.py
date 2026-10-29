# Copyright (c) OpenMMLab. All rights reserved.
# Modifications: explicit valid-pixel supervision for partially labelled data.
# SPDX-License-Identifier: Apache-2.0
"""Mask2Former loss policy for NYUv2's unlabelled pixels.

This is a supervision policy, not an AP-improving claim. Both RGB and RGB-D
experiments must use the same policy. With ``ignore_void=True`` (the default),
``DetDataSample.label_valid.data`` must contain a boolean [1,H,W] map after all
geometric transforms and padding. Direct loss-by-feature callers can instead
put a [H,W] boolean map in ``batch_img_metas[i]['label_valid']``.

For partially valid batches, matching and uncertain-point candidates are drawn
uniformly WITH replacement from valid *target-pixel centres*. GT values are
gathered at integer indices, so interpolation cannot mix in unknown GT labels.
Prediction logits retain bilinear point sampling, preserving sub-grid targets.
Thus unknown GT pixels have no mask-loss supervision; this does NOT promise zero
gradient for every coarse logit/feature that overlaps an unknown region.
Boundary loss further excludes coarse cells whose area or boundary neighbourhood
contains void. Small-object area weights count only valid GT pixels.

All-valid batches take the existing upstream-compatible path bit for bit.
All-void images have connected zero classification and mask losses, rather than
being taught as empty background. Other images retain standard query-level
classification/no-object supervision: an unmatched query on an unlabelled object
can still receive no-object classification. This is not an open-world/partial-
annotation detection algorithm. ``ignore_void=False`` explicitly restores the
old loss policy; never silently use it to treat missing maps as background.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import MODELS
from mmdet.utils import reduce_mean

from optimizations.weighted_boundary_head import WeightedBoundaryMask2FormerHead


def valid_pixel_points(valid, rows, count):
    """Return flattened GT indices and normalized (x,y) pixel-centre coords.

    Sampling is with replacement, so even a single valid pixel is supported.
    Empty maps are rejected here and handled explicitly by the head.
    """
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError('label_valid must be a two-dimensional bool tensor')
    candidates = valid.flatten().nonzero(as_tuple=False).flatten()
    if candidates.numel() == 0:
        raise ValueError('Cannot sample a label_valid map with no valid pixels')
    if rows < 0 or count < 0:
        raise ValueError('rows and count must be nonnegative')
    choice = torch.randint(candidates.numel(), (rows, count), device=valid.device)
    indices = candidates[choice]
    height, width = valid.shape
    coords = torch.stack(((indices % width).float() + 0.5,
                          torch.div(indices, width, rounding_mode='floor').float() + 0.5), -1)
    coords = coords / coords.new_tensor([width, height])
    return indices, coords


@MODELS.register_module()
class VoidAwareMask2FormerHead(WeightedBoundaryMask2FormerHead):
    """Existing independent loss switches plus explicit void-pixel handling."""

    def __init__(self, *args, ignore_void=True, **kwargs):
        if not isinstance(ignore_void, bool):
            raise TypeError('ignore_void must be bool')
        super().__init__(*args, **kwargs)
        self.ignore_void = ignore_void

    def _valid_map(self, gt_instances, img_meta, device):
        value = img_meta.get('label_valid', gt_instances.get('label_valid', None))
        if value is None:
            raise ValueError('ignore_void=True requires an explicit label_valid map')
        if isinstance(value, PixelData):
            value = value.data
        valid = torch.as_tensor(value, device=device)
        if valid.ndim == 3 and valid.shape[0] == 1:
            valid = valid[0]
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError('label_valid must be bool [H,W] or [1,H,W]')
        expected = gt_instances.masks.shape[-2:]
        if tuple(valid.shape) != tuple(expected):
            raise ValueError(f'label_valid {tuple(valid.shape)} != padded GT {tuple(expected)}')
        return valid

    def _all_valid(self, batch_gt_instances, batch_img_metas, device):
        return all(bool(self._valid_map(gt, meta, device).all())
                   for gt, meta in zip(batch_gt_instances, batch_img_metas))

    def loss(self, x, batch_data_samples):
        if not self.ignore_void:
            return super().loss(x, batch_data_samples)
        metas, instances, semantics = [], [], []
        for sample in batch_data_samples:
            meta = dict(sample.metainfo)
            if 'label_valid' in sample:
                meta['label_valid'] = sample.label_valid.data
            metas.append(meta)
            instances.append(sample.gt_instances)
            semantics.append(sample.gt_sem_seg if 'gt_sem_seg' in sample else None)
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        instances = self.preprocess_gt(instances, semantics)
        return self.loss_by_feat(all_cls_scores, all_mask_preds, instances, metas)

    def _use_original_matching(self, valid):
        return bool(valid.all())

    def _matching_points(self, valid, masks):
        return valid_pixel_points(valid, 1, self.num_points)

    def _get_targets_single(self, cls_score, mask_pred, gt_instances, img_meta):
        if not self.ignore_void:
            return super()._get_targets_single(cls_score, mask_pred, gt_instances, img_meta)
        valid = self._valid_map(gt_instances, img_meta, mask_pred.device)
        if self._use_original_matching(valid):
            return super()._get_targets_single(cls_score, mask_pred, gt_instances, img_meta)
        has_valid = bool(valid.any())
        labels = gt_instances.labels if has_valid else gt_instances.labels[:0]
        masks = gt_instances.masks * valid if has_valid else gt_instances.masks[:0]
        clean_gt = InstanceData(labels=labels, masks=masks)
        num_queries, num_gts = cls_score.shape[0], labels.shape[0]
        if has_valid:
            indices, coords = self._matching_points(valid, masks)
            predicted = point_sample(
                mask_pred.unsqueeze(1), coords.expand(num_queries, -1, -1)).squeeze(1)
            targets = masks.flatten(1).gather(1, indices.expand(num_gts, -1)).float()
        else:
            # The assigner handles zero GT. These point arrays are never used
            # for a mask loss and consume no random sampling state.
            predicted = mask_pred.new_zeros((num_queries, self.num_points))
            targets = mask_pred.new_zeros((0, self.num_points))
        assignment = self.assigner.assign(
            pred_instances=InstanceData(scores=cls_score, masks=predicted),
            gt_instances=InstanceData(labels=labels, masks=targets), img_meta=img_meta)
        sampled = self.sampler.sample(
            assign_result=assignment,
            pred_instances=InstanceData(scores=cls_score, masks=mask_pred),
            gt_instances=clean_gt)
        if not has_valid:
            # MaskPseudoSampler normally clamps each image's num_pos to one.
            # A completely unlabelled image must not dilute other images'
            # mask loss by adding this artificial instance to the divisor.
            sampled.avg_factor = 0
        pos, neg = sampled.pos_inds, sampled.neg_inds
        query_labels = labels.new_full((self.num_queries,), self.num_classes)
        query_labels[pos] = labels[sampled.pos_assigned_gt_inds]
        query_weights = labels.new_full((self.num_queries,), int(has_valid))
        target_masks = masks[sampled.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos] = 1
        return (query_labels, query_weights, target_masks, mask_weights, pos, neg, sampled)

    def _uncertain_valid_points(self, predictions, valid):
        num_masks = predictions.shape[0]
        sampled = int(self.num_points * self.oversample_ratio)
        indices, coords = valid_pixel_points(valid, num_masks, sampled)
        uncertainty = -point_sample(predictions.unsqueeze(1), coords).squeeze(1).abs()
        important = int(self.num_points * self.importance_sample_ratio)
        chosen = uncertainty.topk(important, dim=1).indices
        selected_indices = indices.gather(1, chosen)
        selected_coords = coords.gather(1, chosen.unsqueeze(-1).expand(-1, -1, 2))
        if important < self.num_points:
            random_indices, random_coords = valid_pixel_points(
                valid, num_masks, self.num_points - important)
            selected_indices = torch.cat([selected_indices, random_indices], 1)
            selected_coords = torch.cat([selected_coords, random_coords], 1)
        return selected_indices, selected_coords

    def _void_boundary_loss(self, mask_preds, mask_targets, valid_maps, num_total_masks):
        if mask_preds.shape[0] == 0:
            return mask_preds.sum() * 0
        with torch.no_grad():
            valid = torch.stack(valid_maps).float().unsqueeze(1)
            valid_grid = F.interpolate(valid, mask_preds.shape[-2:], mode='area') >= 1.0
            targets = F.interpolate(
                mask_targets.unsqueeze(1).float(), mask_preds.shape[-2:], mode='area')
            radius = self.boundary_loss['radius']
            kernel = 2 * radius + 1
            dilated = F.max_pool2d(targets, kernel, stride=1, padding=radius)
            eroded = -F.max_pool2d(-targets, kernel, stride=1, padding=radius)
            # Exclude artificial GT boundaries adjacent to unknown cells.
            safe_band = F.max_pool2d(
                (~valid_grid).float(), kernel, stride=1, padding=radius) == 0
            band = ((dilated - eroded) > 1e-6) & safe_band
            targets = targets.squeeze(1)
            band = band.squeeze(1).float()
            peak = (targets * valid_grid.squeeze(1)).flatten(1).amax(1)
            band *= (peak >= self.boundary_loss['min_target_peak'])[:, None, None]
        per_pixel = F.binary_cross_entropy_with_logits(
            mask_preds.float(), targets, reduction='none')
        per_instance = (per_pixel * band).flatten(1).sum(1) / band.flatten(1).sum(1).clamp_min(1)
        return per_instance.sum() / num_total_masks * self.boundary_loss['loss_weight']

    def _sample_loss_points(self, predictions, valid, targets):
        """Extension point; the v1 default preserves its exact sampling/RNG."""
        return self._uncertain_valid_points(predictions, valid)

    def _loss_with_additions(self, cls_scores, mask_preds, batch_gt_instances,
                             batch_img_metas, include_boundary=False):
        if not self.ignore_void or self._all_valid(
                batch_gt_instances, batch_img_metas, mask_preds.device):
            return super()._loss_with_additions(
                cls_scores, mask_preds, batch_gt_instances, batch_img_metas, include_boundary)
        count = cls_scores.shape[0]
        (labels_list, weights_list, targets_list, mask_weights_list, average) = self.get_targets(
            list(cls_scores.unbind(0)), list(mask_preds.unbind(0)),
            batch_gt_instances, batch_img_metas)
        labels = torch.stack(labels_list).flatten()
        weights = torch.stack(weights_list).flatten()
        flat_scores = cls_scores.flatten(0, 1)
        class_weights = cls_scores.new_tensor(self.class_weight)
        cls_divisor = (class_weights[labels] * weights).sum().clamp_min(1e-8)
        loss_cls = self.loss_cls(flat_scores, labels, weights, avg_factor=cls_divisor)
        total_masks = max(reduce_mean(cls_scores.new_tensor([average])), 1)
        target_masks = torch.cat(targets_list, 0)
        positive_weights = torch.stack(mask_weights_list) > 0
        predictions = mask_preds[positive_weights]
        if target_masks.shape[0] == 0:
            zero = predictions.sum()
            return loss_cls, zero, zero, zero if include_boundary else None

        point_predictions, point_targets, matched_valid = [], [], []
        offset = 0
        for image_index in range(count):
            num_masks = targets_list[image_index].shape[0]
            if num_masks == 0:
                continue
            group_predictions = predictions[offset:offset + num_masks]
            group_targets = target_masks[offset:offset + num_masks]
            valid = self._valid_map(batch_gt_instances[image_index],
                                    batch_img_metas[image_index], mask_preds.device)
            with torch.no_grad():
                indices, coords = self._sample_loss_points(group_predictions, valid, group_targets)
                targets = group_targets.flatten(1).gather(1, indices).float()
            point_predictions.append(point_sample(group_predictions.unsqueeze(1), coords).squeeze(1))
            point_targets.append(targets)
            matched_valid.extend([valid] * num_masks)
            offset += num_masks
        predicted_points, target_points = torch.cat(point_predictions), torch.cat(point_targets)
        instance_weights = self._instance_weights(target_masks) if self.small_object_loss['enabled'] else None
        dice = self.loss_dice(predicted_points, target_points, weight=instance_weights, avg_factor=total_masks)
        point_weights = (instance_weights[:, None].expand(-1, self.num_points).reshape(-1)
                         if instance_weights is not None else None)
        mask = self.loss_mask(predicted_points.reshape(-1), target_points.reshape(-1),
                              weight=point_weights, avg_factor=total_masks * self.num_points)
        boundary = (self._void_boundary_loss(predictions, target_masks, matched_valid, total_masks)
                    if include_boundary else None)
        return loss_cls, mask, dice, boundary

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances, batch_img_metas):
        if not self.ignore_void or self._all_valid(
                batch_gt_instances, batch_img_metas, mask_preds.device):
            return super()._loss_by_feat_single(cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        return self._loss_with_additions(
            cls_scores, mask_preds, batch_gt_instances, batch_img_metas)[:3]

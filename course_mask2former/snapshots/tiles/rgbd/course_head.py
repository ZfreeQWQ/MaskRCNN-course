"""Opt-in foreground point quota for fully annotated course masks.

Retain the standard COCO matching, classification, evaluation and inference.
For each matched target of 1..area_threshold augmented pixels, replace at most
quota points at the random tail with foreground pixel centres (with replacement).
The uncertainty-selected prefix and total point budget are unchanged. This is
an intentionally foreground-biased mask BCE/Dice objective, not an unbiased
estimator, and it does not fix missed targets at the assignment stage.
"""
import math

import torch
from mmdet.registry import MODELS
from rgbd.foreground_head import ForegroundQuotaMask2FormerHead


@MODELS.register_module()
class CoursePointQuotaMask2FormerHead(ForegroundQuotaMask2FormerHead):
    def __init__(self, *args, course_point_sampling=None, **kwargs):
        options = dict(enabled=False, quota=64, area_threshold=1024.)
        supplied = dict(course_point_sampling or {})
        if set(supplied) - set(options):
            raise ValueError('Unknown course point sampling option')
        options.update(supplied)
        if type(options['enabled']) is not bool or type(options['quota']) is not int or options['quota'] < 1:
            raise ValueError('Course point switch must be bool; quota must be positive integer')
        if not math.isfinite(options['area_threshold']) or options['area_threshold'] <= 0:
            raise ValueError('Course point area threshold must be finite and positive')
        super().__init__(*args, **kwargs)
        if self.ignore_void or self.foreground_sampling['enabled']:
            raise ValueError('Course quota requires full supervision and no NYUv2 foreground policy')
        if options['enabled'] and (self.num_points < 4 or self.importance_sample_ratio >= 1):
            raise ValueError('Course quota needs a nonempty random point tail')
        self.course_point_sampling = options
        self.course_point_counters = dict(calls=0, small_targets=0, replaced_points=0)

    def _mask_loss_point_coords(self, mask_preds, mask_targets):
        coords = super()._mask_loss_point_coords(mask_preds, mask_targets)
        if not self.course_point_sampling['enabled'] or not len(mask_targets):
            return coords
        quota = min(self.course_point_sampling['quota'], self.num_points // 4,
                    self.num_points - int(self.num_points * self.importance_sample_ratio))
        positive = mask_targets > .5
        areas = positive.flatten(1).sum(1)
        rows = ((areas > 0) & (areas <= self.course_point_sampling['area_threshold'])).nonzero().flatten()
        height, width = mask_targets.shape[-2:]
        for row in rows.unbind():
            pixels = positive[row].flatten().nonzero().flatten()
            selected = pixels[torch.randint(pixels.numel(), (quota,), device=pixels.device)]
            centres = torch.stack(((selected % width).float() + .5,
                                   (selected // width).float() + .5), -1)
            coords[row, -quota:] = centres / centres.new_tensor([width, height])
        self.course_point_counters['calls'] += 1
        self.course_point_counters['small_targets'] += rows.numel()
        self.course_point_counters['replaced_points'] += rows.numel() * quota
        return coords

    def _loss_by_feat_single(self, cls_scores, mask_preds, batch_gt_instances, batch_img_metas):
        if not self.course_point_sampling['enabled']:
            return super()._loss_by_feat_single(cls_scores, mask_preds, batch_gt_instances, batch_img_metas)
        return self._loss_with_additions(cls_scores, mask_preds, batch_gt_instances, batch_img_metas)[:3]

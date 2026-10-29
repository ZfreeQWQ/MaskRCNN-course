"""Final-layer boundary BCE restricted to augmented visible small instances.

No new parameters, inference operations, assignment or RNG consumption. Area
selection happens AFTER augmentation (not COCO's original-area APs grouping).
Unresolved subpixel instances retain normal mask/Dice losses but no auxiliary
boundary loss. The divisor is selected, resolvable instances, not all matches.
"""
import math

import torch
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.utils import reduce_mean
from rgbd.course_head import CoursePointQuotaMask2FormerHead


def checked_small_boundary(options=None):
    result = dict(enabled=False, area_threshold=1024., loss_weight=.1,
                  radius=1, min_target_peak=.5)
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError('small_boundary_loss must be a dict')
        if set(options) - set(result):
            raise ValueError('Unknown small boundary option')
        result.update(options)
    if type(result['enabled']) is not bool:
        raise ValueError('Small boundary enabled must be bool')
    for key in ('area_threshold', 'loss_weight', 'min_target_peak'):
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid small boundary ' + key)
    if result['min_target_peak'] > 1:
        raise ValueError('Small boundary peak must be <= 1')
    if type(result['radius']) is not int or result['radius'] < 1:
        raise ValueError('Small boundary radius must be a positive integer')
    return result


def empty_counters():
    return dict(calls=0, area_eligible=0, resolvable=0, skipped_subpixel=0,
                with_boundary=0, nonzero_calls=0)


@MODELS.register_module()
class CourseSmallBoundaryMask2FormerHead(CoursePointQuotaMask2FormerHead):
    def __init__(self, *args, small_boundary_loss=None, **kwargs):
        options = checked_small_boundary(small_boundary_loss)
        super().__init__(*args, **kwargs)
        if options['enabled'] and self.boundary_loss['enabled']:
            raise ValueError('Generic and small-only boundary are alternative objectives')
        self.small_boundary_loss = options
        self.small_boundary_counters = empty_counters()

    def _has_boundary_loss(self):
        return self.small_boundary_loss['enabled'] or super()._has_boundary_loss()

    def _boundary_loss_key(self):
        return ('loss_small_boundary' if self.small_boundary_loss['enabled']
                else super()._boundary_loss_key())

    def loss_by_feat(self, *args, **kwargs):
        if self.small_boundary_loss['enabled']:
            self.small_boundary_counters['calls'] += 1
        return super().loss_by_feat(*args, **kwargs)

    def _boundary_region_loss(self, mask_preds, mask_targets, num_total_masks):
        if not self.small_boundary_loss['enabled']:
            return super()._boundary_region_loss(mask_preds, mask_targets, num_total_masks)
        options = self.small_boundary_loss
        with torch.no_grad():
            areas = mask_targets.float().flatten(1).sum(1)
            selected = (areas > 0) & (areas <= options['area_threshold'])
            count = int(selected.sum().item())
            self.small_boundary_counters['area_eligible'] += count
            targets = mask_targets[selected].float()
            if count:
                targets = F.interpolate(targets[:, None], size=mask_preds.shape[-2:], mode='area')
                resolvable = targets.flatten(1).amax(1) >= options['min_target_peak']
                radius = options['radius']
                kernel = 2 * radius + 1
                band = (F.max_pool2d(targets, kernel, 1, radius)
                        + F.max_pool2d(-targets, kernel, 1, radius)) > 1e-6
                band = band[:, 0].float() * resolvable[:, None, None]
                targets = targets[:, 0]
                resolved = int(resolvable.sum().item())
                self.small_boundary_counters['resolvable'] += resolved
                self.small_boundary_counters['skipped_subpixel'] += count - resolved
                self.small_boundary_counters['with_boundary'] += int((band.flatten(1).sum(1) > 0).sum().item())
            else:
                resolved = 0
            # Same collective on every rank, including empty local selections.
            divisor = reduce_mean(mask_preds.new_tensor(float(resolved))).clamp_min(1)
        if not count:
            return mask_preds.sum() * 0
        per_pixel = F.binary_cross_entropy_with_logits(mask_preds[selected].float(), targets, reduction='none')
        per_instance = (per_pixel * band).flatten(1).sum(1) / band.flatten(1).sum(1).clamp_min(1)
        loss = per_instance.sum() / divisor * options['loss_weight']
        self.small_boundary_counters['nonzero_calls'] += int(loss.detach().item() > 0)
        return loss

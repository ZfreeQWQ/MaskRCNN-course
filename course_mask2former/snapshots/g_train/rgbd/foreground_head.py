"""An opt-in matched-small-instance foreground quota within a fixed point budget.

Only mask BCE/Dice supervision changes; Hungarian matching and inference do not.
At most min(quota, floor(P/4)) of the last (random) points are replaced by valid
foreground pixel centres for each matched target with area <= threshold.
Sampling is with replacement, so one-pixel instances are supported. This is an
intentional foreground-biased training objective, NOT an unbiased estimator of
the original pixel loss. Unknown labels are never sampled as foreground.
"""
import math
import torch
from mmdet.registry import MODELS
from rgbd.void_head import VoidAwareMask2FormerHead


@MODELS.register_module()
class ForegroundQuotaMask2FormerHead(VoidAwareMask2FormerHead):
    def __init__(self, *args, foreground_sampling=None, **kwargs):
        options = dict(enabled=False, quota=64, area_threshold=1024.0)
        supplied = dict(foreground_sampling or {})
        if set(supplied) - set(options):
            raise ValueError('Unknown foreground sampling option')
        options.update(supplied)
        if type(options['enabled']) is not bool or type(options['quota']) is not int or options['quota'] < 1:
            raise ValueError('Foreground enabled must be bool and quota must be a positive integer')
        if not math.isfinite(options['area_threshold']) or options['area_threshold'] <= 0:
            raise ValueError('Foreground area threshold must be finite and positive')
        super().__init__(*args, **kwargs)
        if options['enabled'] and not self.ignore_void:
            raise ValueError('Foreground quota requires the common void policy')
        self.foreground_sampling = options

    def _all_valid(self, batch_gt_instances, batch_img_metas, device):
        # Enabled mode must use the extension point even for fully labeled maps.
        if self.foreground_sampling['enabled']:
            return False
        return super()._all_valid(batch_gt_instances, batch_img_metas, device)

    def _sample_loss_points(self, predictions, valid, targets):
        indices, coords = super()._sample_loss_points(predictions, valid, targets)
        if not self.foreground_sampling['enabled'] or targets.shape[0] == 0:
            return indices, coords
        positive = (targets > 0.5) & valid.unsqueeze(0)
        area = positive.flatten(1).sum(1)
        rows = ((area > 0) & (area <= self.foreground_sampling['area_threshold'])).nonzero().flatten()
        quota = min(self.foreground_sampling['quota'], self.num_points // 4,
                    self.num_points - int(self.num_points * self.importance_sample_ratio))
        if rows.numel() == 0 or quota == 0:
            return indices, coords
        weights = positive.flatten(1)[rows].float()
        selected = torch.multinomial(weights, quota, replacement=True)
        height, width = valid.shape
        centres = torch.stack(((selected % width).float() + 0.5,
                               torch.div(selected, width, rounding_mode='floor').float() + 0.5), -1)
        centres /= centres.new_tensor([width, height])
        indices[rows, -quota:] = selected
        coords[rows, -quota:] = centres
        return indices, coords

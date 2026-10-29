"""Small-instance-aware shared point sampling for Hungarian assignment.

Same fixed point budget and shared cost grid for every query/target pair. This
intentionally changes the matching objective, not just final BCE/Dice weights.
No additional inference work. Default off delegates to the exact v2 method.
"""
import math
import torch
from mmdet.registry import MODELS
from rgbd.foreground_head import ForegroundQuotaMask2FormerHead


def balanced_small_points(valid, masks, indices, coords, quota, area_threshold):
    positive = (masks > .5) & valid.unsqueeze(0)
    areas = positive.flatten(1).sum(1)
    rows = ((areas > 0) & (areas <= area_threshold)).nonzero().flatten()
    if quota == 0 or rows.numel() == 0:
        return indices, coords, 0
    # Random order avoids favoring annotation order if quota < instance count.
    order = rows[torch.randperm(rows.numel(), device=rows.device)]
    selected = []
    for i, row in enumerate(order):
        count = quota // rows.numel() + int(i < quota % rows.numel())
        if count:
            pixels = positive[row].flatten().nonzero().flatten()
            selected.append(pixels[torch.randint(pixels.numel(), (count,), device=pixels.device)])
    chosen = torch.cat(selected)
    indices[:, -quota:] = chosen
    height, width = valid.shape
    centres = torch.stack(((chosen % width).float() + .5, (chosen // width).float() + .5), -1)
    coords[:, -quota:] = centres / centres.new_tensor([width, height])
    return indices, coords, quota


@MODELS.register_module()
class SmallAwareMatchingHead(ForegroundQuotaMask2FormerHead):
    def __init__(self, *args, small_matching=None, **kwargs):
        options = dict(enabled=False, fraction=.25, area_threshold=1024.)
        supplied = dict(small_matching or {})
        if set(supplied) - set(options):
            raise ValueError('Unknown matching option')
        options.update(supplied)
        if type(options['enabled']) is not bool or not 0 < options['fraction'] <= .5:
            raise ValueError('Matching enabled must be bool and fraction in (0, .5]')
        if not math.isfinite(options['area_threshold']) or options['area_threshold'] <= 0:
            raise ValueError('Invalid matching area threshold')
        super().__init__(*args, **kwargs)
        if options['enabled'] and not self.ignore_void:
            raise ValueError('Small matching requires common void policy')
        self.small_matching = options
        self.matching_counters = dict(calls=0, augmented_calls=0, foreground_points=0)

    def _use_original_matching(self, valid):
        if self.small_matching['enabled']:
            return False
        return super()._use_original_matching(valid)

    def _matching_points(self, valid, masks):
        indices, coords = super()._matching_points(valid, masks)
        if not self.small_matching['enabled']:
            return indices, coords
        self.matching_counters['calls'] += 1
        quota = int(self.num_points * self.small_matching['fraction'])
        indices, coords, count = balanced_small_points(valid, masks, indices, coords, quota,
                                                       self.small_matching['area_threshold'])
        if count:
            self.matching_counters['augmented_calls'] += 1
            self.matching_counters['foreground_points'] += count
        return indices, coords

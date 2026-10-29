"""Optional training-only, small-instance-aware crop for MMDetection 3.3.0.

This module changes the sampling distribution of crop windows, not COCO's area
annotations or evaluator. It is an experimental augmentation, not a demonstrated
accuracy improvement. Import it via ``custom_imports`` before building a config.
"""

import math
from typing import Optional, Tuple

import numpy as np
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import RandomCrop
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SmallObjectAwareCrop(RandomCrop):
    """Occasionally retain one eligible small instance in a random crop.

    Only the crop offset is changed. MMDetection's ``RandomCrop`` performs all
    image, box, mask, label, ignore-flag, instance-ID and semantic-map updates,
    including its original negative-crop behavior and homography bookkeeping.

    An eligible instance is non-ignored, has a positive area no greater than
    ``small_area_thr``, and can be completely enclosed by an integer-aligned
    crop window. We first sample an eligible instance uniformly, then sample a
    crop offset uniformly among windows that contain its entire tight mask box
    (or its annotation box when masks are absent). This avoids cutting the
    selected object; other instances can still be cut as with ``RandomCrop``.

    Args:
        prob: Probability of guided sampling when eligible instances exist.
            At zero, no additional random numbers are drawn: the result and RNG
            state exactly match the upstream crop with the same seed.
        small_area_thr: Positive sampling threshold in square pixels, inclusive.
            This is an augmentation hyperparameter, NOT a COCO APs definition.
        area_mode: ``original`` (default) divides current mask area by
            ``scale_factor[0] * scale_factor[1]``. This approximately recovers
            pre-resize area when preceded only by loading, flipping and resizing.
            Rasterization can introduce rounding error. Missing scale_factor
            means no resize (factor one). Previous crops, mosaics and geometric
            warps invalidate this approximation and should not precede this
            transform. ``current`` explicitly uses area after resizing.
        **kwargs: The usual RandomCrop options, including crop_size,
            crop_type, allow_negative_crop and recompute_bbox.

    Notes:
        Masks supply both area and tight bounds when available; otherwise bbox
        area is a documented approximation. With no eligible instance, the
        original crop is used without extra random draws. Objects whose extent
        exceeds the crop are not guided even if their mask area is small.
        Use only in the TRAIN pipeline; keep validation/test preprocessing and
        CocoMetric unchanged. Like upstream RandomCrop, this does not transform
        a separately stored depth array: RGB-D integration needs paired geometry.
    """

    def __init__(self, crop_size: tuple, prob: float = 0.5,
                 small_area_thr: float = 32.0 ** 2,
                 area_mode: str = 'original', **kwargs) -> None:
        super().__init__(crop_size=crop_size, **kwargs)
        if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
            raise ValueError('prob must be finite and in [0, 1]')
        if not math.isfinite(small_area_thr) or small_area_thr <= 0:
            raise ValueError('small_area_thr must be finite and positive')
        if area_mode not in ('original', 'current'):
            raise ValueError("area_mode must be 'original' or 'current'")
        self.prob = float(prob)
        self.small_area_thr = float(small_area_thr)
        self.area_mode = area_mode
        self._guided_offset = None

    def _candidate_ranges(self, results: dict,
                          crop_size: Tuple[int, int]) -> np.ndarray:
        """Return [min_y, max_y, min_x, max_x] for each eligible instance."""
        boxes = results.get('gt_bboxes')
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=np.int64)
        masks = results.get('gt_masks')
        if masks is not None:
            if len(masks) != len(boxes):
                raise ValueError('gt_masks and gt_bboxes must be instance-aligned')
            areas = np.asarray(masks.areas, dtype=np.float64)
            bounds = masks.get_bboxes('hbox').numpy()
        else:
            bounds = boxes.numpy()
            areas = np.maximum(bounds[:, 2] - bounds[:, 0], 0) * np.maximum(
                bounds[:, 3] - bounds[:, 1], 0)

        if self.area_mode == 'original':
            scale = np.asarray(results.get('scale_factor', (1.0, 1.0)),
                               dtype=np.float64).reshape(-1)
            if scale.size == 1:
                scale = np.repeat(scale, 2)
            if scale.size not in (2, 4) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError('scale_factor must contain 1, 2 or 4 positive finite values')
            areas = areas / (scale[0] * scale[1])

        crop_h, crop_w = crop_size
        img_h, img_w = results['img'].shape[:2]
        margin_h, margin_w = max(img_h - crop_h, 0), max(img_w - crop_w, 0)
        eligible = (np.isfinite(areas) & (areas > 0) &
                    (areas <= self.small_area_thr) &
                    np.isfinite(bounds).all(axis=1))
        ignored = results.get('gt_ignore_flags')
        if ignored is not None:
            eligible &= ~np.asarray(ignored, dtype=bool)

        ranges = []
        for x1, y1, x2, y2 in bounds[eligible]:
            if x2 <= x1 or y2 <= y1:
                continue
            # For exclusive right/bottom bounds, containment requires
            # x2-crop_w <= offset_x <= x1 (and likewise for y).
            min_x = max(0, math.ceil(float(x2) - crop_w))
            max_x = min(margin_w, math.floor(float(x1)))
            min_y = max(0, math.ceil(float(y2) - crop_h))
            max_y = min(margin_h, math.floor(float(y1)))
            if min_x <= max_x and min_y <= max_y:
                ranges.append((min_y, max_y, min_x, max_x))
        return np.asarray(ranges, dtype=np.int64).reshape(-1, 4)

    @cache_randomness
    def _sample_guided_offset(self, ranges: np.ndarray) -> Optional[Tuple[int, int]]:
        # No Bernoulli draw at the endpoints: prob=0 is upstream-identical and
        # prob=1 always retains a candidate. All new randomness lives here so
        # MMCV's cache_random_params can reuse it consistently.
        if not len(ranges) or self.prob == 0:
            return None
        if self.prob < 1 and np.random.random() >= self.prob:
            return None
        min_y, max_y, min_x, max_x = ranges[np.random.randint(len(ranges))]
        return (int(np.random.randint(min_y, max_y + 1)),
                int(np.random.randint(min_x, max_x + 1)))

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool):
        if self.prob == 0:
            return super()._crop_data(results, crop_size, allow_negative_crop)
        ranges = self._candidate_ranges(results, crop_size)
        self._guided_offset = self._sample_guided_offset(ranges)
        try:
            return super()._crop_data(results, crop_size, allow_negative_crop)
        finally:
            # A worker reuses the transform on many images. Never leak an
            # offset into a later sample, including after an exception/None.
            self._guided_offset = None

    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        if self._guided_offset is not None:
            return self._guided_offset
        return super()._rand_offset(margin)

    def __repr__(self) -> str:
        return (super().__repr__()[:-1] +
                f', prob={self.prob}, small_area_thr={self.small_area_thr}, '
                f'area_mode={self.area_mode!r})')

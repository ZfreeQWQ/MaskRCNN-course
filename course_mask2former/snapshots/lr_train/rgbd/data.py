"""Aligned RGB-D transforms for local COCO-format RGB-D datasets.

RGB remains three-channel BGR until packing. Only explicit paired transforms
may change geometry; photometric transforms can therefore safely act on RGB.
Depth is measured in metres, raw_valid indicates *measured* depth (not whether
the official inpainted depth is usable), and label_valid identifies pixels
whose labels can supervise the loss. Their three meanings are never conflated.
"""

from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.transforms import LoadImageFromFile
from mmengine.structures import PixelData
from mmdet.datasets import CocoDataset
from mmdet.datasets.transforms import PackDetInputs, RandomCrop, RandomFlip, Resize
from mmdet.registry import DATASETS, TRANSFORMS

from optimizations.small_object_crop import SmallObjectAwareCrop


_PAIRED_KEYS = ('rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid')


def _check_aligned(results):
    shape = results['img'].shape[:2]
    if results['img'].ndim != 3 or results['img'].shape[2] != 3:
        raise ValueError('RGB-D transforms require a three-channel RGB/BGR image')
    for key in _PAIRED_KEYS:
        if key not in results or results[key].shape != shape:
            raise ValueError(f'{key} must be a 2-D array aligned to RGB {shape}')


@DATASETS.register_module()
class RGBDCocoDataset(CocoDataset):
    """Keep RGB-D image fields that the upstream COCO parser discards.

``depth_file``, ``raw_valid_file``, and ``label_valid_file`` in each COCO image
entry must be safe relative file paths under data_root. They do not use the
RGB-specific data_prefix. ``scene`` is retained as diagnostic metadata.
    """

    def parse_data_info(self, raw_data_info):
        parsed = super().parse_data_info(raw_data_info)
        image = raw_data_info['raw_img_info']
        root = Path(self.data_root).resolve()
        for key in ('depth_file', 'raw_valid_file', 'label_valid_file'):
            relative = Path(image[key])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'{key} must be relative to data_root: {relative}')
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f'{key} must remain under data_root: {relative}')
            parsed[key] = str(resolved)
        if 'scene' in image:
            parsed['scene'] = image['scene']
        return parsed


def _load_plane(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        plane = np.load(path, allow_pickle=False)
    else:
        plane = mmcv.imread(str(path), flag='unchanged')
    if plane is None or plane.ndim != 2:
        raise ValueError(f'Expected a readable 2-D array: {path}')
    return plane


@TRANSFORMS.register_module()
class LoadRGBDImageFromFile(LoadImageFromFile):
    """Load RGB, metre-depth, raw-measurement and label-validity planes.

Nonpositive/nonfinite depth is zeroed. Raw-valid flags are intersected with
numerical depth validity. Official finite inpainted depth remains available
even where raw_valid is false; the model receives that missingness indicator.
Depth range clipping is a separate, explicit preprocessor hyperparameter.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('to_float32', True)
        super().__init__(**kwargs)

    def transform(self, results):
        results = super().transform(results)
        if results is None:
            return None
        depth = _load_plane(results['depth_file']).astype(np.float32)
        raw_valid = _load_plane(results['raw_valid_file'])
        label_valid = _load_plane(results['label_valid_file'])
        valid = np.isfinite(depth) & (depth > 0)
        results['rgbd_depth'] = np.where(valid, depth, 0).astype(np.float32)
        results['rgbd_raw_valid'] = (np.isfinite(raw_valid) & (raw_valid > 0) & valid)
        results['rgbd_label_valid'] = np.isfinite(label_valid) & (label_valid > 0)
        _check_aligned(results)
        return results


@TRANSFORMS.register_module()
class RGBDResize(Resize):
    """Upstream RGB/annotation resize plus explicitly paired RGB-D planes.

Depth uses validity-weighted bilinear interpolation so numerical invalids do
not bleed zero metres into nearby valid depth. The nearest source depth's
numerical-valid flag gates the result, keeping missing support zero. Sensor
and label validity independently use nearest-neighbour interpolation.
    """

    def _resize_img(self, results):
        _check_aligned(results)
        super()._resize_img(results)
        height, width = results['img'].shape[:2]
        depth = results['rgbd_depth']
        valid = np.isfinite(depth) & (depth > 0)
        target = (width, height)
        support = mmcv.imresize(valid.astype(np.float32), target, interpolation='bilinear')
        numerator = mmcv.imresize(np.where(valid, depth, 0).astype(np.float32),
                                  target, interpolation='bilinear')
        nearest = mmcv.imresize(valid.astype(np.uint8), target, interpolation='nearest').astype(bool)
        results['rgbd_depth'] = np.divide(numerator, support,
                                           out=np.zeros_like(numerator),
                                           where=(support > 1e-6) & nearest)
        for key in ('rgbd_raw_valid', 'rgbd_label_valid'):
            results[key] = mmcv.imresize(results[key].astype(np.uint8), target,
                                         interpolation='nearest').astype(bool)
        results['rgbd_raw_valid'] &= nearest


@TRANSFORMS.register_module()
class RGBDRandomFlip(RandomFlip):
    """Sample once with upstream RandomFlip and apply that flip to every plane."""

    def _flip(self, results):
        _check_aligned(results)
        super()._flip(results)
        for key in _PAIRED_KEYS:
            results[key] = mmcv.imflip(results[key], direction=results['flip_direction']).copy()


class _PairedCropMixin:
    """Stack only during slicing: exact shared crop, no validity interpolation."""

    def _crop_data(self, results, crop_size, allow_negative_crop):
        _check_aligned(results)
        rgb_dtype = results['img'].dtype
        results['img'] = np.concatenate(
            [results['img'], *[results[key][..., None] for key in _PAIRED_KEYS]], axis=2)
        try:
            output = super()._crop_data(results, crop_size, allow_negative_crop)
        finally:
            stacked = results['img']
            results['img'] = stacked[..., :3].astype(rgb_dtype, copy=False)
            results['rgbd_depth'] = stacked[..., 3].astype(np.float32, copy=False)
            results['rgbd_raw_valid'] = stacked[..., 4].astype(bool)
            results['rgbd_label_valid'] = stacked[..., 5].astype(bool)
        return output


@TRANSFORMS.register_module()
class RGBDRandomCrop(_PairedCropMixin, RandomCrop):
    """Original RandomCrop with exact RGB/depth/validity alignment."""


@TRANSFORMS.register_module()
class RGBDSmallObjectAwareCrop(_PairedCropMixin, SmallObjectAwareCrop):
    """Independently switchable guided crop, retaining the original crop RNG."""


@TRANSFORMS.register_module()
class PackRGBDDetInputs(PackDetInputs):
    """Pack BGR(3), depth-metres(1), raw-valid(1); preserve label-valid PixelData."""

    def transform(self, results):
        _check_aligned(results)
        packed_results = dict(results)
        packed_results['img'] = np.concatenate(
            [results['img'].astype(np.float32), results['rgbd_depth'][..., None],
             results['rgbd_raw_valid'][..., None].astype(np.float32)], axis=2)
        output = super().transform(packed_results)
        valid = np.ascontiguousarray(results['rgbd_label_valid'][None].astype(bool))
        output['data_samples'].label_valid = PixelData(data=torch.from_numpy(valid))
        return output

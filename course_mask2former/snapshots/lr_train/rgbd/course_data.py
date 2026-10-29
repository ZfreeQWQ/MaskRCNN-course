"""Per-image uint16 depth conversion for the component course dataset."""
from pathlib import Path

import mmcv
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS, TRANSFORMS

from rgbd.data import _check_aligned


@DATASETS.register_module()
class CourseRGBDCocoDataset(CocoDataset):
    def parse_data_info(self, raw_data_info):
        parsed = super().parse_data_info(raw_data_info)
        image = raw_data_info['raw_img_info']
        relative = Path(image['depth_file_name'])
        root = Path(self.data_root).resolve()
        depth = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not depth.is_relative_to(root):
            raise ValueError('Depth path must remain under course data root')
        parsed.update(depth_file=str(depth), depth_scale_m=image['depth_scale_m'],
                      depth_invalid_code=image['depth_invalid_code'], scene=image['scene_id'])
        return parsed


@TRANSFORMS.register_module()
class LoadCourseRGBDImage(LoadImageFromFile):
    def __init__(self, **kwargs):
        kwargs.setdefault('to_float32', True)
        super().__init__(**kwargs)

    def transform(self, results):
        results = super().transform(results)
        raw = mmcv.imread(results['depth_file'], flag='unchanged')
        if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
            raise ValueError('Course depth must be uint16 PNG')
        scale = results['depth_scale_m']
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid per-image scale')
        valid = (raw != results['depth_invalid_code']) & (raw > 0)
        results['rgbd_depth'] = np.where(valid, raw.astype(np.float32) * scale, 0).astype(np.float32)
        results['rgbd_raw_valid'] = valid
        # Full component annotations supervise the entire image. Sensor
        # missingness and support_rle do not restrict ground-truth supervision.
        results['rgbd_label_valid'] = np.ones(raw.shape, dtype=bool)
        _check_aligned(results)
        return results

"""Value-equivalent inference preprocessing; no geometry/math/model change.

Use exact installed nearest-neighbor indices cached once per shape, and write
the CHW tensor directly. RGB keeps the same PIL bilinear resize as V3. This
avoids repeated float HWC<->CHW interpolation and the 11-channel HWC temporary.
"""
from functools import lru_cache
import copy
import time
import numpy as np
import torch
from detectron2.data import transforms as T
from course.data_v3 import FastGeometryMapper
from course.geometry_v2 import encode_geometry, standardize_geometry


@lru_cache(maxsize=4)
def nearest_indices(h, w, size):
    assert h*w < 2**24, "Float32 index probe must represent every source pixel exactly"
    probe = np.arange(h*w, dtype=np.float32).reshape(h, w, 1)
    transform = T.ResizeTransform(h, w, size, size)
    full = transform.apply_segmentation(probe).reshape(size, size).astype(np.int64)
    rows, cols = full[:, 0] // w, full[0] % w
    assert np.array_equal(full, rows[:, None]*w + cols[None, :]), "Resize is not separable nearest sampling"
    rows.setflags(write=False)
    cols.setflags(write=False)
    return rows, cols


class CompactGeometryMapper(FastGeometryMapper):
    def __call__(self, record):
        if self.training:
            return super().__call__(record)
        started = time.perf_counter()
        result = copy.deepcopy(record)
        rgb, depth, metric = self.read_inputs(record)
        read_finished = time.perf_counter()
        camera = self.cameras["cameras"][str(record["image_id"])]
        assert camera["split"] == record["split"]
        geometry, result["geometry_audit"] = encode_geometry(metric, camera["K"], record["image_id"])
        geometry = standardize_geometry(geometry, self.statistics["mean"], self.statistics["std"])
        geometry_finished = time.perf_counter()
        h, w = rgb.shape[:2]
        size = self.cfg.COURSE.TEST_SIZE
        assert h == w and size in (1024, 1280, 1536)
        warmup = 0.
        if size != h:
            rgb = T.ResizeTransform(h, w, size, size).apply_image(rgb)
            tick = time.perf_counter()
            rows, cols = nearest_indices(h, w, size)
            warmup = time.perf_counter()-tick
            depth = depth[rows[:, None], cols[None, :]]
            geometry = geometry[rows[:, None], cols[None, :], :]
        image = np.empty((11, size, size), dtype=np.float32)
        image[:3] = rgb.transpose(2, 0, 1)
        image[3] = depth
        image[4:] = geometry.transpose(2, 0, 1)
        result["image"] = torch.from_numpy(image)
        result["mapper_seconds"] = time.perf_counter()-started
        result["preprocess_profile"] = dict(read_seconds=read_finished-started,
                                             geometry_seconds=geometry_finished-read_finished,
                                             transform_seconds=result["mapper_seconds"]-(geometry_finished-started))
        result["nearest_index_lookup_seconds"] = warmup
        return result

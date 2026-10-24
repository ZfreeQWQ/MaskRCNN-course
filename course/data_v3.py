"""Single-read, value-equivalent RGB-D input; explicit aligned test scaling."""
import numpy as np
from PIL import Image
from detectron2.data import transforms as T
from course.data_v2 import GeometryROIMapper


def read_rgbd_once(record):
    with Image.open(record["file_name"]) as stream:
        rgb = np.asarray(stream.convert("RGB"))
    with Image.open(record["depth_path"]) as stream:
        raw = np.asarray(stream, dtype=np.uint16)
    assert raw.shape == rgb.shape[:2] == (record["height"], record["width"])
    # Keep exactly the baseline order/dtype of arithmetic for the stem input.
    metric = raw.astype(np.float32) * record["depth_scale_m"]
    depth = np.clip(metric / 1.0, 0, 1) * 255
    invalid = raw == record["depth_invalid_code"]
    depth[invalid] = 0
    metric[invalid] = 0
    return rgb, depth, metric


class FastGeometryMapper(GeometryROIMapper):
    def read_inputs(self, record):
        return read_rgbd_once(record)

    def transform_rgb(self, rgb):
        if self.training:
            return super().transform_rgb(rgb)
        h, w = rgb.shape[:2]
        size = self.cfg.COURSE.TEST_SIZE
        assert h == w and size in (1024, 1280, 1536)
        transforms = T.TransformList([] if size == h else [
            T.ResizeTransform(h, w, size, size, interp=Image.Resampling.BILINEAR)])
        return transforms.apply_image(rgb), transforms

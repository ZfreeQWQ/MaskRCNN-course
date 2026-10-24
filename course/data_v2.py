"""V2 mapper. Original train/val split and baseline RGB-D transforms unchanged."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

from course.augmentation import decode_annotations
from course.data import RGBDMapper, read_rgbd
from course.geometry_v2 import GEOMETRY_VERSION, encode_geometry, standardize_geometry


class GeometryROIMapper(RGBDMapper):
    def __init__(self, cfg, training=True, cache="data_cache"):
        super().__init__(cfg, training)
        self.cfg = cfg
        self.geometry = cfg.COURSE.V2.GEOMETRY
        if self.geometry:
            cache = Path(cache)
            self.cameras = json.loads((cache / "geometry_v2_intrinsics.json").read_text())
            self.statistics = json.loads((cache / "geometry_v2_stats.json").read_text())
            signature = hashlib.sha256((cache / "index.json").read_bytes()).hexdigest()
            assert self.cameras["index_sha256"] == self.statistics["index_sha256"] == signature
            assert self.statistics["train_only"]
            assert self.statistics["geometry_version"] == GEOMETRY_VERSION
            assert self.statistics["accumulation_dtype"] == "float64"

    def read_inputs(self, record):
        rgb, depth = read_rgbd(record)
        return rgb, depth, None

    def transform_rgb(self, rgb):
        aug_input = T.AugInput(rgb)
        transforms = self.augmentations(aug_input)
        return aug_input.image, transforms

    def __call__(self, record):
        started = time.perf_counter()
        result = copy.deepcopy(record)
        rgb, depth, metric = self.read_inputs(record)
        read_finished = time.perf_counter()
        geometry = None
        if self.geometry:
            camera = self.cameras["cameras"][str(record["image_id"])]
            assert camera["split"] == record["split"]
            if metric is None:
                raw = np.asarray(Image.open(record["depth_path"]), dtype=np.uint16)
                metric = raw.astype(np.float32) * record["depth_scale_m"]
                metric[raw == record["depth_invalid_code"]] = 0
            geometry, result["geometry_audit"] = encode_geometry(metric, camera["K"], record["image_id"])
            # XYZ normals remain in the original camera coordinate frame.
            # Geometry is computed with original K first, then spatially warped
            # with masks/depth. RGB pixels and geometric vectors stay aligned.
            geometry = standardize_geometry(geometry, self.statistics["mean"], self.statistics["std"])
        geometry_finished = time.perf_counter()
        rgb, transforms = self.transform_rgb(rgb)
        depth = transforms.apply_segmentation(depth)
        parts = [rgb.astype(np.float32), depth[..., None]]
        if geometry is not None:
            # Nearest sampling keeps discontinuities and the validity channel;
            # padding happens after normalization and is exactly zero.
            parts.append(transforms.apply_segmentation(geometry))
        result["image"] = torch.from_numpy(np.ascontiguousarray(np.concatenate(parts, -1).transpose(2, 0, 1)))
        if self.training:
            assert record["split"] == "train", "Never train on val/test"
            annotations = [a for a in self.reader.get(record) if not a.get("iscrowd", 0)]
            masks, flags = decode_annotations(annotations, (record["height"], record["width"]), self.cfg.COURSE.SMALL_AREA)
            original_area = np.asarray([a["area"] for a in annotations], np.float32)
            masks = (np.stack([transforms.apply_segmentation(m.astype(np.uint8)).astype(bool) for m in masks])
                     if len(masks) else np.zeros((0, *rgb.shape[:2]), bool))
            keep = masks.sum((1, 2)) > 0
            instances = Instances(rgb.shape[:2])
            instances.gt_masks = BitMasks(torch.from_numpy(np.ascontiguousarray(masks[keep])))
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances.gt_masks = instances.gt_masks.tensor
            instances.gt_classes = torch.zeros(int(keep.sum()), dtype=torch.int64)
            instances.gt_small = torch.from_numpy(flags[keep].copy())
            instances.gt_original_area = torch.from_numpy(original_area[keep].copy())
            result["instances"] = instances
        result["mapper_seconds"] = time.perf_counter() - started
        result["preprocess_profile"] = dict(read_seconds=read_finished-started,
                                             geometry_seconds=geometry_finished-read_finished,
                                             transform_seconds=result["mapper_seconds"]-(geometry_finished-started))
        return result

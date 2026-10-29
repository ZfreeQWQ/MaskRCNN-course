"""RGB-D instance-segmentation dataset mapper for Mask2Former (docs/plan.md §4, §9).

Train recipe follows the official COCOInstanceNewBaselineDatasetMapper's LSJ-style
`ResizeScale(min_scale, max_scale, target, target) + FixedSizeCrop(target)` resize_and_crop,
plus horizontal/vertical flip, exact discrete 90-degree rotation, and (RGB-only) color jitter.
It differs from the official mapper in two ways required by this project: (1) our GT
segmentation is compressed RLE / bitmask, not polygons, so annotations go through
`annotations_to_instances(..., mask_format="bitmask")` like rgbd_seg.data.RGBDMapper, not
`convert_coco_poly_to_mask`; (2) when PROJECT.USE_DEPTH is set, it additionally concatenates 6 depth-geometry channels
(height-above-plane, 3 normal components, discontinuity, valid -- see rgbd_seg.depth_geometry)
onto `image`, which then has 9 channels total. This keeps MaskFormer.forward() completely
unmodified (it just treats "image" as a wider tensor; MODEL.PIXEL_MEAN/STD get 9 entries, with
0/1 for the depth channels since they're pre-standardized here). rgbd_seg.m2f.depth_fusion's
backbone splits the 9 channels back apart. The depth channels are geometrically aligned with the
RGB ones through the exact same `transforms`, standardized to RGB-like magnitude using
train-only stats.

Geometric channels are computed once from the *native-resolution* raw depth (their intrinsics-
based derivation needs the unresampled image), then carried through the same resize/crop/flip/
rotate pipeline as the RGB image and masks via `transforms.apply_segmentation`, which forces
nearest-neighbor resampling for every op that actually resamples (Resize, Rot90) -- consistent
with how rgbd_seg.data.RGBDMapper already treats depth, and correct here too: height/normals are
physical per-surface-point quantities that don't change when we merely resample or permute which
pixel holds them (see rgbd_seg/depth_geometry.py's module docstring).
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks

from .. import depth_geometry as dg
from .geo_transforms import RandomRot90

__all__ = ["RGBDInstanceDatasetMapper"]


class RGBDInstanceDatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.root = Path(cfg.PROJECT.DATASET_ROOT)
        self.use_depth = cfg.PROJECT.USE_DEPTH

        if self.use_depth:
            with open(cfg.PROJECT.DEPTH_GEOMETRY_STATS) as stream:
                self.geo_stats = json.load(stream)
            if self.geo_stats.get("train_only") is not True or self.geo_stats.get("split") != "train":
                raise ValueError("Depth-geometry statistics must explicitly originate only from the training split")

        if is_train:
            augs = []
            flip = cfg.INPUT.RANDOM_FLIP
            if flip in ("horizontal", "both"):
                augs.append(T.RandomFlip(horizontal=True, vertical=False))
            if flip in ("vertical", "both"):
                augs.append(T.RandomFlip(horizontal=False, vertical=True))
            if cfg.INPUT.ROTATE_90:
                augs.append(RandomRot90())
            augs.append(T.ResizeScale(cfg.INPUT.MIN_SCALE, cfg.INPUT.MAX_SCALE,
                                      cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE))
            augs.append(T.FixedSizeCrop((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE),
                                        pad_value=128.0, seg_pad_value=0.0))
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=self.img_format))
            self.augs = T.AugmentationList(augs)
        else:
            self.augs = T.AugmentationList([T.ResizeShortestEdge(cfg.INPUT.TEST_IMAGE_SIZE, cfg.INPUT.TEST_IMAGE_SIZE)])

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _depth_geometry_native(self, record):
        """Compute the standardized 6-channel depth-geometry map at native image resolution."""
        if (not record.get("depth_file_name") or record.get("depth_scale_m") is None
                or record.get("depth_invalid_code") is None):
            raise ValueError("RGB-D input requires explicit depth_file_name, depth_scale_m and "
                              "depth_invalid_code; no test metadata guessing")
        with Image.open(self._path(record["depth_file_name"])) as depth_image:
            raw = np.array(depth_image)
        scale = float(record["depth_scale_m"])
        invalid_code = record["depth_invalid_code"]
        metric = raw.astype(np.float32) * scale
        valid = (raw != invalid_code) & np.isfinite(metric) & (metric > 0)
        encoded = dg.encode(metric, valid)
        return dg.standardize(encoded, self.geo_stats)  # (H, W, 6) float32

    def __call__(self, record):
        record = copy.deepcopy(record)
        image = utils.read_image(str(self._path(record["file_name"])), format=self.img_format)
        utils.check_image_size(record, image)
        record.setdefault("height", image.shape[0])
        record.setdefault("width", image.shape[1])

        depth_geo = self._depth_geometry_native(record) if self.use_depth else None

        aug_input = T.AugInput(image)
        transforms = self.augs(aug_input)
        image = aug_input.image

        if depth_geo is not None:
            depth_geo = transforms.apply_segmentation(depth_geo)

        image_shape = image.shape[:2]
        if depth_geo is not None:
            # Concatenate as extra channels so MaskFormer.forward()'s plain
            # `(x["image"] - pixel_mean) / pixel_std` -> backbone(x) path needs no changes;
            # rgbd_seg.m2f.depth_fusion's backbone splits RGB/depth back apart.
            image = np.concatenate([image.astype(np.float32), depth_geo], axis=2)
        else:
            image = image.astype(np.float32)
        record["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            record.pop("annotations", None)
            return record

        annos = [utils.transform_instance_annotations(a, transforms, image_shape)
                 for a in record.pop("annotations", []) if a.get("iscrowd", 0) == 0]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.mask_format)
        instances = utils.filter_empty_instances(instances)
        if hasattr(instances, "gt_masks") and isinstance(instances.gt_masks, BitMasks):
            # Mask2Former's targets/criterion expect a plain dense tensor, not the BitMasks
            # wrapper (see maskformer_model.py's prepare_targets: gt_masks.shape[0]).
            instances.gt_masks = instances.gt_masks.tensor
        record["instances"] = instances
        return record

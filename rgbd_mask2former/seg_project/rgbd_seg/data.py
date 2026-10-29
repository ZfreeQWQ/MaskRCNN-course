import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils, transforms as T
from detectron2.data.datasets import load_coco_json


def _load_split(root, split, name):
    path = root / "annotations" / f"instances_{split}.json"
    records = load_coco_json(str(path), str(root), name, extra_annotation_keys=["semantic_type"])
    # Detectron2's COCO loader intentionally drops custom image metadata.
    with path.open() as stream:
        images = {im["id"]: im for im in json.load(stream)["images"]}
    for record in records:
        info = images[record["image_id"]]
        for key in ("depth_scale_m", "depth_invalid_code", "scene_id"):
            if key in info:
                record[key] = info[key]
        if "depth_file_name" in info:
            record["depth_file_name"] = str(root / info["depth_file_name"])
    return records


def register_datasets(cfg):
    """Register only train/val. Never opens or registers test annotations."""
    root = Path(cfg.PROJECT.DATASET_ROOT).resolve()
    for split in ("train", "val"):
        name = f"blender_{split}"
        json_file = str(root / "annotations" / f"instances_{split}.json")
        if name in DatasetCatalog.list():
            if MetadataCatalog.get(name).json_file != json_file:
                raise ValueError(f"{name} already registered with a different dataset root")
            continue
        DatasetCatalog.register(name, lambda r=root, s=split, n=name: _load_split(r, s, n))
        MetadataCatalog.get(name).set(json_file=json_file, image_root=str(root),
                                      evaluator_type="coco", thing_classes=["component"],
                                      thing_dataset_id_to_contiguous_id={1: 0})


def normalize_depth(raw, scale_m, invalid_code, p1, p99, neutral=0.5):
    """Return normalized depth and validity; invalid depth becomes zero after model normalization."""
    if not np.isfinite(scale_m) or scale_m <= 0 or not np.isfinite(p1 + p99) or p99 <= p1:
        raise ValueError("Invalid metric depth scale or training depth quantiles")
    metric = raw.astype(np.float32) * float(scale_m)
    valid = (raw != invalid_code) & np.isfinite(metric) & (metric > 0)
    depth = np.clip((metric - p1) / (p99 - p1), 0.0, 1.0)
    depth[~valid] = neutral
    return depth.astype(np.float32), valid


class RGBDMapper:
    """One sampled geometry transform shared by RGB, depth, validity, boxes and masks."""
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.root = Path(cfg.PROJECT.DATASET_ROOT)
        self.use_depth = cfg.PROJECT.USE_DEPTH
        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.augs = T.AugmentationList(utils.build_augmentation(cfg, is_train))
        if cfg.INPUT.CROP.ENABLED:
            raise ValueError("Cropping is not implemented in this controlled baseline")
        if self.use_depth:
            with open(cfg.PROJECT.DEPTH_STATS) as stream:
                stats = json.load(stream)
            if stats.get("train_only") is not True or stats.get("split") != "train":
                raise ValueError("Depth statistics must explicitly originate only from the training split")
            self.p1, self.p99 = float(stats["p1"]), float(stats["p99"])
            self.neutral = float(cfg.MODEL.PIXEL_MEAN[3])

    def _path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __call__(self, record):
        record = copy.deepcopy(record)
        image = utils.read_image(str(self._path(record["file_name"])), format=self.image_format)
        utils.check_image_size(record, image)
        # Keep original dimensions for Detectron2 postprocessing.
        record.setdefault("height", image.shape[0])
        record.setdefault("width", image.shape[1])
        if self.use_depth:
            if (not record.get("depth_file_name") or record.get("depth_scale_m") is None
                    or record.get("depth_invalid_code") is None):
                raise ValueError("RGB-D input requires explicit depth_file_name, depth_scale_m and depth_invalid_code; no test metadata guessing")
            with Image.open(self._path(record["depth_file_name"])) as depth_image:
                raw = np.array(depth_image)
            if raw.ndim != 2 or raw.shape != image.shape[:2] or raw.dtype != np.uint16:
                raise ValueError(f"Expected aligned uint16 depth, got {raw.dtype} {raw.shape}")
            depth, valid = normalize_depth(raw, float(record["depth_scale_m"]),
                                           record["depth_invalid_code"], self.p1, self.p99,
                                           self.neutral)
        aug_input = T.AugInput(image)
        transforms = self.augs(aug_input)
        image = aug_input.image.astype(np.float32)
        if self.use_depth:
            # Nearest sampling avoids interpolation across invalid/background depth boundaries.
            depth = transforms.apply_segmentation(depth)
            valid = transforms.apply_segmentation(valid.astype(np.uint8)).astype(bool)
            depth[~valid] = self.neutral
            image = np.concatenate([image, depth[..., None]], axis=2)
            record["depth_valid_mask"] = torch.as_tensor(np.ascontiguousarray(valid))
        record["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.is_train and "annotations" in record:
            annotations = [utils.transform_instance_annotations(a, transforms, image.shape[:2])
                           for a in record.pop("annotations") if a.get("iscrowd", 0) == 0]
            instances = utils.annotations_to_instances(annotations, image.shape[:2], mask_format=self.mask_format)
            record["instances"] = utils.filter_empty_instances(instances)
        else:
            record.pop("annotations", None)
        return record

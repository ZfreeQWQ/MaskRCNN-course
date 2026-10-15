"""RGB-D Blender dataset support for the Matterport Mask R-CNN codebase.

The dataset contains one COCO category (``component``), RGB PNGs, uint16
depth PNGs, and one gzip-compressed COCO JSON file per scene. Annotation
shards are decoded on demand rather than retaining all masks in memory.
"""

from __future__ import print_function

import argparse
import gzip
import json
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# Allow direct execution from the repository root, matching the bundled
# Matterport examples.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import utils


class BlenderConfig(Config):
    """Configuration for RGB-D component instance segmentation."""

    NAME = "blender_rgbd"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # Background + component
    IMAGE_CHANNEL_COUNT = 4
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "square"
    # The fourth channel is depth normalized to [0, 255].
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 32.0])
    MAX_GT_INSTANCES = 128
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100


class BlenderDataset(utils.Dataset):
    """Dataset adapter for the COCO-style Blender RGB-D data."""

    SOURCE = "blender"
    CLASS_ID = 1
    CLASS_NAME = "component"
    DEPTH_MAX_M = 1.0
    SHARD_CACHE_SIZE = 2

    def __init__(self):
        super(BlenderDataset, self).__init__()
        self._annotation_shards = {}
        self._shard_cache = OrderedDict()

    def load_blender(self, dataset_dir, subset, depth_max_m=None):
        """Index a train or val split without loading all masks in memory."""
        if subset not in ("train", "val"):
            raise ValueError("Only train and val are supported for training")
        dataset_dir = os.path.abspath(dataset_dir)
        split_dir = os.path.join(dataset_dir, "annotations", subset)
        shard_paths = sorted(
            os.path.join(split_dir, name)
            for name in os.listdir(split_dir)
            if name.endswith(".json.gz")
        )
        if not shard_paths:
            raise IOError("No .json.gz annotation shards found in {}".format(split_dir))
        if depth_max_m is not None:
            self.DEPTH_MAX_M = float(depth_max_m)
        self.add_class(self.SOURCE, self.CLASS_ID, self.CLASS_NAME)

        for shard_path in shard_paths:
            with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
                shard = json.load(stream)
            for image in shard["images"]:
                rgb_path = os.path.join(dataset_dir, image["file_name"].replace("/", os.sep))
                depth_name = image.get("depth_file_name")
                if not depth_name:
                    raise ValueError("Missing depth_file_name for image {}".format(image["id"]))
                depth_path = os.path.join(dataset_dir, depth_name.replace("/", os.sep))
                if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
                    raise IOError("Missing RGB/depth pair for {}".format(image["file_name"]))
                self._annotation_shards[image["id"]] = shard_path
                self.add_image(
                    self.SOURCE,
                    image_id=image["id"],
                    path=rgb_path,
                    depth_path=depth_path,
                    width=image["width"],
                    height=image["height"],
                    coco_id=image["id"],
                    annotation_shard=shard_path,
                    depth_scale_m=float(image["depth_scale_m"]),
                    depth_invalid_code=int(image.get("depth_invalid_code", 0)),
                )

    def _read_shard(self, shard_path):
        if shard_path in self._shard_cache:
            shard = self._shard_cache.pop(shard_path)
            self._shard_cache[shard_path] = shard
            return shard
        with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
            raw = json.load(stream)
        annotations = defaultdict(list)
        for annotation in raw.get("annotations", []):
            annotations[annotation["image_id"]].append(annotation)
        self._shard_cache[shard_path] = annotations
        while len(self._shard_cache) > self.SHARD_CACHE_SIZE:
            self._shard_cache.popitem(last=False)
        return annotations

    def load_image(self, image_id):
        """Return RGB plus one normalized metric-depth channel."""
        info = self.image_info[image_id]
        if info["source"] != self.SOURCE:
            return super(BlenderDataset, self).load_image(image_id)
        rgb = np.asarray(Image.open(info["path"]).convert("RGB"), dtype=np.uint8)
        raw_depth = np.asarray(Image.open(info["depth_path"]), dtype=np.uint16)
        if raw_depth.shape != rgb.shape[:2]:
            raise ValueError("RGB/depth shape mismatch for {}".format(info["path"]))
        depth_m = raw_depth.astype(np.float32) * info["depth_scale_m"]
        invalid = raw_depth == info["depth_invalid_code"]
        depth = np.clip(depth_m / self.DEPTH_MAX_M, 0.0, 1.0) * 255.0
        depth[invalid] = 0.0
        return np.concatenate([rgb, depth[..., np.newaxis].astype(np.uint8)], axis=-1)

    def load_mask(self, image_id):
        """Decode compressed COCO-RLE instance masks for one image."""
        info = self.image_info[image_id]
        if info["source"] != self.SOURCE:
            return super(BlenderDataset, self).load_mask(image_id)
        annotations = self._read_shard(info["annotation_shard"])[info["coco_id"]]
        masks = []
        class_ids = []
        for annotation in annotations:
            if annotation.get("category_id") != self.CLASS_ID:
                continue
            segmentation = annotation["segmentation"]
            if isinstance(segmentation.get("counts"), str):
                segmentation = {
                    "size": segmentation["size"],
                    "counts": segmentation["counts"].encode("ascii"),
                }
            mask = mask_utils.decode(segmentation)
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != (info["height"], info["width"]):
                raise ValueError("Mask shape mismatch for image {}".format(info["coco_id"]))
            if np.any(mask):
                masks.append(mask.astype(np.bool_))
                class_ids.append(self.CLASS_ID)
        if not masks:
            return (
                np.empty((info["height"], info["width"], 0), dtype=np.bool_),
                np.empty((0,), dtype=np.int32),
            )
        return np.stack(masks, axis=-1), np.asarray(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id].get("path", "")


def build_datasets(dataset_dir):
    """Create prepared train and validation datasets."""
    train = BlenderDataset()
    train.load_blender(dataset_dir, "train")
    train.prepare()
    val = BlenderDataset()
    val.load_blender(dataset_dir, "val")
    val.prepare()
    return train, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the Blender RGB-D reader")
    parser.add_argument("dataset", help="Blender dataset root")
    args = parser.parse_args()
    train, val = build_datasets(args.dataset)
    image = train.load_image(0)
    masks, class_ids = train.load_mask(0)
    print("train images: {}".format(train.num_images))
    print("val images: {}".format(val.num_images))
    print("sample image: {}, masks: {}, classes: {}".format(
        image.shape, masks.shape, class_ids.tolist()))

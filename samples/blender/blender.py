"""为 Matterport Mask R-CNN 代码库提供 RGB-D Blender 数据集支持。

该数据集包含一个 COCO 类别（``component``）、RGB PNG 图像、uint16 深度 PNG 图像，
以及每个场景一个 gzip 压缩的 COCO JSON 文件。训练和验证标注分片按需解码，
测试集则从根目录的 ``test_images.json`` 读取元数据，不读取测试标注。
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

# 允许直接从仓库根目录执行，与自带的 Matterport 示例保持一致。
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import utils


class BlenderConfig(Config):
    """RGB-D 组件实例分割的配置。"""

    NAME = "blender_rgbd"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2  # 背景 + 组件（component）
    IMAGE_CHANNEL_COUNT = 4
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_RESIZE_MODE = "square"
    # 第四个通道是归一化到 [0, 255] 的深度。
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 32.0])
    MAX_GT_INSTANCES = 128
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100


class BlenderDataset(utils.Dataset):
    """面向 COCO 风格 Blender RGB-D 数据的数据集适配器。"""

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
        """索引一个划分；train/val 读取分片，test 读取 test_images.json。"""
        if subset not in ("train", "val", "test"):
            raise ValueError("subset 必须是 train、val 或 test")
        dataset_dir = os.path.abspath(dataset_dir)

        # 测试集没有实例标注，只使用根目录中的测试图像元数据。
        if subset == "test":
            metadata_path = os.path.join(dataset_dir, "test_images.json")
            if not os.path.isfile(metadata_path):
                raise IOError("找不到测试集元数据文件: {}".format(metadata_path))
            with open(metadata_path, "r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            images = metadata.get("images")
            if not isinstance(images, list):
                raise ValueError("test_images.json 缺少 images 列表")
            self.add_class(self.SOURCE, self.CLASS_ID, self.CLASS_NAME)
            for image in images:
                self._add_image_from_metadata(dataset_dir, image, None)
            return

        split_dir = os.path.join(dataset_dir, "annotations", subset)
        shard_paths = sorted(
            os.path.join(split_dir, name)
            for name in os.listdir(split_dir)
            if name.endswith(".json.gz")
        )
        if not shard_paths:
            raise IOError("找不到标注分片目录或文件: {}".format(split_dir))
        if depth_max_m is not None:
            self.DEPTH_MAX_M = float(depth_max_m)
        self.add_class(self.SOURCE, self.CLASS_ID, self.CLASS_NAME)

        for shard_path in shard_paths:
            with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
                shard = json.load(stream)
            for image in shard["images"]:
                self._annotation_shards[image["id"]] = shard_path
                self._add_image_from_metadata(dataset_dir, image, shard_path)

    def _add_image_from_metadata(self, dataset_dir, image, annotation_shard):
        """根据一条元数据记录加入 RGB、深度和标注路径。"""
        required = ("id", "file_name", "depth_file_name", "width", "height", "depth_scale_m")
        missing = [key for key in required if key not in image]
        if missing:
            raise ValueError("图像元数据缺少字段: {}".format(", ".join(missing)))
        rgb_path = os.path.join(dataset_dir, image["file_name"].replace("/", os.sep))
        depth_path = os.path.join(dataset_dir, image["depth_file_name"].replace("/", os.sep))
        if not os.path.isfile(rgb_path) or not os.path.isfile(depth_path):
            raise IOError("找不到 RGB/深度图像对: {}".format(image["file_name"]))
        self.add_image(
            self.SOURCE,
            image_id=image["id"],
            path=rgb_path,
            depth_path=depth_path,
            width=int(image["width"]),
            height=int(image["height"]),
            coco_id=image["id"],
            annotation_shard=annotation_shard,
            has_annotations=annotation_shard is not None,
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
        """返回 RGB 图像加一个归一化到 [0, 255] 的深度通道。"""
        info = self.image_info[image_id]
        if info["source"] != self.SOURCE:
            return super(BlenderDataset, self).load_image(image_id)
        with Image.open(info["path"]) as rgb_file:
            rgb = np.asarray(rgb_file.convert("RGB"), dtype=np.uint8)
        with Image.open(info["depth_path"]) as depth_file:
            raw_depth = np.asarray(depth_file, dtype=np.uint16)
        if raw_depth.shape != rgb.shape[:2]:
            raise ValueError("RGB 与深度图尺寸不一致: {}".format(info["path"]))
        depth_m = raw_depth.astype(np.float32) * info["depth_scale_m"]
        invalid = raw_depth == info["depth_invalid_code"]
        depth = np.clip(depth_m / self.DEPTH_MAX_M, 0.0, 1.0) * 255.0
        depth[invalid] = 0.0
        return np.concatenate([rgb, depth[..., np.newaxis].astype(np.uint8)], axis=-1)

    def load_mask(self, image_id):
        """解码单张图像上的 COCO-RLE 实例掩码；测试图像返回空掩码。"""
        info = self.image_info[image_id]
        if info["source"] != self.SOURCE:
            return super(BlenderDataset, self).load_mask(image_id)
        if not info.get("has_annotations", False):
            return (
                np.empty((info["height"], info["width"], 0), dtype=np.bool_),
                np.empty((0,), dtype=np.int32),
            )
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
                raise ValueError("掩码尺寸与图像不一致: {}".format(info["coco_id"]))
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
    """创建已准备好的训练集和验证集数据集。"""
    train = BlenderDataset()
    train.load_blender(dataset_dir, "train")
    train.prepare()
    val = BlenderDataset()
    val.load_blender(dataset_dir, "val")
    val.prepare()
    return train, val


def build_test_dataset(dataset_dir):
    """创建只包含测试 RGB-D 图像的无标注数据集。"""
    test = BlenderDataset()
    test.load_blender(dataset_dir, "test")
    test.prepare()
    return test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 Blender RGB-D 数据读取器")
    parser.add_argument("dataset", help="Blender 数据集根目录")
    parser.add_argument(
        "--subset", choices=("train", "val", "test"), default="train",
        help="要检查的划分，默认是 train",
    )
    args = parser.parse_args()
    if args.subset == "test":
        dataset = build_test_dataset(args.dataset)
    else:
        dataset = BlenderDataset()
        dataset.load_blender(args.dataset, args.subset)
        dataset.prepare()
    image = dataset.load_image(0)
    masks, class_ids = dataset.load_mask(0)
    print("{} 图像数: {}".format(args.subset, dataset.num_images))
    print("sample image: {}, masks: {}, classes: {}".format(
        image.shape, masks.shape, class_ids.tolist()))

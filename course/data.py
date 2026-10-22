"""Train/val-only RGB-D data preparation and synchronized augmentation.

An indexed SQLite cache avoids decompressing an entire annotation shard on
each randomly selected image. It is derived solely from the supplied data.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np
from PIL import Image


def prepare_cache(dataset, cache):
    root, cache = Path(dataset).resolve(), Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    shards = {split: sorted((root / "annotations" / split).glob("*.json.gz"))
              for split in ("train", "val")}
    fingerprint = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                   for paths in shards.values() for p in paths]
    signature = json.loads(json.dumps(fingerprint))
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_files"] != signature or manifest["dataset"] != str(root):
            raise RuntimeError("Dataset changed; build a new cache directory")
        return manifest
    if not all(shards.values()):
        raise FileNotFoundError("Missing train/val annotation shards")
    db_path = cache / "annotations.sqlite3.tmp"
    if db_path.exists():
        raise RuntimeError("Incomplete cache exists; inspect before rebuilding: " + str(db_path))
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE instances(split TEXT, image_id INTEGER, data BLOB, PRIMARY KEY(split,image_id))")
    indexes, statistics, split_ids, split_paths = {}, {}, {}, {}
    for split, paths in shards.items():
        records, ids, annotation_ids, rgb_paths = [], set(), set(), set()
        val_images, val_annotations = [], []
        total_annotations = 0
        for n, path in enumerate(paths):
            with gzip.open(path, "rt", encoding="utf-8") as source:
                shard = json.load(source)
            by_image = {int(im["id"]): [] for im in shard["images"]}
            for annotation in shard["annotations"]:
                assert annotation["category_id"] == 1
                assert int(annotation["id"]) not in annotation_ids, "Duplicate annotation ID"
                annotation_ids.add(int(annotation["id"]))
                by_image[int(annotation["image_id"])].append(annotation)
            for image in shard["images"]:
                image_id = int(image["id"])
                assert image_id not in ids, "Duplicate image ID"
                assert image["width"] == 1024 and image["height"] == 1024
                ids.add(image_id)
                rgb_paths.add(image["file_name"])
                record = dict(image_id=image_id, file_name=str(root / image["file_name"]),
                              depth_path=str(root / image["depth_file_name"]),
                              height=1024, width=1024, depth_scale_m=float(image["depth_scale_m"]),
                              depth_invalid_code=int(image.get("depth_invalid_code", 0)),
                              split=split, annotation_db=str(cache / "annotations.sqlite3"))
                assert Path(record["file_name"]).is_file() and Path(record["depth_path"]).is_file()
                records.append(record)
                annotations = by_image[image_id]
                total_annotations += len(annotations)
                connection.execute("INSERT INTO instances VALUES(?,?,?)", (
                    split, image_id, zlib.compress(json.dumps(annotations, separators=(",", ":")).encode())))
                if split == "val":
                    val_images.append(dict(id=image_id, width=1024, height=1024,
                                           file_name=image["file_name"]))
                    val_annotations.extend(annotations)
            if (n + 1) % 100 == 0:
                print("CACHE", split, n + 1, "/", len(paths), flush=True)
        indexes[split] = records
        statistics[split] = dict(images=len(records), instances=total_annotations)
        split_ids[split], split_paths[split] = ids, rgb_paths
        if split == "val":
            with (cache / "coco_val.json").open("w") as stream:
                json.dump(dict(info={}, licenses=[], images=val_images, annotations=val_annotations,
                               categories=[dict(id=1, name="component")]), stream)
    assert not split_ids["train"].intersection(split_ids["val"])
    assert not split_paths["train"].intersection(split_paths["val"])
    connection.commit()
    connection.close()
    os.replace(db_path, cache / "annotations.sqlite3")
    (cache / "index.json").write_text(json.dumps(indexes), encoding="utf-8")
    manifest = dict(dataset=str(root), source_files=signature, statistics=statistics,
                    train_val_ids_and_paths_disjoint=True, cache_version=1)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("CACHE_READY", json.dumps(statistics), flush=True)
    return manifest


class AnnotationReader:
    def __init__(self):
        self.connection = None
        self.pid = None

    def get(self, record):
        if self.connection is None or self.pid != os.getpid():
            self.connection = sqlite3.connect("file:" + record["annotation_db"] + "?mode=ro", uri=True)
            self.pid = os.getpid()
        row = self.connection.execute("SELECT data FROM instances WHERE split=? AND image_id=?",
                                      (record["split"], record["image_id"])).fetchone()
        if row is None:
            raise KeyError(record["image_id"])
        return json.loads(zlib.decompress(row[0]))


def read_rgbd(record, depth_max_m=1.0):
    with Image.open(record["file_name"]) as source:
        rgb = np.asarray(source.convert("RGB"))
    with Image.open(record["depth_path"]) as source:
        raw = np.asarray(source, dtype=np.uint16)
    assert raw.shape == rgb.shape[:2] == (record["height"], record["width"])
    depth = np.clip(raw.astype(np.float32) * record["depth_scale_m"] / depth_max_m, 0, 1) * 255
    depth[raw == record["depth_invalid_code"]] = 0
    return rgb, depth


class RGBDMapper:
    def __init__(self, cfg, training=True):
        from detectron2.data import transforms as T
        self.training = training
        self.reader = AnnotationReader()
        self.augmentations = T.AugmentationList([
            T.RandomFlip(horizontal=True, vertical=False),
            T.ResizeScale(min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE,
                          target_height=cfg.INPUT.IMAGE_SIZE, target_width=cfg.INPUT.IMAGE_SIZE),
            T.FixedSizeCrop(crop_size=(cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE), seg_pad_value=0),
        ]) if training else T.AugmentationList([])

    def __call__(self, record):
        import torch
        from detectron2.data import detection_utils as utils, transforms as T
        from detectron2.structures import BoxMode
        result = copy.deepcopy(record)
        rgb, depth = read_rgbd(record)
        aug_input = T.AugInput(rgb)
        transformations = self.augmentations(aug_input)
        rgb = aug_input.image
        # Nearest-neighbor geometry retains depth discontinuities; RGB is bilinear.
        depth = transformations.apply_segmentation(depth)
        image = np.concatenate([rgb.astype(np.float32), depth[..., None]], axis=-1)
        result["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.training:
            annotations = []
            for original in self.reader.get(record):
                if original.get("iscrowd", 0):
                    continue
                annotation = copy.deepcopy(original)
                annotation["category_id"] = 0  # COCO ID 1 -> contiguous class 0
                annotation["bbox_mode"] = BoxMode.XYWH_ABS
                annotations.append(utils.transform_instance_annotations(annotation, transformations, image.shape[:2]))
            instances = utils.annotations_to_instances(annotations, image.shape[:2], mask_format="bitmask")
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            instances.gt_masks = instances.gt_masks.tensor
            result["instances"] = instances
        return result


def register(cache):
    from detectron2.data import DatasetCatalog, MetadataCatalog
    cache = Path(cache).resolve()
    index = json.loads((cache / "index.json").read_text())
    for split in ("train", "val"):
        name = "blender_rgbd_" + split
        DatasetCatalog.register(name, lambda split=split: index[split])
        MetadataCatalog.get(name).set(thing_classes=["component"],
                                     thing_dataset_id_to_contiguous_id={1: 0}, evaluator_type="coco",
                                     json_file=str(cache / "coco_val.json") if split == "val" else "")
    return index


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--dataset", default="/workspace/blender")
    arguments.add_argument("--cache", default="data_cache")
    options = arguments.parse_args()
    prepare_cache(options.dataset, options.cache)

"""Train-only small-instance augmentation, with aligned RGB, depth and masks.

Default mapper remains untouched. Areas/flags refer to original annotations,
not resized fragment sizes. Copy-Paste uses donor pixels at their native size
and accepts only near-depth target positions; front-most valid depth wins.
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

from course.data import RGBDMapper, read_rgbd


def decode_annotations(annotations, shape, small_area):
    masks, flags = [], []
    for ann in annotations:
        if ann.get("iscrowd", 0):
            continue
        segmentation = ann["segmentation"]
        if isinstance(segmentation, list):
            segmentation = mask_utils.merge(mask_utils.frPyObjects(segmentation, *shape))
        elif isinstance(segmentation.get("counts"), list):
            segmentation = mask_utils.frPyObjects(segmentation, *shape)
        mask = mask_utils.decode(segmentation).astype(bool)
        assert mask.shape == shape
        masks.append(mask)
        flags.append(0 < ann["area"] <= small_area)
    return (np.stack(masks) if masks else np.zeros((0, *shape), bool),
            np.asarray(flags, bool))


def paste_instance(rgb, depth, masks, flags, donor_rgb, donor_depth, donor_mask,
                   x, y, max_instances=100, minimum_visible=0.8):
    """Pure array operation, also tested with synthetic aligned pixel values."""
    h, w = donor_mask.shape
    if len(masks) >= max_instances or x < 0 or y < 0 or x + w > rgb.shape[1] or y + h > rgb.shape[0]:
        return rgb, depth, masks, flags, False
    target = depth[y:y+h, x:x+w]
    valid = donor_mask & (donor_depth > 0)
    if not valid.any():
        return rgb, depth, masks, flags, False
    # Depth was mapped linearly from [0, 1m] to [0, 255]. Require a nearby
    # receiving surface (<= 10cm) rather than pasting an arbitrary floating object.
    receiving = target[valid & (target > 0)]
    if not len(receiving) or abs(float(np.median(receiving)) - float(np.median(donor_depth[valid]))) > 25.5:
        return rgb, depth, masks, flags, False
    visible = valid & ((target <= 0) | (donor_depth <= target))
    if visible.sum() < minimum_visible * donor_mask.sum():
        return rgb, depth, masks, flags, False
    full = np.zeros(rgb.shape[:2], bool)
    full[y:y+h, x:x+w] = visible
    new_masks = masks & ~full[None]
    old_area = masks.sum(axis=(1, 2))
    new_area = new_masks.sum(axis=(1, 2))
    # Reject partially shredding a host instance, rather than silently creating
    # a large collection of tiny annotation fragments.
    if np.any((new_area > 0) & (new_area < old_area * 0.3)):
        return rgb, depth, masks, flags, False
    keep = new_area > 0
    rgb[y:y+h, x:x+w][visible] = donor_rgb[visible]
    depth[y:y+h, x:x+w][visible] = donor_depth[visible]
    return rgb, depth, np.concatenate([new_masks[keep], full[None]]), np.r_[flags[keep], True], True


class ExperimentMapper(RGBDMapper):
    def __init__(self, cfg, training=True, cache="data_cache"):
        super().__init__(cfg, training)
        self.cfg = cfg
        self.records = []
        if training and cfg.COURSE.COPY_PASTE_PROB:
            self.records = json.loads((Path(cache) / "index.json").read_text())["train"]
            assert all(x["split"] == "train" for x in self.records)
            stats = json.loads((Path(cache) / "train_small_stats.json").read_text())["records"]
            assert [r["image_id"] for r in stats] == [r["image_id"] for r in self.records]
            self.records = [r for r, s in zip(self.records, stats) if s["small"]]
            assert self.records, "No small-instance donor images in train"

    def __call__(self, record):
        result = copy.deepcopy(record)
        rgb, depth = read_rgbd(record)
        if not self.training:
            size = self.cfg.COURSE.TEST_SIZE
            if size != rgb.shape[0]:
                rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))
                depth = np.asarray(Image.fromarray(depth).resize((size, size), Image.Resampling.NEAREST))
            result["image"] = torch.from_numpy(np.ascontiguousarray(
                np.concatenate([rgb.astype(np.float32), depth[..., None]], -1).transpose(2, 0, 1)))
            # Keep original height/width: official inference restores masks to 1024.
            return result
        assert record["split"] == "train", "Never augment val/test as training data"
        rgb = rgb.copy()  # PIL-backed np.asarray is read-only; pasting owns this buffer.
        masks, flags = decode_annotations(self.reader.get(record), rgb.shape[:2], self.cfg.COURSE.SMALL_AREA)
        accepted = 0
        if self.records and np.random.random() < self.cfg.COURSE.COPY_PASTE_PROB:
            donor = self.records[np.random.randint(len(self.records))]
            drgb, dd = read_rgbd(donor)
            dm, df = decode_annotations(self.reader.get(donor), drgb.shape[:2], self.cfg.COURSE.SMALL_AREA)
            candidates = np.flatnonzero(df)
            np.random.shuffle(candidates)
            for j in candidates[:self.cfg.COURSE.COPY_PASTE_MAX]:
                ys, xs = np.where(dm[j])
                if not len(xs):
                    continue
                sy, sx = slice(ys.min(), ys.max()+1), slice(xs.min(), xs.max()+1)
                h, w = dm[j, sy, sx].shape
                donor_valid = dm[j] & (dd > 0)
                if not donor_valid.any():
                    continue
                median = float(np.median(dd[donor_valid]))
                receiver_y, receiver_x = np.where((depth > 0) & (depth >= median) & (depth - median <= 25.5))
                if not len(receiver_y):
                    continue
                for attempt in range(20):
                    position = np.random.randint(len(receiver_y))
                    px = int(np.clip(receiver_x[position] - w//2, 0, rgb.shape[1]-w))
                    py = int(np.clip(receiver_y[position] - h//2, 0, rgb.shape[0]-h))
                    rgb, depth, masks, flags, ok = paste_instance(
                        rgb, depth, masks, flags, drgb[sy, sx], dd[sy, sx], dm[j, sy, sx],
                        px, py)
                    if ok:
                        accepted += 1
                        break
        transform_list = []
        if flags.any() and np.random.random() < self.cfg.COURSE.CROP_PROB:
            chosen = np.random.choice(np.flatnonzero(flags))
            ys, xs = np.where(masks[chosen])
            side = self.cfg.COURSE.CROP_SIZE
            location = np.random.randint(len(xs))
            cx, cy = xs[location], ys[location]
            x0 = int(np.clip(cx - side * np.random.uniform(.35, .65), 0, rgb.shape[1]-side))
            y0 = int(np.clip(cy - side * np.random.uniform(.35, .65), 0, rgb.shape[0]-side))
            crop = T.CropTransform(x0, y0, side, side)
            transform_list.append(crop)
            rgb = crop.apply_image(rgb)
        aug_input = T.AugInput(rgb)
        transform_list.extend(self.augmentations(aug_input).transforms)
        transforms = T.TransformList(transform_list)
        rgb = aug_input.image
        depth = transforms.apply_segmentation(depth)
        masks = np.stack([transforms.apply_segmentation(m.astype(np.uint8)).astype(bool) for m in masks]) if len(masks) else np.zeros((0, *rgb.shape[:2]), bool)
        keep = masks.sum(axis=(1, 2)) > 0
        instances = Instances(rgb.shape[:2])
        instances.gt_masks = BitMasks(torch.from_numpy(np.ascontiguousarray(masks[keep])))
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances.gt_classes = torch.zeros(int(keep.sum()), dtype=torch.int64)
        instances.gt_small = torch.from_numpy(flags[keep].copy())
        instances.gt_masks = instances.gt_masks.tensor
        result["image"] = torch.from_numpy(np.ascontiguousarray(
            np.concatenate([rgb.astype(np.float32), depth[..., None]], -1).transpose(2, 0, 1)))
        result["instances"] = instances
        result["copy_paste_accepted"] = accepted
        return result

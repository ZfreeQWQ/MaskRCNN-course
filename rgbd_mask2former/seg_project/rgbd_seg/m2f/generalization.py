"""Opt-in small/occluded-instance fine tuning; existing training code stays frozen.

The auxiliary loss adds foreground, boundary and local-background samples to the
last ordinary-query layer. It never replaces standard point losses or matching,
and never supervises the GT-piloted branch twice. This is a project experiment,
not a reproduction of PointRend or SAHI.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from detectron2.config import CfgNode as CN
from detectron2.data import build_detection_train_loader, detection_utils as utils, transforms as T
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.engine import HookBase
from detectron2.structures import BitMasks, BoxMode
from detectron2.utils import comm
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.projects.point_rend.point_features import point_sample

from .config import get_cfg
from .geo_transforms import RandomRot90
from .mapper import RGBDInstanceDatasetMapper
from .mpformer import MPCriterion, RGBDMPFormer
from .algorithm_trainer import AlgorithmTrainer
from .round2_trainer import Round2Trainer


def setup_cfg(config_file, opts=()):
    cfg = get_cfg()
    cfg.PROJECT.GENERALIZATION = CN()
    knobs = cfg.PROJECT.GENERALIZATION
    knobs.CROP_PROBABILITY = 0.0
    knobs.CROP_SIZES = [640, 768, 896]
    knobs.LOCAL_LOSS = False
    knobs.LOCAL_WEIGHT = 0.5
    knobs.MAX_LOCAL_PAIRS = 12
    cfg.merge_from_file(str(config_file))
    cfg.merge_from_list(list(opts))
    if cfg.DATASETS.TRAIN != ("blender_train",) or cfg.DATASETS.TEST != ("blender_val",):
        raise ValueError("Only train supervision and val selection are allowed")
    if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES != 1:
        raise ValueError("Expected the single component category")
    k = cfg.PROJECT.GENERALIZATION
    if not 0 <= k.CROP_PROBABILITY <= 1 or not k.CROP_SIZES or min(k.CROP_SIZES) < 64:
        raise ValueError("Invalid focus-crop policy")
    if k.LOCAL_WEIGHT <= 0 or k.MAX_LOCAL_PAIRS < 1:
        raise ValueError("Invalid local-supervision budget")
    if cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED or cfg.PROJECT.HIRES_MASK.ENABLED:
        raise ValueError("Existing S/H branches must stay disabled in these experiments")
    cfg.freeze()
    return cfg


def containing_crop(box, height, width, sizes, rng=np.random):
    """Random square crop containing the complete visible instance plus context."""
    x0, y0, x1, y1 = [float(v) for v in box]
    choices = [int(s) for s in sizes if s <= min(height, width)
               and s >= max(x1 - x0, y1 - y0) + 16]
    if not choices:
        return None
    side = int(rng.choice(choices))
    xmin, xmax = max(0, math.ceil(x1 - side)), min(math.floor(x0), width - side)
    ymin, ymax = max(0, math.ceil(y1 - side)), min(math.floor(y0), height - side)
    if xmax < xmin or ymax < ymin:
        return None
    return (int(rng.randint(xmin, xmax + 1)), int(rng.randint(ymin, ymax + 1)), side)


class FocusCropMapper(RGBDInstanceDatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.focus_probability = cfg.PROJECT.GENERALIZATION.CROP_PROBABILITY if is_train else 0
        self.focus_sizes = list(cfg.PROJECT.GENERALIZATION.CROP_SIZES)
        augs = []
        if cfg.INPUT.RANDOM_FLIP in ("horizontal", "both"):
            augs.append(T.RandomFlip(horizontal=True, vertical=False))
        if cfg.INPUT.RANDOM_FLIP in ("vertical", "both"):
            augs.append(T.RandomFlip(horizontal=False, vertical=True))
        if cfg.INPUT.ROTATE_90:
            augs.append(RandomRot90())
        # Exact resize prevents a subsequent random crop discarding the focus GT.
        augs.append(T.Resize((cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)))
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=self.img_format))
        self.focus_augs = T.AugmentationList(augs)

    def _focus(self, record):
        candidates, weights = [], []
        for ann in record.get("annotations", []):
            if ann.get("iscrowd", 0) or not isinstance(ann.get("segmentation"), dict):
                continue
            rle = ann["segmentation"]
            if isinstance(rle["counts"], list):
                rle = mask_utils.frPyObjects(rle, record["height"], record["width"])
            area = float(mask_utils.area(rle))
            box = BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYXY_ABS)
            fill = area / max((box[2] - box[0]) * (box[3] - box[1]), 1)
            if 0 < area <= 1024 or (0 < area <= 4096 and fill < .25):
                crop = containing_crop(box, record["height"], record["width"], self.focus_sizes)
                if crop is not None:
                    candidates.append(crop)
                    weights.append(min(4., math.sqrt(1024. / max(area, 16.))) * (1.5 if fill < .25 else 1.))
        if not candidates:
            return None
        weights = np.asarray(weights)
        return candidates[int(np.random.choice(len(candidates), p=weights / weights.sum()))]

    def __call__(self, record):
        if not self.is_train or np.random.random() >= self.focus_probability:
            result = super().__call__(record)
            result["focus_crop"] = False
            return result
        crop = self._focus(record)
        if crop is None:
            result = super().__call__(record)
            result["focus_crop"] = False
            return result
        record = copy.deepcopy(record)
        image = utils.read_image(str(self._path(record["file_name"])), format=self.img_format)
        utils.check_image_size(record, image)
        # Encode depth at NATIVE resolution before any resampling/cropping.
        depth = self._depth_geometry_native(record) if self.use_depth else None
        x0, y0, side = crop
        first = T.CropTransform(x0, y0, side, side)
        aug_input = T.AugInput(first.apply_image(image))
        rest = self.focus_augs(aug_input)
        transforms = T.TransformList([first] + rest.transforms)
        image = aug_input.image.astype(np.float32)
        if depth is not None:
            image = np.concatenate([image, transforms.apply_segmentation(depth)], axis=2)
        record["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        annos = [utils.transform_instance_annotations(a, transforms, image.shape[:2])
                 for a in record.pop("annotations", []) if not a.get("iscrowd", 0)]
        instances = utils.annotations_to_instances(annos, image.shape[:2], mask_format=self.mask_format)
        if instances.has("gt_masks"):
            # Crop can remove box extremities: keep boxes consistent with visible masks.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = utils.filter_empty_instances(instances)
        if instances.has("gt_masks") and isinstance(instances.gt_masks, BitMasks):
            instances.gt_masks = instances.gt_masks.tensor
        record["instances"] = instances
        record["focus_crop"] = True
        return record


@torch.no_grad()
def local_coordinates(mask, foreground=96, boundary=96, background=64):
    """Pixel-center foreground/boundary samples plus uniform expanded-box context.

    No resizing of GT before sampling: even single-pixel disconnected pieces can
    be sampled. Coordinates use align_corners=False in the padded training image.
    """
    height, width = mask.shape
    pixels = (mask > .5).nonzero()
    if not len(pixels):
        raise ValueError("Cannot sample an empty mask")
    low, high = pixels.amin(0), pixels.amax(0)
    y0, x0 = (low - 3).clamp(min=0).tolist()
    y1, x1 = torch.minimum(high + 4, high.new_tensor([height, width])).tolist()
    patch = mask[y0:y1, x0:x1][None, None].float()
    padded = F.pad(patch, (1, 1, 1, 1))
    dilated = F.max_pool2d(padded, 3, 1)
    eroded = -F.max_pool2d(-padded, 3, 1)
    band = ((dilated - eroded)[0, 0] > 0).nonzero()
    band = band + pixels.new_tensor([y0, x0])
    if not len(band):
        band = pixels
    selected = torch.cat([pixels[torch.randint(len(pixels), (foreground,), device=mask.device)],
                          band[torch.randint(len(band), (boundary,), device=mask.device)]])
    # prepare_targets preserves boolean GT; normalization constants must be float.
    scale = torch.tensor([width, height], dtype=torch.float32, device=mask.device)
    xy = (selected[:, [1, 0]].float() + .5) / scale
    low_xy = torch.tensor([x0, y0], dtype=torch.float32, device=mask.device)
    extent = torch.tensor([x1 - x0, y1 - y0], dtype=torch.float32, device=mask.device)
    context = (low_xy + torch.rand(background, 2, device=mask.device) * extent) / scale
    return torch.cat([xy, context])


class LocalMPCriterion(MPCriterion):
    def __init__(self, *args, max_pairs=12, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pairs = max_pairs
        self.capture = False
        self.main_indices = None
        self.last_local_stats = None

    def loss_masks(self, outputs, targets, indices, num_masks):
        if self.capture and self.main_indices is None:
            self.main_indices = indices
        return super().loss_masks(outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        self.capture, self.main_indices = True, None
        try:
            losses = super().forward(outputs, targets)
            indices = self.main_indices
        finally:
            self.capture = False
            self.main_indices = None
        if indices is None:
            raise RuntimeError("Main-layer matching was not captured")
        selected = []
        # Limit candidate examination as well as differentiable sample count.
        candidates = []
        for batch, (src, tgt) in enumerate(indices):
            for query, gt in zip(src.tolist(), tgt.tolist()):
                mask = targets[batch]["masks"][gt]
                area = float(mask.sum())
                if 0 < area <= 16384:
                    candidates.append((batch, query, mask, area))
        if candidates:
            order = torch.randperm(len(candidates)).tolist()
            for pos in order:
                batch, query, mask, area = candidates[pos]
                if area > 4096:
                    pixels = (mask > .5).nonzero()
                    extent = pixels.amax(0) - pixels.amin(0) + 1
                    if area / float(extent.prod()) >= .25:
                        continue
                selected.append((batch, query, mask))
                if len(selected) >= self.max_pairs:
                    break
        count = outputs["pred_masks"].new_tensor([len(selected)])
        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(count)
        normalizer = max(float(count / comm.get_world_size()), 1.)
        zero = outputs["pred_masks"].sum() * 0
        if not selected:
            losses.update(loss_local_bce=zero, loss_local_dice=zero.clone())
        else:
            coords = torch.stack([local_coordinates(mask) for _, _, mask in selected])
            pred = torch.stack([outputs["pred_masks"][b, q] for b, q, _ in selected])
            gt = torch.stack([mask for _, _, mask in selected]).float()
            logits = point_sample(pred[:, None], coords, align_corners=False)[:, 0]
            with torch.no_grad():
                labels = point_sample(gt[:, None], coords, align_corners=False)[:, 0]
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(1)
            probability = logits.sigmoid()
            dice = 1 - (2 * (probability * labels).sum(1) + 1) / (probability.sum(1) + labels.sum(1) + 1)
            losses.update(loss_local_bce=bce.sum() / normalizer, loss_local_dice=dice.sum() / normalizer)
        self.last_local_stats = {"selected_pairs": len(selected), "global_selected_pairs": int(count),
                                 "points_per_pair": 256, "ordinary_final_layer_only": True,
                                 "foreground_label_mean": float(labels[:, :96].mean()) if selected else None,
                                 "sampled_logits_abs_mean": float(logits.detach().abs().mean()) if selected else None}
        return losses


@META_ARCH_REGISTRY.register()
class GeneralizationMPFormer(RGBDMPFormer):
    @classmethod
    def from_config(cls, cfg):
        result = super().from_config(cfg)
        if not cfg.PROJECT.GENERALIZATION.LOCAL_LOSS:
            return result
        old = result["criterion"]
        weights = dict(old.weight_dict)
        weights.update(loss_local_bce=cfg.PROJECT.GENERALIZATION.LOCAL_WEIGHT,
                       loss_local_dice=cfg.PROJECT.GENERALIZATION.LOCAL_WEIGHT)
        result["criterion"] = LocalMPCriterion(
            old.num_classes, old.matcher, weights, old.eos_coef, old.losses,
            old.num_points, old.oversample_ratio, old.importance_sample_ratio,
            max_pairs=cfg.PROJECT.GENERALIZATION.MAX_LOCAL_PAIRS)
        return result

    def forward(self, batched_inputs):
        if self.training:
            self.last_generalization_stats = {
                "images": len(batched_inputs),
                "focus_crops": sum(bool(r.get("focus_crop")) for r in batched_inputs),
                "instances": sum(len(r["instances"]) for r in batched_inputs),
            }
        return super().forward(batched_inputs)


class GeneralizationDiagnostics(HookBase):
    def after_step(self):
        if (self.trainer.iter + 1) % 20 and self.trainer.iter + 1 != self.trainer.max_iter:
            return
        import json
        from pathlib import Path
        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        report = dict(model.last_generalization_stats, rank=comm.get_rank(),
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        report["local_supervision"] = getattr(model.criterion, "last_local_stats", None)
        records = comm.gather(report, dst=0)
        if comm.is_main_process():
            with (Path(self.trainer.cfg.OUTPUT_DIR) / "generalization_diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps({"completed_iterations": self.trainer.iter + 1, "ranks": records}) + "\n")


class GeneralizationTrainer(AlgorithmTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(0, GeneralizationDiagnostics())
        return hooks

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=FocusCropMapper(cfg, is_train=True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        # Frequency/label modules are already trained. Use the same mature-module
        # LR in every arm, rather than the earlier new-module 10x multiplier.
        return Round2Trainer.build_optimizer(cfg, model)

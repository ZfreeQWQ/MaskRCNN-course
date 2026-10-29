"""COCO-style segmentation evaluation on explicitly labelled NYUv2 pixels.

This is a custom DEVELOPMENT protocol, not official NYUv2 or standard COCO
scores. Ground truth must already exclude unlabelled pixels. Predictions are
intersected with the original-image ``label_valid_file`` from each annotation
image entry; masks that become empty are removed. Scores and class labels are
not recomputed. COCO's IoU thresholds, visible-GT-mask area ranges, maxDets and
matching otherwise remain unchanged. Prediction mask area is the clipped RLE
area, not its old bounding-box area. Never compare these values directly with
standard COCO/NYUv2 tables, or compare RGB and RGB-D with different protocols.

This metric does not fix training's remaining query-level no-object supervision
on unannotated objects. That separate limitation is documented in void_head.py.
"""
from copy import deepcopy
from pathlib import Path

import mmcv
import numpy as np
import torch

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def clip_masks_to_valid(masks, valid):
    """Return clipped binary masks and keep flags; never mutate caller data."""
    masks = _numpy(masks)
    valid = np.asarray(valid)
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError('Original label_valid must be a two-dimensional bool array')
    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(valid.shape):
        raise ValueError(f'Prediction masks {masks.shape} must have original shape [N,{valid.shape[0]},{valid.shape[1]}]')
    if not np.isfinite(masks).all() or not np.isin(masks, [0, 1]).all():
        raise ValueError('Prediction masks must already be binary, not probabilities or RLE')
    clipped = masks.astype(bool) & valid[None]
    keep = clipped.reshape(clipped.shape[0], valid.size).any(axis=1)
    return clipped[keep], keep


@METRICS.register_module()
class VoidAwareCocoMetric(CocoMetric):
    """Original-coordinate void masking with an explicitly distinct prefix."""

    default_prefix = 'nyuv2_void'

    def __init__(self, ann_file, data_root, metric='segm', **kwargs):
        if not ann_file:
            raise ValueError('Void-aware evaluation requires an exported ann_file')
        if metric not in ('segm', ['segm']):
            raise ValueError('VoidAwareCocoMetric supports segm only, not bbox/proposals')
        if kwargs.get('backend_args') is not None:
            raise ValueError('This development metric requires local/NFS data_root files')
        self.data_root = Path(data_root).resolve()
        super().__init__(ann_file=ann_file, metric=metric, **kwargs)
        self._validated_gt = set()

    def _original_valid(self, image_id):
        if image_id not in self._coco_api.imgs:
            raise ValueError(f'Prediction image {image_id} is absent from ann_file')
        image = self._coco_api.imgs[image_id]
        if 'label_valid_file' not in image:
            raise ValueError(f'Image {image_id} has no label_valid_file')
        relative = Path(image['label_valid_file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('label_valid_file must be relative to data_root')
        path = (self.data_root / relative).resolve()
        if not path.is_relative_to(self.data_root):
            raise ValueError('label_valid_file must remain inside data_root')
        raw = (np.load(path, allow_pickle=False) if path.suffix.lower() == '.npy'
               else mmcv.imread(str(path), flag='unchanged'))
        if raw is None or raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError(f'Unreadable/nonfinite 2-D label_valid_file: {path}')
        expected = (image['height'], image['width'])
        if raw.shape != expected:
            raise ValueError(f'Original label_valid {raw.shape} != annotation image {expected}')
        valid = raw > 0
        if image_id not in self._validated_gt:
            for ann in self._coco_api.imgToAnns.get(image_id, []):
                mask = self._coco_api.annToMask(ann).astype(bool)
                if np.any(mask & ~valid):
                    raise ValueError(f'GT annotation {ann["id"]} contains unlabelled pixels')
                if abs(float(ann['area']) - int(mask.sum())) > 1e-4:
                    raise ValueError(f'GT annotation {ann["id"]} area is not visible mask area')
            self._validated_gt.add(image_id)
        return valid

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            # MMEngine Evaluator normally converts DetDataSample to a dict.
            # Accept the object too, useful for standalone pipeline checks.
            if hasattr(sample, 'to_dict'):
                sample = sample.to_dict()
            image_id = int(sample['img_id'])
            valid = self._original_valid(image_id)
            if tuple(sample['ori_shape'][:2]) != tuple(valid.shape):
                raise ValueError('ori_shape disagrees with original annotation coordinates')
            pred = sample['pred_instances']
            if hasattr(pred, 'to_dict'):
                pred = pred.to_dict()
            if 'masks' not in pred:
                raise ValueError('Void-aware segmentation evaluation requires predicted masks')
            masks, keep = clip_masks_to_valid(pred['masks'], valid)
            result = dict(img_id=image_id, masks=encode_mask_results(masks))
            for field in ('bboxes', 'scores', 'labels', 'mask_scores'):
                if field in pred:
                    values = _numpy(pred[field])
                    if values.shape[0] != keep.shape[0]:
                        raise ValueError(f'Prediction {field} count disagrees with masks')
                    result[field] = values[keep]
                elif field != 'mask_scores':
                    raise ValueError(f'Missing predicted {field}')
            gt = dict(img_id=image_id, width=valid.shape[1], height=valid.shape[0])
            self.results.append((gt, result))

    def compute_metrics(self, results):
        if not results or self.format_only or any(len(item[1]['scores']) for item in results):
            return super().compute_metrics(results)
        # CocoMetric's loadRes([]) catches IndexError and returns {}. Explicitly
        # evaluate an empty detection set so a completely failed model reports
        # AP=0 for represented GT ranges, and -1 for absent ranges, not silence.
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()
        empty = COCO()
        empty.dataset = dict(images=deepcopy(self._coco_api.dataset['images']),
                             categories=deepcopy(self._coco_api.dataset['categories']), annotations=[])
        empty.createIndex()
        evaluation = COCOeval(self._coco_api, empty, 'segm')
        evaluation.params.catIds = self.cat_ids
        evaluation.params.imgIds = self.img_ids
        evaluation.params.maxDets = list(self.proposal_nums)
        evaluation.params.iouThrs = self.iou_thrs
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        indices = dict(mAP=0, mAP_50=1, mAP_75=2, mAP_s=3, mAP_m=4, mAP_l=5,
                       **{'AR@100': 6, 'AR@300': 7, 'AR@1000': 8,
                          'AR_s@1000': 9, 'AR_m@1000': 10, 'AR_l@1000': 11})
        items = self.metric_items or ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
        if any(item not in indices for item in items):
            raise KeyError('Unsupported metric_items for COCO-style evaluation')
        result = {f'segm_{item}': float(f'{evaluation.stats[indices[item]]:.3f}') for item in items}
        if self.classwise:
            for index, category in enumerate(self.cat_ids):
                precision = evaluation.eval['precision'][:, :, index, 0, -1]
                values = precision[precision > -1]
                name = self._coco_api.cats[category]['name']
                result[f'{name}_precision'] = round(float(values.mean()), 3) if values.size else float('nan')
        return result

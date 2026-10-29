"""Opt-in, train-only small-instance Copy-Paste with aligned metric depth.

Insert immediately after LoadAnnotations, *before* any flip/resize/crop. Donor
images and masks come solely from the very same course train annotation file
as the destination dataset. This is deliberately conservative: no depth shift,
resizing, partial donor mask, RGB-only paste or guessed depth is permitted.

An accepted placement needs valid measured depth on every donor/target mask
pixel, a donor 90th depth percentile in front of the target 10th percentile,
and a per-pixel front-depth check. This reduces obvious depth contradictions;
it does not prove physical scene plausibility or accuracy improvement.

Pack ``course_copy_paste_activity`` in PackRGBDDetInputs.meta_keys to aggregate
per-sample counts in the main-process hook. Worker-local counters are not used
as evidence. The disabled transform returns the exact input object without
filesystem access, RNG draws, or metadata changes.
"""

import json
from pathlib import Path

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks
from pycocotools import mask as mask_utils

from rgbd.data import _check_aligned


def _under(root, value, prefix):
    """Reject unsafe/cross-split paths before opening any referenced file."""
    relative = Path(value)
    target = (root / relative).resolve()
    if (relative.is_absolute() or '..' in relative.parts
            or relative.parts[:len(prefix)] != tuple(prefix)
            or not target.is_relative_to(root.joinpath(*prefix))):
        raise ValueError('Copy-Paste paths must remain inside the same train root')
    return target


def _new_activity():
    return dict(calls=1, probability_skipped=0, donor_attempts=0,
                eligible_objects=0, invalid_depth_objects=0,
                placement_attempts=0, depth_rejected_placements=0,
                pasted_instances=0, removed_instances=0, cap_skipped=0,
                donor_image_ids=[], donor_annotation_ids=[])


@TRANSFORMS.register_module()
class CourseRGBDSmallCopyPaste(BaseTransform):
    """Bounded uniform small-donor sampling from the official train split.

    ``ann_file`` and ``data_root`` must be copied from the enclosing training
    dataset config. Only ``annotations/instances_train.json`` or the explicit
    ``annotations/smoke_train.json`` short-validation index is accepted. Each
    destination's ID, RGB/depth paths and original geometry are checked against
    that index; accidentally inserting this transform into val fails closed.

    A chosen object keeps its original size and depth. Up to ``max_paste``
    successful translations are made without scaling. All destination masks
    are clipped by the visible pasted mask; fully occluded masks are removed
    and all boxes are recomputed. The count limit is checked before mutation.
    """

    def __init__(self, enabled=False, data_root=None, ann_file=None, prob=.5,
                 max_paste=3, area_threshold=1024, max_instances=100,
                 max_donor_attempts=12, max_position_attempts=8,
                 front_margin_m=.001):
        if type(enabled) is not bool:
            raise TypeError('Copy-Paste enabled must be a boolean')
        self.enabled = enabled
        self.data_root = data_root
        self.ann_file = ann_file
        self.prob = prob
        self.max_paste = max_paste
        self.area_threshold = area_threshold
        self.max_instances = max_instances
        self.max_donor_attempts = max_donor_attempts
        self.max_position_attempts = max_position_attempts
        self.front_margin_m = front_margin_m
        self._images = None
        self._candidates = None
        self._root = None
        if (not np.isfinite(prob) or not 0 <= prob <= 1
                or not np.isfinite(front_margin_m) or front_margin_m < 0):
            raise ValueError('Invalid Copy-Paste probability/depth margin')
        for name, value, limit in (
                ('max_paste', max_paste, 3), ('area_threshold', area_threshold, 1024),
                ('max_instances', max_instances, 100),
                ('max_donor_attempts', max_donor_attempts, 64),
                ('max_position_attempts', max_position_attempts, 32)):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'Invalid bounded Copy-Paste {name}')

    def _load_index(self):
        if self._images is not None:
            return
        if self.data_root is None or self.ann_file is None:
            raise ValueError('Enabled Copy-Paste requires its training root/index')
        root = Path(self.data_root).resolve()
        ann = Path(self.ann_file)
        # MMEngine may resolve ann_file before constructing the pipeline.
        if ann.is_absolute():
            try:
                ann = ann.relative_to(root)
            except ValueError as exc:
                raise ValueError('Copy-Paste annotation must be in the train root') from exc
        if ann.as_posix() not in ('annotations/instances_train.json',
                                  'annotations/smoke_train.json'):
            raise ValueError('Copy-Paste accepts only the same explicit train annotation index')
        ann_path = _under(root, str(ann), ('annotations',))
        annotation = json.loads(ann_path.read_text())
        if annotation['categories'] != [dict(id=1, name='component', supercategory='component')]:
            raise ValueError('Copy-Paste supports only the course component category')
        images = {}
        for image in annotation['images']:
            image = dict(image)
            ident = image['id']
            if ident in images or image.get('split', 'train') != 'train':
                raise ValueError('Duplicate or non-train donor image')
            image['_rgb'] = _under(root, image['file_name'], ('rgb', 'train'))
            image['_depth'] = _under(root, image['depth_file_name'], ('depth', 'train'))
            if (image['width'] <= 0 or image['height'] <= 0
                    or not np.isfinite(image['depth_scale_m']) or image['depth_scale_m'] <= 0):
                raise ValueError('Invalid train donor geometry/depth scale')
            images[ident] = image
        candidates = []
        seen = set()
        for obj in annotation['annotations']:
            if (obj['id'] in seen or obj['image_id'] not in images
                    or obj['category_id'] != 1):
                raise ValueError('Invalid donor annotation identity/category')
            seen.add(obj['id'])
            if not obj.get('iscrowd', 0) and not obj.get('ignore', False):
                if not np.isfinite(obj['area']) or obj['area'] <= 0:
                    raise ValueError('Invalid donor mask area')
                if obj['area'] <= self.area_threshold:
                    candidates.append(obj)
        self._root, self._images, self._candidates = root, images, candidates

    def _check_destination(self, results):
        _check_aligned(results)
        image = self._images.get(results.get('img_id'))
        if image is None:
            raise ValueError('Copy-Paste destination is not in the same train index')
        if (Path(results['img_path']).resolve() != image['_rgb']
                or Path(results['depth_file']).resolve() != image['_depth']):
            raise ValueError('Copy-Paste destination paths do not match the train index')
        if (results.get('depth_scale_m') != image['depth_scale_m']
                or results.get('depth_invalid_code') != image['depth_invalid_code']):
            raise ValueError('Copy-Paste destination depth interpretation differs from the train index')
        shape = (image['height'], image['width'])
        if (results['img'].shape[:2] != shape or results.get('homography_matrix') is not None
                or tuple(results.get('scale_factor', (1., 1.))) != (1., 1.)
                or results.get('flip', False)):
            raise ValueError('Copy-Paste must run in original coordinates before all geometry')
        if not isinstance(results['gt_masks'], BitmapMasks):
            raise TypeError('Copy-Paste requires loaded BitmapMasks')
        masks = results['gt_masks'].masks
        count = len(masks)
        if masks.shape != (count, *shape) or count > self.max_instances:
            raise ValueError('Invalid destination mask geometry/instance cap')
        for key in ('gt_bboxes', 'gt_bboxes_labels', 'gt_ignore_flags'):
            if len(results[key]) != count:
                raise ValueError('Destination instance fields have inconsistent cardinality')
        if np.any(results['gt_bboxes_labels'] != 0):
            raise ValueError('Destination category is not the course component')
        if not np.all(results['rgbd_label_valid']):
            raise ValueError('Course Copy-Paste requires full label-valid supervision')
        if 'gt_instances_ids' in results:
            ids = results['gt_instances_ids']
            if len(ids) != count or not np.issubdtype(ids.dtype, np.signedinteger) or len(set(ids.tolist())) != count:
                raise ValueError('Destination instance IDs must be unique signed integers')

    def _load_object(self, obj):
        image = self._images[obj['image_id']]
        h, w = image['height'], image['width']
        segmentation = obj['segmentation']
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, h, w))
        elif isinstance(segmentation.get('counts'), list):
            rle = mask_utils.frPyObjects(segmentation, h, w)
        else:
            rle = segmentation
        mask = mask_utils.decode(rle).astype(bool)
        if mask.shape != (h, w) or not 1 <= mask.sum() <= self.area_threshold:
            raise ValueError('Donor decoded mask is inconsistent with the small-object contract')
        if abs(float(mask.sum()) - obj['area']) > 1e-6:
            raise ValueError('Donor annotation area does not match its decoded mask')
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgb = mmcv.imread(str(image['_rgb']), flag='color')
        raw = mmcv.imread(str(image['_depth']), flag='unchanged')
        if rgb is None or rgb.shape != (h, w, 3) or raw is None or raw.shape != (h, w) or raw.dtype != np.uint16:
            raise ValueError('Donor RGB/depth files are invalid or misaligned')
        valid = (raw != image['depth_invalid_code']) & (raw > 0)
        depth = np.where(valid, raw.astype(np.float32) * image['depth_scale_m'], 0).astype(np.float32)
        valid &= np.isfinite(depth) & (depth > 0)
        sl = np.s_[y0:y1, x0:x1]
        return dict(img=rgb[sl], rgbd_depth=depth[sl], rgbd_raw_valid=valid[sl],
                    rgbd_label_valid=np.ones((y1-y0, x1-x0), dtype=bool), mask=mask[sl])

    def _placement_is_front(self, results, donor, y, x):
        mask = donor['mask']
        h, w = mask.shape
        depth = results['rgbd_depth'][y:y+h, x:x+w][mask]
        valid = results['rgbd_raw_valid'][y:y+h, x:x+w][mask]
        foreground = donor['rgbd_depth'][mask]
        if not np.all(valid & np.isfinite(depth) & (depth > 0)):
            return False
        return (np.percentile(foreground, 90) + self.front_margin_m < np.percentile(depth, 10)
                and np.all(foreground + self.front_margin_m < depth))

    def _paste(self, results, donor, y, x, activity, obj):
        """Only called after validity/geometry tests; return False at the cap."""
        height, width = results['img'].shape[:2]
        dh, dw = donor['mask'].shape
        visible = np.zeros((height, width), dtype=bool)
        visible[y:y+dh, x:x+dw] = donor['mask']
        current = results['gt_masks'].masks.astype(bool)
        clipped = current & ~visible[None]
        keep = clipped.reshape(len(current), height * width).any(axis=1)
        if int(keep.sum()) + 1 > self.max_instances:
            activity['cap_skipped'] += 1
            return False
        new_masks = np.concatenate([clipped[keep], visible[None]], axis=0).astype(np.uint8)
        for key in ('img', 'rgbd_depth', 'rgbd_raw_valid', 'rgbd_label_valid'):
            region = results[key][y:y+dh, x:x+dw]
            region[donor['mask']] = donor[key][donor['mask']]
        masks = BitmapMasks(new_masks, height, width)
        results['gt_masks'] = masks
        boxes = results['gt_bboxes']
        if isinstance(boxes, np.ndarray):
            results['gt_bboxes'] = masks.get_bboxes('hbox').tensor.numpy().astype(boxes.dtype)
        else:
            new_boxes = masks.get_bboxes(type(boxes))
            results['gt_bboxes'] = new_boxes.to(device=boxes.device, dtype=boxes.dtype)
        for key, value in (('gt_bboxes_labels', 0), ('gt_ignore_flags', False)):
            source = results[key]
            results[key] = np.concatenate([source[keep], np.array([value], dtype=source.dtype)])
        if 'gt_instances_ids' in results:
            ids = results['gt_instances_ids']
            fresh_id = min(-1, int(ids.min()) - 1) if len(ids) else -1
            if fresh_id < np.iinfo(ids.dtype).min:
                raise ValueError('Cannot allocate a unique synthetic instance ID')
            results['gt_instances_ids'] = np.concatenate([ids[keep], np.array([fresh_id], dtype=ids.dtype)])
        activity['pasted_instances'] += 1
        activity['removed_instances'] += int((~keep).sum())
        activity['donor_image_ids'].append(int(obj['image_id']))
        activity['donor_annotation_ids'].append(int(obj['id']))
        return True

    def transform(self, results):
        if not self.enabled:
            return results
        self._load_index()
        self._check_destination(results)
        activity = _new_activity()
        results['course_copy_paste_activity'] = activity
        if self.prob == 0 or (self.prob < 1 and np.random.random() >= self.prob):
            activity['probability_skipped'] = 1
            return results
        if not self._candidates:
            return results
        # Bounded rejection sampling avoids allocating a copy of a large donor
        # list for every sample. Duplicate annotations are not pasted twice.
        attempted = set()
        height, width = results['img'].shape[:2]
        for _ in range(self.max_donor_attempts):
            if activity['pasted_instances'] >= self.max_paste:
                break
            obj = self._candidates[int(np.random.randint(len(self._candidates)))]
            activity['donor_attempts'] += 1
            if obj['image_id'] == results['img_id'] or obj['id'] in attempted:
                continue
            attempted.add(obj['id'])
            donor = self._load_object(obj)
            activity['eligible_objects'] += 1
            mask = donor['mask']
            dh, dw = mask.shape
            values = donor['rgbd_depth'][mask]
            if not np.all(donor['rgbd_raw_valid'][mask] & np.isfinite(values) & (values > 0)):
                activity['invalid_depth_objects'] += 1
                continue
            if dh > height or dw > width:
                continue
            for _ in range(self.max_position_attempts):
                activity['placement_attempts'] += 1
                y = int(np.random.randint(height-dh+1))
                x = int(np.random.randint(width-dw+1))
                if not self._placement_is_front(results, donor, y, x):
                    activity['depth_rejected_placements'] += 1
                    continue
                if self._paste(results, donor, y, x, activity, obj):
                    break
        return results

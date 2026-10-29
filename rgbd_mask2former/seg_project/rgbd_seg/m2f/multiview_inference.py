"""Prediction-only native-coordinate full/tile inference; no extra dependencies."""
import numpy as np
from pycocotools import mask as mask_util


def windows(height, width, size=640, overlap=.4):
    def axis(length):
        if length <= size:
            return [0]
        step = max(1, round(size * (1 - overlap)))
        return sorted(set(list(range(0, length - size + 1, step)) + [length - size]))
    return [(x, y, min(size, width-x), min(size, height-y))
            for y in axis(height) for x in axis(width)]


def touches_internal_edge(mask, window, shape, margin=3):
    x, y, w, h = window
    H, W = shape
    return bool((x > 0 and mask[:, :margin].any()) or
                (y > 0 and mask[:margin, :].any()) or
                (x+w < W and mask[:, -margin:].any()) or
                (y+h < H and mask[-margin:, :].any()))


def tile_rows(instances, image_id, window, shape, view, max_area=4096):
    """Reject cut-edge masks, restore native coordinates, preserve each identity."""
    x, y, w, h = window
    rows = []
    instances = instances.to('cpu')
    for index, local in enumerate(instances.pred_masks.numpy()):
        area = int(local.sum())
        if not 0 < area <= max_area or touches_internal_edge(local, window, shape):
            continue
        full = np.zeros(shape, dtype=np.uint8)
        full[y:y+h, x:x+w] = local
        rle = mask_util.encode(np.asfortranarray(full))
        box = mask_util.toBbox(rle).tolist()
        rle['counts'] = rle['counts'].decode('ascii')
        rows.append(dict(image_id=int(image_id), category_id=1, score=float(instances.scores[index]),
                         segmentation=rle, bbox=box, view=view))
    return rows


def merge_views(rows, tile_weight=.9, iou_threshold=.55, ios_threshold=.85, max_dets=100):
    """Cross-view mask NMS; overlapping bboxes alone never suppress an instance.

    No union/connected-component split: disjoint visible fragments retain their
    query's instance identity. Within-view predictions keep the original policy.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        return []
    for row in rows:
        row['score'] *= 1. if row.get('view', 'full') == 'full' else tile_weight
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]['score'], i))
    rles = [r['segmentation'] for r in rows]
    areas = np.asarray(mask_util.area(rles), dtype=np.float64)
    ious = mask_util.iou(rles, rles, [0] * len(rows))
    intersection = ious * (areas[:, None] + areas[None, :]) / (1 + ious)
    ios = intersection / np.maximum(np.minimum(areas[:, None], areas[None, :]), 1)
    kept = []
    for i in order:
        if areas[i] <= 0:
            continue
        if any(rows[i].get('view', 'full') != rows[j].get('view', 'full') and
               (ious[i, j] >= iou_threshold or ios[i, j] >= ios_threshold) for j in kept):
            continue
        kept.append(i)
        if len(kept) == max_dets:
            break
    return [{k: v for k, v in rows[i].items() if k != 'view'} for i in kept]


def resize_view(native, size, window=None):
    """Encode depth before slicing; bilinear RGB and nearest geometry, as mapper."""
    import torch
    from detectron2.data import transforms as T
    image = native['image'].numpy().transpose(1, 2, 0)
    H, W = image.shape[:2]
    x, y, w, h = window or (0, 0, W, H)
    image = image[y:y+h, x:x+w]
    rgb = image[:, :, :3].astype(np.uint8)
    transform = T.ResizeShortestEdge(size, size).get_transform(rgb)
    rgb = transform.apply_image(rgb).astype(np.float32)
    if image.shape[2] > 3:
        rgb = np.concatenate([rgb, transform.apply_segmentation(image[:, :, 3:])], axis=2)
    record = {k: native[k] for k in ('image_id',) if k in native}
    record.update(height=h, width=w, image=torch.as_tensor(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
    return record


class MultiViewPredictor:
    def __init__(self, cfg, model):
        from detectron2.data import transforms as T
        from .mapper import RGBDInstanceDatasetMapper
        self.mapper = RGBDInstanceDatasetMapper(cfg, is_train=False)
        self.mapper.augs = T.AugmentationList([])
        self.model = model
        self.cfg = cfg

    def __call__(self, record, variant='all', verify_reference=False):
        import torch
        from tools.common import encode_instances
        from .round2_inference import paired_prediction
        native = self.mapper(record)
        shape = (native['height'], native['width'])
        output, diagnostics = {}, {'image_id': int(record['image_id']), 'tile_candidates': 0}
        if variant in ('all', 'full2048'):
            prediction = paired_prediction(self.model, resize_view(native, 2048))
            output['full2048'] = encode_instances(prediction['instances'], record['image_id'])
        if variant in ('all', 'tiles090', 'tiles100'):
            full = paired_prediction(self.model, resize_view(native, 1536))['instances']
            if verify_reference:
                from .mapper import RGBDInstanceDatasetMapper
                reference_mapper = RGBDInstanceDatasetMapper(self.cfg, is_train=False)
                reference = paired_prediction(self.model, reference_mapper(record))['instances']
                if not torch.equal(full.pred_masks, reference.pred_masks):
                    raise AssertionError('Native/view full-image masks differ from original mapper')
                torch.testing.assert_close(full.scores, reference.scores, rtol=0, atol=0)
                diagnostics['full_view_matches_existing_inference'] = True
            rows = [dict(r, view='full') for r in encode_instances(full, record['image_id'])]
            del full
            for view, window in enumerate(windows(*shape)):
                pred = paired_prediction(self.model, resize_view(native, 1536, window))['instances']
                added = tile_rows(pred, record['image_id'], window, shape, str(view))
                diagnostics['tile_candidates'] += len(added)
                rows.extend(added)
                del pred
            for name, weight in (('tiles090', .9), ('tiles100', 1.)):
                if variant in ('all', name):
                    output[name] = merge_views(rows, tile_weight=weight)
        return output, diagnostics

"""Read-only train/val audit; writes new planning evidence, never starts training."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/analysis/optimization_audit_20261018.json'


def quantiles(values):
    return dict(zip(['p10', 'p25', 'p50', 'p75', 'p90', 'p95'],
                    np.percentile(values, [10, 25, 50, 75, 90, 95]).tolist()))


def geometry(anns):
    boxes = np.array([a['bbox'][2:] for a in anns], dtype=float)
    areas = np.array([a['area'] for a in anns])
    return {'count': len(anns), 'mask_area': quantiles(areas),
            'sqrt_bbox_area': quantiles(np.sqrt(boxes.prod(1))),
            'bbox_aspect_long_over_short': quantiles(boxes.max(1) / boxes.min(1)),
            'mask_fill_fraction': quantiles(areas / boxes.prod(1)),
            'bbox_short_le_4': int((boxes.min(1) <= 4).sum()),
            'aspect_gt_3': int((boxes.max(1) / boxes.min(1) > 3).sum()),
            'mask_area_le_64': int((areas <= 64).sum()),
            'sqrt_bbox_area_gt_32': int((boxes.prod(1) > 1024).sum()),
            'semantic_types': dict(Counter(a.get('semantic_type', 'unknown') for a in anns))}


result = {'scope': 'train/val only; no test input or labels read',
          'small_definition': 'GT mask area <= 1024, matching COCO small inclusive boundary',
          'splits': {}, 'runs': {}, 'coverage_diagnostics': {}}
model_ids = {}
for split in ['train', 'val']:
    data = json.loads((ROOT / f'datasets/blender/annotations/instances_{split}.json').read_text())
    anns, images = data['annotations'], data['images']
    small = [a for a in anns if a['area'] <= 1024]
    model_ids[split] = {str(a.get('model_id')) for a in anns}
    summary = {'images': len(images), 'all': geometry(anns), 'small': geometry(small),
               'image_metadata': {k: dict(Counter(str(im.get(k)) for im in images))
                                  for k in ['composition', 'density', 'renderer']},
               'unique_model_ids': len(model_ids[split])}
    samples = []
    for ix in np.linspace(0, len(images)-1, 24, dtype=int):
        im = images[ix]
        raw = np.asarray(Image.open(ROOT / 'datasets/blender' / im['depth_file_name']))
        rgb = np.asarray(Image.open(ROOT / 'datasets/blender' / im['file_name']), dtype=np.float32)
        valid = raw != im['depth_invalid_code']
        d = raw.astype(np.float32) * im['depth_scale_m']
        norm = (np.clip((d[valid] - .055) / (1.0256 - .055), 0, 1) - .5) / .25
        samples.append({'image_id': im['id'], 'rgb_channel_std': rgb.std((0, 1)).tolist(),
                        'normalized_valid_depth_std': float(norm.std()),
                        'depth_m_p10_p50_p90': np.percentile(d[valid], [10, 50, 90]).tolist()})
    summary['depth_rgb_sample_24_sorted_evenly'] = samples
    result['splits'][split] = summary
    if split == 'val':
        val = data
    print(split, 'geometry complete', flush=True)
result['train_val_model_id_overlap'] = len(model_ids['train'] & model_ids['val'])

gt_by_image = defaultdict(list)
for a in val['annotations']:
    gt_by_image[a['image_id']].append(a)
for tag in ['b0', 'b1', 'm1', 'a1']:
    folder = ROOT / 'experiments/maskrcnn' / f'1x_{tag}'
    best = json.loads((folder / 'best_metrics.json').read_text())
    rows = [json.loads(s) for s in (folder / 'metrics.json').read_text().splitlines() if s.strip()]
    result['runs'][tag] = {'best': best,
        'validation_curve': [{k: r[k] for k in ['iteration', 'segm/APs', 'segm/AP']}
                             for r in rows if 'segm/APs' in r]}
    preds = json.loads((folder / 'best_val_predictions.json').read_text())
    by_image = defaultdict(list)
    for p in preds:
        by_image[p['image_id']].append(p)
    counts = {str(t): Counter() for t in [.05, .5]}
    semantics = {str(t): defaultdict(Counter) for t in [.05, .5]}
    examples = []
    for iid, all_gt in gt_by_image.items():
        gt = [a for a in all_gt if a['area'] <= 1024]
        if not gt:
            continue
        pp = sorted(by_image[iid], key=lambda p: -p['score'])[:100]
        if pp:
            mask_ious = mask_util.iou([p['segmentation'] for p in pp], [g['segmentation'] for g in gt], [0]*len(gt))
            box_ious = mask_util.iou(np.array([p['bbox'] for p in pp], dtype=float),
                                     np.array([g['bbox'] for g in gt], dtype=float), [0]*len(gt))
        for threshold in [.05, .5]:
            active = [i for i, p in enumerate(pp) if p['score'] >= threshold]
            mi = mask_ious[active] if active else np.zeros((1, len(gt)))
            bi = box_ious[active] if active else np.zeros((1, len(gt)))
            bm, bb = mi.max(0), bi.max(0)
            same = np.any((bi >= .5) & (mi >= .5), axis=0)
            for j, g in enumerate(gt):
                flags = {'gt': 1, 'any_mask_iou_ge_05': int(bm[j] >= .5),
                         'any_mask_iou_ge_075': int(bm[j] >= .75),
                         'any_box_iou_ge_05': int(bb[j] >= .5),
                         'box_covered_but_no_mask_iou_ge_05': int(bb[j] >= .5 and bm[j] < .5),
                         'same_prediction_box_and_mask_ge_05': int(same[j])}
                counts[str(threshold)].update(flags)
                semantics[str(threshold)][g.get('semantic_type', 'unknown')].update(flags)
                if tag == 'm1' and threshold == .5 and bb[j] >= .5 and bm[j] < .5 and len(examples) < 12:
                    examples.append({'image_id': iid, 'annotation_id': g['id'], 'area': g['area'],
                                     'bbox': g['bbox'], 'semantic_type': g.get('semantic_type'),
                                     'best_box_iou': float(bb[j]), 'best_mask_iou': float(bm[j])})
    result['coverage_diagnostics'][tag] = {
        'definition': 'Non-exclusive GT coverage by any retained prediction. NOT COCO AP/AR, not one-to-one matching, not RPN recall. All targets grouped by GT MASK area, including box comparisons.',
        'thresholds': {k: dict(v) for k, v in counts.items()},
        'by_semantic_type': {t: {k: dict(v) for k, v in sem.items()} for t, sem in semantics.items()},
        'box_covered_mask_missed_examples': examples}
    print(tag, 'coverage', dict(counts['0.05']), flush=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print('Wrote', OUT, flush=True)

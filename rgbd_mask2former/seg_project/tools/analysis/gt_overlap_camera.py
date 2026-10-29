#!/usr/bin/env python3
"""GT box-overlap (NMS risk) and camera-intrinsics constancy per split."""
import ast, gzip, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

ROOT = Path('/private/rgzndzy')
ANN = ROOT / 'seg_project/datasets/blender/annotations'


def load(split):
    if split in ('train', 'val'):
        return json.load(open(ANN / f'instances_{split}.json'))
    d = {'images': [], 'annotations': []}
    for p in sorted((ROOT / 'blender/annotations/test').glob('*.json.gz')):
        part = json.load(gzip.open(p)); d['images'] += part['images']; d['annotations'] += part['annotations']
    return d


out = {}
for split in ('train', 'val', 'test'):
    d = load(split)
    Ks = Counter()
    for im in d['images']:
        c = im['camera']; c = ast.literal_eval(c) if isinstance(c, str) else c
        Ks[json.dumps(np.round(np.array(c['K']), 3).tolist())] += 1
    by = defaultdict(list)
    for a in d['annotations']: by[a['image_id']].append(a)
    mx_small, mx_all = [], []
    for anns in by.values():
        b = [a['bbox'] for a in anns]
        iou = np.asarray(mu.iou(b, b, [0] * len(b)))
        np.fill_diagonal(iou, 0)
        m = iou.max(1)
        for a, v in zip(anns, m):
            mx_all.append(v)
            if a['area'] <= 1024: mx_small.append(v)
    mx_small, mx_all = np.array(mx_small), np.array(mx_all)
    out[split] = {'distinct_K': len(Ks), 'K_top': Ks.most_common(2),
                  'image_keys': sorted(d['images'][0].keys()),
                  **{f'small_maxBoxIoU>{t}': round(float((mx_small > t).mean()), 4) for t in (0.3, 0.5, 0.6, 0.7)},
                  **{f'all_maxBoxIoU>{t}': round(float((mx_all > t).mean()), 4) for t in (0.5, 0.7)}}
    print(split, json.dumps(out[split]))
json.dump(out, open(Path(__file__).resolve().parents[2] / 'reports/analysis/gt_overlap_camera.json', 'w'), indent=1)

#!/usr/bin/env python3
"""CPU-only composition analysis of train/val/test annotations (no mask decoding)."""
import gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/private/rgzndzy')
ANN = ROOT / 'seg_project/datasets/blender/annotations'
OUT = Path(__file__).resolve().parents[2] / 'reports/analysis/split_composition.json'


def load(split):
    if split in ('train', 'val'):
        return json.load(open(ANN / f'instances_{split}.json'))
    d = {'images': [], 'annotations': []}
    for p in sorted((ROOT / 'blender/annotations/test').glob('*.json.gz')):
        part = json.load(gzip.open(p))
        d['images'] += part['images']; d['annotations'] += part['annotations']
    return d


def q(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    return {f'p{k}': round(float(np.percentile(x, k)), 2) for k in qs} if len(x) else {}


def analyse(d):
    ims = {im['id']: im for im in d['images']}
    anns = d['annotations']
    area = np.array([a['area'] for a in anns], float)
    bw = np.array([a['bbox'][2] for a in anns], float); bh = np.array([a['bbox'][3] for a in anns], float)
    bx = np.array([a['bbox'][0] for a in anns], float); by = np.array([a['bbox'][1] for a in anns], float)
    fill = area / np.maximum(bw * bh, 1)
    ar = np.maximum(bw, bh) / np.maximum(np.minimum(bw, bh), 1)
    border = (bx <= 1) | (by <= 1) | (bx + bw >= 1023) | (by + bh >= 1023)
    z = np.array([a.get('center_camera_cv', [0, 0, 0])[2] for a in anns], float)
    small = area <= 32 ** 2; med = (area > 32 ** 2) & (area <= 96 ** 2); large = area > 96 ** 2
    tiny = area <= 16 ** 2
    st = np.array([a['semantic_type'] for a in anns]); mid = np.array([a['model_id'] for a in anns])
    per_img = Counter(a['image_id'] for a in anns)
    small_img = Counter(a['image_id'] for a, s in zip(anns, small) if s)
    cnt = np.array([per_img.get(i, 0) for i in ims])
    r = {
        'images': len(ims), 'instances': len(anns), 'scenes': len({im['scene_id'] for im in ims.values()}),
        'inst_per_img': {**q(cnt, (0, 50, 95, 100)), 'mean': round(float(cnt.mean()), 2),
                         'imgs_gt_100': int((cnt > 100).sum()), 'imgs_gt_80': int((cnt > 80).sum())},
        'small_per_img': q([small_img.get(i, 0) for i in ims], (0, 50, 95, 100)),
        'imgs_without_small': int(sum(1 for i in ims if small_img.get(i, 0) == 0)),
        'area_bins': {'small': int(small.sum()), 'medium': int(med.sum()), 'large': int(large.sum()),
                      'small_frac': round(float(small.mean()), 4), 'tiny_le16sq': int(tiny.sum())},
        'small_area': q(area[small]),
        'small_bbox_sqrt': q(np.sqrt(bw * bh)[small]),
        'small_bbox_short_side': q(np.minimum(bw, bh)[small]),
        'small_fill': q(fill[small]), 'all_fill': q(fill),
        'small_aspect_gt3_frac': round(float((ar[small] > 3).mean()), 4),
        'small_border_frac': round(float(border[small].mean()), 4), 'all_border_frac': round(float(border.mean()), 4),
        'small_fill_lt_0p3_frac': round(float((fill[small] < 0.3).mean()), 4),
        'center_z_m': q(z), 'small_center_z_m': q(z[small]),
        'depth_scale_m': dict(Counter(str(im.get('depth_scale_m')) for im in ims.values())),
        'depth_invalid_code': dict(Counter(str(im.get('depth_invalid_code')) for im in ims.values())),
        'renderer': dict(Counter(im.get('renderer') for im in ims.values())),
        'composition': dict(Counter(im.get('composition') for im in ims.values())),
        'density': dict(Counter(im.get('density') for im in ims.values())),
        'views_per_scene': dict(Counter(Counter(im['scene_id'] for im in ims.values()).values())),
        'semantic_type_all': dict(Counter(st).most_common()),
        'semantic_type_small': dict(Counter(st[small]).most_common()),
        'model_ids': sorted(set(mid.tolist())),
        'small_by_density': dict(Counter(ims[a['image_id']]['density'] for a, s in zip(anns, small) if s)),
        'small_by_composition': dict(Counter(ims[a['image_id']]['composition'] for a, s in zip(anns, small) if s)),
    }
    # small fraction within each semantic type
    tot = Counter(st); sm = Counter(st[small])
    r['small_rate_by_type'] = {k: round(sm[k] / tot[k], 3) for k in tot}
    # per-type small fill/aspect: why are they small
    r['small_by_type_shape'] = {k: {'n': int(sm[k]), 'fill_p50': round(float(np.median(fill[small & (st == k)])), 3),
                                    'bbox_sqrt_p50': round(float(np.median(np.sqrt(bw * bh)[small & (st == k)])), 1),
                                    'aspect_gt3': round(float((ar[small & (st == k)] > 3).mean()), 3),
                                    'border': round(float(border[small & (st == k)].mean()), 3)}
                                for k in sm if sm[k] >= 20}
    # model ids contributing small instances
    msm = Counter(mid[small]); r['small_models_top'] = dict(msm.most_common(15))
    r['n_models_with_small'] = len(msm)
    r['model_type'] = {m: t for m, t in zip(mid.tolist(), st.tolist())}
    return r


res = {s: analyse(load(s)) for s in ('train', 'val', 'test')}
mods = {s: set(res[s]['model_ids']) for s in res}
res['model_overlap'] = {f'{a}&{b}': len(mods[a] & mods[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
res['n_models'] = {s: len(mods[s]) for s in mods}
# semantic type coverage
types = {s: set(res[s]['semantic_type_all']) for s in ('train', 'val', 'test')}
res['types_only_in'] = {s: sorted(types[s] - set().union(*(types[t] for t in types if t != s))) for s in types}
res['types'] = {s: sorted(types[s]) for s in types}
for s in ('train', 'val', 'test'):
    res[s].pop('model_ids'); res[s].pop('model_type')
json.dump(res, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(OUT)

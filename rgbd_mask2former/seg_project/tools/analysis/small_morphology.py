#!/usr/bin/env python3
"""Decode small GT masks: fragments, thickness, visibility proxy; camera distance per image."""
import ast, gzip, json, random
from collections import defaultdict, Counter
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
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


def seg(a):
    s = a['segmentation']
    return ast.literal_eval(s) if isinstance(s, str) else s


def morph(rle):
    m = mu.decode(rle)
    x, y, w, h = cv2.boundingRect(m)
    m = m[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    er = cv2.erode(m, np.ones((3, 3), np.uint8)).sum() / max(m.sum(), 1)
    return n - 1, areas.max() / max(areas.sum(), 1), float(dt.max()), float(er)


def q(x, ks=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float); return {f'p{k}': round(float(np.percentile(x, k)), 3) for k in ks}


def main():
    out = {}
    random.seed(0)
    for split in ('train', 'val', 'test'):
        d = load(split)
        ims = {im['id']: im for im in d['images']}
        cam = []
        for im in ims.values():
            c = im['camera']; c = ast.literal_eval(c) if isinstance(c, str) else c
            t = np.array(c['c2w_blender'])[:3, 3]
            cam.append((np.linalg.norm(t), t[2]))
        cam = np.array(cam)
        # visibility proxy: area / max area of same model in same image (>=3 copies)
        grp = defaultdict(list)
        for a in d['annotations']:
            grp[(a['image_id'], a['model_id'])].append(a['area'])
        vis_small, vis_all = [], []
        for a in d['annotations']:
            g = grp[(a['image_id'], a['model_id'])]
            if len(g) >= 3:
                v = a['area'] / max(g)
                vis_all.append(v)
                if a['area'] <= 1024: vis_small.append(v)
        small = [a for a in d['annotations'] if a['area'] <= 1024]
        if len(small) > 10000: small = random.sample(small, 10000)
        with Pool(48) as p:
            res = np.array(p.map(morph, [seg(a) for a in small], chunksize=64))
        ncomp = res[:, 0]
        out[split] = {
            'camera_dist_m': q(cam[:, 0]), 'camera_height_m': q(cam[:, 1]),
            'visibility_proxy_small': q(vis_small), 'visibility_proxy_all': q(vis_all),
            'small_vis_lt_0p25_frac': round(float(np.mean(np.array(vis_small) < 0.25)), 3),
            'n_small_sampled': len(small),
            'small_multi_fragment_frac': round(float((ncomp > 1).mean()), 3),
            'small_ncomp_hist': dict(Counter(np.minimum(ncomp, 5).astype(int).tolist())),
            'small_largest_comp_frac': q(res[:, 1]),
            'small_max_inscribed_radius_px': q(res[:, 2]),
            'small_radius_le_2px_frac': round(float((res[:, 2] <= 2).mean()), 3),
            'small_erosion_survival': q(res[:, 3]),
        }
        print(split, json.dumps(out[split]))
    json.dump(out, open(Path(__file__).resolve().parents[2] / 'reports/analysis/small_morphology.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

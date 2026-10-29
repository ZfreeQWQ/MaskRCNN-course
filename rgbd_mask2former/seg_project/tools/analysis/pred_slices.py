#!/usr/bin/env python3
"""Diagnostic COCOeval slices of existing val predictions (CPU). Out-of-slice GT -> iscrowd (ignored)."""
import ast, contextlib, copy, io, json, sys
from collections import defaultdict, Counter
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path('/private/rgzndzy/seg_project')
OUT = Path(__file__).resolve().parents[2] / 'reports/analysis/pred_slices.json'
GT = json.load(open(ROOT / 'datasets/blender/annotations/instances_val.json'))
for a in GT['annotations']:
    if isinstance(a['segmentation'], str): a['segmentation'] = ast.literal_eval(a['segmentation'])
IMS = {im['id']: im for im in GT['images']}


def ncomp(a):
    m = mu.decode(a['segmentation']); return cv2.connectedComponents(m, connectivity=8)[0] - 1


def attrs():
    grp = defaultdict(list)
    for a in GT['annotations']: grp[(a['image_id'], a['model_id'])].append(a['area'])
    with Pool(64) as p: nc = p.map(ncomp, GT['annotations'], chunksize=256)
    cam = {}
    for i, im in IMS.items():
        c = im['camera']; c = ast.literal_eval(c) if isinstance(c, str) else c
        cam[i] = float(np.array(c['c2w_blender'])[2, 3])
    at = {}
    for a, n in zip(GT['annotations'], nc):
        g = grp[(a['image_id'], a['model_id'])]
        vis = a['area'] / max(g) if len(g) >= 3 else None
        at[a['id']] = dict(vis=vis, ncomp=n, type=a['semantic_type'], density=IMS[a['image_id']]['density'],
                          comp=IMS[a['image_id']]['composition'], cam=cam[a['image_id']])
    return at


SLICES = {
    'all': lambda t: True,
    'vis<0.1': lambda t: t['vis'] is not None and t['vis'] < 0.1,
    'vis0.1-0.3': lambda t: t['vis'] is not None and 0.1 <= t['vis'] < 0.3,
    'vis>=0.3': lambda t: t['vis'] is not None and t['vis'] >= 0.3,
    'frag1': lambda t: t['ncomp'] <= 1, 'frag2-3': lambda t: 2 <= t['ncomp'] <= 3, 'frag4+': lambda t: t['ncomp'] >= 4,
    'dens_low': lambda t: t['density'] == 'low', 'dens_med': lambda t: t['density'] == 'medium', 'dens_high': lambda t: t['density'] == 'high',
    'cam<0.8': lambda t: t['cam'] < 0.8, 'cam0.8-1.05': lambda t: 0.8 <= t['cam'] < 1.05, 'cam>=1.05': lambda t: t['cam'] >= 1.05,
    'header_connector': lambda t: t['type'] == 'header_connector', 'fpc_connector': lambda t: t['type'] == 'fpc_connector',
    'terminal_block': lambda t: t['type'] == 'terminal_block', 'unknown': lambda t: t['type'] == 'unknown',
    'choke': lambda t: t['type'] == 'choke', 'switch': lambda t: t['type'] == 'switch',
}


def run_eval(gt, dt, iou):
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt; cg.createIndex()
        cd = cg.loadRes(dt) if dt else COCO()
        e = COCOeval(cg, cd, iou); e.evaluate(); e.accumulate(); e.summarize()
    return [round(100 * float(x), 2) for x in e.stats[:6]]  # AP AP50 AP75 APs APm APl


def job(args):
    model, name, iou, rescore = args
    at = ATTR
    gt = copy.deepcopy(GT)
    keep = SLICES[name]
    n_small = 0
    for a in gt['annotations']:
        if not keep(at[a['id']]): a['iscrowd'] = 1
        elif a['area'] <= 1024: n_small += 1
    dt = load_preds(model)
    if rescore:  # oracle: score = best mask IoU with any GT (upper bound on ranking quality)
        dt = ORACLE[model]
    if iou == 'bbox':
        dt = [{k: v for k, v in d.items() if k != 'segmentation'} for d in dt]
    else:
        dt = [{k: v for k, v in d.items() if k != 'bbox'} for d in dt]
    s = run_eval(gt, copy.deepcopy(dt), iou)
    return model, name, iou, rescore, n_small, s


_P = {}
def load_preds(model):
    if model not in _P:
        p = json.load(open(ROOT / f'experiments/maskrcnn/1x_{model}/best_val_predictions.json'))
        for d in p:
            if isinstance(d['segmentation'], str): d['segmentation'] = ast.literal_eval(d['segmentation'])
            for k in ('image_id', 'category_id'):
                d[k] = int(d[k])
            d['score'] = float(d['score'])
            if isinstance(d['bbox'], str): d['bbox'] = ast.literal_eval(d['bbox'])
        _P[model] = p
    return _P[model]


def oracle(model):
    p = copy.deepcopy(load_preds(model))
    by_img = defaultdict(list)
    for a in GT['annotations']: by_img[a['image_id']].append(a['segmentation'])
    pi = defaultdict(list)
    for k, d in enumerate(p): pi[d['image_id']].append(k)
    for img, ks in pi.items():
        g = by_img.get(img, [])
        if not g: continue
        ious = mu.iou([p[k]['segmentation'] for k in ks], g, [0] * len(g))
        for k, v in zip(ks, np.asarray(ious).max(1)): p[k]['score'] = float(v) + 1e-6 * p[k]['score']
    return p


def stats(model):
    p = load_preds(model)
    c = Counter(d['image_id'] for d in p)
    gtc = Counter(a['image_id'] for a in GT['annotations'])
    dense = [i for i in IMS if gtc[i] >= 70]
    return {'dets_per_img_max': max(c.values()), 'imgs_at_100_dets': sum(v >= 100 for v in c.values()),
            'dense_imgs_ge70gt': len(dense), 'dense_imgs_at_100': sum(c[i] >= 100 for i in dense)}


if __name__ == '__main__':
    ATTR = attrs()
    models = sys.argv[1:] or ['b0', 'a1']
    for m in models: load_preds(m)
    ORACLE = {m: oracle(m) for m in models}
    jobs = [(m, s, iou, False) for m in models for s in SLICES for iou in ('segm', 'bbox')]
    jobs += [(m, 'all', 'segm', True) for m in models]
    res = {'det_stats': {m: stats(m) for m in models}, 'slices': []}
    with Pool(min(len(jobs), 48)) as pool:
        for m, name, iou, rs, n, s in pool.imap_unordered(job, jobs):
            r = dict(model=m, slice=name, iou=iou, oracle_score=rs, n_small_gt=n,
                     AP=s[0], AP50=s[1], AP75=s[2], APs=s[3], APm=s[4], APl=s[5])
            res['slices'].append(r); print(json.dumps(r), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps(res['det_stats']))

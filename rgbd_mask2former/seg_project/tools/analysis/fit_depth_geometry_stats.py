#!/usr/bin/env python3
"""Fit train-only depth-geometry channel stats for rgbd_seg.depth_geometry.standardize().

Samples a subset of train images (RANSAC plane fit is the expensive step), computes height/
normal/discontinuity statistics over valid pixels, writes datasets/blender/annotations/
depth_geometry_stats.json. Train-only, matches the depth_stats.json convention.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg import depth_geometry as dg

ROOT = Path('/private/rgzndzy/seg_project')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-images', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default=str(ROOT / 'datasets/blender/annotations/depth_geometry_stats.json'))
    args = p.parse_args()

    d = json.load(open(ROOT / 'datasets/blender/annotations/instances_train.json'))
    images = d['images']
    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.n_images, len(images)))

    heights, normal_sq, discs = [], [], []
    n_plane_fail = 0
    root = ROOT / 'datasets/blender'
    for im in sample:
        depth_raw = np.array(Image.open(root / im['depth_file_name']), dtype=np.uint16)
        scale = float(im['depth_scale_m'])
        invalid_code = im['depth_invalid_code']
        metric = depth_raw.astype(np.float32) * scale
        valid = (depth_raw != invalid_code) & (metric > 0)
        out = dg.encode(metric, valid, rng=np.random.default_rng(args.seed))
        if out['plane'] is None:
            n_plane_fail += 1
            continue
        v = out['valid'].astype(bool)
        heights.append(out['height'][v])
        normal_sq.append((out['normals'][v] ** 2).sum(axis=-1) / 3.0)  # per-channel variance proxy
        discs.append(out['discontinuity'][v])

    heights = np.concatenate(heights)
    normal_meansq = np.concatenate(normal_sq)
    discs = np.concatenate(discs)
    stats = {
        'n_images': len(sample), 'n_plane_fit_failed': n_plane_fail,
        'height': {'mean': float(heights.mean()), 'std': float(heights.std()),
                   'p1': float(np.percentile(heights, 1)), 'p99': float(np.percentile(heights, 99))},
        'normals': {'std': float(np.sqrt(normal_meansq.mean()))},
        'discontinuity': {'p99': float(np.percentile(discs, 99)), 'p999': float(np.percentile(discs, 99.9)),
                           'frac_nonzero': float((discs > 1e-6).mean())},
        'split': 'train', 'train_only': True,
        'method': 'RANSAC support-plane height + local-gradient normals + relative neighbor depth '
                  'jump, sampled over a random train subset (rgbd_seg.depth_geometry.encode)',
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(args.output, 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()

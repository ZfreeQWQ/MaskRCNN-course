#!/usr/bin/env python3
"""Merge test shards for evaluation/model selection only (2026-10-20 policy); never a training set.

Writes new files and leaves the old null-calibration test_images.json untouched:
  instances_test.json      full COCO GT for evaluation
  test_images_depth.json   input metadata with depth_scale_m / depth_invalid_code from shard images
"""
import gzip
import json
from pathlib import Path

from prepare_data import EXPECTED, image_identity, write_json

ROOT = Path('/private/rgzndzy')
SOURCE = ROOT / 'blender'
TARGET = ROOT / 'seg_project/datasets/blender/annotations'
INPUT_KEYS = ('id', 'file_name', 'depth_file_name', 'height', 'width', 'scene_id', 'view', 'split',
              'depth_scale_m', 'depth_invalid_code')


def main():
    merged = {'images': [], 'annotations': [], 'categories': None}
    for path in sorted((SOURCE / 'annotations/test').glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            part = json.load(f)
        merged['categories'] = merged['categories'] or part['categories']
        assert part['categories'] == merged['categories'], path
        merged['images'] += part['images']
        merged['annotations'] += part['annotations']
    ids = [im['id'] for im in merged['images']]
    assert len(ids) == len(set(ids)) == EXPECTED['test'][0], len(ids)
    for im in merged['images']:
        assert image_identity(im['file_name'])[0] == im['id'] and im['split'] == 'test'
        assert im['depth_scale_m'] > 0 and im['height'] == im['width'] == 1024
        for key in ('file_name', 'depth_file_name'):
            assert (SOURCE / im[key]).is_file(), im[key]
    ann_ids = [a['id'] for a in merged['annotations']]
    assert len(ann_ids) == len(set(ann_ids)) and set(a['image_id'] for a in merged['annotations']) <= set(ids)
    assert all(a['category_id'] == 1 and a['iscrowd'] == 0 for a in merged['annotations'])
    # Template order must match the submission CSV ids.
    old = json.load(open(TARGET / 'test_images.json'))
    assert sorted(ids) == sorted(im['id'] for im in old['images'])
    write_json(TARGET / 'instances_test.json', merged)
    write_json(TARGET / 'test_images_depth.json', {
        'images': [{k: im[k] for k in INPUT_KEYS} for im in sorted(merged['images'], key=lambda x: x['id'])],
        'depth_calibration_available': True,
        'provenance': 'depth_scale_m/depth_invalid_code from blender/annotations/test shard images fields',
        'policy': 'test labels may assist model selection only; never used for training (user decision 2026-10-20)'})
    print(len(ids), 'images', len(ann_ids), 'annotations')


if __name__ == '__main__':
    main()

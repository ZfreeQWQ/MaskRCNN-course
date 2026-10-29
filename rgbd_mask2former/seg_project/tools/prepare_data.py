#!/usr/bin/env python3
"""Merge only public train/val annotations, preserve source fields, index test inputs.

Never enumerates or reads annotations/test. Missing test depth calibration stays null.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

EXPECTED = {'train': (14432, 448469), 'val': (2584, 73421), 'test': (2112, None)}
NAME = re.compile(r'(?P<scene>\d{7})_v(?P<view>\d{3})\.png$')


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    tmp.replace(path)


def image_identity(name):
    m = NAME.fullmatch(Path(name).name)
    if not m:
        raise ValueError(f'Unexpected image name: {name}')
    scene, view = m['scene'], int(m['view'])
    return int(scene) * 1000 + view + 1, scene, view


def merge_split(source, target, split):
    files = sorted((source / 'annotations' / split).glob('*.json.gz'))
    if not files:
        raise ValueError(f'No annotation shards for {split}')
    merged, image_ids, ann_ids, scenes = {}, set(), set(), set()
    for path in files:
        with gzip.open(path, 'rt') as f:
            part = json.load(f)
        for key, value in part.items():
            if key in ('images', 'annotations'):
                merged.setdefault(key, []).extend(value)
            elif key not in merged:
                merged[key] = value
            elif merged[key] != value:
                raise ValueError(f'Conflicting non-list field {key} in {path}')
        assert part['categories'] == [{'id': 1, 'name': 'component', 'supercategory': 'component'}], path
        local_ids = set()
        for im in part['images']:
            iid, scene, view = image_identity(im['file_name'])
            assert iid == im['id'] and scene == im['scene_id'] and im['view'] == view, im['id']
            assert im['split'] == split and im['id'] not in image_ids, im['id']
            assert Path(im['file_name']).name == Path(im['depth_file_name']).name, im['id']
            for key in ('file_name', 'depth_file_name'):
                p = (source / im[key]).resolve()
                assert p.is_relative_to(source) and p.is_file(), p
            assert im['depth_scale_m'] > 0 and im['height'] == im['width'] == 1024, im['id']
            image_ids.add(im['id']); local_ids.add(im['id']); scenes.add(scene)
        for ann in part['annotations']:
            assert ann['id'] not in ann_ids and ann['image_id'] in local_ids, ann['id']
            assert ann['category_id'] == 1 and ann['iscrowd'] == 0, ann['id']
            ann_ids.add(ann['id'])
    assert (len(image_ids), len(ann_ids)) == EXPECTED[split], (split, len(image_ids), len(ann_ids))
    for kind, field in [('rgb', 'file_name'), ('depth', 'depth_file_name')]:
        actual = {p.name for p in (source / kind / split).glob('*.png')}
        expected = {Path(im[field]).name for im in merged['images']}
        assert actual == expected, (split, kind, 'file set mismatch')
    write_json(target / 'annotations' / f'instances_{split}.json', merged)
    return {'images': len(image_ids), 'annotations': len(ann_ids), 'shards': len(files),
            'scenes': len(scenes), 'scene_min': min(scenes), 'scene_max': max(scenes)}, scenes, image_ids


def index_test(source, target):
    # Only whitelisted standalone image metadata, image input files and CSV IDs.
    # In particular, never recover metadata from annotations/test, even if present.
    source_meta = source / 'test_images.json'
    if source_meta.exists():
        raw = json.loads(source_meta.read_text())
        assert not isinstance(raw, dict) or not raw.get('annotations'), 'test image metadata contains annotations'
        images = raw['images'] if isinstance(raw, dict) else raw
        metadata_source = 'source test_images.json (image metadata only)'
    else:
        from PIL import Image
        images = []
        for p in sorted((source / 'rgb' / 'test').glob('*.png')):
            iid, scene, view = image_identity(p.name)
            depth = source / 'depth' / 'test' / p.name
            assert depth.is_file(), depth
            with Image.open(p) as img:
                w, h = img.size
            with Image.open(depth) as dep:
                assert dep.size == (w, h), depth
            images.append({'id': iid, 'file_name': f'rgb/test/{p.name}',
                           'depth_file_name': f'depth/test/{p.name}', 'height': h, 'width': w,
                           'scene_id': scene, 'view': view, 'split': 'test',
                           'depth_scale_m': None, 'depth_invalid_code': None})
        metadata_source = 'filename ID rule verified against submission.csv; PNG headers for size; no depth calibration available'
    csv_path = source / 'sample_submission.csv'
    if not csv_path.exists():
        csv_path = source / 'submission.csv'
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    csv_ids = [int(row['image_id']) for row in rows]  # PredictionString intentionally never consumed.
    ids = [im['id'] for im in images]
    assert len(ids) == len(set(ids)) == len(csv_ids) == len(set(csv_ids)) == EXPECTED['test'][0]
    assert set(ids) == set(csv_ids), 'Test image IDs disagree with CSV'
    for im in images:
        assert image_identity(im['file_name'])[0] == im['id']
        for field in ('file_name', 'depth_file_name'):
            path = (source / im[field]).resolve()
            assert path.is_relative_to(source) and path.is_file(), path
    assert {Path(im['depth_file_name']).name for im in images} == {p.name for p in (source / 'depth' / 'test').glob('*.png')}
    available = all(im.get('depth_scale_m') is not None and im.get('depth_invalid_code') is not None for im in images)
    write_json(target / 'annotations' / 'test_images.json', {'images': images,
        'provenance': metadata_source, 'depth_calibration_available': available,
        'policy': 'Test annotations were never read. Missing calibration must not be guessed.'})
    return {'images': len(images), 'id_source': str(csv_path), 'metadata_source': metadata_source,
            'depth_calibration_available': available}, {im['scene_id'] for im in images}, set(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'blender')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'datasets/blender')
    args = parser.parse_args(); source = args.source.resolve(); target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    for kind in ('rgb', 'depth'):
        link = target / kind
        if not link.exists():
            link.symlink_to(source / kind, target_is_directory=True)
        assert link.resolve() == source / kind, link
    report, scenes, ids = {}, {}, {}
    for split in ('train', 'val'):
        report[split], scenes[split], ids[split] = merge_split(source, target, split)
        print(split, report[split], flush=True)
    report['test'], scenes['test'], ids['test'] = index_test(source, target)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not scenes[a] & scenes[b] and not ids[a] & ids[b], (a,b,'split leakage')
    report['scene_disjoint'] = True
    report['test_annotations_read'] = False
    report['original_image_fields_preserved'] = True
    report['merged_sha256'] = {}
    for split in ('train', 'val'):
        path = target / 'annotations' / f'instances_{split}.json'
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        report['merged_sha256'][split] = digest.hexdigest()
    write_json(Path(__file__).resolve().parents[1] / 'reports/data/data_preparation.json', report)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()

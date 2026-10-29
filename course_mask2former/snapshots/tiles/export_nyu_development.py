"""Export ONLY NYUv2 official-train frames into a fixed scene-disjoint dev protocol.

No network access, no official-test pixel reads, no class/area-dependent sampling.
Use with prepare_nyuv2.py and a human-fixed taxonomy JSON in the same directory.
The completed manifest and completion marker are written last and verified on reuse.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


_HELPER = importlib.util.spec_from_file_location(
    'nyu_prepare_helper', Path(__file__).with_name('prepare_nyuv2.py'))
NYU = importlib.util.module_from_spec(_HELPER)
_HELPER.loader.exec_module(NYU)
PROTOCOL = 'nyuv2-official-train-scene-dev-indoor8-v1'
COCO80_NAMES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush')
SOURCE_TO_COCO_ALIAS = {'sofa': 'couch'}
VOID_POLICY = {
    'label_valid': 'source labels > 0, exported as uint8 in {0,1}',
    'unlabeled': 'source label 0 is void, not a trustworthy negative target',
    'unselected_labeled': 'source label > 0 outside the fixed whitelist is non-target background',
    'training': 'Exporter only records label_valid. A training loss must explicitly consume it; '
                'stock COCO training DOES NOT automatically ignore void pixels.',
    'evaluation': 'COCO AP over selected GT masks; ordinary COCO metrics do not automatically '
                  'ignore void regions. This is a custom development metric, not official NYUv2 AP.'}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def split_scenes(official_split, scenes, seed=20260923, val_fraction=0.2):
    """Rank scene strings by seed-prefixed SHA256; use ceil(fraction * scenes) for val.

    80/20 refers to scene count, not exact frame count. Both outputs retain ascending
    original 1-based frame IDs. No pixel values, class counts or instance areas enter
    this function, and each official-train frame is preserved exactly once.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('seed must be an integer')
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be finite and between 0 and 1')
    for name in ('train', 'test'):
        ids = official_split[name]
        if not ids or len(ids) != len(set(ids)) or any(
                not isinstance(i, int) or isinstance(i, bool) or not 1 <= i <= len(scenes)
                for i in ids):
            raise ValueError(f'Invalid/duplicate 1-based {name} frame IDs')
    train_ids, test_ids = set(official_split['train']), set(official_split['test'])
    if train_ids & test_ids:
        raise ValueError('Official train and test frame IDs overlap')
    train_scenes = {scenes[i - 1] for i in train_ids}
    test_scenes = {scenes[i - 1] for i in test_ids}
    if any(not isinstance(s, str) or not s for s in train_scenes | test_scenes):
        raise ValueError('Scene names must be nonempty strings')
    if train_scenes & test_scenes:
        raise ValueError('Official train and test scenes overlap; refuse export')
    if len(train_scenes) < 2:
        raise ValueError('At least two official-train scenes are required')
    ranked = sorted(train_scenes, key=lambda s: (hashlib.sha256(
        f'{seed}\0{s}'.encode()).hexdigest(), s))
    val_count = max(1, min(len(ranked) - 1, math.ceil(len(ranked) * val_fraction)))
    val_scenes = set(ranked[:val_count])
    splits = {'train': sorted(i for i in train_ids if scenes[i - 1] not in val_scenes),
              'val': sorted(i for i in train_ids if scenes[i - 1] in val_scenes)}
    split_scene_ids = {name: sorted({scenes[i - 1] for i in ids})
                       for name, ids in splits.items()}
    assert set(splits['train']) | set(splits['val']) == train_ids
    assert not set(split_scene_ids['train']) & set(split_scene_ids['val'])
    return dict(seed=seed, val_scene_fraction_requested=val_fraction,
                algorithm='rank unique official-train scene strings by SHA256(seed + NUL + scene); '
                          'first ceil(fraction * scene_count) scenes are val',
                frame_ids=splits, scene_ids=split_scene_ids,
                official_test_frame_ids_excluded=sorted(test_ids),
                official_test_scenes_excluded=sorted(test_scenes),
                no_frame_leakage=True, no_scene_leakage=True,
                official_test_pixels_read=False)


def load_taxonomy(path, names):
    taxonomy = json.loads(Path(path).read_text())
    if not isinstance(taxonomy.get('description'), str) or not taxonomy['description'].strip():
        raise ValueError('Taxonomy description is required')
    rows = taxonomy.get('categories')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Taxonomy categories must be a non-empty ordered list')
    seen_sources, seen_coco = set(), set()
    categories = []
    for target_id, row in enumerate(rows, 1):
        source, name = row['id'], row['name']
        coco_index = row['coco80_class_index']
        if not isinstance(source, int) or isinstance(source, bool) or not 1 <= source <= len(names):
            raise ValueError('Invalid source category ID')
        if names[source - 1] != name:
            raise ValueError(f'NYUv2 ID/name mismatch for {source}: {name}')
        if not isinstance(coco_index, int) or isinstance(coco_index, bool) or not 0 <= coco_index < 80:
            raise ValueError('coco80_class_index must be a zero-based integer in [0,79]')
        coco_name = COCO80_NAMES[coco_index]
        if row.get('coco_name') != coco_name or SOURCE_TO_COCO_ALIAS.get(name, name) != coco_name:
            raise ValueError(f'COCO classifier mapping does not match semantic category {name}')
        if source in seen_sources or coco_index in seen_coco:
            raise ValueError('Duplicate NYUv2 or COCO category in taxonomy')
        seen_sources.add(source)
        seen_coco.add(coco_index)
        categories.append(dict(id=target_id, name=name, supercategory='thing',
                               nyuv2_source_id=source, coco80_class_index=coco_index,
                               coco_name=coco_name))
    return taxonomy, categories


def file_record(root, relative):
    path = Path(root) / relative
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': NYU.sha256(path)}


def verify_export(root):
    root = Path(root)
    complete = json.loads((root / 'export_complete.json').read_text())
    manifest_path = root / 'manifest.json'
    if complete.get('manifest_sha256') != NYU.sha256(manifest_path):
        raise ValueError('Completed manifest SHA256 mismatch')
    manifest = json.loads(manifest_path.read_text())
    if (complete.get('signature') != manifest.get('signature') or
            manifest.get('signature') != canonical_sha(manifest['specification'])):
        raise ValueError('Completion/specification signature mismatch')
    if manifest['specification'].get('protocol') != PROTOCOL:
        raise ValueError('Unrecognized export protocol')
    files = manifest['files']
    if len(files) != len({entry['path'] for entry in files}):
        raise ValueError('Duplicate exported file records')
    for entry in files:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe path in export manifest')
        path = root / relative
        if not path.is_file() or path.is_symlink() or file_record(root, entry['path']) != entry:
            raise ValueError(f'Exported file bytes/SHA256 mismatch: {entry["path"]}')
    expected = {entry['path'] for entry in files} | {'manifest.json', 'export_complete.json'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
              and not p.name.endswith('.partial')}
    if actual != expected:
        raise ValueError('Unexpected or missing files in completed export')
    return manifest


def export_development(data, official_split, things_path, output, source_manifest,
                       seed=20260923, val_fraction=0.2):
    """The caller holds the output lock; only train/val IDs enter read_frame."""
    from PIL import Image
    from pycocotools import mask as mask_api

    output = Path(output)
    names, scenes = NYU.matlab_strings(data, 'names'), NYU.matlab_strings(data, 'scenes')
    taxonomy, categories = load_taxonomy(things_path, names)
    split = split_scenes(official_split, scenes, seed, val_fraction)
    specification = dict(protocol=PROTOCOL, source=source_manifest, taxonomy=taxonomy,
                         categories=categories, split=split, void_policy=VOID_POLICY,
                         rgb='unaltered uint8 PNG in RGB order',
                         depth='official inpainted depths; float32 meters; invalid -> 0',
                         depth_validity='raw/filled: finite and > 0; NOT sensor confidence',
                         instance_semantics='(source category, source instance) pairs; retain '
                                            'instance 0; no component splitting or area filtering')
    signature = canonical_sha(specification)
    marker = output / 'export_spec.json'
    if (output / 'export_complete.json').exists():
        previous = verify_export(output)
        if previous['signature'] != signature:
            raise ValueError('Different completed protocol/taxonomy/source/split; use a new output root')
        print('Verified existing complete scene-development export; no files rewritten', flush=True)
        return previous
    if output.exists() and any(output.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()).get('signature') != signature:
            raise ValueError('Non-empty output is unowned or has a different specification; use a new output root')
    output.mkdir(parents=True, exist_ok=True)
    NYU.write_json(marker, dict(signature=signature, specification=specification))
    records = [file_record(output, 'export_spec.json')]
    NYU.write_json(output / 'development_split.json', split)
    records.append(file_record(output, 'development_split.json'))
    category_map = {row['nyuv2_source_id']: row['id'] for row in categories}
    statistics = {}
    for split_name, frame_ids in split['frame_ids'].items():
        images, annotations, areas, per_category, frame_statistics = [], [], [], Counter(), []
        for offset, frame_id in enumerate(frame_ids):
            if frame_id not in official_split['train'] or frame_id in official_split['test']:
                raise AssertionError('Refuse to read official-test or non-training pixels')
            frame = NYU.read_frame(data, frame_id)
            stem = f'{frame_id:04d}'
            paths = dict(file_name=f'rgb/{split_name}/{stem}.png',
                         depth_file=f'depth/{split_name}/{stem}.npy',
                         raw_valid_file=f'valid_raw/{split_name}/{stem}.npy',
                         filled_valid_file=f'valid_filled/{split_name}/{stem}.npy',
                         label_valid_file=f'label_valid/{split_name}/{stem}.npy')
            with NYU.atomic_target(output / paths['file_name']) as temporary:
                Image.fromarray(frame['rgb']).save(temporary, format='PNG')
            arrays = {
                'depth_file': np.where(frame['valid_filled'], frame['depths'], 0).astype(np.float32),
                'raw_valid_file': frame['valid_raw'].astype(np.uint8),
                'filled_valid_file': frame['valid_filled'].astype(np.uint8),
                'label_valid_file': (frame['labels'] > 0).astype(np.uint8)}
            for key, value in arrays.items():
                NYU.save_array(output / paths[key], value)
            records.extend(file_record(output, path) for path in paths.values())
            height, width = frame['rgb'].shape[:2]
            images.append(dict(id=frame_id, width=width, height=height,
                               scene=scenes[frame_id - 1], **paths))
            selected_areas = []
            for source_category, instance, area in NYU.instance_pairs(frame['labels'], frame['instances']):
                if source_category not in category_map:
                    continue
                binary = (frame['labels'] == source_category) & (frame['instances'] == instance)
                rle = mask_api.encode(np.asfortranarray(binary.astype(np.uint8)))
                if int(mask_api.area(rle)) != area:
                    raise ValueError('Instance RLE area mismatch')
                bbox = mask_api.toBbox(rle).tolist()
                rle['counts'] = rle['counts'].decode('ascii')
                target = category_map[source_category]
                annotations.append(dict(id=len(annotations) + 1, image_id=frame_id,
                                        category_id=target, segmentation=rle, bbox=bbox, area=area,
                                        iscrowd=0, nyuv2_instance_id=instance,
                                        nyuv2_category_id=source_category))
                selected_areas.append(area)
                per_category[target] += 1
            areas.extend(selected_areas)
            frame_statistics.append(dict(frame_id=frame_id, instances=len(selected_areas),
                                         labeled_pixel_fraction=float((frame['labels'] > 0).mean()),
                                         raw_depth_valid_fraction=float(frame['valid_raw'].mean()),
                                         filled_depth_valid_fraction=float(frame['valid_filled'].mean())))
            if (offset + 1) % 50 == 0 or offset + 1 == len(frame_ids):
                print(f'Exported {split_name} {offset + 1}/{len(frame_ids)}', flush=True)
        coco = dict(info={'description': PROTOCOL, 'note': 'Custom development labels; NOT official NYUv2 AP'},
                    images=images, annotations=annotations, categories=categories)
        annotation_path = f'annotations/instances_{split_name}.json'
        index_path = f'{split_name}_rgbd_index.json'
        NYU.write_json(output / annotation_path, coco)
        NYU.write_json(output / index_path, images)
        records.extend(file_record(output, p) for p in (annotation_path, index_path))
        statistics[split_name] = dict(images=len(images), scenes=len(split['scene_ids'][split_name]),
                                     instances=NYU.size_summary(areas),
                                     category_instance_counts={row['name']: per_category[row['id']]
                                                               for row in categories},
                                     images_with_no_selected_instances=sum(row['instances'] == 0
                                                                          for row in frame_statistics),
                                     frame_statistics=frame_statistics)
    manifest = dict(signature=signature, specification=specification, statistics=statistics,
                    files=sorted(records, key=lambda entry: entry['path']),
                    exported_images=sum(item['images'] for item in statistics.values()),
                    export_scope='All official train frames; official test IDs/scenes recorded but pixels never read',
                    caveats=['80/20 is by scenes, not exact image count.',
                             'No class/size balancing, small-instance filtering or empty-image dropping.',
                             'Development statistics are descriptive only and do not alter the fixed protocol.',
                             'COCO80 classifier indices are zero-based; COCO output category IDs are contiguous one-based.',
                             'label_valid must be explicitly consumed by training; exporter alone implements no loss masking.',
                             'COCO evaluator does not automatically ignore predictions inside unannotated regions.'])
    NYU.write_json(output / 'manifest.json', manifest)
    NYU.write_json(output / 'export_complete.json', dict(signature=signature,
                                                       manifest_sha256=NYU.sha256(output / 'manifest.json')))
    return verify_export(output)


def validated_sources(root):
    """Verify already-downloaded sources without calling validation that reads test pixels."""
    import h5py

    source = json.loads((root / 'source_manifest.json').read_text())
    for key, name in (('mat', 'nyu_depth_v2_labeled.mat'), ('split', 'splits.mat')):
        path, record = root / name, source[key]
        if path.stat().st_size != record['bytes'] or NYU.sha256(path) != record['sha256']:
            raise ValueError(f'Source file does not match its integrity record: {name}')
    mat = root / 'nyu_depth_v2_labeled.mat'
    if not h5py.is_hdf5(mat):
        raise ValueError('Source MAT is not HDF5')
    with h5py.File(mat, 'r') as data:
        expected = {'images': (1449, 3, 640, 480)}
        expected.update({key: (1449, 640, 480) for key in ('depths', 'rawDepths', 'labels', 'instances')})
        for key, shape in expected.items():
            if key not in data or data[key].shape != shape:
                raise ValueError(f'Unexpected HDF5 shape for {key}')
        if data['images'].dtype != np.uint8 or data['scenes'].size != 1449:
            raise ValueError('Unexpected image dtype or scene count')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/data/nyuv2'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--things-json', type=Path,
                        default=Path(__file__).with_name('nyuv2_things_v1.json'))
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    output = args.output_root or args.root / 'development_indoor8_v1'
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f'.{output.name}.export.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another export owns this destination')
        if args.verify_only:
            manifest = verify_export(output)
        else:
            import h5py
            source = validated_sources(args.root)
            official = NYU.read_split(args.root / 'splits.mat')
            with h5py.File(args.root / 'nyu_depth_v2_labeled.mat', 'r') as data:
                manifest = export_development(data, official, args.things_json, output, source,
                                              args.seed, args.val_fraction)
    print(json.dumps({'output': str(output), 'signature': manifest['signature'],
                      'statistics': {name: {k: v for k, v in stats.items() if k != 'frame_statistics'}
                                     for name, stats in manifest['statistics'].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

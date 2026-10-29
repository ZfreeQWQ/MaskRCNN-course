"""Inspect official NYUv2 RGB-D data; optionally export an explicitly chosen thing taxonomy.

This is a development-data adapter, NOT an official NYUv2 instance benchmark.
Default mode never invents a things/stuff split and never exports training labels.
Dependencies: numpy, h5py, scipy; export additionally needs Pillow, pycocotools.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np


MAT_URL = 'https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
SPLIT_URL = 'https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
MAT_BYTES = 2_972_037_809  # HTTPS HEAD checked 2026-09-22; no upstream hash supplied.
SPLIT_BYTES = 2626
SPLIT_SHA256 = '6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee'
SOURCE_PAGE = 'https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html'
TOOLBOX_URL = 'https://cs.nyu.edu/~fergus/datasets/toolbox_nyu_depth_v2.zip'
FIELDS = ('images', 'depths', 'rawDepths', 'labels', 'instances', 'names', 'scenes')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_target(path):
    """Only replace a destination after the entire new file is written successfully."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_target(path) as temporary:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate_mat(path):
    import h5py

    if not h5py.is_hdf5(path):
        raise ValueError(f'{path}: expected a MATLAB v7.3/HDF5 file, not HTML or a truncated download')
    with h5py.File(path, 'r') as data:
        missing = sorted(set(FIELDS) - set(data))
        if missing:
            raise ValueError(f'Missing NYUv2 fields: {missing}')
        # MATLAB axis order is reversed by HDF5: N,C,W,H and N,W,H.
        expected = {'images': (1449, 3, 640, 480)}
        expected.update({key: (1449, 640, 480) for key in ('depths', 'rawDepths', 'labels', 'instances')})
        for key, shape in expected.items():
            if data[key].shape != shape:
                raise ValueError(f'Unexpected {key} layout {data[key].shape}, expected {shape}; inspect before adapting')
        if data['images'].dtype != np.uint8:
            raise ValueError('RGB images are not uint8')
        if data['scenes'].size != 1449:
            raise ValueError('Expected one scene reference per image')
        # Read actual data chunks as well as metadata to catch simple truncation.
        for key in ('images', 'depths', 'rawDepths', 'labels', 'instances'):
            _ = data[key][0]
            _ = data[key][-1]


def read_split(path):
    from scipy.io import loadmat

    raw = loadmat(path)
    split = {}
    for key, count, name in (('trainNdxs', 795, 'train'), ('testNdxs', 654, 'test')):
        values = np.asarray(raw[key]).reshape(-1)
        if not np.all(np.isfinite(values)) or not np.all(values == values.astype(np.int64)):
            raise ValueError(f'{key} contains non-integer indices')
        values = values.astype(np.int64)
        if len(values) != count or len(set(values.tolist())) != count:
            raise ValueError(f'{key} must contain {count} unique 1-based frame IDs')
        split[name] = values.tolist()
    if set(split['train']) & set(split['test']):
        raise ValueError('Official split has overlapping frame IDs')
    if set(split['train']) | set(split['test']) != set(range(1, 1450)):
        raise ValueError('Official split does not cover exactly the 1449 labeled frames')
    return split


def acquire(url, target, size, validator, expected_hash=None):
    """HTTPS only; validate size/content/hash and promote atomically. Reruns reuse verified files.

    A failed initial download is removed, never mistaken for a complete file. For
    simplicity this restarts failed transfers; it does not promise byte resuming.
    """
    target = Path(target)
    metadata_path = target.with_suffix(target.suffix + '.source.json')
    if target.exists():
        if target.stat().st_size != size:
            raise ValueError(f'Existing {target} has unexpected size; preserve it and resolve manually')
        actual_hash = sha256(target)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f'Existing {target}: SHA256 mismatch')
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous['sha256'] != actual_hash:
                raise ValueError(f'Existing {target} changed since its recorded manifest')
        validator(target)
        metadata = dict(url=url, bytes=size, sha256=actual_hash,
                        checksum_authority='pinned locally observed official split' if expected_hash else 'local integrity record; not an upstream-published checksum')
        write_json(metadata_path, metadata)
        return metadata
    if not url.startswith('https://'):
        raise ValueError('Only verified HTTPS downloads are supported')
    print(f'Downloading {url} ({size / 1e9:.3f} GB)', flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'AI-Class-NYUv2-Preparation/1.0'})
    with atomic_target(target) as temporary:
        digest = hashlib.sha256()
        copied = 0
        last_print = time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            if not response.geturl().startswith('https://'):
                raise ValueError('Refusing an HTTPS-to-HTTP redirect')
            header_size = response.headers.get('Content-Length')
            if header_size and int(header_size) != size:
                raise ValueError(f'Unexpected Content-Length {header_size}; expected {size}')
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                copied += len(block)
                if copied > size:
                    raise ValueError('Download is larger than the verified source size')
                digest.update(block)
                output.write(block)
                if time.monotonic() - last_print > 15:
                    print(f'{copied / 1e6:.1f}/{size / 1e6:.1f} MB', flush=True)
                    last_print = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        if copied != size:
            raise ValueError(f'Incomplete download: {copied}/{size} bytes')
        actual_hash = digest.hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f'Download SHA256 mismatch: {actual_hash}')
        validator(temporary)
    metadata = dict(url=url, bytes=copied, sha256=actual_hash,
                    checksum_authority='pinned locally observed official split' if expected_hash else 'local integrity record; not an upstream-published checksum')
    write_json(metadata_path, metadata)
    return metadata


def matlab_strings(data, key):
    """Read MATLAB cell strings, which h5py exposes as object references."""
    values = []
    for reference in np.asarray(data[key]).reshape(-1):
        chars = np.asarray(data[reference]).astype('<u2').reshape(-1)
        values.append(chars.tobytes().decode('utf-16-le').rstrip('\x00'))
    return values


def read_frame(data, frame_id):
    index = frame_id - 1  # Retain original MATLAB 1-based IDs in COCO/manifest.
    rgb = np.asarray(data['images'][index]).transpose(2, 1, 0)
    result = {'rgb': np.ascontiguousarray(rgb)}
    for key in ('depths', 'rawDepths', 'labels', 'instances'):
        result[key] = np.ascontiguousarray(np.asarray(data[key][index]).T)
    result['depths'] = result['depths'].astype(np.float32)
    result['rawDepths'] = result['rawDepths'].astype(np.float32)
    result['valid_raw'] = np.isfinite(result['rawDepths']) & (result['rawDepths'] > 0)
    result['valid_filled'] = np.isfinite(result['depths']) & (result['depths'] > 0)
    for key, value in result.items():
        if key != 'rgb' and value.shape != rgb.shape[:2]:
            raise ValueError(f'RGB/depth/label shape mismatch in frame {frame_id}')
    return result


def instance_pairs(labels, instances):
    """Official get_instance_masks.m groups by (class, instance), NOT instance alone.

    Its only removed pair is (0,0). We additionally exclude class 0 because the
    official dataset calls that unlabeled. Class>0 with instance==0 is retained.
    Inputs must be equally shaped 2-D nonnegative integer maps fitting uint32.
    Packing each pair into uint64 makes unique a 1-D integer operation rather
    than an expensive structured-row comparison, with identical ordering.
    """
    labels, instances = np.asarray(labels), np.asarray(instances)
    if labels.ndim != 2 or labels.shape != instances.shape:
        raise ValueError('labels and instances must be equally shaped 2-D maps')
    for name, value in (('labels', labels), ('instances', instances)):
        if value.dtype.kind not in 'iu':
            raise TypeError(f'{name} must have an integer dtype, not {value.dtype}')
        if value.size and ((value.dtype.kind == 'i' and value.min() < 0) or value.max() > 0xFFFFFFFF):
            raise ValueError(f'{name} values must be nonnegative integers fitting uint32')
    packed = labels.reshape(-1).astype(np.uint64, copy=False) << np.uint64(32)
    packed |= instances.reshape(-1).astype(np.uint64, copy=False)
    keys, counts = np.unique(packed, return_counts=True)
    categories, instance_ids = keys >> np.uint64(32), keys & np.uint64(0xFFFFFFFF)
    return [(int(category), int(instance), int(count))
            for category, instance, count in zip(categories, instance_ids, counts) if category > 0]


def size_summary(areas):
    values = np.asarray(areas, dtype=np.int64)
    return dict(count=int(values.size), less_than_32_squared=int((values < 32 ** 2).sum()),
                from_32_squared_to_96_squared=int(((values >= 32 ** 2) & (values < 96 ** 2)).sum()),
                at_least_96_squared=int((values >= 96 ** 2).sum()),
                median=float(np.median(values)) if values.size else None,
                note='Descriptive non-overlapping bins at ORIGINAL resolution, not an AP calculation.')


def inspect(data, split, sample_limit):
    names = matlab_strings(data, 'names')
    scenes = matlab_strings(data, 'scenes')
    train_scenes = {scenes[i - 1] for i in split['train']}
    test_scenes = {scenes[i - 1] for i in split['test']}
    overlap = sorted(train_scenes & test_scenes)
    class_counts, areas, frames = Counter(), [], []
    selected = range(1, 1450) if not sample_limit else range(1, min(sample_limit, 1449) + 1)
    for frame_id in selected:
        frame = read_frame(data, frame_id)
        pairs = instance_pairs(frame['labels'], frame['instances'])
        for category, _, area in pairs:
            if category > len(names):
                raise ValueError(f'Unknown class {category} in frame {frame_id}')
            class_counts[category] += 1
            areas.append(area)
        valid = frame['rawDepths'][frame['valid_raw']]
        frames.append(dict(frame_id=frame_id, scene=scenes[frame_id - 1], height=frame['rgb'].shape[0],
                           width=frame['rgb'].shape[1], labeled_pairs=len(pairs),
                           invalid_raw_fraction=float(1.0 - frame['valid_raw'].mean()),
                           invalid_filled_fraction=float(1.0 - frame['valid_filled'].mean()),
                           raw_depth_min_m=float(valid.min()) if valid.size else None,
                           raw_depth_max_m=float(valid.max()) if valid.size else None,
                           unlabeled_fraction=float((frame['labels'] == 0).mean())))
        if frame_id % 100 == 0:
            print(f'Inspected {frame_id} / {len(selected)} frames', flush=True)
    return dict(
        purpose='Public RGB-D development data, not a course or official instance-segmentation benchmark',
        hdf5_layout={key: {'shape': list(data[key].shape), 'dtype': str(data[key].dtype)} for key in FIELDS},
        conversion='images N,C,W,H -> H,W,C RGB uint8; other maps N,W,H -> H,W',
        depth_units='meters; depths is inpainted; rawDepths preserves missing measurements',
        validity='isfinite(rawDepths) & (rawDepths > 0); NOT measured-depth confidence or sensor accuracy',
        split={key: len(value) for key, value in split.items()},
        split_indexing='original 1-based MATLAB frame IDs',
        scene_leakage={'train_scenes': len(train_scenes), 'test_scenes': len(test_scenes),
                       'overlap': overlap, 'scene_disjoint': not overlap},
        inspected_frames=len(frames), partial_statistics=bool(sample_limit and sample_limit < 1449),
        categories=[{'id': i, 'name': name, 'inspected_labeled_pairs': class_counts[i]}
                    for i, name in enumerate(names, start=1)],
        all_labeled_pair_areas=size_summary(areas),
        area_scope='Includes all labeled category-instance pairs, including stuff; NOT thing-only statistics',
        frames=frames,
    )


def load_things(path, names):
    value = json.loads(Path(path).read_text())
    if not isinstance(value.get('description'), str) or not value['description'].strip():
        raise ValueError('Taxonomy JSON needs a non-empty description of this custom thing selection')
    selected = value.get('categories')
    if not isinstance(selected, list) or not selected:
        raise ValueError('Taxonomy JSON needs a non-empty categories list: [{"id": original_id, "name": exact_name}]')
    ids = []
    for category in selected:
        class_id = category['id']
        if not isinstance(class_id, int) or isinstance(class_id, bool) or not 1 <= class_id <= len(names):
            raise ValueError(f'Invalid source class ID: {class_id}')
        if names[class_id - 1] != category['name']:
            raise ValueError(f'Class/name mismatch for source ID {class_id}; inspect categories.json first')
        ids.append(class_id)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate thing category IDs')
    return value, {source_id: i for i, source_id in enumerate(sorted(ids), start=1)}


def save_array(path, array):
    with atomic_target(path) as temporary, temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)


def export(data, split, things_path, output, source_manifest):
    from PIL import Image
    from pycocotools import mask as mask_api

    names = matlab_strings(data, 'names')
    scenes = matlab_strings(data, 'scenes')
    taxonomy, category_map = load_things(things_path, names)
    specification = dict(protocol='custom-nyuv2-things-development-v1', source=source_manifest,
                         taxonomy=taxonomy, depth_variant='official inpainted meters float32',
                         validity='separate raw measured-depth finite-and-positive mask',
                         split='official 795 train / 654 test; test is a holdout, NOT tuning validation')
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'manifest.json'
    marker = output / 'export_spec.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['signature'] != signature:
            raise ValueError('Output contains a different export; choose a new --export-root')
        print('Rechecking/rebuilding the same deterministic export', flush=True)
    elif output.exists() and any(output.iterdir()) and not marker.exists():
        raise ValueError('Output is non-empty and is not owned by this adapter; choose a new --export-root')
    if marker.exists() and json.loads(marker.read_text())['signature'] != signature:
        raise ValueError('An interrupted export used a different specification; choose a new output')
    output.mkdir(parents=True, exist_ok=True)
    write_json(marker, dict(signature=signature, specification=specification))
    categories = [dict(id=target, name=names[source - 1], nyuv2_source_id=source, supercategory='thing')
                  for source, target in category_map.items()]
    statistics = {}
    for split_name, frame_ids in split.items():
        images, annotations, areas, index = [], [], [], []
        for frame_id in frame_ids:
            frame = read_frame(data, frame_id)
            stem = f'{frame_id:04d}'
            rgb_path = f'rgb/{split_name}/{stem}.png'
            depth_path = f'depth/{split_name}/{stem}.npy'
            valid_path = f'valid_raw/{split_name}/{stem}.npy'
            with atomic_target(output / rgb_path) as temporary:
                Image.fromarray(frame['rgb']).save(temporary, format='PNG')
            # Invalid filled depth is set to zero, with its own validity map retained.
            depth = np.where(frame['valid_filled'], frame['depths'], 0).astype(np.float32)
            save_array(output / depth_path, depth)
            save_array(output / valid_path, frame['valid_raw'].astype(np.uint8))
            filled_valid_path = f'valid_filled/{split_name}/{stem}.npy'
            save_array(output / filled_valid_path, frame['valid_filled'].astype(np.uint8))
            # Unlabeled/unselected pixels must be known by an eventual custom RGB-D
            # training pipeline. Stock COCO ignores this field and treats them as
            # non-target background; the adapter does not conceal that limitation.
            label_valid_path = f'label_valid/{split_name}/{stem}.npy'
            save_array(output / label_valid_path, (frame['labels'] > 0).astype(np.uint8))
            height, width = frame['rgb'].shape[:2]
            item = dict(id=frame_id, file_name=rgb_path, height=height, width=width,
                        depth_file=depth_path, raw_valid_file=valid_path,
                        filled_valid_file=filled_valid_path, label_valid_file=label_valid_path,
                        scene=scenes[frame_id - 1])
            images.append(item)
            index.append(item)
            for category, instance, area in instance_pairs(frame['labels'], frame['instances']):
                if category not in category_map:
                    continue
                binary = (frame['labels'] == category) & (frame['instances'] == instance)
                rle = mask_api.encode(np.asfortranarray(binary.astype(np.uint8)))
                rle_area = int(mask_api.area(rle))
                if area != rle_area:
                    raise ValueError('RLE area mismatch')
                bbox = mask_api.toBbox(rle).tolist()
                rle['counts'] = rle['counts'].decode('ascii')
                annotations.append(dict(id=len(annotations) + 1, image_id=frame_id,
                                        category_id=category_map[category], segmentation=rle,
                                        area=area, bbox=bbox, iscrowd=0,
                                        nyuv2_instance_id=instance, nyuv2_category_id=category))
                areas.append(area)
        coco = dict(info={'description': specification['protocol'], 'note': 'Custom thing selection; not an official NYUv2 AP protocol'},
                    images=images, annotations=annotations, categories=categories)
        write_json(output / f'annotations/instances_{split_name}.json', coco)
        write_json(output / f'{split_name}_rgbd_index.json', index)
        statistics[split_name] = dict(images=len(images), instances=size_summary(areas),
                                     categories={str(category['id']): sum(a['category_id'] == category['id'] for a in annotations)
                                                 for category in categories})
    manifest = dict(signature=signature, specification=specification, statistics=statistics,
                    caveats=['No small-object filtering or connected-component splitting is applied.',
                             'Same (category, instance) disconnected regions remain one instance, matching the toolbox.',
                             'All source images are retained, including images with no selected thing.',
                             'COCO JSON contains RGB paths plus nonstandard depth/validity fields; stock MMDetection does not consume depth automatically.',
                             'Unlabeled source pixels are recorded in label_valid maps, but stock COCO loss does not ignore them automatically.',
                             'Do not tune on test; derive a scene-disjoint development split from train first.',
                             'NYUv2 is public prototyping data, not the course-provided dataset.'])
    write_json(manifest_path, manifest)
    return manifest


def self_test():
    """Small semantic regression tests; no network or dataset download."""
    labels = np.array([[1, 1, 2], [1, 2, 0]], dtype=np.uint16)
    instances = np.array([[0, 1, 1], [0, 1, 3]], dtype=np.uint16)
    assert instance_pairs(labels, instances) == [(1, 0, 2), (1, 1, 1), (2, 1, 2)]
    rng = np.random.default_rng(20260922)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64, np.int32, np.int64):
        for shape in ((0, 0), (1, 1), (17, 31), (64, 80)):
            random_labels = rng.integers(0, 101, size=shape).astype(dtype)
            random_instances = rng.integers(0, 13, size=shape).astype(dtype)
            old_pairs, old_counts = np.unique(
                np.stack((random_labels.reshape(-1), random_instances.reshape(-1)), axis=1),
                axis=0, return_counts=True)
            reference = [(int(category), int(instance), int(count))
                         for (category, instance), count in zip(old_pairs, old_counts) if category > 0]
            assert instance_pairs(random_labels, random_instances) == reference
    boundary = np.array([[0, 1, 0xFFFFFFFF]], dtype=np.uint64)
    assert instance_pairs(boundary, boundary) == [(1, 1, 1), (0xFFFFFFFF, 0xFFFFFFFF, 1)]
    for invalid in (np.array([[-1]], dtype=np.int64),
                    np.array([[0x100000000]], dtype=np.uint64),
                    np.array([[1.0]], dtype=np.float32), np.array([[True]])):
        try:
            instance_pairs(invalid, np.zeros_like(invalid))
            raise AssertionError('Invalid integer-map input unexpectedly accepted')
        except (TypeError, ValueError):
            pass
    # Same instance ID in two categories must never be merged. Class 0 is ignored.
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    raw = np.array([[0, 1, np.nan], [2, -1, np.inf]], dtype=np.float32)
    fake = {'images': rgb.transpose(2, 1, 0)[None], 'rawDepths': raw.T[None],
            'depths': np.ones((1, 3, 2), dtype=np.float32),
            'labels': labels.T[None], 'instances': instances.T[None]}
    frame = read_frame(fake, 1)
    assert np.array_equal(frame['rgb'], rgb)
    assert np.array_equal(frame['labels'], labels)
    assert int(frame['valid_raw'].sum()) == 2
    assert frame['depths'].dtype == np.float32
    assert size_summary([1, 1023, 1024, 9215, 9216])['less_than_32_squared'] == 2
    with tempfile.TemporaryDirectory(prefix='nyuv2-adapter-test-') as temporary:
        path = Path(temporary) / 'value.json'
        write_json(path, {'ok': True})
        assert json.loads(path.read_text()) == {'ok': True}
        taxonomy = Path(temporary) / 'things.json'
        write_json(taxonomy, {'description': 'unit test, not real taxonomy',
                              'categories': [{'id': 2, 'name': 'chair'}]})
        _, mapping = load_things(taxonomy, ['wall', 'chair'])
        assert mapping == {2: 1}
    print('PASS: class+instance pairing, 24 randomized old/new equivalence cases, uint32 boundaries, invalid-input rejection, '
          'instance zero, axis conversion, depth validity, area bins, atomic JSON, taxonomy checks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/data/nyuv2'))
    parser.add_argument('--download', action='store_true', help='Allow ~2.97 GB official labeled MAT download; no raw archive')
    parser.add_argument('--mode', choices=('inspect', 'export'), default='inspect')
    parser.add_argument('--sample-limit', type=int, default=0, help='Inspect first N frames; 0=all. Never changes official split.')
    parser.add_argument('--things-json', type=Path, help='Required for export; exact source class IDs/names plus description')
    parser.add_argument('--export-root', type=Path, help='Default ROOT/custom_things; choose another path to change taxonomy')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.sample_limit < 0:
        parser.error('--sample-limit must be nonnegative')
    if args.mode == 'export' and not args.things_json:
        parser.error('Export requires --things-json; no official things/stuff split is assumed')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / '.prepare.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another preparation process owns this data root')
        mat_path = args.root / 'nyu_depth_v2_labeled.mat'
        split_path = args.root / 'splits.mat'
        if not args.download and (not mat_path.exists() or not split_path.exists()):
            raise SystemExit('Source files are missing. Use --download to explicitly permit downloading them.')
        # Verify the small split first so a broken metadata link does not waste a large download.
        split_source = acquire(SPLIT_URL, split_path, SPLIT_BYTES, read_split, SPLIT_SHA256)
        mat_source = acquire(MAT_URL, mat_path, MAT_BYTES, validate_mat)
        split = read_split(split_path)
        source_manifest = dict(mat=mat_source, split=split_source, homepage=SOURCE_PAGE,
                               instance_semantics_source=TOOLBOX_URL,
                               note='Local hashes detect changes; MAT hash is not a checksum published by the data author.')
        import h5py
        with h5py.File(mat_path, 'r') as data:
            report = inspect(data, split, args.sample_limit)
            report['sources'] = source_manifest
            write_json(args.root / 'inspection.json', report)
            write_json(args.root / 'categories.json', report['categories'])
            write_json(args.root / 'official_split.json', split)
            write_json(args.root / 'source_manifest.json', source_manifest)
            summary = {key: report[key] for key in ('inspected_frames', 'partial_statistics', 'scene_leakage', 'all_labeled_pair_areas')}
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            if args.mode == 'export':
                if not report['scene_leakage']['scene_disjoint']:
                    raise ValueError('Scene overlap detected: inspection preserved; export aborted to prevent leakage')
                destination = args.export_root or args.root / 'custom_things'
                export(data, split, args.things_json, destination, source_manifest)
                print(f'Custom development export complete: {destination}', flush=True)
            else:
                print('Inspection complete. No training annotations exported; choose a documented thing taxonomy first.', flush=True)


if __name__ == '__main__':
    main()

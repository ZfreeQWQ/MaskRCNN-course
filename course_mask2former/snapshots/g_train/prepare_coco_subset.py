"""Prepare a deterministic, unbiased COCO 2017 *development* subset.

Downloads the official annotation ZIP and only the sampled JPEG images. This
subset is not the complete COCO benchmark and has no measured depth images.
An existing subset with another seed/count/source is never silently replaced.
Requires Pillow; all other dependencies are in the Python standard library.
"""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile


# COCO's original domain currently has a TLS hostname mismatch. Its *same*
# public S3 bucket supports path-style HTTPS with a valid s3.amazonaws.com
# certificate. Do not disable certificate verification or use HTTP downloads.
# HEAD checks on 2026-09-22 matched ETag, length and Last-Modified against the
# official HTTP URLs for the annotation ZIP and val2017/000000000139.jpg.
COCO_TRANSPORT_BASE = 'https://s3.amazonaws.com/images.cocodataset.org/'
ANNOTATIONS_SOURCE_URL = (
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
)
ANNOTATIONS_URL = COCO_TRANSPORT_BASE + 'annotations/annotations_trainval2017.zip'
SCHEMA_VERSION = 1
SPLITS = ('train2017', 'val2017')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
            + '\n').encode('utf-8')


def write_once(path: Path, payload: bytes) -> None:
    """Atomic publication; matching existing output is allowed, conflicts fail."""
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f'Refusing to overwrite conflicting file: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another preparation process owns {path}') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for split in SPLITS:
            info = archive.getinfo(f'annotations/instances_{split}.json')
            if info.is_dir() or info.file_size <= 0:
                raise ValueError(f'Invalid annotation ZIP member: {info.filename}')
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f'ZIP CRC verification failed: {bad_member}')


def download(url: str, destination: Path, validator, retries: int = 4,
             stop_event: threading.Event | None = None) -> None:
    """HTTPS only, resumable partial files, validation before atomic rename.

    HTTP Range is honored only if the server returns the requested start offset.
    Servers that ignore Range cause a clean restart of the partial file. A fully
    downloaded but invalid partial is discarded; existing final files are not.
    """
    def check_cancelled():
        if stop_event is not None and stop_event.is_set():
            raise CancelledError('Another image download failed; preserving resumable partial')

    check_cancelled()
    if urlparse(url).scheme != 'https':
        raise ValueError(f'Refusing unencrypted download: {url}')
    if destination.exists():
        validator(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    last_error = None
    for attempt in range(retries):
        transfer_complete = False
        try:
            check_cancelled()
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {'User-Agent': 'AI-Class-COCO-subset/1.0',
                       'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = Request(url, headers=headers)
            try:
                response = urlopen(request, timeout=60)
            except HTTPError as error:
                if error.code != 416 or not offset:
                    raise
                # An interrupted process can leave an already complete partial.
                transfer_complete = True
                validator(temporary)
                os.replace(temporary, destination)
                return
            with response:
                if urlparse(response.geturl()).scheme != 'https':
                    raise RuntimeError('Refusing a redirect away from HTTPS')
                if response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not match or int(match[1]) != offset:
                        raise RuntimeError(f'Unexpected Content-Range: {content_range}')
                    expected_total = int(match[3])
                    mode = 'ab' if offset else 'wb'
                elif response.status == 200:
                    offset = 0
                    mode = 'wb'
                    expected_total = int(response.headers.get('Content-Length', '0'))
                else:
                    raise RuntimeError(f'Unexpected HTTP status: {response.status}')
                copied, last_message = offset, time.monotonic()
                with temporary.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        check_cancelled()
                        output.write(chunk)
                        copied += len(chunk)
                        if time.monotonic() - last_message >= 15:
                            print(f'{destination.name}: {copied / 1e6:.1f}'
                                  f'/{expected_total / 1e6:.1f} MB', flush=True)
                            last_message = time.monotonic()
                    output.flush()
                    os.fsync(output.fileno())
                if expected_total and copied != expected_total:
                    raise RuntimeError(f'Incomplete response: {copied}/{expected_total}')
            transfer_complete = True
            validator(temporary)
            os.replace(temporary, destination)
            return
        except CancelledError:
            raise
        except Exception as error:
            last_error = error
            if transfer_complete and temporary.exists():
                temporary.unlink()  # Only this function's uncommitted download.
            if attempt + 1 < retries:
                print(f'Retrying {destination.name} ({attempt + 1}/{retries}): '
                      f'{error}', flush=True)
                delay = min(2 ** attempt, 8)
                if stop_event is not None:
                    if stop_event.wait(delay):
                        check_cancelled()
                else:
                    time.sleep(delay)
    raise RuntimeError(f'Download failed after {retries} attempts: {url}') from last_error


def prepare_sources(cache: Path) -> dict:
    """Extract only exact known members, never trust ZIP paths with extractall."""
    with exclusive_lock(cache / '.prepare.lock'):
        zip_path = cache / 'annotations_trainval2017.zip'
        download(ANNOTATIONS_URL, zip_path, validate_zip)
        record = {
            'url': ANNOTATIONS_URL, 'official_source_url': ANNOTATIONS_SOURCE_URL,
            'transport': 'Same official images.cocodataset.org S3 bucket; '
                         'path-style HTTPS with TLS certificate verification enabled',
            'zip_sha256': sha256(zip_path), 'files': {},
        }
        with zipfile.ZipFile(zip_path) as archive:
            for split in SPLITS:
                name = f'instances_{split}.json'
                destination = cache / name
                partial = destination.with_name(destination.name + '.partial')
                with archive.open(f'annotations/{name}') as source, partial.open('wb') as out:
                    shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
                extracted_hash = sha256(partial)
                if destination.exists():
                    if sha256(destination) != extracted_hash:
                        raise RuntimeError(f'Cached JSON differs from official ZIP: {destination}')
                    partial.unlink()
                else:
                    os.replace(partial, destination)
                record['files'][split] = {'name': name, 'sha256': extracted_hash}
        write_once(cache / 'source_manifest.json', json_bytes(record))
        return record


def valid_image_record(record: dict, split: str) -> None:
    name = record['file_name']
    if not re.fullmatch(r'\d{12}\.jpg', name):
        raise ValueError(f'Unsafe or unexpected image filename: {name!r}')
    if not isinstance(record['id'], int):
        raise ValueError('Image IDs must be integers')
    if record['width'] <= 0 or record['height'] <= 0:
        raise ValueError(f'Invalid image dimensions: {name}')
    parsed = urlparse(record['coco_url'])
    if (parsed.scheme not in ('http', 'https')
            or parsed.netloc != 'images.cocodataset.org'
            or parsed.path != f'/{split}/{name}' or parsed.query or parsed.fragment):
        raise ValueError(f'Unexpected official coco_url: {record["coco_url"]}')


def validate_annotation(annotation: dict, image_ids: set, category_ids: set) -> None:
    if annotation['image_id'] not in image_ids or annotation['category_id'] not in category_ids:
        raise ValueError(f'Annotation has dangling reference: {annotation["id"]}')
    segmentation = annotation.get('segmentation')
    if isinstance(segmentation, list):
        if (not segmentation or any(not isinstance(polygon, list)
                                   or len(polygon) < 6 or len(polygon) % 2
                                   for polygon in segmentation)):
            raise ValueError(f'Missing or malformed polygons: {annotation["id"]}')
    elif isinstance(segmentation, dict):
        if not segmentation.get('counts') or len(segmentation.get('size', [])) != 2:
            raise ValueError(f'Malformed RLE segmentation: {annotation["id"]}')
    else:
        raise ValueError(f'Missing instance segmentation: {annotation["id"]}')
    area = annotation['area']
    if not isinstance(area, (int, float)) or not math.isfinite(area) or area < 0:
        raise ValueError(f'Invalid instance area: {annotation["id"]}')


def sample_split(source: dict, split: str, count: int, seed: int) -> dict:
    """Uniform random image sampling, without filtering object size or labels."""
    images = sorted(source['images'], key=lambda image: image['id'])
    all_ids = [image['id'] for image in images]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(f'Duplicate source image IDs in {split}')
    if not 1 <= count <= len(images):
        raise ValueError(f'{split} count must be in [1, {len(images)}], got {count}')
    selected = sorted(random.Random(f'{seed}:{split}').sample(images, count),
                      key=lambda image: image['id'])
    image_ids = {image['id'] for image in selected}
    category_ids = {category['id'] for category in source['categories']}
    annotations = sorted((ann for ann in source['annotations']
                          if ann['image_id'] in image_ids), key=lambda ann: ann['id'])
    if len({ann['id'] for ann in annotations}) != len(annotations):
        raise ValueError(f'Duplicate annotation IDs in {split}')
    for image in selected:
        valid_image_record(image, split)
    for annotation in annotations:
        validate_annotation(annotation, image_ids, category_ids)
    result = {key: source[key] for key in ('info', 'licenses', 'categories') if key in source}
    result.update(images=selected, annotations=annotations)
    return result


def statistics(subset: dict) -> dict:
    annotations = subset['annotations']
    # COCOeval's small area range is inclusive [0, 32**2]. Crowd annotations
    # are counted separately because they are ignored as ordinary positives.
    small = [ann for ann in annotations if 0 <= ann['area'] <= 32 ** 2]
    noncrowd = [ann for ann in annotations if not ann.get('iscrowd', 0)]
    return {
        'images': len(subset['images']),
        'instances_including_crowd': len(annotations),
        'instances_noncrowd': len(noncrowd),
        'small_instances_including_crowd': len(small),
        'small_instances_noncrowd': sum(not ann.get('iscrowd', 0) for ann in small),
        'images_with_small_instances': len({ann['image_id'] for ann in small}),
        'images_without_annotations': len(subset['images'])
            - len({ann['image_id'] for ann in annotations}),
        'categories_present': len({ann['category_id'] for ann in annotations}),
        'small_area_rule': 'COCOeval inclusive mask annotation area [0, 1024] px^2',
    }


def validate_image(path: Path, record: dict) -> None:
    from PIL import Image
    with Image.open(path) as image:
        if image.format != 'JPEG' or image.size != (record['width'], record['height']):
            raise ValueError(f'Image format/dimensions do not match annotations: {path}')
        image.load()  # Fully decode; do not enable LOAD_TRUNCATED_IMAGES.


def prepare_images(root: Path, subsets: dict, previous_manifest: dict | None,
                   workers: int) -> dict:
    previous = {}
    if previous_manifest:
        previous = {item['path']: item for item in previous_manifest['images']}

    stop_event = threading.Event()

    def prepare(split, record):
        relative = f'{split}/{record["file_name"]}'
        path = root / relative
        url = COCO_TRANSPORT_BASE + relative
        download(url, path, lambda candidate: validate_image(candidate, record),
                 stop_event=stop_event)
        item = {'path': relative, 'image_id': record['id'], 'url': url,
                'official_source_url': record['coco_url'],
                'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if relative in previous and item != previous[relative]:
            raise RuntimeError(f'Image differs from completed manifest: {path}')
        return item

    result, last_message = [], time.monotonic()
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(prepare, split, record)
                   for split in SPLITS for record in subsets[split]['images']]
        for future in as_completed(futures):
            result.append(future.result())
            if time.monotonic() - last_message >= 10 or len(result) == len(futures):
                print(f'Images downloaded and verified: {len(result)}/{len(futures)}', flush=True)
                last_message = time.monotonic()
    except BaseException:
        stop_event.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        # Pending images are cancelled, not allowed to drain the full queue.
        # Running transfers notice stop_event between chunks/retries; a blocked
        # network read still has its 60-second timeout before thread shutdown.
        executor.shutdown(wait=True, cancel_futures=True)
    return sorted(result, key=lambda item: item['path'])


def prepare(args) -> dict:
    root = Path(args.data_root).expanduser().resolve()
    cache = root.parent / '_coco2017_cache'
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / '.prepare.lock'):
        selection_path = root / 'selection.json'
        unexpected = [path.name for path in root.iterdir()
                      if path.name not in {'.prepare.lock', 'selection.json.partial'}]
        if not selection_path.exists() and unexpected:
            raise RuntimeError(f'Refusing nonempty unowned subset directory: {root}; {unexpected}')
        if selection_path.exists():
            previous_selection = json.loads(selection_path.read_text())
            requested = {'train2017': args.train_count, 'val2017': args.val_count}
            if (previous_selection.get('seed') != args.seed
                    or previous_selection.get('counts') != requested):
                raise RuntimeError('Subset seed/count conflict. Choose a NEW --data-root; '
                                   'existing subsets are never overwritten.')
        sources = prepare_sources(cache)
        subsets = {}
        source_counts = {}
        for split, count in zip(SPLITS, (args.train_count, args.val_count)):
            source = json.loads((cache / sources['files'][split]['name']).read_text())
            source_counts[split] = {'images': len(source['images']),
                                    'instances': len(source['annotations'])}
            subsets[split] = sample_split(source, split, count, args.seed)
            del source
        train_ids = {image['id'] for image in subsets['train2017']['images']}
        val_ids = {image['id'] for image in subsets['val2017']['images']}
        if train_ids & val_ids:
            raise RuntimeError('Train/validation image ID overlap detected')
        if subsets['train2017']['categories'] != subsets['val2017']['categories']:
            raise RuntimeError('Train/validation category definitions differ')
        plan = {
            'schema_version': SCHEMA_VERSION, 'dataset': 'COCO 2017 development subset',
            'seed': args.seed, 'counts': {'train2017': args.train_count, 'val2017': args.val_count},
            'sampling': 'uniform images; separate Random(f"{seed}:{split}"); no size/label filter',
            'sources': sources,
            'image_ids': {split: [image['id'] for image in subsets[split]['images']]
                          for split in SPLITS},
        }
        write_once(selection_path, json_bytes(plan))
        annotations = {}
        for split in SPLITS:
            path = root / 'annotations' / f'instances_{split}.json'
            write_once(path, json_bytes(subsets[split]))
            annotations[split] = {'path': str(path.relative_to(root)), 'sha256': sha256(path)}
        manifest_path = root / 'manifest.json'
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        images = prepare_images(root, subsets, previous, args.workers)
        manifest = {
            **plan, 'status': 'complete',
            'prepared_at_utc': previous['prepared_at_utc'] if previous
                else datetime.now(timezone.utc).isoformat(),
            'source_counts': source_counts,
            'statistics': {split: statistics(subsets[split]) for split in SPLITS},
            'subset_annotations': annotations, 'images': images,
            'images_total_bytes': sum(image['bytes'] for image in images),
            'leakage_checks': {'train_val_image_id_overlap': 0,
                               'train_val_file_name_overlap': len(
                                   {i['file_name'] for i in subsets['train2017']['images']}
                                   & {i['file_name'] for i in subsets['val2017']['images']})},
            'limitations': [
                'Development subset; metrics are NOT the complete official COCO benchmark.',
                'Standard COCO instance data supplies RGB images, not measured depth.',
                'COCO-pretrained checkpoints have seen the training split; this is a '
                'pipeline/fine-tuning experiment, not a new held-out pretraining evaluation.',
                'Keep validation images out of training; final coursework requires its '
                'designated RGB-D dataset and split.',
                'Uniform sampling may omit rare classes; do not tune on validation repeatedly '
                'and treat resulting small-subset gains as confirmed generalization.',
            ],
        }
        if manifest['leakage_checks']['train_val_file_name_overlap']:
            raise RuntimeError('Train/validation filename overlap detected')
        write_once(manifest_path, json_bytes(manifest))
        print(json.dumps({'data_root': str(root), 'manifest': str(manifest_path),
                          'statistics': manifest['statistics'],
                          'images_total_bytes': manifest['images_total_bytes'],
                          'leakage_checks': manifest['leakage_checks']}, indent=2), flush=True)
        return manifest


def verify_subset(data_root) -> dict:
    """Read-only, offline validation of a completed portable subset.

    Checks the transferred files against their existing manifest, without
    downloading or consulting the original annotation ZIP/JSON cache. Source
    hashes remain provenance records; this mode does not reverify those absent
    source files or independently regenerate the random sample from full COCO.
    """
    root = Path(data_root).expanduser().resolve()
    selection = json.loads((root / 'selection.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('schema_version') != SCHEMA_VERSION:
        raise RuntimeError('Verification requires a completed supported subset manifest')
    for key, value in selection.items():
        if manifest.get(key) != value:
            raise RuntimeError(f'Selection/manifest mismatch: {key}')
    if set(selection) != {'schema_version', 'dataset', 'seed', 'counts',
                          'sampling', 'sources', 'image_ids'}:
        raise RuntimeError('Incomplete or unexpected selection metadata')

    def checked_path(relative):
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise RuntimeError(f'Missing or unsafe subset file: {relative}')
        return path

    subsets, expected_images = {}, {}
    for split in SPLITS:
        ann_record = manifest['subset_annotations'][split]
        relative = f'annotations/instances_{split}.json'
        if ann_record['path'] != relative:
            raise RuntimeError(f'Unexpected subset annotation path: {ann_record["path"]}')
        path = checked_path(relative)
        if sha256(path) != ann_record['sha256']:
            raise RuntimeError(f'Annotation SHA256 mismatch: {relative}')
        subset = json.loads(path.read_text())
        ids = [image['id'] for image in subset['images']]
        category_ids = {category['id'] for category in subset['categories']}
        if len(category_ids) != len(subset['categories']):
            raise RuntimeError(f'Duplicate category IDs: {split}')
        if (len(set(ids)) != len(ids) or ids != selection['image_ids'][split]
                or len(ids) != selection['counts'][split]):
            raise RuntimeError(f'Image selection/count/uniqueness mismatch: {split}')
        annotation_ids = [annotation['id'] for annotation in subset['annotations']]
        if len(set(annotation_ids)) != len(annotation_ids):
            raise RuntimeError(f'Duplicate annotation IDs: {split}')
        for annotation in subset['annotations']:
            validate_annotation(annotation, set(ids), category_ids)
        for image in subset['images']:
            valid_image_record(image, split)
            relative_image = f'{split}/{image["file_name"]}'
            if relative_image in expected_images:
                raise RuntimeError(f'Duplicate image filename: {relative_image}')
            expected_images[relative_image] = image
        actual_names = {path.name for path in (root / split).iterdir()}
        if actual_names != {image['file_name'] for image in subset['images']}:
            raise RuntimeError(f'Missing or unexpected image files: {split}')
        if statistics(subset) != manifest['statistics'][split]:
            raise RuntimeError(f'Subset statistics mismatch: {split}')
        subsets[split] = subset

    if subsets[SPLITS[0]]['categories'] != subsets[SPLITS[1]]['categories']:
        raise RuntimeError('Train/validation category definitions differ')
    leakage = {
        'train_val_image_id_overlap': len(
            set(selection['image_ids'][SPLITS[0]]) & set(selection['image_ids'][SPLITS[1]])),
        'train_val_file_name_overlap': len(
            {image['file_name'] for image in subsets[SPLITS[0]]['images']}
            & {image['file_name'] for image in subsets[SPLITS[1]]['images']}),
    }
    if any(leakage.values()) or leakage != manifest['leakage_checks']:
        raise RuntimeError('Train/validation leakage check failed')
    records = {record['path']: record for record in manifest['images']}
    if len(records) != len(manifest['images']) or set(records) != set(expected_images):
        raise RuntimeError('Manifest image list differs from subset annotations')
    total_bytes = 0
    for index, (relative, image) in enumerate(sorted(expected_images.items()), start=1):
        record = records[relative]
        path = checked_path(relative)
        if (record['image_id'] != image['id']
                or record['official_source_url'] != image['coco_url']
                or record['url'] != COCO_TRANSPORT_BASE + relative):
            raise RuntimeError(f'Image metadata/source mismatch: {relative}')
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Image size/SHA256 mismatch: {relative}')
        validate_image(path, image)
        total_bytes += record['bytes']
        if index % 250 == 0 or index == len(expected_images):
            print(f'Offline image verification: {index}/{len(expected_images)}', flush=True)
    if total_bytes != manifest['images_total_bytes']:
        raise RuntimeError('Manifest images_total_bytes mismatch')
    print(json.dumps({
        'status': 'verified', 'data_root': str(root), 'offline': True,
        'original_source_cache_checked': False,
        'statistics': manifest['statistics'], 'leakage_checks': leakage,
        'images_total_bytes': total_bytes,
    }, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/workspace/data/coco_subset')
    parser.add_argument('--train-count', type=int, default=1000)
    parser.add_argument('--val-count', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true',
                        help='Offline read-only verification of an existing completed subset; '
                             'does not require the full COCO annotation cache')
    args = parser.parse_args()
    if args.verify_only:
        verify_subset(args.data_root)
        return
    if args.train_count < 1 or args.val_count < 1 or not 1 <= args.workers <= 16:
        parser.error('counts must be positive; workers must be in [1, 16]')
    prepare(args)


if __name__ == '__main__':
    main()

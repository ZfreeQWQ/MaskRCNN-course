"""Course data indexing and RGB-D decoding, usable without TensorFlow.

Only train/val annotations are accepted. Test labels are never opened.
Compressed RLE follows the COCO maskApi.c format (COCO API, BSD license).
"""
import gzip
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def rle_counts(segmentation):
    encoded = segmentation['counts']
    if isinstance(encoded, list):
        return encoded
    if isinstance(encoded, bytes):
        encoded = encoded.decode('ascii')
    result, pos = [], 0
    while pos < len(encoded):
        value, shift = 0, 0
        while True:
            byte = ord(encoded[pos]) - 48
            pos += 1
            if not 0 <= byte <= 63:
                raise ValueError('Invalid COCO compressed RLE character')
            value |= (byte & 31) << (5 * shift)
            shift += 1
            if not byte & 32:
                if byte & 16:
                    value |= -1 << (5 * shift)
                break
        if len(result) > 2:
            value += result[-2]
        result.append(value)
    return result


def decode_mask(segmentation):
    """Portable decoder, also used for CPU inspection when pycocotools is absent."""
    height, width = map(int, segmentation['size'])
    if height <= 0 or width <= 0:
        raise ValueError('Invalid mask dimensions')
    runs = rle_counts(segmentation)
    if not runs or min(runs) < 0 or sum(runs) != height * width:
        raise ValueError('RLE runs do not cover the declared image')
    flat = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    for number, length in enumerate(runs):
        if number % 2:
            flat[pos:pos + length] = 1
        pos += length
    return flat.reshape((height, width), order='F').astype(np.bool_)


def normalized_depth(raw, scale_m, invalid_code, max_m):
    """Map metric depth to [0,255]; invalid pixels become the neutral 127.5."""
    if not np.isfinite(scale_m) or scale_m <= 0 or not np.isfinite(max_m) or max_m <= 0:
        raise ValueError('Depth scale and clipping maximum must be positive')
    valid = raw != invalid_code
    depth_m = raw.astype(np.float32) * float(scale_m)
    depth = np.clip(depth_m / float(max_m), 0.0, 1.0) * 255.0
    depth[~valid] = 127.5
    return depth.astype(np.float32), valid


def depth_array(raw, expected_shape):
    # Older Pillow versions expose a 16-bit PNG as mode I / int32.
    if (raw.shape != expected_shape or not np.issubdtype(raw.dtype, np.integer)
            or raw.dtype.itemsize < 2 or raw.min() < 0 or raw.max() > 65535):
        raise ValueError('Expected a registered 16-bit integer depth image')
    return raw.astype(np.uint16, copy=False)


class BlenderIndex:
    """Compact image metadata and a small LRU cache of raw annotation shards."""
    def __init__(self, root, subset, limit=0, small_area=1024):
        if subset not in ('train', 'val'):
            raise ValueError('Only train/val annotations are supported; test labels are excluded')
        if limit < 0 or small_area <= 0:
            raise ValueError('Invalid limit or small-area threshold')
        self.root = Path(root).resolve()
        self.subset = subset
        self.small_area = small_area
        self.records = []
        self._cache = OrderedDict()
        paths = sorted((self.root / 'annotations' / subset).glob('*.json.gz'))
        if not paths:
            raise ValueError('No annotation shards in {}'.format(self.root / 'annotations' / subset))
        seen = set()
        for path in paths:
            with gzip.open(str(path), 'rt', encoding='utf-8') as stream:
                raw = json.load(stream)
            if any(c['id'] != 1 or c['name'] != 'component' for c in raw.get('categories', [])):
                raise ValueError('Expected the single component category in {}'.format(path))
            grouped = defaultdict(list)
            for annotation in raw.get('annotations', []):
                if annotation.get('category_id') == 1:
                    grouped[annotation['image_id']].append(annotation)
            for metadata in raw['images']:
                if metadata['id'] in seen:
                    raise ValueError('Duplicate image id {}'.format(metadata['id']))
                seen.add(metadata['id'])
                # Drop large support_rle/camera fields; retain only what this pipeline uses.
                keys = ('id', 'file_name', 'depth_file_name', 'width', 'height',
                        'depth_scale_m', 'depth_invalid_code', 'scene_id', 'model_ids',
                        'density', 'composition')
                record = {k: metadata[k] for k in keys if k in metadata}
                for key in ('file_name', 'depth_file_name', 'depth_scale_m', 'depth_invalid_code'):
                    if key not in record:
                        raise ValueError('Missing {} for image {}'.format(key, metadata['id']))
                annotations = grouped.get(metadata['id'], [])
                record['shard'] = str(path)
                record['instances'] = sum(float(a['area']) > 0 for a in annotations)
                record['small_count'] = sum(0 < float(a['area']) <= small_area for a in annotations)
                self.records.append(record)
                if limit and len(self.records) >= limit:
                    return
        if not self.records:
            raise ValueError('The selected split has no images')

    def path(self, relative):
        candidate = (self.root / relative.replace('\\', '/')).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise ValueError('Image path leaves the dataset root')
        return candidate

    def annotations(self, index):
        record = self.records[index]
        shard = record['shard']
        if shard not in self._cache:
            with gzip.open(shard, 'rt', encoding='utf-8') as stream:
                raw = json.load(stream)
            grouped = defaultdict(list)
            for annotation in raw.get('annotations', []):
                if annotation.get('category_id') == 1:
                    grouped[annotation['image_id']].append(annotation)
            self._cache[shard] = grouped
            while len(self._cache) > 2:
                self._cache.popitem(last=False)
        self._cache.move_to_end(shard)
        return self._cache[shard].get(record['id'], [])

    def load_masks(self, index):
        record = self.records[index]
        masks = []
        for annotation in self.annotations(index):
            mask = decode_mask(annotation['segmentation'])
            if mask.shape != (record['height'], record['width']):
                raise ValueError('Mask/image shape mismatch for {}'.format(record['id']))
            if int(mask.sum()) != int(annotation['area']):
                raise ValueError('Mask/area mismatch for annotation {}'.format(annotation['id']))
            if mask.any():
                masks.append(mask)
        if not masks:
            return np.empty((record['height'], record['width'], 0), dtype=np.bool_), np.empty(0, dtype=np.int32)
        return np.stack(masks, axis=-1), np.ones(len(masks), dtype=np.int32)

    def load_image(self, index, modality='rgbd', depth_max_m=2.0):
        record = self.records[index]
        with Image.open(str(self.path(record['file_name']))) as image:
            rgb = np.array(image.convert('RGB'), dtype=np.float32)
        if rgb.shape[:2] != (record['height'], record['width']):
            raise ValueError('RGB metadata dimensions do not match')
        if modality == 'rgb':
            return rgb
        if modality != 'rgbd':
            raise ValueError('Unknown modality {}'.format(modality))
        with Image.open(str(self.path(record['depth_file_name']))) as image:
            raw = np.array(image)
        raw = depth_array(raw, rgb.shape[:2])
        depth, _ = normalized_depth(raw, record['depth_scale_m'],
                                    record['depth_invalid_code'], depth_max_m)
        return np.concatenate([rgb, depth[..., None]], axis=-1)

    def statistics(self):
        records = self.records
        return {'subset': self.subset, 'images': len(records),
                'instances': sum(r['instances'] for r in records),
                'small_instances': sum(r['small_count'] for r in records),
                'images_with_small': sum(r['small_count'] > 0 for r in records),
                'max_instances_per_image': max(r['instances'] for r in records),
                'scenes': len(set(r.get('scene_id') for r in records)),
                'density': dict(Counter(r.get('density') for r in records)),
                'composition': dict(Counter(r.get('composition') for r in records))}

"""Validate COCO RLE and Kaggle per-image JSON CSV; empty predictions must be []."""
import argparse
import csv
import math
import json
import numpy as np
from pycocotools import mask as mask_util
from common import load_images, read_json, write_json, submission_instances


def validate(metadata, predictions, csv_path, expected_images=2112, manifest=None, split='test',
             csv_format='mask-json'):
    if csv_format not in ('mask-json', 'legacy-boxes'):
        raise ValueError('Unknown CSV format')
    images = load_images(metadata, split)
    if len(images) != expected_images:
        raise ValueError(f'Expected {expected_images} metadata images, got {len(images)}')
    by_id = {row['id']: row for row in images}
    if split == 'test' and expected_images == 2112:
        if any((im['height'], im['width']) != (1024, 1024) for im in images):
            raise ValueError('Submission masks must be 1024 x 1024')
    rows = read_json(predictions)
    if not isinstance(rows, list):
        raise ValueError('COCO results must be a list')
    expected_boxes = {image_id: [] for image_id in by_id}
    expected_instances = {image_id: [] for image_id in by_id}
    for index, row in enumerate(rows):
        image_id = row['image_id']
        if type(image_id) is not int or image_id not in by_id or type(row['category_id']) is not int or row['category_id'] != 1:
            raise ValueError(f'Result {index}: unknown id/class')
        info = by_id[image_id]
        box = np.asarray(row['bbox'], dtype=float)
        if type(row['score']) not in (int, float):
            raise ValueError(f'Result {index}: score must be a number')
        score = float(row['score'])
        if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(score):
            raise ValueError(f'Result {index}: invalid bbox/score')
        x, y, width, height = box
        if not 0 <= score <= 1 or min(x, y) < 0 or width <= 0 or height <= 0:
            raise ValueError(f'Result {index}: score/bbox out of range')
        if x + width > info['width'] or y + height > info['height']:
            raise ValueError(f'Result {index}: bbox outside image')
        rle = row['segmentation']
        if not isinstance(rle['counts'], str) or rle['size'] != [info['height'], info['width']]:
            raise ValueError(f'Result {index}: compressed RLE size/counts invalid')
        mask = mask_util.decode(rle)
        if mask.shape != (info['height'], info['width']) or not mask.any():
            raise ValueError(f'Result {index}: empty or invalid decoded mask')
        if not np.allclose(mask_util.toBbox(rle), box, rtol=0, atol=1e-5):
            raise ValueError(f'Result {index}: bbox differs from mask bounds')
        expected_boxes[image_id].append([score, x, y, x + width, y + height])
        expected_instances[image_id].append(row)
        if len(expected_instances[image_id]) > 100:
            raise ValueError(f'{image_id}: at most 100 instances permitted')
    seen = set()
    # A 100-instance cell with complex RLEs can exceed csv's 128 KiB default.
    csv.field_size_limit(64 * 1024 * 1024)
    with open(csv_path, newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['image_id', 'PredictionString']:
            raise ValueError('CSV header must be image_id,PredictionString')
        for line in reader:
            if None in line or any(value is None for value in line.values()):
                raise ValueError('CSV row has missing/extra columns')
            image_id = int(line['image_id'])
            if image_id in seen or image_id not in by_id:
                raise ValueError('Duplicate/unknown CSV image_id')
            seen.add(image_id)
            if csv_format == 'mask-json':
                raw = line['PredictionString']
                if not raw.strip():
                    raise ValueError(f'{image_id}: empty PredictionString; use []')
                try:
                    value = json.loads(raw, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{image_id}: invalid PredictionString JSON') from exc
                if not isinstance(value, list) or len(value) > 100:
                    raise ValueError(f'{image_id}: PredictionString must be a JSON list of at most 100 instances')
                for item in value:
                    if not isinstance(item, dict) or set(item) != {'category_id', 'score', 'segmentation'}:
                        raise ValueError(f'{image_id}: invalid instance fields')
                    if type(item['category_id']) is not int or item['category_id'] != 1:
                        raise ValueError(f'{image_id}: invalid category_id')
                    if type(item['score']) not in (int, float) or not math.isfinite(item['score']) or not 0 <= item['score'] <= 1:
                        raise ValueError(f'{image_id}: invalid score')
                    rle = item['segmentation']
                    if not isinstance(rle, dict) or set(rle) != {'size', 'counts'} or not isinstance(rle['counts'], str):
                        raise ValueError(f'{image_id}: invalid compressed RLE')
                    if rle['size'] != [by_id[image_id]['height'], by_id[image_id]['width']]:
                        raise ValueError(f'{image_id}: invalid mask size')
                if value != submission_instances(expected_instances[image_id]):
                    raise ValueError(f'{image_id}: CSV masks/scores differ from source JSON')
                continue
            values = np.asarray([float(v) for v in line['PredictionString'].split()])
            if len(values) % 5 or not np.isfinite(values).all():
                raise ValueError(f'{image_id}: PredictionString requires groups of 5 finite numbers')
            got = values.reshape(-1, 5)
            expected = np.asarray(sorted(expected_boxes[image_id], key=lambda box: -box[0])).reshape(-1, 5)
            # The exporter defines descending-score order; compare exact boxes with rounding tolerance.
            if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-9, atol=1e-6):
                raise ValueError(f'{image_id}: CSV score/xyxy boxes differ from JSON')
    if seen != set(by_id):
        raise ValueError(f'CSV missing {len(set(by_id)-seen)} images')
    if manifest:
        info = read_json(manifest)
        if info.get('split') != split or (split == 'test' and info.get('smoke')) or set(info['image_ids']) != seen:
            raise ValueError('Manifest is smoke/partial/wrong split')
    return dict(valid=True, num_images=len(images), num_instances=len(rows),
                csv_rows=len(seen), empty_images=sum(not boxes for boxes in expected_boxes.values()),
                split=split, csv_format=csv_format,
                submission_ready=split == 'test' and expected_images == 2112 and csv_format == 'mask-json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--manifest')
    p.add_argument('--output')
    p.add_argument('--csv-format', choices=['mask-json', 'legacy-boxes'], default='mask-json',
                   help='Current Kaggle format; legacy-boxes is for historical artifacts only')
    p.add_argument('--split', choices=['test', 'val'], default='test', help='Val is format smoke only, never a final submission')
    p.add_argument('--expected-images', type=int, default=2112, help='Override for synthetic smoke tests only')
    args = p.parse_args()
    result = validate(args.metadata, args.predictions, args.csv, args.expected_images, args.manifest, args.split, args.csv_format)
    if args.output:
        write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()

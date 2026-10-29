"""Shared I/O. Test code accepts only image metadata, never test annotations."""
from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_util

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
for path in (str(PROJECT), str(ROOT / 'detectron2')):
    if path not in sys.path:
        sys.path.insert(0, path)


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_images(path, split=None):
    """Read a metadata JSON. Reject annotation-bearing files for test inference."""
    obj = read_json(path)
    if split == 'test' and isinstance(obj, dict) and 'annotations' in obj:
        raise ValueError('Test metadata must not contain annotations; use test_images.json only')
    images = obj['images'] if isinstance(obj, dict) else obj
    allowed = ('id', 'file_name', 'depth_file_name', 'height', 'width',
               'depth_scale_m', 'depth_invalid_code', 'scene_id', 'split')
    images = [{key: row[key] for key in allowed if key in row} for row in images]
    ids = [row['id'] for row in images]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image ids')
    if split and any(row.get('split', split) != split for row in images):
        raise ValueError('Metadata split mismatch')
    return sorted(images, key=lambda row: row['id'])


def image_record(info, data_root):
    record = dict(info)
    record['image_id'] = info['id']
    for key in ('file_name', 'depth_file_name'):
        if key in record:
            record[key] = str(Path(data_root) / record[key])
    return record


def encode_instances(instances, image_id):
    """Use mask bounds for both formats, avoiding bbox/mask disagreement."""
    instances = instances.to('cpu')
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    results = []
    for mask, score, category in zip(masks, scores, classes):
        if int(category) != 0:
            raise ValueError(f'Expected one class (contiguous id 0), got {category}')
        if not math.isfinite(float(score)):
            raise ValueError('Non-finite score')
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        if int(mask_util.area(rle)) == 0:
            continue
        bbox = mask_util.toBbox(rle).tolist()
        rle['counts'] = rle['counts'].decode('ascii')
        results.append(dict(image_id=int(image_id), category_id=1, bbox=bbox,
                            score=float(score), segmentation=rle))
    return results


def submission_instances(predictions):
    """Kaggle cell schema: category, confidence and compressed COCO RLE."""
    if len(predictions) > 100:
        raise ValueError('At most 100 instances per image are permitted')
    return [dict(category_id=p['category_id'], score=p['score'],
                 segmentation=p['segmentation'])
            for p in sorted(predictions, key=lambda p: -p['score'])]


def write_csv(path, images, predictions, csv_format='mask-json'):
    if csv_format not in ('mask-json', 'legacy-boxes'):
        raise ValueError(f'Unknown CSV format: {csv_format}')
    grouped = {row['id']: [] for row in images}
    for prediction in predictions:
        grouped[prediction['image_id']].append(prediction)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id', 'PredictionString'])
        for info in images:
            if csv_format == 'mask-json':
                value = json.dumps(submission_instances(grouped[info['id']]),
                                   separators=(',', ':'), ensure_ascii=True, allow_nan=False)
                writer.writerow([info['id'], value])
                continue
            boxes = []
            for prediction in sorted(grouped[info['id']], key=lambda p: -p['score']):
                x, y, width, height = prediction['bbox']
                boxes.extend([prediction['score'], x, y, x + width, y + height])
            writer.writerow([info['id'], ' '.join(format(float(x), '.12g') for x in boxes)])


def require_depth_metadata(images):
    for info in images:
        scale = info.get('depth_scale_m')
        invalid = info.get('depth_invalid_code')
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"image_id={info['id']}: missing/invalid depth_scale_m; RGB-D inference requires original per-image calibration metadata, never guessed values")
        if not info.get('depth_file_name') or not isinstance(invalid, int):
            raise ValueError(f"image_id={info['id']}: missing depth_file_name/depth_invalid_code; provide original input metadata")


def predictor_from_args(args, images=None):
    from rgbd_seg.config import setup_cfg
    from rgbd_seg.inference import load_predictor
    opts = ('MODEL.ROI_HEADS.SCORE_THRESH_TEST', str(args.threshold))
    cfg = setup_cfg(args.config, opts)
    if cfg.PROJECT.USE_DEPTH and images is not None:
        # Fail before constructing a model / allocating GPU memory when test calibration is absent.
        require_depth_metadata(images)
    return load_predictor(args.config, args.weights, device=args.device, opts=opts)


def predictor_from_args_m2f(args, images=None):
    """Mask2Former counterpart of predictor_from_args. Unlike Mask R-CNN's ROI_HEADS, stock
    MaskFormer.instance_inference always returns exactly TEST.DETECTIONS_PER_IMAGE instances
    per image regardless of score -- there is no equivalent config-level score cutoff to pass
    at construction time. Thresholding for the final submission happens post-hoc on the decoded
    results list (see tools/infer_m2f.py's --score-threshold), not here."""
    from rgbd_seg.m2f.config import setup_cfg
    from rgbd_seg.m2f.inference import load_predictor
    cfg = setup_cfg(args.config)
    if cfg.PROJECT.USE_DEPTH and images is not None:
        require_depth_metadata(images)
    return load_predictor(args.config, args.weights, device=args.device)


def add_model_args(parser):
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Fix on val; 0.05 retains detections for COCO AP')

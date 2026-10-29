"""RGB-D comparison sheet from saved predictions. GT is restricted to train/val."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util
from common import image_record, load_images, read_json, require_depth_metadata


def decode_annotation(annotation, height, width):
    segmentation = annotation['segmentation']
    if isinstance(segmentation, list):
        segmentation = mask_util.merge(mask_util.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get('counts'), list):
        segmentation = mask_util.frPyObjects(segmentation, height, width)
    return mask_util.decode(segmentation).astype(bool)


def overlay(rgb, annotations, threshold=0.0):
    array = np.asarray(rgb).copy()
    draw_boxes = []
    for index, annotation in enumerate(annotations):
        if annotation.get('score', 1) < threshold:
            continue
        mask = decode_annotation(annotation, array.shape[0], array.shape[1])
        color = np.array([(53 + index * 73) % 220 + 20,
                          (89 + index * 47) % 220 + 20, (127 + index * 91) % 220 + 20])
        array[mask] = (0.45 * array[mask] + 0.55 * color).astype(np.uint8)
        if 'bbox' in annotation:
            x, y, w, h = annotation['bbox']
            draw_boxes.append((x, y, x+w, y+h))
    result = Image.fromarray(array)
    draw = ImageDraw.Draw(result)
    for box in draw_boxes:
        draw.rectangle(box, outline='white', width=1)
    return result


def depth_view(path, info):
    require_depth_metadata([info])
    raw = np.asarray(Image.open(path))
    if raw.dtype.kind not in 'ui' or raw.ndim != 2:
        raise ValueError('Expected one-channel integer PNG depth')
    valid = raw != info.get('depth_invalid_code', 0)
    meters = raw.astype(np.float32) * float(info['depth_scale_m'])
    pixels = np.zeros((*raw.shape, 3), dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(meters[valid], [1, 99])
        norm = np.clip((meters - low) / max(high-low, 1e-8), 0, 1)
        from matplotlib import colormaps
        pixels = (colormaps['viridis'](norm)[..., :3] * 255).astype(np.uint8)
        pixels[~valid] = 0
        caption = f'Depth meters (p1={low:.3f}, p99={high:.3f}, invalid={1-valid.mean():.1%})'
    else:
        caption = 'Depth: all pixels invalid'
    return Image.fromarray(pixels), caption


def make_demo(info, data_root, predictions, output, ground_truth=None, threshold=0.5):
    record = image_record(info, data_root)
    rgb = Image.open(record['file_name']).convert('RGB')
    depth, caption = depth_view(record['depth_file_name'], info)
    panels = [('RGB', rgb), (caption, depth)]
    if ground_truth is not None:
        panels.append(('Ground truth (val/train only)', overlay(rgb, ground_truth)))
    for label, rows in predictions:
        selected = [row for row in rows if row['image_id'] == info['id']]
        panels.append((f'{label} (score >= {threshold:g})', overlay(rgb, selected, threshold)))
    size = 512
    sheet = Image.new('RGB', (size * len(panels), size + 64), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(panels):
        sheet.paste(panel.resize((size, size)), (index*size, 64))
        draw.text((index*size+8, 8), label, fill='black')
        draw.text((index*size+8, 30), f"image_id={info['id']}", fill='black')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--image-id', type=int, required=True)
    p.add_argument('--data-root', default='/private/rgzndzy/blender')
    p.add_argument('--predictions', action='append', default=[], metavar='LABEL=JSON')
    p.add_argument('--ground-truth', help='Val/train COCO JSON only; forbidden for test')
    p.add_argument('--threshold', type=float, default=0.5, help='Display only, does not alter predictions')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.split == 'test' and args.ground_truth:
        p.error('Test ground truth is forbidden')
    images = load_images(args.metadata, args.split)
    info = next((row for row in images if row['id'] == args.image_id), None)
    if info is None:
        raise ValueError('Unknown image id')
    predictions = []
    for item in args.predictions:
        label, path = item.split('=', 1)
        predictions.append((label, read_json(path)))
    gt = None
    if args.ground_truth:
        gt_data = read_json(args.ground_truth)
        gt_image = next(row for row in gt_data['images'] if row['id'] == info['id'])
        if gt_image.get('split', args.split) != args.split:
            raise ValueError('GT split mismatch')
        gt = [row for row in gt_data['annotations'] if row['image_id'] == info['id']]
    make_demo(info, args.data_root, predictions, args.output, gt, args.threshold)


if __name__ == '__main__':
    main()

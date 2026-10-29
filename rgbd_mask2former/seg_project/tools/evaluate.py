"""Official COCO bbox and segm AP. Predictions must cover the full val image set."""
import argparse
import copy
from common import load_images, read_json, write_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(annotations, predictions, image_ids=None):
    gt = COCO(str(annotations))
    # COCO.loadRes requires info, even though this field is optional in datasets.
    gt.dataset.setdefault('info', {})
    rows = read_json(predictions) if isinstance(predictions, (str, bytes)) else predictions
    valid_ids = set(gt.getImgIds())
    if any(row['image_id'] not in valid_ids for row in rows):
        raise ValueError('Prediction has image_id outside validation annotations')
    metrics = {}
    for kind in ('segm', 'bbox'):
        formatted = copy.deepcopy(rows)
        if kind == 'segm':
            # loadRes prefers bbox area if present: remove it for true mask-area APs.
            for row in formatted:
                row.pop('bbox', None)
        if formatted:
            dt = gt.loadRes(formatted)
        else:
            dt = COCO()
            dt.dataset = dict(images=copy.deepcopy(gt.dataset['images']),
                              categories=copy.deepcopy(gt.dataset['categories']), annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, kind)
        if image_ids is not None:
            evaluator.params.imgIds = sorted(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        metrics[kind] = {name: (float(value * 100) if value >= 0 else None)
                         for name, value in zip(names, evaluator.stats)}
    return {'metrics': metrics, 'units': 'percent (0-100); null=undefined',
            'primary_metric': 'segm/APs', 'max_detections': [1, 10, 100],
            'num_images': len(valid_ids if image_ids is None else image_ids),
            'num_predictions': len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', help='Inference manifest; checks val and full image coverage')
    p.add_argument('--smoke-limit', type=int, default=0, help='Explicit partial evaluation, not final metrics')
    p.add_argument('--split', choices=['val', 'test'], default='val',
                   help='test GT is for model selection only (2026-10-20 policy), never training')
    args = p.parse_args()
    images = load_images(args.annotations, None)
    if any(row.get('split', args.split) != args.split for row in images):
        raise ValueError('Annotation split mismatch')
    ids = [row['id'] for row in images]
    if args.smoke_limit:
        ids = ids[:args.smoke_limit]
    if args.manifest:
        manifest = read_json(args.manifest)
        if manifest['split'] != args.split or set(manifest['image_ids']) != set(ids):
            raise ValueError('Manifest must cover the evaluation image set exactly')
    rows = read_json(args.predictions)
    if args.smoke_limit:
        rows = [row for row in rows if row['image_id'] in set(ids)]
    result = evaluate(args.annotations, rows, ids)
    result.update(annotations=args.annotations, predictions=args.predictions,
                  split=args.split, smoke=bool(args.smoke_limit))
    write_json(args.output, result)


if __name__ == '__main__':
    main()

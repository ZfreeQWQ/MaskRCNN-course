"""Mask2Former counterpart of tools/infer.py; test never loads any annotation file.

Mirrors infer.py's structure and output format exactly (same COCO-results JSON / mask-JSON CSV /
manifest schema, same `encode_instances` -- so validate_submission.py and merge_predictions.py
work unchanged on this tool's output). The one real difference: stock MaskFormer always returns
exactly TEST.DETECTIONS_PER_IMAGE instances per image regardless of score (there is no
Mask-R-CNN-style score-threshold config to pass at model-construction time), so this tool
applies --score-threshold as an explicit post-hoc filter on the decoded results list instead.
"""
import argparse
import time

from common import (encode_instances, image_record, load_images,
                    predictor_from_args_m2f, write_csv, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default=None)
    p.add_argument('--metadata', required=True)
    p.add_argument('--data-root', default='/private/rgzndzy/blender')
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--output', required=True, help='COCO results JSON')
    p.add_argument('--csv', help='Optional Kaggle mask-JSON CSV; empty predictions are []')
    p.add_argument('--score-threshold', type=float, default=0.5,
                   help='Post-hoc filter on the decoded results; MaskFormer has no model-level '
                        'equivalent to --threshold. Default 0.5 chosen from the D-experiment '
                        'val predictions: at 0.5 the average surviving detections/image (31.56) '
                        'matches this dataset\'s documented ~31 instances/image train density, '
                        'i.e. it is the natural real-object/background-query separation point '
                        'for this model, not an arbitrary round number. Not yet swept for '
                        'optimal val COCO AP/APs; treat as provisional for a first submission.')
    p.add_argument('--limit', type=int, default=0, help='Smoke subset; never final submission')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    if args.limit < 0 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid limit/shard settings')
    if not 0 <= args.score_threshold <= 1:
        p.error('--score-threshold must be in [0, 1]')
    all_images = load_images(args.metadata, args.split)
    images = all_images[args.shard_index::args.num_shards]
    if args.limit:
        images = images[:args.limit]
    predictor = predictor_from_args_m2f(args, images)
    results = []
    start = time.time()
    for index, info in enumerate(images):
        output = predictor(image_record(info, args.data_root))
        encoded = encode_instances(output['instances'], info['id'])
        results.extend(row for row in encoded if row['score'] >= args.score_threshold)
        if (index + 1) % 50 == 0:
            print(f'{index + 1}/{len(images)} images', flush=True)
    write_json(args.output, results)
    if args.csv:
        write_csv(args.csv, images, results)
    write_json(args.output + '.manifest.json', {
        'split': args.split, 'metadata': args.metadata, 'config': args.config,
        'weights': args.weights, 'score_threshold': args.score_threshold,
        'image_ids': [row['id'] for row in images], 'num_images': len(images),
        'num_instances': len(results), 'num_shards': args.num_shards,
        'shard_index': args.shard_index, 'smoke': bool(args.limit),
        'wall_seconds': time.time() - start, 'tta': False, 'tiling': False})
    print(f'Saved {len(results)} instances across {len(images)} images to {args.output}')


if __name__ == '__main__':
    main()

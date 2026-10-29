"""Predict RGB-D metadata; test never loads any annotation file."""
import argparse
import time
from common import (add_model_args, encode_instances, image_record, load_images,
                    predictor_from_args, write_csv, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument('--metadata', required=True)
    p.add_argument('--data-root', default='/private/rgzndzy/blender')
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--output', required=True, help='COCO results JSON')
    p.add_argument('--csv', help='Optional Kaggle mask-JSON CSV; empty predictions are []')
    p.add_argument('--limit', type=int, default=0, help='Smoke subset; never final submission')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    if args.limit < 0 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid limit/shard settings')
    all_images = load_images(args.metadata, args.split)
    images = all_images[args.shard_index::args.num_shards]
    if args.limit:
        images = images[:args.limit]
    predictor = predictor_from_args(args, images)
    results = []
    start = time.time()
    for index, info in enumerate(images):
        output = predictor(image_record(info, args.data_root))
        results.extend(encode_instances(output['instances'], info['id']))
        if (index + 1) % 50 == 0:
            print(f'{index + 1}/{len(images)} images', flush=True)
    write_json(args.output, results)
    if args.csv:
        write_csv(args.csv, images, results)
    write_json(args.output + '.manifest.json', {
        'split': args.split, 'metadata': args.metadata, 'config': args.config,
        'weights': args.weights, 'score_threshold': args.threshold,
        'image_ids': [row['id'] for row in images], 'num_images': len(images),
        'num_instances': len(results), 'num_shards': args.num_shards,
        'shard_index': args.shard_index, 'smoke': bool(args.limit),
        'wall_seconds': time.time() - start, 'tta': False, 'tiling': False})
    print(f'Saved {len(results)} instances across {len(images)} images to {args.output}')


if __name__ == '__main__':
    main()

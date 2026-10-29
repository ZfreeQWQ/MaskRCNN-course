"""Merge inference shards; manifests prove that zero-detection images were processed."""
import argparse
from common import load_images, read_json, write_csv, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--shards', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--csv')
    args = p.parse_args()
    images = load_images(args.metadata, args.split)
    expected = {row['id'] for row in images}
    covered, results, manifests = set(), [], []
    for path in args.shards:
        manifest = read_json(path + '.manifest.json')
        ids = set(manifest['image_ids'])
        if manifest['split'] != args.split or manifest['smoke'] or covered & ids or not ids <= expected:
            raise ValueError(f'Invalid, overlapping, smoke or wrong-split shard: {path}')
        rows = read_json(path)
        if any(row['image_id'] not in ids for row in rows):
            raise ValueError('Shard predictions disagree with manifest')
        if manifests:
            for key in ('weights', 'config', 'score_threshold', 'tta', 'tiling'):
                if manifest[key] != manifests[0][key]:
                    raise ValueError(f'Incompatible shards: {key}')
        covered |= ids
        results.extend(rows)
        manifests.append(manifest)
    if covered != expected:
        raise ValueError(f'Missing {len(expected-covered)} image ids')
    results.sort(key=lambda row: (row['image_id'], -row['score']))
    write_json(args.output, results)
    manifest = dict(manifests[0])
    manifest.update(image_ids=sorted(covered), num_images=len(covered), num_instances=len(results),
                    shards=args.shards, num_shards=1, shard_index=0,
                    wall_seconds=sum(row['wall_seconds'] for row in manifests))
    write_json(args.output + '.manifest.json', manifest)
    if args.csv:
        write_csv(args.csv, images, results)


if __name__ == '__main__':
    main()

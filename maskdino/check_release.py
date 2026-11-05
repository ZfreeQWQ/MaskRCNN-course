"""Replay cached, label-free 43D features against independently archived outputs."""
import argparse
import json
from pathlib import Path
import torch
from soft_feature_scoring import FrozenFeatureScorer


def main(a):
    torch.set_num_threads(4)
    features = Path(a.features)
    expected = Path(a.expected)
    paths = sorted(features.glob('*.json'))
    assert paths, 'No feature caches'
    if a.limit:
        paths = paths[:a.limit]
    scorer = FrozenFeatureScorer(a.weights, a.device)
    maximum = 0.
    count = 0
    for path in paths:
        payload = json.loads(path.read_text())
        actual = scorer.score(payload, {'features': scorer.features})
        reference = json.loads((expected/path.name).read_text())
        assert len(actual) == len(reference), path.name
        for got, want in zip(actual, reference):
            assert all(got[k] == want[k] for k in ('image_id', 'category_id', 'segmentation')), path.name
            error = abs(got['score']-want['score'])
            maximum = max(maximum, error)
            assert error <= 1e-5, (path.name, error)
        count += len(actual)
    result = dict(status='passed', images=len(paths), instances=count, mask_and_order_exact=True,
                  max_score_absolute_error=maximum, device=a.device,
                  scope='Cached inference replay, not a new complete model/AP evaluation')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--expected', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weights')
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=0)
    main(p.parse_args())

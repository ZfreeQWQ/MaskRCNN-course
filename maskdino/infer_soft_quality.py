"""RGB-D MaskDINO: RGB-D images -> frozen masks -> submission CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Small engineering check; not a full score')
    parser.add_argument('--template', help='Competition template; mandatory for test')
    parser.add_argument('--save-features', action='store_true', help='Save unscored 43-dimensional inference features')
    return parser.parse_args()


def main(a):
    import settings
    import refine_inference_20261103 as h
    from scale_data import GeometryData
    from soft_runtime import build, verify_weights
    from soft_quality import SoftCapture, FEATURES, current_features, clean
    from soft_feature_scoring import FrozenFeatureScorer
    from submission_io import export

    assert a.limit >= 0
    assert a.split != 'test' or a.template, 'Test export requires the competition template'
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = GeometryData(a.split, 'geometry', settings.GEOMETRY_STATS, side=1024, limit=a.limit)
    ids = [row['id'] for row in data.index.records]
    assert ids and all(row['width'] == row['height'] == 1024 for row in data.index.records)
    manifest = dict(release='soft_quality', split=a.split, ids=ids,
                    weights=verify_weights(), geometry_stats_sha256=h.d.sha(settings.GEOMETRY_STATS),
                    image_metadata_sha256=h.d.sha(settings.IMAGE_METADATA) if a.split == 'test' else None,
                    records_sha256=hashlib.sha256(json.dumps(data.index.records, sort_keys=True).encode()).hexdigest(),
                    source_hashes={p.name: h.d.sha(p) for p in Path(__file__).parent.glob('*.py')},
                    input_side=2048, native_views=[[0,0,1024],[0,0,640],[384,0,640],[0,384,640],[384,384,640]],
                    nms_iou=.5, max_detections=100, save_features=a.save_features)
    if (out/'manifest.json').exists():
        assert h.read(out/'manifest.json') == manifest, 'Use a new output directory for a changed run'
    else:
        h.write(out/'manifest.json', manifest)
    cache = out/'images'
    cache.mkdir(exist_ok=True)
    scorer = FrozenFeatureScorer(settings.POSTPROCESS_WEIGHTS/'soft_quality_weights.pth', device='cuda')
    model = capture = None
    predictions = []
    with h.torch.inference_mode():
        for i, image_id in enumerate(ids):
            path = cache/f'{i:05d}.json'
            if path.exists():
                pred = h.read(path)
            else:
                if model is None:
                    model = build(out/'model')
                    capture = SoftCapture(model)
                selected, x, positions, _, _ = current_features(model, capture, data, i)
                payload = dict(index=i, image_id=image_id, features=x.tolist(), positions=positions,
                               raw_predictions=[clean(p) for p in selected])
                pred = scorer.score(payload, {'features': FEATURES})
                h.write(path, pred)
                if a.save_features:
                    feature_dir = out/'features'
                    feature_dir.mkdir(exist_ok=True)
                    h.write(feature_dir/f'{i:05d}.json', payload)
            assert len(pred) <= 100 and all(p['image_id'] == image_id for p in pred)
            predictions.extend(pred)
            h.write(out/'progress.json', dict(done=i+1, total=len(ids)))
            print(f'{a.split}: {i+1}/{len(ids)}', flush=True)
    h.write(out/'predictions.json', predictions)
    if a.save_features:
        h.write(out/'provenance.json', dict(features=FEATURES, weights=manifest['weights'],
                                         scope='Inference-only raw features; no annotation fields'))
    metrics = h.score(data, list(range(len(ids))), predictions, out/'validation.json') if a.split == 'val' else None
    if a.template:
        with Path(a.template).open(newline='', encoding='utf-8-sig') as f:
            ids = [row['image_id'] for row in csv.DictReader(f)]
        assert set(ids) == set(map(str, manifest['ids'])) and len(ids) == len(manifest['ids'])
    exported = export(predictions, ids, out/'submission.csv')
    result = dict(status='completed', images=len(ids), csv_sha256=h.d.sha(out/'submission.csv'),
                  csv_validation=exported, validation_APs_0_to_1=metrics['segm']['APs']/100 if metrics else None,
                  competition_score=None, complete_dataset_evaluation=a.limit == 0)
    h.write(out/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main(arguments())

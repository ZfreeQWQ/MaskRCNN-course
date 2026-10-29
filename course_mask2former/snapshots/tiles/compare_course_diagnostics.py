"""Aggregate fixed-case diagnostics on the intersection of matched small GT IDs."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def compare(paths):
    results, sources = {}, {}
    scene_ids = annotation_hash = thresholds = None
    for path in paths:
        data = json.loads(path.read_text())
        assert data['status'] == 'complete'
        assert not data['test_inputs_used'] and not data['test_annotations_used']
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name, result in data['results'].items():
            assert name not in results, 'Duplicate model name'
            ids = [(c['image_id'], c['small_instances']) for c in result['cases']]
            if scene_ids is None:
                scene_ids, annotation_hash, thresholds = ids, result['validation_annotations_sha256'], data['diagnostic_thresholds']
            assert ids == scene_ids
            assert result['validation_annotations_sha256'] == annotation_hash
            assert data['diagnostic_thresholds'] == thresholds
            results[name] = result
    matches = {name: {m['annotation_id']: m['boundary_f1']
        for case in result['cases'] for m in case['matched_boundary_diagnostics'] if m['small']}
        for name, result in results.items()}
    common = sorted(set.intersection(*(set(m) for m in matches.values())))
    assert common, 'No common matched small instances'
    rows = []
    for name, result in results.items():
        assert len(matches[name]) == result['small_matched']
        rows.append(dict(name=name,small_instances=result['small_instances'],
            small_matched=result['small_matched'],small_missed=result['small_instances']-result['small_matched'],
            unmatched_predictions=sum(c['unmatched_predictions'] for c in result['cases']),
            boundary_f1_common_matched_small=mean(matches[name][i] for i in common),
            inference_mean_ms=result['timing']['mean_ms']))
    return dict(protocol='Fixed eight val images; score>=0.3, greedy IoU>=0.5; 2px Manhattan boundary tolerance. Not AP.',
        sources_sha256=sources,validation_annotations_sha256=annotation_hash,
        common_matched_small_instances=len(common),common_annotation_ids=common,results=rows,
        limitations='Conditional boundary mean excludes missed objects. Single-seed, eight-image diagnostics are not causal or full-val accuracy claims.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = compare(args.summaries)
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('common_annotation_ids','sources_sha256')},ensure_ascii=False,indent=2))

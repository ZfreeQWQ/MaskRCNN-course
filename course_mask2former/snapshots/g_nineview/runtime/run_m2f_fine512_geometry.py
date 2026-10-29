"""Bounded CPU cached refinement pilot, then at most one new full-val AP."""
import collections
import hashlib
import json
import os
import statistics
import time
import unittest
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
import run_m2f_tiny_positive as w
from m2f_fine512_geometry import refine

START = 1792570934  # 2026-10-21 16:22:14 Beijing; includes initial analysis.
PREP_END, PILOT_END, COMPUTE_END, DEADLINE = (START + s for s in (1200, 1800, 3000, 3600))
NAME = 'course_m2f_fine512_geometry_20261021_v2'
ROOT = Path('/workspace')
OUT = ROOT/'outputs'/NAME
ADAPT = ROOT/'outputs/course_m2f_fine512_adapt_20261021_v2'
FULL = ROOT/'outputs/course_m2f_fine512_fullval_20261021_v2'
RAM = ROOT/'outputs/course_m2f_dual512_ram_20261021_v1'
BEST = ROOT/'outputs/course_m2f_tiny_hflip_20261021_v1'
BEST_APS = .41399673960953054
INPUTS = {}


def cutoff(end=COMPUTE_END):
    assert time.time() < end, 'Predeclared absolute phase deadline'


def read_bound(path, expected=None):
    path = Path(path); raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    assert expected is None or digest == expected, str(path)
    key = str(path.relative_to(ROOT))
    assert key not in INPUTS or INPUTS[key] == digest
    INPUTS[key] = digest
    return json.loads(raw)


def save(name, value):
    w.save(OUT/name, value)


def mark(phase, **kw):
    cutoff(); value = dict(phase=phase, unix=time.time(), **kw)
    save('progress.json', value); print(json.dumps(value), flush=True)


def maximum_iou(rows, anns, coco):
    answer = np.zeros(len(anns))
    for j, ann in enumerate(anns):
        same = [r['segmentation'] for r in rows if r['category_id'] == ann['category_id']]
        if same:
            answer[j] = float(mu.iou(same, [coco.annToRLE(ann)], [0]).max())
    return answer


def space_and_cases(coco, ident, before, after):
    anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
            if not a.get('iscrowd', 0) and not a.get('ignore', 0) and 0 < a['area'] <= 1024]
    b, a = maximum_iou(before, anns, coco), maximum_iou(after, anns, coco)
    # Original or this fixed proposed geometry only; oracle is a space gate,
    # never used to select the actual output or a threshold.
    upper = np.maximum(b, a)
    out = []
    thresholds = np.arange(.5, 1., .05)
    for ann, x, y, z in zip(anns, b, a, upper):
        out.append(dict(image_id=ident, gt_id=ann['id'], area=ann['area'],
                        before_iou=float(x), after_iou=float(y),
                        oracle_extra_crossings=int(np.sum(z >= thresholds)-np.sum(x >= thresholds)),
                        actual_crossings=int(np.sum(y >= thresholds)-np.sum(x >= thresholds))))
    return out


def comparison(coco, ids, before, after, baseline_metrics, baseline_matching):
    current = w.metrics(coco, ids, after)
    matching = w.matched(coco, ids, after)
    b, a = baseline_matching, matching
    result = dict(baseline=baseline_metrics, current=current,
                  delta_points={k:100*(current[k]-baseline_metrics[k]) for k in current},
                  tiny_gained=sorted(set(a['tiny'])-set(b['tiny'])),
                  tiny_lost=sorted(set(b['tiny'])-set(a['tiny'])),
                  small_gained=sorted(set(a['small'])-set(b['small'])),
                  small_lost=sorted(set(b['small'])-set(a['small'])),
                  fp_before=len(b['false_positives']), fp_after=len(a['false_positives']), image_ids=ids)
    return result, matching


def terminal(outcome, fullval=0, compare=None):
    cutoff()
    save('binding.json', dict(input_sha256=INPUTS, inference_receives_gt=False,
                             cache_only=True, baseline_aps=BEST_APS))
    # Inputs were SHA checked when loaded; confirm final top-level bindings
    # and original read-only mount, not another full cache scan.
    plan = w.read(OUT/'plan.json')
    assert all(w.sha(OUT/'source'/n) == h for n,h in plan['source_sha256'].items())
    artifacts = {p.name:w.sha(p) for p in OUT.glob('*.json') if p.name not in
                 ('plan.json', 'status.json', 'progress.json', 'launch.json', 'report.json',
                  'terminal_acceptance.json', 'command.json', 'container.json')}
    report = dict(status='passed', outcome=outcome, start_unix=START, deadline_unix=DEADLINE,
                  finished_unix=time.time(), baseline_aps=BEST_APS, comparison=compare,
                  full_val_evaluations=fullval, model_loads=0, weight_updates=0, gpu_used=False,
                  new_model_predictions=0, cached_postprocessing_only=True, maskrcnn_paused=True,
                  artifacts_sha256=artifacts, development_checks_reused=True,
                  inference_cost='CPU refinement only measured; no new end-to-end or GPU memory measurement')
    save('report.json', report)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not Path('/dev/nvidiactl').exists()
    cutoff(PREP_END)
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_m2f_fine512_geometry'))
    assert tests.wasSuccessful() and not tests.skipped
    ar = read_bound(ADAPT/'report.json', 'e1d6fa8d6b30514771fa7efec17649b51d83f65be70b2b93fb0e73ab612e8a53')
    rr = read_bound(RAM/'report.json', '432333b3e2f5db30955c4c33df54cf8b0a545169a5ba1993ee75c88ad86a67e2')
    fr = read_bound(FULL/'report.json', '89b80f5704d4328b41d53804024ffab68d78309fdad2b2f31edebbfd4116f8fa')
    br = read_bound(BEST/'report.json', '8a73c0241f429c23daad53a2f9cf5674bada77b2dae114fb38e26f586aefb9e6')
    assert br['new_aps'] == BEST_APS
    pm = read_bound(ADAPT/'patch_manifest.json', ar['artifacts_sha256']['patch_manifest.json'])
    past = read_bound(RAM/'internal_gate.json', rr['artifacts_sha256']['internal_gate.json'])
    previous_matching = {g:read_bound(ADAPT/(g+'_matching.json'), ar['artifacts_sha256'][g+'_matching.json'])['best5']
                         for g in ('check', 'continuity')}
    coco = w.coco_load('train')
    records, cases, counts, spaces = [], [], collections.Counter(), collections.Counter()
    for n, f in enumerate(pm['files']):
        cutoff(PREP_END)
        d = read_bound(f['capture'], f['capture_sha256']); ident = f['image_id']
        assert d['image_id'] == d['input_hashes']['image_id'] == ident
        assert ident in past['groups'][f['group']]['ids']
        before = d['before']; counts[len(before)] += 1
        after, activity = refine(before, d['new_raw'])
        rows = space_and_cases(coco, ident, before, after)
        spaces['small_extra_crossings'] += sum(r['oracle_extra_crossings'] for r in rows)
        spaces['tiny_extra_crossings'] += sum(r['oracle_extra_crossings'] for r in rows if r['area'] <= 256)
        spaces['tiny_extra_gt'] += sum(r['oracle_extra_crossings'] > 0 for r in rows if r['area'] <= 256)
        cases += [dict(r, group=f['group']) for r in rows if r['before_iou'] != r['after_iou']]
        records.append(dict(group=f['group'], image_id=ident, before=before, after=after,
                            activity=activity, input_hashes=d['input_hashes']))
        if n%32 == 0:mark('space', images=n+1, expected=256)
    assert len(records) == len({r['image_id'] for r in records}) == 256
    assert counts == {100:256}, 'Initial no-vacancy observation changed; investigate instead of switching rule'
    # Carry out CPU postprocessor timing, not a new model or AP evaluation.
    sample = read_bound(pm['files'][0]['capture'], pm['files'][0]['capture_sha256'])
    timings = []
    for n in range(25):
        t = time.perf_counter(); rows, _ = refine(sample['before'], sample['new_raw'])
        json.dumps(rows, separators=(',', ':'), allow_nan=False)
        if n >= 5:timings.append((time.perf_counter()-t)*1000)
    save('timing.json', dict(warmup=5, repeats=20, milliseconds=timings, mean_ms=statistics.mean(timings),
                            scope='Loaded RLE reciprocal pairing, bbox update and JSON; excludes disk/model/old nineview cost',
                            GPU=False, new_gpu_memory_measurement=False))
    space_passed = spaces['tiny_extra_crossings'] > 0 and spaces['small_extra_crossings'] > 0
    save('space_gate.json', dict(passed=bool(space_passed), counts=dict(spaces), baseline_counts=dict(counts),
                                no_vacancy_append_rejected=True, oracle_is_not_AP=True))
    save('pilot_rows.json', records); save('pilot_geometry_cases.json', cases)
    save('cpu_acceptance.json', dict(status='passed', tests=tests.testsRun, zero_skip=True,
                                    inputs_bound=len(INPUTS), input_images=256, no_GPU=True))
    if not space_passed:
        terminal('stopped_no_train_geometry_space'); return
    mark('pilot')
    groups = {}; allids, allbefore, allafter = [], [], []
    for group in ('check', 'continuity'):
        cutoff(PILOT_END); chosen = [r for r in records if r['group'] == group]
        ids = [r['image_id'] for r in chosen]
        assert ids == past['groups'][group]['ids'] and len(ids) == 128
        before = [x for r in chosen for x in r['before']]; after = [x for r in chosen for x in r['after']]
        groups[group], matching = comparison(coco, ids, before, after, past['groups'][group]['baseline'], previous_matching[group])
        save(group+'_matching.json', matching); allids += ids; allbefore += before; allafter += after
    combined = w.metrics(coco, allids, allafter)
    combined_delta = 100*(combined['APs']-past['combined_before']['APs'])
    passed = all(g['delta_points']['APs'] >= 0 and len(g['tiny_gained']) >= len(g['tiny_lost']) for g in groups.values()) and combined_delta > 0
    save('internal_gate.json', dict(passed=bool(passed), groups=groups, combined=combined,
                                   combined_before=past['combined_before'], combined_delta_aps_points=combined_delta,
                                   full_val=False, no_baseline_AP_rerun=True))
    if not passed:
        terminal('stopped_failed_internal_gate'); return
    # Only this branch may read and transform the complete official val cache.
    mark('fullval_cached_transform')
    manifest = read_bound(FULL/'prepared_manifest.json', fr['artifacts_sha256']['prepared_manifest.json'])
    frozen = read_bound(FULL/'predictions_frozen.json', fr['artifacts_sha256']['predictions_frozen.json'])
    captures = {f['image_id']:f['capture_sha256'] for f in frozen['files']}
    ids = manifest['ids']; val = w.coco_load('val')
    assert len(ids) == len(set(ids)) == 2584 and ids == frozen['image_ids'] and set(ids) == set(val.imgs)
    baseline = read_bound(BEST/'predictions.segm.json', br['artifacts_sha256']['predictions.segm.json'])
    metrics_before = read_bound(BEST/'full_val_metrics.json', br['artifacts_sha256']['full_val_metrics.json'])
    hashes_before = read_bound(BEST/'val_input_hashes.json', br['artifacts_sha256']['val_input_hashes.json'])
    hashes = []; before_rows = []; after_rows = []; activities = []; changed = []
    for n, item in enumerate(manifest['files']):
        cutoff(); ident = item['image_id']
        b = read_bound(FULL/'base_cache'/f'{ident}.json', item['sha256'])
        d = read_bound(FULL/'captures'/f'{ident}.json', captures[ident])
        assert d['image_id'] == ident and d['input_hashes'] == b['input_hashes']
        after, activity = refine(b['before'], d['new_raw'])
        before_rows += b['before']; after_rows += after; hashes.append(b['input_hashes'])
        activities.append(dict(image_id=ident, **activity))
        if activity['changed']:
            changed += [x for x in space_and_cases(val, ident, b['before'], after) if x['before_iou'] != x['after_iou']]
        if n%64 == 0:mark('fullval_cached_transform', images=n+1, expected=2584)
    assert before_rows == baseline and hashes == hashes_before
    save('predictions.segm.json', after_rows); save('val_activity.json', activities)
    save('val_input_hashes.json', hashes); save('fullval_geometry_cases.json', changed)
    mark('one_fullval_AP')
    assert not (OUT/'full_val_metrics.json').exists()
    metrics = w.metrics(val, ids, after_rows)
    save('full_val_metrics.json', dict(metrics=metrics, images=2584, image_ids=ids, evaluations=1))
    # This is a separate IoU.5 error analysis, not another all-threshold AP.
    bmatch, amatch = w.matched(val, ids, baseline), w.matched(val, ids, after_rows)
    transfers = dict(tiny_gained=sorted(set(amatch['tiny'])-set(bmatch['tiny'])),
                     tiny_lost=sorted(set(bmatch['tiny'])-set(amatch['tiny'])),
                     small_gained=sorted(set(amatch['small'])-set(bmatch['small'])),
                     small_lost=sorted(set(bmatch['small'])-set(amatch['small'])),
                     fp_before=len(bmatch['false_positives']), fp_after=len(amatch['false_positives']),
                     original_instances_evicted=0)
    save('new_matching.json', amatch); save('full_val_error_analysis.json', transfers)
    comp = dict(metrics=metrics, baseline=metrics_before['metrics'],
                delta_points={k:100*(metrics[k]-metrics_before['metrics'][k]) for k in metrics},
                any_positive_retained=metrics['APs'] > BEST_APS, repeated_development_val=True)
    save('comparison.json', comp)
    terminal('evaluated', 1, comp)


if __name__ == '__main__':main()

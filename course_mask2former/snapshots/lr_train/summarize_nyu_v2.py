"""Strict partial summaries for the predeclared v2 queue; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path

from launch_nyu_sweep import utc_now
from nyu_v2_plan import FLAGS, VARIANTS, CONTRASTS, experiment_jobs
from summarize_nyu_sweep import (validate_report as base_validate, require, digest,
    stats, formatted, SHARED, atomic_write, number)


def validate_plan(plan, schema_version=2, jobs_builder=experiment_jobs):
    require(plan['schema_version'] == schema_version, 'Wrong schema')
    require(plan['jobs'] == jobs_builder(plan['seeds']), 'Jobs differ from fixed recipes')
    recipe = plan['recipe']
    for k in ('iters', 'val_interval', 'batch_size', 'image_size', 'foreground_quota'):
        require(type(recipe[k]) is int and recipe[k] > 0, 'Invalid ' + k)
    require(recipe['iters'] % recipe['val_interval'] == 0, 'Unequal final validation budget')
    require(recipe['image_size'] % 32 == 0, 'Invalid image size')
    require(type(recipe['workers']) is int and recipe['workers'] >= 0, 'Invalid workers')
    number(recipe['lr'], 'lr', positive=True)
    number(recipe['branch_lr_mult'], 'branch_lr_mult', positive=True)
    require(Path(plan['container_output_root']).is_absolute(), 'Output must be absolute')


def validate_report(report, job, plan):
    row = base_validate(report, job, plan)
    args = report['args']
    for flag in FLAGS:
        require(type(args[flag]) is bool and args[flag] is job['flags'][flag], 'Wrong switch: ' + flag)
    for key, value in plan['recipe'].items():
        require(args[key] == value, 'Recipe differs: ' + key)
    expected = dict(spatial_projection_changed=args['multiscale_depth'],
                    spatial_encoder_changed=args['multiscale_depth'],
                    spatial_gate_changed=args['reliability_gate'])
    for key, value in expected.items():
        require(report[key] is value, 'Branch update mismatch: ' + key)
    digest(report['initial_spatial_sha256'], 'initial_spatial_sha256')
    require(type(report['active_parameters']) is int and 0 < report['active_parameters'] <= report['parameters'],
            'Invalid active parameter count')
    require(type(report['spatial_parameters']) is int and report['spatial_parameters'] > 0, 'Missing branch size')
    curve = report['learning_curve']
    require([r['iteration'] for r in curve] == list(range(args['val_interval'], args['iters'] + 1, args['val_interval'])),
            'Incomplete learning curve')
    for point in curve:
        for metric in ('mAP', 'mAP_s'):
            value = number(point['metrics'][metric], metric)
            require(value <= 1, 'Curve AP outside range')
    for metric in ('mAP', 'mAP_s'):
        require(curve[-1]['metrics'][metric] == report['metrics']['nyuv2_void/segm_' + metric],
                'Final metric disagrees with curve')
    row.update(variant=job['variant'], learning_curve=curve, active_parameters=report['active_parameters'])
    return row


def summarize(root, plan_validator=validate_plan, report_validator=validate_report,
              flags=FLAGS, variants=VARIANTS, contrasts_spec=CONTRASTS, schema_version=2):
    root = Path(root).resolve()
    plan_bytes = (root / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    plan_validator(plan)
    jobs, candidates, shared = [], {}, None
    initial_by_seed, conflict = {}, None
    for job in plan['jobs']:
        row = dict(id=job['id'], variant=job['variant'], seed=job['seed'])
        path = (root / job['output_rel'] / 'report.json').resolve()
        require(path.is_relative_to(root), 'Report escapes root')
        if not path.exists():
            row['status'] = 'missing'
            jobs.append(row)
            continue
        try:
            raw = path.read_bytes()
            row['report_sha256'] = hashlib.sha256(raw).hexdigest()
            report = json.loads(raw)
            row['status'] = report['status']
            if report['status'] == 'failed':
                row['reason'] = report.get('error', 'Runner failed')
            elif report['status'] != 'running':
                candidates[job['id']] = report_validator(report, job, plan)
                conditions = {k: report[k] for k in SHARED}
                conditions['args'] = {k: v for k, v in report['args'].items() if k not in (*flags, 'seed', 'output')}
                if shared is None:
                    shared = conditions
                elif conditions != shared:
                    conflict = 'Shared training conditions disagree across reports'
                previous = initial_by_seed.setdefault(job['seed'], report['initial_spatial_sha256'])
                if previous != report['initial_spatial_sha256']:
                    conflict = 'New-branch initialization differs within a paired seed'
                row['status'] = 'passed'
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            candidates.pop(job['id'], None)
            row.update(status='invalid', reason=str(exc))
        jobs.append(row)
    if conflict:
        candidates.clear()
        for row in jobs:
            if row['status'] == 'passed':
                row.update(status='invalid', reason=conflict)
    runs = list(candidates.values())
    pairs = {(r['seed'], r['variant']): r for r in runs}
    def differences(on, off):
        return [dict(seed=s, **{m: round(pairs[s, on][m] - pairs[s, off][m], 6) for m in ('AP', 'APs')})
                for s in plan['seeds'] if (s, on) in pairs and (s, off) in pairs]
    combinations = []
    for variant, label, _ in variants:
        rows = [r for r in runs if r['variant'] == variant]
        delta = differences(variant, 'rgb')
        combinations.append(dict(variant=variant, label=label, n=len(rows),
            AP=stats([r['AP'] for r in rows]), APs=stats([r['APs'] for r in rows]),
            delta_AP=stats([r['AP'] for r in delta]), delta_APs=stats([r['APs'] for r in delta]),
            paired_deltas_vs_RGB=delta, active_parameters=sorted({r['active_parameters'] for r in rows}),
            curve=[dict(iteration=step, APs=stats([100 * r['learning_curve'][i]['metrics']['mAP_s'] for r in rows]))
                   for i, step in enumerate(range(plan['recipe']['val_interval'], plan['recipe']['iters'] + 1,
                                                  plan['recipe']['val_interval']))]))
    contrasts = []
    for on, off in contrasts_spec:
        delta = differences(on, off)
        contrasts.append(dict(on=on, off=off, paired_deltas=delta,
            delta_AP=stats([r['AP'] for r in delta]), delta_APs=stats([r['APs'] for r in delta])))
    counts = {s: sum(j['status'] == s for j in jobs) for s in ('passed', 'missing', 'running', 'failed', 'invalid')}
    return dict(schema_version=schema_version, status='complete' if counts['passed'] == len(jobs) else 'incomplete',
        updated_at_utc=utc_now(), planned_jobs=len(jobs), counts=counts, invalid_count=counts['invalid'],
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), recipe=plan['recipe'], seeds=plan['seeds'],
        jobs=jobs, validated_runs=runs, combinations=combinations, contrasts=contrasts,
        shared_training_conditions=None if conflict else shared,
        note='Final fixed-budget checkpoint AP, 0–100 scale; gains are percentage points. '
             'Mean ± sample SD, n=3 is exploratory, not significance. Indoor8 dev set, not course benchmark. '
             'No latency claim; this summary validates report evidence and does not rehash checkpoints.')


def markdown(s):
    lines = [f'# NYUv2 v{s["schema_version"]} 固定预算优化验证', '', f'状态：{s["status"]}；{s["counts"]}', '',
             f'每组 {s["recipe"]["iters"]} 步；种子 {s["seeds"]}；最终固定步数为主指标。', '',
             'AP/APs 为 0–100 标度，增益单位为百分点。均值 ± 样本标准差。', '',
             '| 组合 | n | AP | APs | 配对 ΔAP | 配对 ΔAPs |', '|---|---:|---:|---:|---:|---:|']
    for r in s['combinations']:
        lines.append(f'| {r["label"]} | {r["n"]} | {formatted(r["AP"])} | {formatted(r["APs"])} | '
                     f'{formatted(r["delta_AP"], True)} | {formatted(r["delta_APs"], True)} |')
    lines += ['', '## 预先指定的配对对照', '', '| on − off | n | ΔAP | ΔAPs |', '|---|---:|---:|---:|']
    for r in s['contrasts']:
        lines.append(f'| {r["on"]} − {r["off"]} | {r["delta_AP"]["n"]} | '
                     f'{formatted(r["delta_AP"], True)} | {formatted(r["delta_APs"], True)} |')
    steps = [r['iteration'] for r in s['combinations'][0]['curve']]
    lines += ['', '## 小目标 APs 学习曲线', '', '| 组合 | ' + ' | '.join(map(str, steps)) + ' |',
              '|---|' + '---:|' * len(steps)]
    for r in s['combinations']:
        lines.append('| ' + r['label'] + ' | ' + ' | '.join(formatted(p['APs']) for p in r['curve']) + ' |')
    lines += ['', '三种子仅作探索，不能保证显著提升；同一开发集上的选择存在偏差。'
              '本批不使用官方测试集调参，也不等同于课程成绩。耗时不等于推理延迟。', '']
    for j in s['jobs']:
        if j['status'] in ('failed', 'invalid'):
            lines.append(f'- {j["id"]}: {j["status"]} — {j.get("reason", "")}' )
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    result = summarize(args.root)
    root = Path(args.root)
    atomic_write(root / 'summary.json', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_write(root / 'summary.md', markdown(result))
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))


if __name__ == '__main__':
    main()

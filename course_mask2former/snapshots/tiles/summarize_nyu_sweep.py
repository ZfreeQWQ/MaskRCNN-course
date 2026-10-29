"""Refresh a strict, partial-progress summary of a planned NYUv2 factorial sweep.

Standard-library only. Run after every finished job, or manually while the
queue runs. Failed, missing, running and invalid reports never enter AP means.
This validates reported training evidence; it does not reread multi-GB weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import tempfile


FLAGS = ('depth_fusion', 'small_object_crop', 'small_object_loss', 'boundary_loss')
LABELS = ('深度融合', '小目标裁剪', '小目标损失加权', '边界损失')
PROTOCOL = 'nyuv2-official-train-scene-dev-indoor8-v1'
EVALUATION = 'mask prediction intersect label_valid; discard empty predictions; COCO segm'
NAMES = ['book', 'bottle', 'chair', 'bowl', 'cup', 'vase', 'sofa', 'bed']
COCO = [73, 39, 56, 45, 41, 75, 57, 59]
NYU = [1, 2, 5, 22, 35, 78, 83, 157]
RECIPE = ('iters', 'batch_size', 'image_size', 'workers', 'lr')
SHARED = ('protocol', 'dataset_manifest_sha256', 'source_checkpoint_sha256',
          'implementation_sha256', 'categories', 'parameters',
          'mmdetection_commit', 'evaluation_policy', 'training_images')
REQUIRED_IMPLEMENTATION = {
    'run_nyu_experiment.py', 'run_coco_experiment.py', 'export_nyu_development.py',
    'rgbd/data.py', 'rgbd/model.py', 'rgbd/void_head.py', 'rgbd/metric.py',
    'optimizations/weighted_boundary_head.py', 'optimizations/small_object_crop.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f'{label}: invalid integer')
    return value


def number(value, label, minimum=0, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > minimum if positive else value >= minimum),
            f'{label}: invalid finite number')
    return value


def digest(value, label):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
            f'{label}: invalid SHA256')


def flags_for(mask):
    return {flag: bool(mask & (1 << i)) for i, flag in enumerate(FLAGS)}


def label_for(mask):
    return ' + '.join(name for i, name in enumerate(LABELS) if mask & (1 << i)) or 'RGB 基线'


def validate_plan(plan):
    require(plan['schema_version'] == 1, 'Unsupported plan schema')
    require(isinstance(plan['created_at_utc'], str) and plan['created_at_utc'], 'Missing plan creation time')
    for key in ('workspace', 'container_output_root'):
        path = PurePosixPath(plan[key])
        require(path.is_absolute() and '..' not in path.parts, f'{key}: require absolute safe path')
    seeds = plan['seeds']
    require(isinstance(seeds, list) and seeds, 'Empty seeds')
    for seed in seeds:
        integer(seed, 'seed')
    require(len(seeds) == len(set(seeds)), 'Duplicate seeds')
    recipe = plan['recipe']
    for key in ('iters', 'batch_size', 'image_size'):
        integer(recipe[key], key, 1)
    integer(recipe['workers'], 'workers')
    number(recipe['lr'], 'lr', positive=True)
    require(recipe['image_size'] % 32 == 0, 'Image size must be divisible by 32')
    seen, ids, outputs = set(), set(), set()
    for job in plan['jobs']:
        require(isinstance(job['id'], str) and job['id'], 'Invalid job ID')
        integer(job['seed'], 'job seed')
        integer(job['mask'], 'mask')
        require(job['seed'] in seeds and job['mask'] < 16, 'Job outside seed/mask grid')
        require(set(job['flags']) == set(FLAGS)
                and all(type(v) is bool for v in job['flags'].values())
                and job['flags'] == flags_for(job['mask']), 'Job flags disagree with mask bits')
        rel = PurePosixPath(job['output_rel'])
        require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in ('', '.'),
                'output_rel must be a safe relative path')
        pair = job['seed'], job['mask']
        require(pair not in seen and job['id'] not in ids and str(rel) not in outputs,
                'Duplicate job seed/mask, ID or output directory')
        seen.add(pair)
        ids.add(job['id'])
        outputs.add(str(rel))
    require(seen == {(seed, mask) for seed in seeds for mask in range(16)},
            'Plan must contain exactly all 16 combinations for every seed')


def validate_report(report, job, plan):
    require(report['status'] == 'passed', 'Report status is not passed')
    args = report['args']
    require(args['mode'] == 'train' and args['trained_checkpoint'] is False,
            'Require independent fresh training, not evaluation/resume')
    integer(args['seed'], 'reported seed')
    require(args['seed'] == job['seed'], 'Seed differs from plan')
    require(all(type(args[key]) is bool and args[key] is job['flags'][key] for key in FLAGS),
            'Reported switches differ from plan')
    for key in RECIPE:
        if key != 'lr':
            integer(args[key], key)
        else:
            number(args[key], key, positive=True)
        require(args[key] == plan['recipe'][key], f'{key} differs from planned budget')
    output = PurePosixPath(plan['container_output_root']) / job['output_rel']
    require(PurePosixPath(args['output']) == output, 'Reported output path differs from job')
    require(PurePosixPath(report['checkpoint']) == output / f'iter_{args["iters"]}.pth',
            'Checkpoint is not the planned final checkpoint')
    require(report['protocol'] == PROTOCOL and report['evaluation_policy'] == EVALUATION,
            'Unexpected data/evaluation protocol')
    for key in ('dataset_manifest_sha256', 'source_checkpoint_sha256', 'checkpoint_sha256'):
        digest(report[key], key)
    implementation = report['implementation_sha256']
    require(isinstance(implementation, dict) and REQUIRED_IMPLEMENTATION <= set(implementation),
            'Incomplete implementation fingerprint')
    for key, value in implementation.items():
        digest(value, f'implementation {key}')
    require(isinstance(report['mmdetection_commit'], str)
            and re.fullmatch('[0-9a-f]{40}', report['mmdetection_commit']), 'Invalid MMDetection commit')
    categories = report['categories']
    require([c['id'] for c in categories] == list(range(1, 9))
            and [c['name'] for c in categories] == NAMES
            and [c['coco80_class_index'] for c in categories] == COCO
            and [c['nyuv2_source_id'] for c in categories] == NYU,
            'Categories differ from fixed indoor8 protocol')
    integer(report['parameters'], 'parameters', 1)
    require(type(report['training_images']) is int and report['training_images'] == 640,
            'Expected all 640 development training frames')
    integer(report['final_iteration'], 'final_iteration', 1)
    require(report['final_iteration'] == args['iters'], 'Final iteration differs from budget')
    require(isinstance(report['final_optimizer_steps'], list)
            and all(type(s) is int for s in report['final_optimizer_steps'])
            and report['final_optimizer_steps'] == [args['iters']],
            'Optimizer step counts must be exactly [iters]')
    require(report['classifier_changed'] is True, 'Classifier was not updated')
    require(report['depth_adapter_changed'] is args['depth_fusion'],
            'Depth adapter update disagrees with depth switch')
    norm = number(report['depth_adapter_l2'], 'depth_adapter_l2')
    require((norm > 0) == args['depth_fusion'], 'Depth adapter norm disagrees with depth switch')
    integer(report['loss_records'], 'loss_records', 1)
    require(report['loss_records'] == args['iters'], 'Missing training loss records')
    for key in ('first_loss', 'last_loss'):
        number(report[key], key)
    for key in ('seconds_including_evaluation', 'whole_run_peak_allocated_mib'):
        number(report[key], key, positive=True)
    for key in ('small_area_threshold', 'small_loss_max_weight'):
        number(args[key], key, positive=True)
    require(args['small_loss_max_weight'] >= 1, 'small_loss_max_weight must be >= 1')
    require(0 <= number(args['small_crop_prob'], 'small_crop_prob') <= 1,
            'Invalid small_crop_prob')
    number(args['boundary_loss_weight'], 'boundary_loss_weight')
    points = {}
    for suffix, name in (('mAP', 'AP'), ('mAP_s', 'APs')):
        value = number(report['metrics'][f'nyuv2_void/segm_{suffix}'], suffix)
        require(value <= 1, f'{suffix} is outside [0,1]')
        points[name] = round(value * 100, 6)
    for suffix in ('mAP_50', 'mAP_75', 'mAP_m', 'mAP_l'):
        value = report['metrics'][f'nyuv2_void/segm_{suffix}']
        require(type(value) in (int, float) and math.isfinite(value)
                and (value == -1 or 0 <= value <= 1), f'{suffix} is outside [0,1] or missing-GT -1')
    # Extra planned provenance fields are an optional pinned anchor; comparison
    # against other valid reports is still mandatory below.
    for key, expected in plan.get('expected_provenance', {}).items():
        require(key in SHARED and report[key] == expected, f'{key} differs from pinned plan provenance')
    return dict(id=job['id'], seed=job['seed'], mask=job['mask'], flags=job['flags'],
                output_rel=job['output_rel'], **points,
                checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256'],
                seconds_including_evaluation=report['seconds_including_evaluation'],
                whole_run_peak_allocated_mib=report['whole_run_peak_allocated_mib'])


def shared_conditions(report):
    result = {key: report[key] for key in SHARED}
    result['args_except_seed_switches_output'] = {
        key: value for key, value in report['args'].items()
        if key not in ('output', 'seed', *FLAGS)}
    return result


def stats(values):
    return dict(n=len(values), mean=round(statistics.mean(values), 6) if values else None,
                sample_std=round(statistics.stdev(values), 6) if len(values) > 1 else None)


def summarize(root):
    root = Path(root).resolve()
    plan_bytes = (root / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan)
    jobs, candidates, report_objects = [], {}, {}
    for job in plan['jobs']:
        record = dict(id=job['id'], seed=job['seed'], mask=job['mask'], output_rel=job['output_rel'])
        path = (root / job['output_rel'] / 'report.json').resolve()
        require(path.is_relative_to(root), 'Report path escapes root via symlink')
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            record['status'] = 'missing'
            jobs.append(record)
            continue
        record['report_sha256'] = hashlib.sha256(raw).hexdigest()
        try:
            report = json.loads(raw)
            if report['status'] in ('running', 'failed'):
                record['status'] = report['status']
                if report['status'] == 'failed':
                    record['reason'] = str(report.get('error', 'Runner failed without an error field'))
            else:
                candidates[job['id']] = validate_report(report, job, plan)
                report_objects[job['id']] = report
                record['status'] = 'passed'
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            record.update(status='invalid', reason=f'{type(exc).__name__}: {exc}')
        jobs.append(record)
    # An inconsistent set has no unambiguous trusted comparison anchor. Do not
    # cherry-pick the first fingerprint and quietly rank the matching subset.
    shared = None
    conflict = None
    for job_id, report in report_objects.items():
        conditions = shared_conditions(report)
        if shared is None:
            shared = conditions
        elif conditions != shared:
            keys = sorted(key for key in shared if shared[key] != conditions.get(key))
            conflict = f'Cross-report shared-condition conflict at {job_id}: {", ".join(keys)}'
            break
    if conflict:
        for record in jobs:
            if record['status'] == 'passed':
                record.update(status='invalid', reason=conflict)
        candidates.clear()
        shared = None
    rows = list(candidates.values())
    by_pair = {(row['seed'], row['mask']): row for row in rows}
    combinations = []
    for mask in range(16):
        current = [by_pair[(seed, mask)] for seed in plan['seeds'] if (seed, mask) in by_pair]
        paired = []
        for row in current:
            base = by_pair.get((row['seed'], 0))
            if base:
                paired.append(dict(seed=row['seed'], **{
                    metric: round(row[metric] - base[metric], 6) for metric in ('AP', 'APs')}))
        combinations.append(dict(mask=mask, label=label_for(mask), flags=flags_for(mask),
                                 completed_seeds=len(current), expected_seeds=len(plan['seeds']),
                                 AP=stats([r['AP'] for r in current]), APs=stats([r['APs'] for r in current]),
                                 paired_deltas_vs_RGB=paired,
                                 delta_AP=stats([p['AP'] for p in paired]),
                                 delta_APs=stats([p['APs'] for p in paired])))
    effects = []
    for bit, flag in enumerate(FLAGS):
        pairs = []
        for seed in plan['seeds']:
            for off_mask in range(16):
                if off_mask & (1 << bit):
                    continue
                on_mask = off_mask | (1 << bit)
                off, on = by_pair.get((seed, off_mask)), by_pair.get((seed, on_mask))
                if off is not None and on is not None:
                    pairs.append(dict(seed=seed, off_mask=off_mask, on_mask=on_mask, **{
                        metric: round(on[metric] - off[metric], 6) for metric in ('AP', 'APs')}))
        effects.append(dict(flag=flag, label=LABELS[bit], expected_pairs=len(plan['seeds']) * 8,
                            pairs=pairs, delta_AP=stats([p['AP'] for p in pairs]),
                            delta_APs=stats([p['APs'] for p in pairs])))
    counts = {status: sum(j['status'] == status for j in jobs)
              for status in ('passed', 'missing', 'running', 'failed', 'invalid')}
    return dict(schema_version=1, status='complete' if counts['passed'] == len(jobs) else 'incomplete',
                updated_at_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), planned_jobs=len(jobs), counts=counts,
                invalid_count=counts['invalid'],
                recipe=plan['recipe'], seeds=plan['seeds'], protocol=PROTOCOL,
                shared_training_conditions=shared, jobs=jobs, validated_runs=rows,
                combinations=combinations, single_switch_comparisons=[combinations[m] for m in (0, 1, 2, 4, 8)],
                switch_context_paired_effects=effects,
                metric_units='AP/APs on 0–100 scale; deltas are percentage points, not relative percentages.',
                uncertainty='sample_std uses n-1 and is unavailable with fewer than two observations. '
                            'Missing RGB baselines never become zero or unpaired gains. Context-pair '
                            'observations share seeds and are not independent replicates.',
                interpretation='Exploratory short-training, small-data, multi-comparison development sweep. '
                               'Three seeds do not establish stable or significant improvement. Interactions '
                               'can reverse individual-switch effects; averaged context differences are '
                               'descriptive, not causal proof. Not official NYUv2 or coursework benchmark.',
                verification_scope='Runner report provenance, budgets, metrics and optimizer/update evidence '
                                   'validated; report hashes recorded. Checkpoint bytes are not rehashed here; '
                                   'the runner generated checkpoint hashes after verifying training.')


def formatted(stat, signed=False):
    if stat['mean'] is None:
        return '—'
    mean = f'{stat["mean"]:+.3f}' if signed else f'{stat["mean"]:.3f}'
    std = '—' if stat['sample_std'] is None else f'{stat["sample_std"]:.3f}'
    return f'{mean} ± {std}'


def markdown(summary):
    c = summary['counts']
    lines = ['# NYUv2 indoor8 四开关批量消融', '',
             f'更新时间（UTC）：{summary["updated_at_utc"]}', '',
             f'状态：{summary["status"]}；有效完成 {c["passed"]}/{summary["planned_jobs"]}；'
             f'运行中 {c["running"]}；缺报告 {c["missing"]}；失败 {c["failed"]}；无效 {c["invalid"]}。', '',
             f'预算：每组 {summary["recipe"]["iters"]} 步，batch={summary["recipe"]["batch_size"]}，'
             f'尺寸={summary["recipe"]["image_size"]}；种子：{summary["seeds"]}。', '',
             'AP / APs 采用 0–100 标度；Δ 为相对同种子 RGB 的百分点变化，不是相对百分比。'
             '数值为均值 ± 样本标准差（n−1）；只有一个观测时标准差记为 —。', '',
             '## 单独开启每项优化', '',
             '| 组合 | 完成种子 | AP | APs | 配对 n | ΔAP | ΔAPs |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def row(item):
        return (f'| {item["label"]} | {item["completed_seeds"]}/{item["expected_seeds"]} | '
                f'{formatted(item["AP"])} | {formatted(item["APs"])} | {item["delta_AP"]["n"]} | '
                f'{formatted(item["delta_AP"], True)} | {formatted(item["delta_APs"], True)} |')
    lines.extend(row(item) for item in summary['single_switch_comparisons'])
    lines.extend(['', '## 全部 16 种组合', '',
                  '| 组合 | 完成种子 | AP | APs | 配对 n | ΔAP | ΔAPs |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    lines.extend(row(item) for item in summary['combinations'])
    lines.extend(['', '## 每项开关在相同其他开关配置下的配对变化', '',
                  '对每个种子、每种其余三个开关配置，比较此开关开 / 关。'
                  '这些配对共用种子，不是独立重复实验；均值受交互作用和未完成组合影响，不能作为因果或显著性证据。', '',
                  '| 开关 | 已有 / 计划配对 | ΔAP | ΔAPs |', '|---|---:|---:|---:|'])
    for item in summary['switch_context_paired_effects']:
        lines.append(f'| {item["label"]} | {item["delta_AP"]["n"]}/{item["expected_pairs"]} | '
                     f'{formatted(item["delta_AP"], True)} | {formatted(item["delta_APs"], True)} |')
    lines.extend(['', '## 解读限制', '',
                  '- 只有通过来源、预算、种子、开关、权重更新和指标检查的报告进入统计。RGB 缺失时不计算相对增益。',
                  '- 这是自定义 indoor8 开发集、短训练、多重比较结果；三个种子仍不足以证明稳定提升，挑选最好组合会产生选择偏差。',
                  '- 不是官方 NYUv2 分数，也不是课程统一数据集成绩。AP 改善不代表推理更快；本脚本不捏造速度收益。',
                  '- 所有 16 × 种子数 个任务有效完成才显示 complete。检查点更新依据训练器报告；汇总不重复读取大体积权重文件。'])
    bad = [j for j in summary['jobs'] if j['status'] in ('failed', 'invalid')]
    if bad:
        lines.extend(['', '## 失败 / 无效报告', ''])
        for item in bad:
            reason = item['reason'].replace('\n', ' ').replace('`', "'")
            lines.append(f'- `{item["id"]}`：{item["status"]} — {reason}')
    return '\n'.join(lines) + '\n'


def atomic_write(path, contents):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    root = Path(args.root).resolve()
    atomic_write(root / 'summary.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    atomic_write(root / 'summary.md', markdown(result))
    print(json.dumps({key: result[key] for key in ('status', 'planned_jobs', 'counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()

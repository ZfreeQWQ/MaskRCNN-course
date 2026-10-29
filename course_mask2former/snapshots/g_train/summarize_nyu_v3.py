"""v3 fixed-factorial evidence validation and paired descriptive effects."""
import argparse
import json
from pathlib import Path
import summarize_nyu_v2 as v2
from summarize_nyu_sweep import require, stats, formatted, atomic_write
from nyu_v3_plan import FLAGS, NEW_FLAGS, VARIANTS, CONTRASTS, experiment_jobs


def validate_plan(plan):
    v2.validate_plan(plan, schema_version=3, jobs_builder=experiment_jobs)
    require(0 < plan['recipe']['dropout_prob'] < 1, 'Invalid dropout probability')
    require(0 < plan['recipe']['matching_fraction'] <= .5, 'Invalid matching fraction')


def validate_report(report, job, plan):
    row = v2.validate_report(report, job, plan)
    args = report['args']
    for flag in NEW_FLAGS:
        require(type(args[flag]) is bool and args[flag] is job['flags'][flag], 'Wrong switch: ' + flag)
    depth, matching = report['advanced_depth_counters'], report['small_matching_counters']
    for value in (*depth.values(), *matching.values()):
        require(type(value) is int and value >= 0, 'Invalid execution counter')
    count = args['iters'] * args['batch_size']
    require(depth['relative_training_samples'] == (count if args['relative_depth'] else 0), 'Relative transform not exercised')
    require(depth['dropout_training_samples'] == (count if args['modality_dropout'] else 0), 'Dropout not exercised')
    require(0 <= depth['dropped_samples'] <= depth['dropout_training_samples'], 'Invalid dropped count')
    if args['modality_dropout'] and args['iters'] >= 1000:
        require(0 < depth['dropped_samples'] < count, 'No kept/dropped mixture')
    require(matching['augmented_calls'] <= matching['calls'], 'Invalid matching count')
    require(matching['foreground_points'] == matching['augmented_calls'] * int(12544 * args['matching_fraction']),
            'Matching point budget differs')
    if not args['small_matching']:
        require(not any(matching.values()), 'Disabled matching was exercised')
    elif args['iters'] >= 1000:
        require(matching['augmented_calls'] > 0, 'Small matching not exercised')
    row.update(advanced_depth_counters=depth, small_matching_counters=matching)
    return row


def summarize(root):
    result = v2.summarize(root, plan_validator=validate_plan, report_validator=validate_report,
        flags=FLAGS, variants=VARIANTS, contrasts_spec=CONTRASTS, schema_version=3)
    pairs = {(r['seed'], r['variant']): r for r in result['validated_runs']}
    effects = []
    for bit, flag in enumerate(NEW_FLAGS):
        observations = []
        for seed in result['seeds']:
            for off in range(8):
                if off & (1 << bit):
                    continue
                on = off | (1 << bit)
                a, b = pairs.get((seed, f'm{on}')), pairs.get((seed, f'm{off}'))
                if a and b:
                    observations.append(dict(seed=seed, on=f'm{on}', off=f'm{off}',
                        **{m: round(a[m] - b[m], 6) for m in ('AP', 'APs')}))
        effects.append(dict(flag=flag, pairs=observations, delta_AP=stats([r['AP'] for r in observations]),
                            delta_APs=stats([r['APs'] for r in observations])))
    result['switch_context_effects'] = effects
    return result


def markdown(result):
    text = v2.markdown(result)
    lines = ['', '## 三个新开关在其余开关相同条件下的配对变化', '',
             '各上下文共用种子，不是独立重复样本，不能把配对数当成独立 seed 数。', '',
             '| 开关 | 配对数 | ΔAP | ΔAPs |', '|---|---:|---:|---:|']
    for item in result['switch_context_effects']:
        lines.append(f'| {item["flag"]} | {len(item["pairs"])} | {formatted(item["delta_AP"], True)} | '
                     f'{formatted(item["delta_APs"], True)} |')
    return text + '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    result = summarize(root)
    atomic_write(root / 'summary.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    atomic_write(root / 'summary.md', markdown(result))
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))


if __name__ == '__main__':
    main()

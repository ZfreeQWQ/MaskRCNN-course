"""Strict six-cell summaries; reject undeclared training-policy differences."""
import argparse
import json
import math
from pathlib import Path

import summarize_nyu_v2 as v2
from summarize_nyu_sweep import require, atomic_write
from nyu_v4_plan import FLAGS, VARIANTS, CONTRASTS, experiment_jobs, scheduler_config


def validate_plan(plan):
    v2.validate_plan(plan, schema_version=4, jobs_builder=experiment_jobs)
    require(plan['recipe']['schedule'] == 'constant', 'Invalid base schedule')
    require(plan['recipe']['branch_lr_mult'] == 10., 'Invalid base branch LR')
    scheduler_config('decay', plan['recipe']['iters'])


def validate_report(report, job, plan):
    effective = dict(plan, recipe={**plan['recipe'], **job['recipe_overrides']})
    row = v2.validate_report(report, job, effective)
    args = report['args']
    require(report['declared_scheduler'] == scheduler_config(args['schedule'], args['iters']),
            'Wrong declared scheduler')
    factor = 1. if args['schedule'] == 'constant' else .01
    expected = sorted(set(args['lr'] * x * factor for x in (.1, 1., args['branch_lr_mult'])))
    actual = report['final_learning_rates']
    require(len(actual) == len(expected) and all(math.isclose(a, b, rel_tol=1e-7)
            for a, b in zip(actual, expected)), 'Wrong final LR')
    diagnostics = report['training_policy_diagnostics']
    require([r['iteration'] for r in diagnostics] == [r['iteration'] for r in report['learning_curve']],
            'Incomplete diagnostics')
    for r in diagnostics:
        ratios = r['first_val_image_residual_rgb_l2_ratios']
        require((ratios is not None) == args['multiscale_depth'], 'Wrong diagnostic switch')
        if ratios is not None:
            require(len(ratios) == 4 and all(math.isfinite(v) and v >= 0 for v in ratios), 'Invalid ratios')
    return row


def summarize(root):
    result = v2.summarize(root, plan_validator=validate_plan, report_validator=validate_report,
        flags=(*FLAGS, 'schedule', 'branch_lr_mult'), variants=VARIANTS,
        contrasts_spec=CONTRASTS, schema_version=4)
    result['note'] = ('Training-policy screening, final fixed-budget APs primary. n=1 is not confirmation. '
                      'Overall deltas use constant RGB; use predeclared matched-schedule contrasts '
                      'for algorithm benefit. No official test used.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    root = Path(p.parse_args().root)
    result = summarize(root)
    atomic_write(root / 'summary.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    text = v2.markdown(result).replace('三种子仅作探索', '本批仅作探索')
    text += '\n总体表的 Δ 参照固定学习率 RGB；算法收益必须看同学习率策略的配对对照。单种子不证明稳定提升。\n'
    atomic_write(root / 'summary.md', text)
    print(json.dumps(dict(status=result['status'], counts=result['counts'])))


if __name__ == '__main__':
    main()

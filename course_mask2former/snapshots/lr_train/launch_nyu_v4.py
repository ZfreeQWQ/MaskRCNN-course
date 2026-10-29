"""Launch only the six-cell screening stage; no automatic winner selection."""
import argparse
from pathlib import Path
import shutil

from launch_nyu_sweep import prepare
from nyu_v4_plan import experiment_jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start'])
    p.add_argument('--workspace', default='/mnt/nfs/fenghaibin/AI_Class')
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    p.add_argument('--seeds', nargs='+', type=int, default=[20261013])
    for name, default in [('iters', 5000), ('val_interval', 500), ('batch_size', 2),
                          ('image_size', 512), ('workers', 2), ('foreground_quota', 64)]:
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    p.add_argument('--lr', type=float, default=1e-5)
    args = p.parse_args()
    if args.iters < 10 or args.val_interval < 1 or args.iters % args.val_interval:
        p.error('Require >=10 steps and a budget divisible by validation interval')
    jobs = experiment_jobs(args.seeds)
    # Retain all validation checkpoints (including 1k), unlike v2/v3 last-two.
    reserve = 10 + .7 * len(jobs) * args.iters // args.val_interval
    if shutil.disk_usage(Path(args.workspace)).free < reserve * 2**30:
        raise RuntimeError(f'Need at least {reserve:.1f} GiB free for retained checkpoints')
    recipe = {key: getattr(args, key) for key in ('iters', 'val_interval', 'batch_size',
        'image_size', 'workers', 'foreground_quota', 'lr')}
    recipe.update(branch_lr_mult=10., schedule='constant')
    prepare(args, jobs=jobs,
        extra_source_files=('run_nyu_v2.py', 'nyu_v2_plan.py', 'summarize_nyu_v2.py',
            'run_nyu_v4.py', 'nyu_v4_plan.py', 'summarize_nyu_v4.py', 'launch_nyu_v4.py'),
        extra_train_files=('run_nyu_v2.py', 'launch_nyu_sweep.py', 'run_nyu_v4.py', 'nyu_v4_plan.py'),
        plan_extensions=dict(schema_version=4, runner_script='run_nyu_v4.py',
            summary_script='summarize_nyu_v4.py', recipe=recipe, max_job_seconds=10800,
            note='Six-cell training-policy screening. Final fixed-budget APs is primary; AP secondary. '
                 'One seed is exploratory; no automatic winner selection or confirmation. '
                 'Compare depth against RGB with matched schedule. Official test excluded.'))


if __name__ == '__main__':
    main()

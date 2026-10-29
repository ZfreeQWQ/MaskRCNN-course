"""Start the fixed 8-recipe x 3-seed, two-GPU v2 experiment queue."""
import argparse
from launch_nyu_sweep import prepare
from nyu_v2_plan import experiment_jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['start'])
    p.add_argument('--workspace', default='/mnt/nfs/fenghaibin/AI_Class')
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    p.add_argument('--seeds', nargs='+', type=int, default=[20260923, 20260924, 20260925])
    for name, default in [('iters', 5000), ('val_interval', 1000), ('batch_size', 2),
                          ('image_size', 512), ('workers', 2), ('foreground_quota', 64)]:
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--branch-lr-mult', type=float, default=10.)
    args = p.parse_args()
    if (args.val_interval < 1 or args.iters % args.val_interval or args.foreground_quota < 1
            or not 0 < args.branch_lr_mult < float('inf')):
        p.error('Invalid interval or branch/foreground setting')
    recipe = {k: getattr(args, k) for k in ('iters', 'batch_size', 'image_size', 'workers', 'lr',
              'val_interval', 'branch_lr_mult', 'foreground_quota')}
    prepare(args, jobs=experiment_jobs(args.seeds),
        extra_source_files=('run_nyu_v2.py', 'launch_nyu_v2.py', 'summarize_nyu_v2.py', 'nyu_v2_plan.py'),
        extra_train_files=('run_nyu_v2.py', 'launch_nyu_sweep.py'),
        plan_extensions=dict(schema_version=2, runner_script='run_nyu_v2.py', summary_script='summarize_nyu_v2.py',
            recipe=recipe, max_job_seconds=10800,
            note='Predeclared eight recipes, equal fixed budgets and COCO weights. All flags default off. '
                 'Final checkpoint is primary; no interim tuning/early stopping. Three paired seeds. '
                 'Official test excluded from tuning. No guarantee of improvement. No notification scheduled.'))


if __name__ == '__main__':
    main()

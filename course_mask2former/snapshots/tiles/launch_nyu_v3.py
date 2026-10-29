"""Host-side fixed v3 queue. `plan` is read-only; `start` requires idle GPUs."""
import argparse
import json
from launch_nyu_sweep import prepare
from nyu_v3_plan import experiment_jobs

EXTRA_SOURCES = ('run_nyu_v2.py', 'summarize_nyu_v2.py', 'nyu_v2_plan.py',
                 'run_nyu_v3.py', 'summarize_nyu_v3.py', 'nyu_v3_plan.py', 'launch_nyu_v3.py',
                 'run_nyu_v3_all.py')


def stage_plan(stage, seeds):
    if stage not in ('smoke', 'screen', 'confirm'):
        raise ValueError('Unknown stage')
    budget = {'smoke': (20, 20), 'screen': (1000, 1000), 'confirm': (5000, 1000)}[stage]
    if stage != 'confirm':
        seeds = seeds[:1]
    jobs = experiment_jobs(seeds)
    return dict(schema_version=3, stage=stage, seeds=seeds, jobs=jobs,
        recipe=dict(iters=budget[0], val_interval=budget[1], batch_size=2, image_size=512,
                    workers=2, lr=1e-5, branch_lr_mult=10., foreground_quota=64,
                    dropout_prob=.2, matching_fraction=.25),
        runner_script='run_nyu_v3.py', summary_script='summarize_nyu_v3.py', max_job_seconds=10800,
        note='Three-switch full factorial around a predeclared v2 anchor, plus RGB and capacity controls. '
             'No AP-based selection of cells or automatic parameter tuning. Final fixed step is primary. '
             'Official test excluded; no claim of significance or inference speed gain.')


def launch(workspace, output, gpus, stage, seeds):
    plan = stage_plan(stage, seeds)
    args = argparse.Namespace(workspace=workspace, output=output, gpus=gpus, seeds=plan['seeds'], **plan['recipe'])
    prepare(args, jobs=plan['jobs'], extra_source_files=EXTRA_SOURCES,
            extra_train_files=('run_nyu_v2.py', 'run_nyu_v3.py', 'launch_nyu_sweep.py'), plan_extensions=plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'start'])
    p.add_argument('--workspace', default='/mnt/nfs/fenghaibin/AI_Class')
    p.add_argument('--output')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--seeds', type=int, nargs='+', default=[20260923, 20260924, 20260925])
    p.add_argument('--stage', choices=['smoke', 'screen', 'confirm'], default='confirm')
    args = p.parse_args()
    if args.action == 'plan':
        print(json.dumps(stage_plan(args.stage, args.seeds), ensure_ascii=False, indent=2))
    elif not args.output:
        p.error('--output required to start')
    else:
        launch(args.workspace, args.output, args.gpus, args.stage, args.seeds)


if __name__ == '__main__':
    main()

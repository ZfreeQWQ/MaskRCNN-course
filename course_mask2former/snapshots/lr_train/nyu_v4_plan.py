"""Six predeclared training-policy cells, not a new algorithm search."""
from nyu_v2_plan import FLAGS

CELLS = [
    ('rgb', 'RGB / constant', False, 'constant', 10.),
    ('rgb_decay', 'RGB / decay', False, 'decay', 10.),
    ('depth_constant_10', 'Gated depth / constant / x10', True, 'constant', 10.),
    ('depth_decay_10', 'Gated depth / decay / x10', True, 'decay', 10.),
    ('depth_constant_1', 'Gated depth / constant / x1', True, 'constant', 1.),
    ('depth_decay_1', 'Gated depth / decay / x1', True, 'decay', 1.),
]
VARIANTS = [(name, label, ()) for name, label, *_ in CELLS]
CONTRASTS = [('rgb_decay', 'rgb'), ('depth_constant_10', 'rgb'),
             ('depth_constant_1', 'rgb'), ('depth_decay_10', 'rgb_decay'),
             ('depth_decay_1', 'rgb_decay'),
             ('depth_decay_10', 'depth_constant_10'),
             ('depth_decay_1', 'depth_constant_1'),
             ('depth_constant_1', 'depth_constant_10'),
             ('depth_decay_1', 'depth_decay_10')]


def experiment_jobs(seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    jobs = []
    for seed in seeds:
        for name, label, depth, schedule, multiplier in CELLS:
            identity = f's{seed}_{name}'
            jobs.append(dict(id=identity, seed=seed, variant=name, label=label, mask=0,
                flags={flag: depth and flag in ('multiscale_depth', 'reliability_gate') for flag in FLAGS},
                recipe_overrides=dict(schedule=schedule, branch_lr_mult=multiplier),
                output_rel='runs/' + identity))
    return jobs


def scheduler_config(schedule, iters):
    if schedule not in ('constant', 'decay') or type(iters) is not int or iters < 10:
        raise ValueError('Unknown schedule or fewer than 10 steps')
    # Predeclared before seeing v4 metrics: no warmup, all parameter groups
    # decay by 0.1 at 60% and 80% of the fixed training budget.
    return [] if schedule == 'constant' else [dict(type='MultiStepLR', by_epoch=False,
        begin=0, end=iters, milestones=[int(iters * .6), int(iters * .8)], gamma=.1)]

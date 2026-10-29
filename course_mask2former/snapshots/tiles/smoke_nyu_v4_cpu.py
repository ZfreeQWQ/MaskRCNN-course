"""Actual MMEngine config/scheduler and observer checks; NOT GPU training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn
from mmengine.registry import PARAM_SCHEDULERS
from mmdet.utils import register_all_modules

from run_nyu_experiment import categories_from_export
from run_nyu_v4 import parser, build_config, evidence, TrainingPolicyDiagnosticsHook
from nyu_v4_plan import experiment_jobs


def main():
    register_all_modules(init_default_scope=True)
    categories = categories_from_export('/workspace/data/nyuv2/development_indoor8_v1')
    for job in experiment_jobs([20261013]):
        with TemporaryDirectory() as folder:
            cli = ['train', '--output', folder, '--iters', '20', '--val-interval', '10']
            for k, v in job['recipe_overrides'].items():
                cli += ['--' + k.replace('_', '-'), str(v)]
            cli += ['--' + k.replace('_', '-') for k, v in job['flags'].items() if v]
            args = parser().parse_args(cli)
            cfg = build_config(args, categories)
            assert cfg.model.multiscale_depth.enabled == args.multiscale_depth
            assert cfg.default_hooks.checkpoint.max_keep_ckpts == 2
            assert cfg.optim_wrapper.paramwise_cfg.custom_keys.spatial_fusion.lr_mult == args.branch_lr_mult
            optimizer = torch.optim.AdamW([{'params': [nn.Parameter(torch.zeros(1))], 'lr': args.lr * scale}
                                          for scale in (.1, 1., args.branch_lr_mult)])
            schedulers = [PARAM_SCHEDULERS.build(c, default_args=dict(optimizer=optimizer))
                          for c in cfg.param_scheduler]
            class Fusion(nn.Module):
                def forward(self, levels):
                    return tuple(x + .1 for x in levels)
            runner = SimpleNamespace(iter=0, cfg=cfg, work_dir=folder,
                model=SimpleNamespace(spatial_fusion=Fusion()),
                optim_wrapper=SimpleNamespace(optimizer=optimizer))
            hook = TrainingPolicyDiagnosticsHook()
            for step in range(20):
                runner.iter = step
                hook.before_train_iter(runner, step)
                optimizer.step()
                for scheduler in schedulers:
                    scheduler.step()
                runner.iter = step + 1
                if runner.iter % 10 == 0:
                    hook.before_val_epoch(runner)
                    before = torch.get_rng_state().clone()
                    if args.multiscale_depth:
                        runner.model.spatial_fusion(tuple(torch.ones(1, 2, 4, 4) for _ in range(4)))
                    assert torch.equal(before, torch.get_rng_state())
                    hook.after_val_epoch(runner, {'nyuv2_void/book_precision': .1})
            result = evidence(runner, args)
            assert len(result['training_policy_diagnostics']) == 2
            print(job['variant'], 'CPU config/scheduler/diagnostics passed', flush=True)
    print('6/6 passed. Synthetic observer inputs; no AP result, no GPU smoke.', flush=True)


if __name__ == '__main__':
    main()

"""Training-policy diagnosis; no change to model, data, loss or evaluator."""
import json
import math
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

import run_nyu_v2 as v2
from launch_nyu_sweep import atomic_json
from nyu_v4_plan import scheduler_config


@HOOKS.register_module()
class TrainingPolicyDiagnosticsHook(Hook):
    """Record LR and the first deterministic validation image's fusion ratio.

    Observes only: never consumes RNG, writes predictions or changes features.
    Ratio is post-gate residual L2 / RGB feature L2, not a causal explanation.
    """
    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if runner.iter % runner.cfg.train_cfg.val_interval == 0:
            rates = sorted(set(float(g['lr']) for g in runner.optim_wrapper.optimizer.param_groups))
            self.rates = rates
            if not all(math.isfinite(x) and x > 0 for x in rates):
                raise RuntimeError('Invalid optimizer LR')

    def before_val_epoch(self, runner):
        self.ratios = None
        model = runner.model
        def observe(module, inputs, outputs):
            if self.ratios is not None:
                return
            with torch.no_grad():
                self.ratios = [float((out.detach() - rgb.detach()).float().norm()
                                    / rgb.detach().float().norm().clamp_min(1e-12))
                               for rgb, out in zip(inputs[0], outputs)]
        self.handle = model.spatial_fusion.register_forward_hook(observe)

    def after_val_epoch(self, runner, metrics=None):
        self.handle.remove()
        row = dict(iteration=runner.iter, lr_at_interval_start=self.rates,
                   first_val_image_residual_rgb_l2_ratios=self.ratios,
                   class_AP={key: (float(value) if math.isfinite(float(value)) else None)
                             for key, value in (metrics or {}).items() if key.endswith('_precision')})
        if self.ratios is not None and not all(math.isfinite(x) for x in self.ratios):
            raise RuntimeError('Nonfinite fusion ratios')
        path = Path(runner.work_dir) / 'diagnostics.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append(row)
        atomic_json(path, rows)


def parser():
    p = v2.parser()
    p.add_argument('--schedule', choices=['constant', 'decay'], default='constant')
    return p


def build_config(args, categories):
    cfg = v2.build_config(args, categories)
    cfg.param_scheduler = scheduler_config(args.schedule, args.iters)
    cfg.default_hooks.checkpoint.max_keep_ckpts = args.iters // args.val_interval
    cfg.custom_hooks.append(dict(type='TrainingPolicyDiagnosticsHook'))
    return cfg


def evidence(runner, args):
    rows = json.loads((Path(runner.work_dir) / 'diagnostics.json').read_text())
    if [r['iteration'] for r in rows] != list(range(args.val_interval, args.iters + 1, args.val_interval)):
        raise RuntimeError('Incomplete policy diagnostics')
    if any((row['first_val_image_residual_rgb_l2_ratios'] is not None) != args.multiscale_depth for row in rows):
        raise RuntimeError('Fusion diagnostics disagree with switch')
    actual = sorted(set(float(g['lr']) for g in runner.optim_wrapper.optimizer.param_groups))
    factor = 1. if args.schedule == 'constant' else .01
    expected = sorted(set(args.lr * x * factor for x in (.1, 1., args.branch_lr_mult)))
    if len(actual) != len(expected) or any(not math.isclose(a, b, rel_tol=1e-7) for a, b in zip(actual, expected)):
        raise RuntimeError(f'Final optimizer LR mismatch: {actual} != {expected}')
    return dict(training_policy_diagnostics=rows, final_learning_rates=actual,
                declared_scheduler=scheduler_config(args.schedule, args.iters))


if __name__ == '__main__':
    args = parser().parse_args()
    v2.main(args=args, config_builder=build_config,
            extra_sources=('run_nyu_v4.py', 'nyu_v4_plan.py'), report_callback=evidence)

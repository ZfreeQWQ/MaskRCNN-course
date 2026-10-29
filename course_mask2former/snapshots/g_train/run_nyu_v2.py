"""Fresh, fixed-budget NYUv2 v2 runs; every algorithm switch defaults off."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules

from run_nyu_experiment import (build_config as base_config, categories_from_export,
                                map_coco_state, WholeRunPeakHook)
from run_coco_experiment import sha256, repository_commit, optimizer_steps
from export_nyu_development import verify_export
from launch_nyu_sweep import atomic_json

FLAGS = ('depth_fusion', 'small_object_crop', 'small_object_loss', 'boundary_loss',
         'multiscale_depth', 'reliability_gate', 'rgb_control', 'foreground_points')


@HOOKS.register_module()
class ValidationCurveHook(Hook):
    def after_val_epoch(self, runner, metrics=None):
        path = Path(runner.work_dir) / 'learning_curve.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        mandatory = ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l')
        cleaned = {key: float((metrics or {})['nyuv2_void/segm_' + key]) for key in mandatory}
        if not all(math.isfinite(v) for v in cleaned.values()):
            raise RuntimeError('Missing/nonfinite validation metric')
        rows.append(dict(iteration=runner.iter, metrics=cleaned))
        atomic_json(path, rows)


def build_config(args, categories):
    cfg = base_config(args, categories)
    cfg.model.type = 'MultiScaleRGBDMask2Former'
    cfg.model.multiscale_depth = dict(enabled=args.multiscale_depth,
        reliability_gate=args.reliability_gate, source='rgb' if args.rgb_control else 'depth')
    cfg.model.panoptic_head.type = 'ForegroundQuotaMask2FormerHead'
    cfg.model.panoptic_head.foreground_sampling = dict(enabled=args.foreground_points,
        quota=args.foreground_quota, area_threshold=args.small_area_threshold)
    cfg.custom_imports.imports += ['rgbd.multiscale', 'rgbd.foreground_head']
    cfg.custom_hooks.append(dict(type='ValidationCurveHook'))
    cfg.train_cfg.val_interval = args.val_interval
    cfg.default_hooks.checkpoint.interval = args.val_interval
    cfg.default_hooks.checkpoint.max_keep_ckpts = 2
    cfg.optim_wrapper.paramwise_cfg.custom_keys['spatial_fusion'] = dict(
        lr_mult=args.branch_lr_mult, decay_mult=1.0)
    # Preserve the original norm no-weight-decay policy even inside custom keys.
    for name in ('stem.0.1', 'stem.1.1', 'stages.0.1', 'stages.1.1', 'stages.2.1'):
        cfg.optim_wrapper.paramwise_cfg.custom_keys['spatial_fusion.' + name] = dict(
            lr_mult=args.branch_lr_mult, decay_mult=0.0)
    return cfg


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train'])
    p.add_argument('--workspace', default='/workspace')
    p.add_argument('--data-root', default='/workspace/data/nyuv2/development_indoor8_v1')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', default='/workspace/checkpoints/'
                   'mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth')
    p.set_defaults(trained_checkpoint=False)
    for name, default in [('iters', 5000), ('batch_size', 2), ('image_size', 512), ('workers', 2),
                          ('seed', 20260923), ('val_interval', 1000), ('foreground_quota', 64)]:
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    for name, default in [('lr', 1e-5), ('small_area_threshold', 1024.), ('small_crop_prob', .5),
                          ('small_loss_max_weight', 2.), ('boundary_loss_weight', .1), ('branch_lr_mult', 10.)]:
        p.add_argument('--' + name.replace('_', '-'), type=float, default=default)
    for flag in FLAGS:
        p.add_argument('--' + flag.replace('_', '-'), action='store_true')
    return p


def validate_args(args):
    if (min(args.iters, args.batch_size, args.val_interval, args.foreground_quota) < 1
            or args.image_size < 32 or args.image_size % 32 or args.workers < 0 or args.seed < 0
            or not 0 < args.lr < math.inf or not 0 < args.branch_lr_mult < math.inf
            or args.iters % args.val_interval):
        raise ValueError('Invalid budget; iters must be a multiple of val_interval')
    if (not 0 <= args.small_crop_prob <= 1 or not 0 < args.small_area_threshold < math.inf
            or not 1 <= args.small_loss_max_weight < math.inf
            or not 0 <= args.boundary_loss_weight < math.inf):
        raise ValueError('Invalid optimization hyperparameters')
    if args.multiscale_depth and args.depth_fusion:
        raise ValueError('Early and multiscale fusion are alternative switches')
    if (args.reliability_gate or args.rgb_control) and not args.multiscale_depth:
        raise ValueError('Gate/RGB capacity control requires multiscale fusion')


def load_initial(model, coco_state):
    current = model.state_dict()
    missing = set(current) - set(coco_state)
    expected = {k for k in current if k.startswith('spatial_fusion.')}
    if missing != expected or set(coco_state) - set(current):
        raise ValueError('Only newly added spatial branch keys may be missing')
    current.update(coco_state)
    model.load_state_dict(current, strict=True)
    initial = {k: v.detach().cpu().clone() for k, v in current.items()
               if k.startswith(('spatial_fusion.', 'depth_adapter.', 'panoptic_head.cls_embed.'))}
    if any(v.count_nonzero() for k, v in initial.items()
           if k.startswith(('spatial_fusion.projections.', 'spatial_fusion.gates.'))):
        raise RuntimeError('Initial residual projection/gate must be zero')
    return initial


def main(args=None, config_builder=None, extra_sources=(), report_callback=None):
    args = parser().parse_args() if args is None else args
    validate_args(args)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing nonempty experiment output')
    verify_export(Path(args.data_root))
    categories = categories_from_export(args.data_root)
    source = Path(args.checkpoint).resolve()
    state = map_coco_state(torch.load(source, map_location='cpu')['state_dict'], categories)
    register_all_modules(init_default_scope=True)
    cfg = (config_builder or build_config)(args, categories)
    import_modules_from_strings(**cfg.custom_imports)
    origin = Path(__file__).parent
    files = [origin / name for name in ('run_nyu_v2.py', 'run_nyu_experiment.py',
             'run_coco_experiment.py', 'export_nyu_development.py', 'launch_nyu_sweep.py')]
    files += sorted((origin / 'rgbd').glob('*.py')) + sorted((origin / 'optimizations').glob('*.py'))
    files += [origin / name for name in extra_sources]
    provenance = dict(dataset_manifest_sha256=sha256(Path(args.data_root) / 'manifest.json'),
        source_checkpoint_sha256=sha256(source),
        implementation_sha256={str(f.relative_to(origin)): sha256(f) for f in files},
        categories=categories, protocol='nyuv2-official-train-scene-dev-indoor8-v1',
        evaluation_policy='mask prediction intersect label_valid; discard empty predictions; COCO segm')
    cfg.experiment_provenance = provenance
    output.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output / 'config.py'))
    report = dict(status='running', args=vars(args), **provenance,
        mmdetection_commit=repository_commit(Path(args.workspace) / 'mmdetection'),
        gpu=torch.cuda.get_device_name(0), note='Custom indoor8 development AP, not course or official NYUv2 AP. '
        'Final fixed-budget checkpoint is primary; intermediate validation is descriptive, not early stopping.')
    path = output / 'report.json'
    atomic_json(path, report)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        runner = Runner.from_cfg(cfg)
        runner.model.init_weights()
        initial = load_initial(runner.model, state)
        del state
        report.update(parameters=sum(p.numel() for p in runner.model.parameters()),
                      active_parameters=runner.model.active_parameter_count(),
                      spatial_parameters=sum(p.numel() for p in runner.model.spatial_fusion.parameters()),
                      initial_spatial_sha256=hashlib.sha256(b''.join(v.numpy().tobytes()
                          for k, v in sorted(initial.items()) if k.startswith('spatial_fusion.'))).hexdigest())
        runner.train()
        latest = Path((output / 'last_checkpoint').read_text().strip())
        checkpoint = torch.load(latest, map_location='cpu')
        weights = checkpoint['state_dict']
        if not all(torch.isfinite(v).all() for v in weights.values()
                   if torch.is_tensor(v) and v.is_floating_point()):
            raise RuntimeError('Nonfinite trained weights')
        def changed(prefix):
            return any(not torch.equal(v, weights[k]) for k, v in initial.items() if k.startswith(prefix))
        updates = dict(classifier_changed=changed('panoptic_head.cls_embed.'),
                       depth_adapter_changed=changed('depth_adapter.'),
                       spatial_projection_changed=changed('spatial_fusion.projections.'),
                       spatial_encoder_changed=changed(('spatial_fusion.stem.', 'spatial_fusion.stages.')),
                       spatial_gate_changed=changed('spatial_fusion.gates.'))
        expected = dict(classifier_changed=True, depth_adapter_changed=args.depth_fusion,
                        spatial_projection_changed=args.multiscale_depth)
        if args.iters >= 3:
            expected.update(spatial_encoder_changed=args.multiscale_depth,
                            spatial_gate_changed=args.reliability_gate)
        if any(updates[k] != v for k, v in expected.items()):
            raise RuntimeError(f'Switch/weight-update mismatch: {updates} expected {expected}')
        steps = optimizer_steps(checkpoint)
        if checkpoint['meta']['iter'] != args.iters or steps != [args.iters]:
            raise RuntimeError('Checkpoint/optimizer iteration mismatch')
        scalars = runner.message_hub.log_scalars
        for key, history in scalars.items():
            if ('loss' in key or 'grad_norm' in key) and not all(math.isfinite(float(v)) for v in history.data[0]):
                raise RuntimeError('Nonfinite scalar: ' + key)
        metrics = {k.removeprefix('val/'): float(v.current()) for k, v in scalars.items()
                   if k.startswith('val/nyuv2_void/')}
        for suffix in ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'):
            if not math.isfinite(metrics['nyuv2_void/segm_' + suffix]):
                raise RuntimeError('Nonfinite aggregate metric')
        val = json.loads((Path(args.data_root) / 'annotations/instances_val.json').read_text())
        present = {a['category_id'] for a in val['annotations']}
        absent = {f'nyuv2_void/{c["name"]}_precision' for c in categories if c['id'] not in present}
        unavailable = []
        for key, value in metrics.items():
            if not math.isfinite(value):
                if key not in absent:
                    raise RuntimeError('Unexpected nonfinite metric: ' + key)
                metrics[key] = None
                unavailable.append(key)
        curve = json.loads((output / 'learning_curve.json').read_text())
        if [row['iteration'] for row in curve] != list(range(args.val_interval, args.iters + 1, args.val_interval)):
            raise RuntimeError('Incomplete validation curve')
        torch.cuda.synchronize()
        report.update(status='passed', metrics=metrics, metrics_without_ground_truth=unavailable,
            checkpoint=str(latest), checkpoint_sha256=sha256(latest), final_iteration=checkpoint['meta']['iter'],
            final_optimizer_steps=steps, **updates, depth_adapter_l2=float(weights['depth_adapter.weight'].norm()),
            training_images=len(runner.train_dataloader.dataset), loss_records=len(scalars['train/loss'].data[0]),
            first_loss=float(scalars['train/loss'].data[0][0]), last_loss=float(scalars['train/loss'].current()),
            learning_curve=curve, seconds_including_evaluation=time.perf_counter() - started,
            whole_run_peak_allocated_mib=max(torch.cuda.max_memory_allocated() / 2**20,
                runner.message_hub.get_info('whole_run_peak_allocated_mib') or 0))
        if report_callback is not None:
            report.update(report_callback(runner, args))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        atomic_json(path, report)
        raise
    atomic_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

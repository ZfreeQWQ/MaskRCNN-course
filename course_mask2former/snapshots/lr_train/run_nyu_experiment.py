"""Fixed NYUv2 indoor8 development experiment; RGB versus optional depth fusion.

Not official NYUv2 AP or the coursework benchmark. Official test frames are
excluded. Every proposed optimization defaults off; void handling is a common
dataset supervision/evaluation policy, not an optimization ablation.
"""
import argparse
import copy
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

from run_coco_experiment import (build_config as coco_config, optimizer_steps,
                                 repository_commit, sha256)
from export_nyu_development import verify_export


@HOOKS.register_module()
class WholeRunPeakHook(Hook):
    """Capture peaks before LoggerHook resets CUDA peak counters."""
    priority = 'ABOVE_NORMAL'

    def _record(self, runner):
        peak = torch.cuda.max_memory_allocated() / 2**20
        runner.message_hub.update_info('whole_run_peak_allocated_mib', max(
            peak, runner.message_hub.get_info('whole_run_peak_allocated_mib') or 0))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._record(runner)

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._record(runner)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._record(runner)


def categories_from_export(data):
    categories = json.loads((Path(data) / 'annotations/instances_train.json').read_text())['categories']
    other = json.loads((Path(data) / 'annotations/instances_val.json').read_text())['categories']
    if categories != other or [x['id'] for x in categories] != list(range(1, len(categories) + 1)):
        raise ValueError('Train/validation categories must agree and have contiguous IDs')
    return categories


def map_coco_state(state, categories):
    """Map semantic classifier rows, preserve no-object, add zero depth adapter."""
    result = dict(state)
    indices = [c['coco80_class_index'] for c in categories] + [80]
    if len(indices) != len(set(indices)) or any(not 0 <= i <= 80 for i in indices):
        raise ValueError('Invalid or duplicate COCO class mapping')
    for suffix in ('weight', 'bias'):
        key = f'panoptic_head.cls_embed.{suffix}'
        if state[key].shape[0] != 81:
            raise ValueError('Fresh runs require the original 80-class COCO checkpoint')
        result[key] = state[key][indices].clone()
    if 'depth_adapter.weight' in result:
        raise ValueError('Expected original RGB COCO weights, not an already adapted checkpoint')
    result['depth_adapter.weight'] = torch.zeros(3, 2, 1, 1, dtype=result['panoptic_head.cls_embed.weight'].dtype)
    return result


def build_config(args, categories):
    base_args = copy.copy(args)
    base_args.resume = None
    base_args.small_object_crop = base_args.small_object_loss = base_args.boundary_loss = False
    cfg = coco_config(base_args)
    data = Path(args.data_root).resolve()
    classes = tuple(c['name'] for c in categories)
    cfg.num_things_classes = cfg.num_classes = len(classes)
    cfg.num_stuff_classes = 0
    cfg.model.type = 'RGBDMask2Former'
    cfg.model.depth_fusion = dict(enabled=args.depth_fusion)
    cfg.model.data_preprocessor.type = 'RGBDDataPreprocessor'
    cfg.model.data_preprocessor.depth_max = 10.0
    cfg.model.panoptic_head.update(
        type='VoidAwareMask2FormerHead', ignore_void=True,
        num_things_classes=len(classes), num_stuff_classes=0,
        small_object_loss=dict(enabled=args.small_object_loss,
                               area_threshold=args.small_area_threshold,
                               max_weight=args.small_loss_max_weight),
        boundary_loss=dict(enabled=args.boundary_loss, loss_weight=args.boundary_loss_weight,
                           radius=1, min_target_peak=0.5))
    cfg.model.panoptic_head.loss_cls.class_weight = [1.0] * len(classes) + [0.1]
    cfg.model.panoptic_fusion_head.update(num_things_classes=len(classes), num_stuff_classes=0)
    size = args.image_size
    crop = dict(type='RGBDRandomCrop', crop_size=(size, size), crop_type='absolute',
                recompute_bbox=True, allow_negative_crop=True)
    if args.small_object_crop:
        crop.update(type='RGBDSmallObjectAwareCrop', prob=args.small_crop_prob,
                    small_area_thr=args.small_area_threshold, area_mode='original')
    train_pipeline = [
        dict(type='LoadRGBDImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='RGBDRandomFlip', prob=0.5),
        dict(type='RandomResize', scale=(size, size), ratio_range=(0.8, 1.25),
             resize_type='RGBDResize', keep_ratio=True), crop,
        # Despite its name, upstream keep_empty=True *rejects* a newly empty
        # sample. False preserves negative training frames after augmentation.
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5),
             by_mask=True, keep_empty=False),
        dict(type='PackRGBDDetInputs'),
    ]
    test_pipeline = [dict(type='LoadRGBDImageFromFile'),
                     dict(type='RGBDResize', scale=(round(size * 1333 / 800), size), keep_ratio=True),
                     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                     dict(type='PackRGBDDetInputs', meta_keys=(
                         'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]
    def dataset(split, pipeline):
        return dict(type='RGBDCocoDataset', data_root=str(data), metainfo=dict(classes=classes),
                    ann_file=f'annotations/instances_{split}.json', data_prefix=dict(img=''),
                    filter_cfg=None, pipeline=pipeline, test_mode=split != 'train')
    val_loader = dict(batch_size=1, num_workers=args.workers, persistent_workers=args.workers > 0,
                      drop_last=False, sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dataset('val', test_pipeline))
    evaluator = dict(type='VoidAwareCocoMetric', data_root=str(data),
                     ann_file=str(data / 'annotations/instances_val.json'), metric='segm',
                     classwise=True, prefix='nyuv2_void', format_only=False)
    cfg.test_dataloader = copy.deepcopy(val_loader)
    cfg.test_evaluator = copy.deepcopy(evaluator)
    if args.mode == 'train':
        cfg.train_dataloader.dataset = dataset('train', train_pipeline)
        cfg.val_dataloader = val_loader
        cfg.val_evaluator = evaluator
    cfg.custom_imports = dict(imports=['rgbd.data', 'rgbd.model', 'rgbd.void_head', 'rgbd.metric'],
                              allow_failed_imports=False)
    cfg.custom_hooks = ([dict(type='RejectNonfiniteLossHook')] if args.mode == 'train' else [])
    cfg.custom_hooks.append(dict(type='WholeRunPeakHook'))
    cfg.load_from = None  # Explicit strict state loading below: never silently skip classifier rows.
    cfg.resume = False
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'eval'])
    p.add_argument('--workspace', default='/workspace')
    p.add_argument('--data-root', default='/workspace/data/nyuv2/development_indoor8_v1')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', default='/workspace/checkpoints/'
                   'mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth')
    p.add_argument('--trained-checkpoint', action='store_true',
                   help='Evaluate an adapted checkpoint, without remapping its already adapted classifier')
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--depth-fusion', action='store_true')
    p.add_argument('--small-object-crop', action='store_true')
    p.add_argument('--small-object-loss', action='store_true')
    p.add_argument('--boundary-loss', action='store_true')
    p.add_argument('--small-area-threshold', type=float, default=1024.0)
    p.add_argument('--small-crop-prob', type=float, default=0.5)
    p.add_argument('--small-loss-max-weight', type=float, default=2.0)
    p.add_argument('--boundary-loss-weight', type=float, default=0.1)
    args = p.parse_args()
    if (args.iters < 1 or args.batch_size < 1 or args.image_size <= 0 or args.image_size % 32
            or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0):
        p.error('Invalid training budget, image size, worker count or learning rate')
    if (not 0 <= args.small_crop_prob <= 1 or not 0 < args.small_area_threshold < math.inf
            or not 1 <= args.small_loss_max_weight < math.inf
            or not 0 <= args.boundary_loss_weight < math.inf):
        p.error('Invalid optimization hyperparameters')
    if args.trained_checkpoint and args.mode != 'eval':
        p.error('This runner does not yet support resume/fine-tuning adapted weights; fresh train only')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Refusing to mix a new experiment with a nonempty output directory')
    # Verify every exported file before starting the expensive run.
    verify_export(Path(args.data_root))
    categories = categories_from_export(args.data_root)
    source = Path(args.checkpoint).resolve()
    saved = torch.load(source, map_location='cpu')
    state = saved['state_dict'] if args.trained_checkpoint else map_coco_state(saved['state_dict'], categories)
    if args.trained_checkpoint:
        metadata = saved.get('meta', {})
        from mmengine.config import Config
        prior = Config.fromstring(metadata.get('cfg', metadata.get('config', '')), '.py')
        if prior.model.depth_fusion.enabled != args.depth_fusion:
            raise ValueError('Repeat the trained checkpoint depth-fusion switch for evaluation')
        if prior.experiment_provenance.dataset_manifest_sha256 != sha256(Path(args.data_root) / 'manifest.json'):
            raise ValueError('Adapted checkpoint was trained on a different dataset manifest')
    initial_classifier = state['panoptic_head.cls_embed.weight'].clone()
    initial_adapter = state['depth_adapter.weight'].clone()
    register_all_modules(init_default_scope=True)
    cfg = build_config(args, categories)
    import_modules_from_strings(**cfg.custom_imports)
    script_dir = Path(__file__).parent
    files = [Path(__file__), script_dir / 'run_coco_experiment.py', script_dir / 'export_nyu_development.py',
             *sorted((script_dir / 'rgbd').glob('*.py')), *sorted((script_dir / 'optimizations').glob('*.py'))]
    provenance = dict(dataset_manifest_sha256=sha256(Path(args.data_root) / 'manifest.json'),
                       source_checkpoint_sha256=sha256(source),
                       implementation_sha256={str(f.relative_to(script_dir)): sha256(f) for f in files},
                       categories=categories, protocol='nyuv2-official-train-scene-dev-indoor8-v1',
                       evaluation_policy='mask prediction intersect label_valid; discard empty predictions; COCO segm')
    cfg.experiment_provenance = provenance
    output.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output / 'config.py'))
    report = dict(status='running', args=vars(args), **provenance,
                  mmdetection_commit=repository_commit(Path(args.workspace) / 'mmdetection'),
                  gpu=torch.cuda.get_device_name(0),
                  note='Custom indoor8 development AP, not official NYUv2 or course benchmark. '
                       'Official test not used for training/tuning. Mask losses ignore void pixels; '
                       'query no-object classification is not a complete partial-annotation solution.')
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        runner = Runner.from_cfg(cfg)
        # MMEngine train() initializes weights before loading. Initialize once
        # before strict loading; BaseModule prevents the second call overwriting.
        runner.model.init_weights()
        runner.model.load_state_dict(state, strict=True)
        del state, saved
        report['parameters'] = sum(p.numel() for p in runner.model.parameters())
        if args.mode == 'eval':
            metrics = runner.test()
        else:
            runner.train()
            latest = Path((output / 'last_checkpoint').read_text().strip())
            checkpoint = torch.load(latest, map_location='cpu')
            weights = checkpoint['state_dict']
            if not all(torch.isfinite(v).all() for v in weights.values()
                       if torch.is_tensor(v) and v.is_floating_point()):
                raise RuntimeError('Non-finite model weights')
            changed = not torch.equal(initial_classifier, weights['panoptic_head.cls_embed.weight'])
            depth_changed = not torch.equal(initial_adapter, weights['depth_adapter.weight'])
            steps = optimizer_steps(checkpoint)
            if (checkpoint['meta']['iter'] != args.iters or not steps or not changed
                    or depth_changed != args.depth_fusion):
                raise RuntimeError('Iteration, optimizer, classifier or depth-switch update verification failed')
            scalars = runner.message_hub.log_scalars
            for key, history in scalars.items():
                if ('loss' in key or 'grad_norm' in key) and not all(
                        math.isfinite(float(v)) for v in history.data[0]):
                    raise RuntimeError(f'Non-finite training scalar {key}')
            metrics = {k.removeprefix('val/'): float(v.current()) for k, v in scalars.items()
                       if k.startswith('val/nyuv2_void/')}
            report.update(checkpoint=str(latest), checkpoint_sha256=sha256(latest),
                          final_iteration=checkpoint['meta']['iter'], final_optimizer_steps=steps,
                          classifier_changed=changed, depth_adapter_changed=depth_changed,
                          depth_adapter_l2=float(weights['depth_adapter.weight'].norm()),
                          training_images=len(runner.train_dataloader.dataset),
                          loss_records=len(scalars['train/loss'].data[0]),
                          first_loss=float(scalars['train/loss'].data[0][0]),
                          last_loss=float(scalars['train/loss'].current()))
        torch.cuda.synchronize()
        mandatory = [f'nyuv2_void/segm_{key}' for key in ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l')]
        if any(key not in metrics or not math.isfinite(float(metrics[key])) for key in mandatory):
            raise RuntimeError(f'Missing or non-finite aggregate metrics: {metrics}')
        val = json.loads((Path(args.data_root) / 'annotations/instances_val.json').read_text())
        present = {ann['category_id'] for ann in val['annotations']}
        absent = {f'nyuv2_void/{c["name"]}_precision' for c in categories if c['id'] not in present}
        clean_metrics, unavailable = {}, []
        for key, value in metrics.items():
            value = float(value)
            if not math.isfinite(value):
                if key not in absent:
                    raise RuntimeError(f'Unexpected non-finite metric {key}')
                clean_metrics[key] = None
                unavailable.append(key)
            else:
                clean_metrics[key] = value
        report.update(status='passed', metrics=clean_metrics,
                      metrics_without_ground_truth=unavailable,
                      metric_note='COCO -1 means no GT for this area range, not negative accuracy.',
                      seconds_including_evaluation=time.perf_counter() - started,
                      whole_run_peak_allocated_mib=max(torch.cuda.max_memory_allocated() / 2**20,
                          runner.message_hub.get_info('whole_run_peak_allocated_mib') or 0))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        raise
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

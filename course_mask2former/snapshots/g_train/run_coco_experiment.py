"""Small, reproducible RGB COCO engineering experiments (not a paper benchmark).

Uses the actual MMEngine Runner, DataLoader, CocoMetric and checkpoint hooks.
The development recipe deliberately differs from the official 50-epoch recipe.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmdet.utils import register_all_modules
from optimizations.configure import apply_optimizations


@HOOKS.register_module()
class RejectNonfiniteLossHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for key, value in (outputs or {}).items():
            if 'loss' in key and not torch.isfinite(torch.as_tensor(value)).all():
                raise RuntimeError(f'Non-finite {key} at iteration {runner.iter}: {value}')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit(repo):
    """Read this checkout's Git metadata; the reused CUDA image has no git CLI."""
    git_dir = Path(repo) / '.git'
    head = (git_dir / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = head[5:]
        if not ref.startswith('refs/') or '..' in Path(ref).parts:
            raise ValueError('Unexpected Git reference')
        loose_ref = git_dir / ref
        if loose_ref.is_file():
            head = loose_ref.read_text().strip()
        else:
            matches = [line.split()[0] for line in (git_dir / 'packed-refs').read_text().splitlines()
                       if not line.startswith(('#', '^')) and line.endswith(' ' + ref)]
            if len(matches) != 1:
                raise ValueError('Cannot resolve checkout commit')
            head = matches[0]
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError(f'Unexpected commit: {head}')
    return head


def build_config(args):
    root = Path(args.workspace).resolve()
    repo = root / 'mmdetection'
    cfg = Config.fromfile(str(repo / 'configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'))
    data = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    size = args.image_size
    cfg.work_dir = str(output)
    cfg.model.backbone.init_cfg = None
    cfg.model.data_preprocessor.batch_augments[0].size = (size, size)
    train_pipeline = [
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='RandomResize', scale=(size, size), ratio_range=(0.8, 1.25),
             resize_type='Resize', keep_ratio=True),
        dict(type='RandomCrop', crop_size=(size, size), crop_type='absolute',
             recompute_bbox=True, allow_negative_crop=True),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
        dict(type='PackDetInputs'),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile', to_float32=True),
        dict(type='Resize', scale=(round(size * 1333 / 800), size), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor')),
    ]
    cfg.train_dataloader = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        persistent_workers=args.workers > 0,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        batch_sampler=dict(type='AspectRatioBatchSampler'),
        dataset=dict(type='CocoDataset', data_root=str(data),
                     ann_file='annotations/instances_train2017.json',
                     data_prefix=dict(img='train2017/'),
                     filter_cfg=dict(filter_empty_gt=True, min_size=32),
                     pipeline=train_pipeline))
    cfg.val_dataloader = dict(
        batch_size=1, num_workers=args.workers, persistent_workers=args.workers > 0,
        drop_last=False, sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='CocoDataset', data_root=str(data),
                     ann_file='annotations/instances_val2017.json',
                     data_prefix=dict(img='val2017/'), test_mode=True, pipeline=test_pipeline))
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.val_evaluator = dict(type='CocoMetric',
                             ann_file=str(data / 'annotations/instances_val2017.json'),
                             metric=['segm'], format_only=False)
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=args.iters,
                         val_interval=args.iters)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    cfg.optim_wrapper.optimizer.lr = args.lr
    cfg.optim_wrapper.clip_grad.error_if_nonfinite = True
    cfg.param_scheduler = []
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=16)
    cfg.default_hooks.logger.interval = 10
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=False,
                                       interval=args.iters, save_last=True, max_keep_ckpts=-1)
    cfg.custom_hooks = [dict(type='RejectNonfiniteLossHook')]
    cfg.log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.env_cfg.cudnn_benchmark = False
    cfg.launcher = 'none'
    cfg.load_from = str(Path(args.resume or args.checkpoint).resolve())
    cfg.resume = bool(args.resume)
    apply_optimizations(cfg, args)
    if args.mode == 'eval':
        cfg.train_cfg = cfg.train_dataloader = cfg.optim_wrapper = cfg.param_scheduler = None
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
        cfg.custom_hooks = []
    return cfg


def optimizer_steps(checkpoint):
    state = checkpoint.get('optimizer', {}).get('state', {})
    return sorted({int(value['step']) for value in state.values() if 'step' in value})


def verify_resume_recipe(source_config_text, cfg):
    """Never silently disable optimizations or change data when resuming."""
    if not isinstance(source_config_text, str):
        raise ValueError('Resume checkpoint lacks its saved config; recipe consistency cannot be checked')
    previous = Config.fromstring(source_config_text, '.py')
    keys = ('model', 'train_dataloader', 'val_dataloader', 'optim_wrapper',
            'param_scheduler', 'randomness', 'experiment_provenance')
    old_dict, new_dict = previous.to_dict(), cfg.to_dict()
    # Record entrypoint changes, but compare optimization implementation hashes
    # strictly. This allows metadata-reader/reporting fixes without pretending
    # the launcher file is identical; model/data/optimizer recipes still match.
    entry_hashes = []
    for candidate in (old_dict, new_dict):
        implementation = candidate.get('experiment_provenance', {}).get('implementation_sha256', {})
        entry_hashes.append(implementation.pop('run_coco_experiment.py', None))
    changed = [key for key in keys if old_dict.get(key) != new_dict.get(key)]
    if changed:
        raise ValueError('Resume recipe differs: ' + ', '.join(changed)
                         + '. Repeat the same optimization switches/data/seed/hyperparameters; '
                         'use an independent fresh run to change the recipe.')
    return dict(entrypoint_sha256_changed=entry_hashes[0] != entry_hashes[1],
                optimization_module_hashes_match=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'eval'])
    parser.add_argument('--workspace', default='/workspace')
    parser.add_argument('--data-root', default='/workspace/data/coco_subset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', default='/workspace/checkpoints/'
                        'mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth')
    parser.add_argument('--resume', help='Explicit existing checkpoint; restores AdamW and iteration.')
    parser.add_argument('--iters', type=int, default=100, help='Total iterations, including resumed steps.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--small-object-crop', action='store_true', help='Enable small-object-aware training crop')
    parser.add_argument('--small-object-loss', action='store_true', help='Enable small-instance mask loss weights')
    parser.add_argument('--boundary-loss', action='store_true', help='Enable final-decoder boundary auxiliary loss')
    parser.add_argument('--small-area-threshold', type=float, default=1024.0)
    parser.add_argument('--small-crop-prob', type=float, default=0.5)
    parser.add_argument('--small-loss-max-weight', type=float, default=2.0)
    parser.add_argument('--boundary-loss-weight', type=float, default=0.1)
    args = parser.parse_args()
    if args.iters < 1 or args.batch_size < 1 or args.workers < 0 or args.image_size % 32:
        parser.error('iters/batch size must be positive; workers >=0; size a multiple of 32')
    if args.resume and args.mode != 'train':
        parser.error('--resume is only for training')
    if (not 0 <= args.small_crop_prob <= 1 or args.small_area_threshold <= 0
            or args.small_loss_max_weight < 1 or args.boundary_loss_weight < 0):
        parser.error('Invalid optimization probability, area threshold or loss weights')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Refusing to mix a new experiment with a nonempty output directory')
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.resume or args.checkpoint).resolve()
    manifest = Path(args.data_root) / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Prepare and validate the dataset first: {manifest}')
    source_state = torch.load(source, map_location='cpu')
    initial_iter = int(source_state.get('meta', {}).get('iter', 0)) if args.resume else 0
    initial_steps = optimizer_steps(source_state) if args.resume else []
    source_config_text = source_state.get('meta', {}).get('cfg', source_state.get('meta', {}).get('config'))
    if args.resume and (initial_iter >= args.iters or not initial_steps):
        raise ValueError('Resume needs a lower iteration and a checkpoint containing optimizer state')
    initial_classifier = source_state['state_dict']['panoptic_head.cls_embed.weight'].clone()
    del source_state
    register_all_modules(init_default_scope=True)
    cfg = build_config(args)
    if cfg.get('custom_imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    implementation_hashes = {str(path.relative_to(Path(__file__).parent)): sha256(path)
                             for path in [Path(__file__), *sorted((Path(__file__).parent / 'optimizations').glob('*.py'))]}
    cfg.experiment_provenance = dict(dataset_manifest_sha256=sha256(manifest),
                                     implementation_sha256=implementation_hashes)
    resume_recipe = verify_resume_recipe(source_config_text, cfg) if args.resume else None
    cfg.dump(str(output / f'config_{args.mode}_{args.iters}.py'))
    repo = Path(args.workspace) / 'mmdetection'
    git = repository_commit(repo)
    report = dict(
        status='running', mode=args.mode, args=vars(args), mmdetection_commit=git,
        dataset_manifest_sha256=sha256(manifest), input_checkpoint=str(source),
        input_checkpoint_sha256=sha256(source), initial_iteration=initial_iter,
        initial_optimizer_steps=initial_steps, gpu=torch.cuda.get_device_name(0),
        implementation_sha256=implementation_hashes,
        resume_recipe_verification=resume_recipe,
        note='RGB COCO SUBSET development recipe, not the official 50e benchmark; '
             'not RGB-D, not course results, not proof of algorithm improvement. '
             'The initial checkpoint was pretrained on COCO train2017. '
             'Resume restores state, but does not promise bitwise-identical data order.')
    report_path = output / f'report_{args.mode}_{args.iters}.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runner = Runner.from_cfg(cfg)
    if args.mode == 'eval':
        metrics = runner.test()
    else:
        runner.train()
        # Validation runs on the final iteration; no extra validation pass is needed.
        metrics = {key.removeprefix('val/'): value for key, value in
                   runner.message_hub.runtime_info.items() if key.startswith('val/')}
        pointer = output / 'last_checkpoint'
        last = Path(pointer.read_text().strip())
        saved = torch.load(last, map_location='cpu')
        final_iter = int(saved['meta']['iter'])
        final_steps = optimizer_steps(saved)
        if not all(torch.isfinite(tensor).all().item() for tensor in saved['state_dict'].values()
                   if torch.is_tensor(tensor) and tensor.is_floating_point()):
            raise RuntimeError('Saved model contains non-finite parameters or buffers')
        changed = not torch.equal(initial_classifier, saved['state_dict']['panoptic_head.cls_embed.weight'])
        if final_iter != args.iters or not changed or not final_steps:
            raise RuntimeError('Checkpoint iteration/optimizer/parameter-update verification failed')
        if args.resume and min(final_steps) <= min(initial_steps):
            raise RuntimeError('AdamW step counter did not advance after resume')
        report.update(final_iteration=final_iter, final_optimizer_steps=final_steps,
                      classifier_weights_changed=changed, checkpoint=str(last),
                      checkpoint_sha256=sha256(last),
                      training_images_after_filter=len(runner.train_dataloader.dataset))
        # Metrics and every recorded loss are recoverable even across a resumed run.
        scalars = runner.message_hub.log_scalars
        for key, history in scalars.items():
            values = history.data[0]
            if ('loss' in key or 'grad_norm' in key) and not all(math.isfinite(float(x)) for x in values):
                raise RuntimeError(f'Non-finite loss recorded: {key}')
            if key.startswith('val/'):
                metrics[key.removeprefix('val/')] = float(history.current())
        if 'train/loss' in scalars:
            values = scalars['train/loss'].data[0]
            report['loss_records'] = len(values)
            report['first_logged_loss'] = float(values[0])
            report['last_logged_loss'] = float(values[-1])
    torch.cuda.synchronize()
    if not metrics or not all(math.isfinite(float(v)) for v in metrics.values()):
        raise RuntimeError(f'Missing/non-finite evaluation metrics: {metrics}')
    report.update(status='passed', metrics={k: float(v) for k, v in metrics.items()},
                  seconds_including_evaluation=time.perf_counter() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

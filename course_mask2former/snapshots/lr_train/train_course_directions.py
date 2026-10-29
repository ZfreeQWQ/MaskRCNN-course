"""Fresh fixed-budget course ablations; opt-in features and audited coverage.

Old experiment entrypoints and frozen releases remain unchanged. Four new
switches are independent; the bounded launcher tests singles, never a grid.
"""
import copy
import json
import math
from pathlib import Path
import time

import torch
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmdet.utils import register_all_modules

import run_course_experiment as base
from launch_nyu_sweep import atomic_json
from run_coco_experiment import optimizer_steps, repository_commit, sha256

NEW_FLAGS = ('course_detail', 'course_depth_reliability', 'course_copy_paste',
             'course_high_resolution')
SIX_METRICS = tuple('course/segm_mAP' + s for s in ('', '_50', '_75', '_s', '_m', '_l'))
COCO_SHA = '41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879'
CONTROL_RUN = 'outputs/course_budget_20261014_v1/budget_diagnostic/runs/gated_depth_long'


def parser():
    p = base.parser()
    p.set_defaults(iters=30000, val_interval=5000, keep_checkpoints=6)
    for flag in NEW_FLAGS:
        p.add_argument('--' + flag.replace('_', '-'), action='store_true')
    return p


def validate_direction_args(args):
    if any(getattr(args, name) for name in ('depth_fusion', 'small_object_crop',
            'small_object_loss', 'boundary_loss', 'foreground_points', 'rgb_control',
            'course_foreground_points', 'course_large_scale_jitter', 'course_small_boundary')):
        raise ValueError('Legacy extra treatments are not part of this bounded course batch')
    if args.image_size != 1024:
        raise ValueError('Base image-size must be 1024; high-resolution switch explicitly selects 1536')
    if (args.course_depth_reliability or args.course_copy_paste or args.course_high_resolution
            or args.course_detail) and not (args.multiscale_depth and args.reliability_gate):
        raise ValueError('New treatment requires the accepted gated RGB-D baseline')
    if args.functional_smoke:
        if args.iters != 20 or args.val_interval != 20:
            raise ValueError('This batch smoke is exactly 20 steps with one validation')
    elif (args.iters, args.val_interval, args.keep_checkpoints, args.batch_size,
          args.seed, args.lr, args.branch_lr_mult, args.schedule) != (
              30000, 5000, 6, 2, 20261013, 1e-5, 10., 'decay'):
        raise ValueError('Formal recipe must remain the predeclared 30k recipe')


def build_config(args, categories):
    validate_direction_args(args)
    effective = copy.copy(args)
    effective.image_size = 1536 if args.course_high_resolution else 1024
    cfg = base.build_config(effective, categories)
    if args.course_detail or args.course_depth_reliability:
        cfg.custom_imports.imports.append('rgbd.course_feature_models')
        cfg.model.type = 'CourseFeatureRGBDMask2Former'
        cfg.model.course_detail = dict(enabled=args.course_detail)
        cfg.model.course_depth_reliability = dict(enabled=args.course_depth_reliability)
        if args.course_detail:
            cfg.optim_wrapper.paramwise_cfg.custom_keys['course_detail_adapter'] = dict(
                lr_mult=args.branch_lr_mult, decay_mult=1.0)
    if args.course_copy_paste:
        cfg.custom_imports.imports.append('rgbd.course_copy_paste')
        dataset = cfg.train_dataloader.dataset
        if [t['type'] for t in dataset.pipeline[:3]] != [
                'LoadCourseRGBDImage', 'LoadAnnotations', 'RGBDRandomFlip']:
            raise ValueError('Copy-Paste must operate before original-coordinate geometry')
        dataset.pipeline.insert(2, dict(type='CourseRGBDSmallCopyPaste', enabled=True,
            data_root=dataset.data_root, ann_file=dataset.ann_file,
            prob=.5, max_paste=3, area_threshold=1024, max_instances=100))
        pack = dataset.pipeline[-1]
        if pack['type'] != 'PackRGBDDetInputs':
            raise ValueError('Expected paired pack transform')
        pack['meta_keys'] = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'course_copy_paste_activity')
    # This hook audits only; it does not sample RNG or change model/optimizer.
    cfg.custom_hooks.append(dict(type='CourseDirectionsAuditHook'))
    return cfg


def verify_reference_recipe(args, categories, cfg):
    """Reconstruct shared 30k base and compare to the accepted frozen config."""
    control_args = copy.copy(args)
    for name in NEW_FLAGS:
        setattr(control_args, name, False)
    control_args.multiscale_depth = control_args.reliability_gate = True
    control_args.functional_smoke = False
    control_args.iters, control_args.val_interval, control_args.keep_checkpoints = 30000, 5000, 6
    reference_path = Path(args.workspace)/CONTROL_RUN/'config.py'
    reference = Config.fromfile(str(reference_path)).to_dict()
    rebuilt = base.build_config(control_args, categories).to_dict()
    keys = ('model', 'train_cfg', 'optim_wrapper', 'param_scheduler', 'randomness',
            'train_dataloader', 'val_dataloader', 'val_evaluator', 'auto_scale_lr', 'env_cfg')
    if any(rebuilt[k] != reference[k] for k in keys):
        changed = [k for k in keys if rebuilt[k] != reference[k]]
        raise ValueError('Shared base drifted from archived 30k recipe: ' + ', '.join(changed))
    def differences(a, b, path=''):
        if isinstance(a, dict) and isinstance(b, dict):
            result = []
            for key in sorted(set(a) | set(b)):
                result += differences(a.get(key), b.get(key), path+'.'+str(key))
            return result
        if a == b:
            return []
        return [dict(path=path, archived=a, treatment=b)]
    actual = cfg.to_dict()
    return dict(status='passed', shared_base_matches=True,
        reference_config=str(reference_path), reference_config_sha256=sha256(reference_path),
        compared_keys=list(keys), differences=[row for k in keys
            for row in differences(reference[k], actual[k], k)])


def finite_tree(value):
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def initial_state(model, state, detail=False):
    current = model.state_dict()
    missing = set(current) - set(state)
    allowed = {k for k in current if k.startswith('spatial_fusion.')}
    if detail:
        allowed |= {k for k in current if k.startswith('course_detail_adapter.')}
        if not any(k.startswith('course_detail_adapter.') for k in allowed):
            raise ValueError('Detail switch has no new parameter')
    if missing != allowed or set(state) - set(current):
        raise ValueError('Unexpected missing/unexpected initialization keys')
    current.update(state)
    model.load_state_dict(current, strict=True)
    tracked = ('spatial_fusion.', 'depth_adapter.', 'panoptic_head.cls_embed.', 'course_detail_adapter.')
    initial = {k: v.detach().cpu().clone() for k, v in current.items() if k.startswith(tracked)}
    if any(bool(v.count_nonzero()) for k, v in initial.items()
           if k.startswith(('spatial_fusion.projections.', 'spatial_fusion.gates.', 'course_detail_adapter.'))):
        raise ValueError('New projection/gate/detail must initialize at zero')
    return initial


@HOOKS.register_module()
class CourseDirectionsAuditHook(Hook):
    def __init__(self):
        self.coverage = []
        self.train_iterations = 0
        self.copy_activity = {}

    def before_train(self, runner):
        runner.course_directions_audit = self
        self.expected_ids = [im['id'] for im in json.loads(
            Path(runner.cfg.val_evaluator.ann_file).read_text())['images']]
        if len(self.expected_ids) != len(set(self.expected_ids)):
            raise ValueError('Validation annotation has duplicate IDs')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not finite_tree(outputs or {}):
            raise RuntimeError('Nonfinite training output')
        grad = runner.message_hub.get_scalar('train/grad_norm').current()
        if not math.isfinite(float(grad)):
            raise RuntimeError('Nonfinite gradient norm')
        self.train_iterations += 1
        for sample in (data_batch or {}).get('data_samples', []):
            activity = sample.metainfo.get('course_copy_paste_activity')
            if activity is not None:
                for key, value in activity.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.copy_activity[key] = self.copy_activity.get(key, 0) + value
        step = runner.iter + 1
        if step % 100 == 0 or step == runner.max_iters:
            feature = getattr(runner.model, 'feature_activity', {})
            if hasattr(runner.model, 'feature_activity_snapshot'):
                feature = runner.model.feature_activity_snapshot()
            else:
                feature = {k: v.detach().cpu().item() if torch.is_tensor(v) else v
                           for k, v in feature.items()}
            atomic_json(Path(runner.work_dir)/'direction_activity.json', dict(
                iteration=step, finite_train_iterations=self.train_iterations,
                feature=feature, copy_paste=self.copy_activity))

    def before_val_epoch(self, runner):
        self.val_ids = []

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for sample in outputs:
            self.val_ids.append(int(sample.img_id))
            pred = sample.pred_instances
            if len(pred) > 100 or not finite_tree(dict(scores=pred.scores, bboxes=pred.bboxes)):
                raise RuntimeError('Invalid validation predictions')
            if tuple(pred.masks.shape[-2:]) != tuple(sample.ori_shape[:2]):
                raise RuntimeError('Prediction masks are not in original image coordinates')
            if (pred.masks.dtype != torch.bool or not bool((pred.labels == 0).all())
                    or not bool(((pred.scores >= 0) & (pred.scores <= 1)).all())):
                raise RuntimeError('Invalid mask/label/score contract')

    def after_val_epoch(self, runner, metrics=None):
        if self.val_ids != self.expected_ids:
            raise RuntimeError('Validation IDs incomplete, duplicated or reordered')
        if any(k not in (metrics or {}) or not math.isfinite(float(metrics[k])) for k in SIX_METRICS):
            raise RuntimeError('Missing/nonfinite COCO metric')
        self.coverage.append(dict(iteration=runner.iter, completed_images=len(self.val_ids),
                                 unique_images=len(set(self.val_ids))))
        atomic_json(Path(runner.work_dir)/'validation_coverage.json', self.coverage)


def main():
    args = parser().parse_args()
    validate_direction_args(args)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing nonempty output')
    data = Path(args.data_root).resolve()
    if data.name != 'prepared' or 'course_blender_20261013' not in data.parts:
        raise ValueError('Only the prepared official course data is permitted')
    if sha256(args.checkpoint) != COCO_SHA:
        raise ValueError('Require pinned original COCO initialization')
    completion = json.loads((data/'prepare_complete.json').read_text())
    if completion['manifest_sha256'] != sha256(data/'manifest.json'):
        raise ValueError('Prepared manifest changed')
    prefix = 'smoke' if args.functional_smoke else 'instances'
    train_path, val_path = [data/'annotations'/f'{prefix}_{s}.json' for s in ('train', 'val')]
    train, val = [json.loads(p.read_text()) for p in (train_path, val_path)]
    categories = [dict(id=1, name='component', supercategory='component')]
    if train['categories'] != categories or val['categories'] != categories:
        raise ValueError('Expected the single component category')
    if {im['scene_id'] for im in train['images']} & {im['scene_id'] for im in val['images']}:
        raise ValueError('Train/val scene leakage')
    if not args.functional_smoke and (len(train['images']), len(val['images'])) != (14432, 2584):
        raise ValueError('Require complete official train/val')
    for doc, split in ((train, 'train'), (val, 'val')):
        for image in doc['images']:
            for key, directory in (('file_name', 'rgb'), ('depth_file_name', 'depth')):
                rel = Path(image[key])
                if rel.is_absolute() or '..' in rel.parts or not (data/rel).resolve().is_relative_to(data/directory/split):
                    raise ValueError('Image path outside the declared train/val split')
    register_all_modules(init_default_scope=True)
    cfg = build_config(args, categories)
    import_modules_from_strings(**cfg.custom_imports)
    config_checks = verify_reference_recipe(args, categories, cfg)
    origin = Path(__file__).parent
    files = [origin/name for name in ('train_course_directions.py', 'run_course_experiment.py',
        'run_nyu_v2.py', 'run_nyu_experiment.py', 'run_coco_experiment.py', 'nyu_v4_plan.py')]
    files += sorted((origin/'rgbd').glob('*.py')) + sorted((origin/'optimizations').glob('*.py'))
    provenance = dict(train_annotations_sha256=sha256(train_path), val_annotations_sha256=sha256(val_path),
        source_checkpoint_sha256=sha256(args.checkpoint),
        implementation_sha256={str(f.relative_to(origin)): sha256(f) for f in files},
        class_initialization='mean of 80 foreground COCO rows; preserve no-object',
        protocol='component-official-train-val-standard-coco-mask-maxdets100',
        test_inputs_used=False, test_annotations_used=False)
    cfg.experiment_provenance = provenance
    output.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output/'config.py'))
    atomic_json(output/'config_checks.json', config_checks)
    report = dict(status='running', args=vars(args), **provenance,
        training_images=len(train['images']), validation_images=len(val['images']),
        effective_train_size=1536 if args.course_high_resolution else 1024,
        effective_eval_size=1536 if args.course_high_resolution else 1024,
        config_checks=config_checks,
        mmdetection_commit=repository_commit(Path(args.workspace)/'mmdetection'),
        note='20-step functional test, NOT accuracy evidence.' if args.functional_smoke else
             'Single seed, fixed final 30k primary; no early stopping, test access or automatic tuning. '
             'High-resolution training compares to archived 1024-trained/1536-evaluated control; not equal-compute.')
    atomic_json(output/'report.json', report)
    started = time.monotonic()
    try:
        runner = Runner.from_cfg(cfg)
        runner.model.init_weights()
        state = base.component_state(torch.load(args.checkpoint, map_location='cpu')['state_dict'])
        initial = initial_state(runner.model, state, args.course_detail)
        del state
        report.update(parameters=sum(p.numel() for p in runner.model.parameters()),
            active_parameters=runner.model.active_parameter_count(),
            state_tensor_bytes=sum(t.numel()*t.element_size() for t in runner.model.state_dict().values()))
        atomic_json(output/'report.json', report)
        runner.train()
        latest = Path((output/'last_checkpoint').read_text().strip())
        checkpoint = torch.load(latest, map_location='cpu')
        weights = checkpoint['state_dict']
        def changed(prefix):
            return any(not torch.equal(v, weights[k]) for k, v in initial.items() if k.startswith(prefix))
        updates = dict(classifier_changed=changed('panoptic_head.cls_embed.'),
            depth_adapter_changed=changed('depth_adapter.'),
            spatial_projection_changed=changed('spatial_fusion.projections.'),
            spatial_encoder_changed=changed(('spatial_fusion.stem.', 'spatial_fusion.stages.')),
            spatial_gate_changed=changed('spatial_fusion.gates.'),
            detail_adapter_changed=changed('course_detail_adapter.'))
        expected = dict(classifier_changed=True, depth_adapter_changed=False,
            spatial_projection_changed=args.multiscale_depth, spatial_encoder_changed=args.multiscale_depth,
            spatial_gate_changed=args.reliability_gate, detail_adapter_changed=args.course_detail)
        if updates != expected:
            raise RuntimeError(f'Parameter update mismatch: {updates} != {expected}')
        if checkpoint['meta']['iter'] != args.iters or optimizer_steps(checkpoint) != [args.iters]:
            raise RuntimeError('Final optimizer iteration mismatch')
        audit = runner.course_directions_audit
        scalars = runner.message_hub.log_scalars
        finite = dict(model=finite_tree(weights), optimizer=finite_tree(checkpoint['optimizer']),
            loss=all(math.isfinite(float(v)) for k, h in scalars.items() if 'loss' in k for v in h.data[0]),
            grad=all(math.isfinite(float(v)) for v in scalars['train/grad_norm'].data[0]))
        if not all(finite.values()) or audit.train_iterations != args.iters:
            raise RuntimeError('Finite-value/complete iteration audit failed')
        metrics = {k.removeprefix('val/'): float(v.current()) for k, v in scalars.items() if k.startswith('val/course/')}
        if any(k not in metrics or not math.isfinite(metrics[k]) for k in SIX_METRICS):
            raise RuntimeError('Missing/nonfinite final six COCO metrics')
        curve = json.loads((output/'learning_curve.json').read_text())
        if [p['iteration'] for p in curve] != list(range(args.val_interval, args.iters+1, args.val_interval)):
            raise RuntimeError('Incomplete curve')
        activity = json.loads((output/'direction_activity.json').read_text())
        if [p['iteration'] for p in audit.coverage] != list(range(args.val_interval,args.iters+1,args.val_interval)):
            raise RuntimeError('Incomplete validation coverage audit')
        if args.course_copy_paste and (activity['copy_paste'].get('pasted_instances', 0) <= 0
                or activity['copy_paste'].get('calls') != args.iters*args.batch_size):
            raise RuntimeError('Copy-Paste enabled but not exercised or missing consumed-sample metadata')
        feature = activity['feature']
        if args.course_detail or args.course_depth_reliability:
            if feature.get('train_calls') != args.iters:
                raise RuntimeError('Feature training call count differs from optimizer iterations')
        if args.course_detail and feature.get('detail_train_samples') != args.iters*args.batch_size:
            raise RuntimeError('Detail switch not exercised on every training sample')
        if args.course_depth_reliability and (
                feature.get('reliability_train_samples') != args.iters*args.batch_size
                or feature.get('training_sensor_valid_pixels', 0) <= 0
                or feature.get('training_reliability_suppressed_pixels', 0) <= 0
                or not 0 < feature.get('training_mean_reliability_on_valid', float('nan')) < 1):
            raise RuntimeError('Reliability enabled but had no verified nontrivial effect')
        # Final file integrity is rechecked, not assumed from the initial report.
        if sha256(train_path) != provenance['train_annotations_sha256'] or sha256(val_path) != provenance['val_annotations_sha256']:
            raise RuntimeError('Annotations changed during training')
        if sha256(args.checkpoint) != COCO_SHA or any(sha256(origin/k) != v for k, v in provenance['implementation_sha256'].items()):
            raise RuntimeError('Initialization or frozen implementation changed')
        report.update(status='passed', **updates, finite_checks=finite,
            finite_train_iterations=audit.train_iterations, validation_coverage=audit.coverage,
            direction_activity=activity, metrics=metrics, learning_curve=curve,
            final_iteration=args.iters, final_optimizer_steps=optimizer_steps(checkpoint),
            checkpoint=str(latest), checkpoint_sha256=sha256(latest),
            seconds_including_evaluation=time.monotonic()-started)
    except Exception as exc:
        report.update(status='failed', error=type(exc).__name__+': '+str(exc),
                      seconds_including_evaluation=time.monotonic()-started)
        atomic_json(output/'report.json', report)
        raise
    atomic_json(output/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

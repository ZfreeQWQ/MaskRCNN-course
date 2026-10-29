"""Component RGB-D course training; fresh initialization and standard COCO AP."""
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

import run_nyu_v2 as v2
from run_coco_experiment import sha256, optimizer_steps, repository_commit
from nyu_v4_plan import scheduler_config
from launch_nyu_sweep import atomic_json


@HOOKS.register_module()
class CourseCurveHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        counters = getattr(runner.model.panoptic_head, 'small_boundary_counters', None)
        step = runner.iter + 1
        if counters is not None and (step % 100 == 0 or step == runner.max_iters):
            atomic_json(Path(runner.work_dir) / 'small_boundary_activity.json',
                        dict(iteration=step, counters=dict(counters)))

    def after_val_epoch(self, runner, metrics=None):
        row = dict(iteration=runner.iter, metrics={k: float(v) for k, v in metrics.items()})
        if not all(math.isfinite(v) for v in row['metrics'].values()):
            raise RuntimeError('Nonfinite course validation metric')
        path = Path(runner.work_dir) / 'learning_curve.json'
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append(row)
        atomic_json(path, rows)


def parser():
    p = v2.parser()
    p.set_defaults(data_root='/workspace/data/course_blender_20261013/prepared',
                   image_size=1024, iters=10000, val_interval=2500, seed=20261013)
    p.add_argument('--schedule', choices=['constant', 'decay'], default='decay')
    p.add_argument('--depth-max', type=float, default=10.)
    p.add_argument('--functional-smoke', action='store_true')
    p.add_argument('--keep-checkpoints', type=int, default=2,
                   help='Retain this many periodic checkpoints; does not alter training or model selection')
    p.add_argument('--course-foreground-points', action='store_true',
                   help='Fixed-budget foreground quota for fully annotated small course masks')
    p.add_argument('--course-large-scale-jitter', action='store_true',
                   help='Train-only resize ratio 0.8–2.0 instead of 0.8–1.25; inference unchanged')
    p.add_argument('--course-small-boundary', action='store_true',
                   help='Final-layer small-only boundary BCE; fixed weight 0.1, radius 1, peak >=0.5')
    return p


def build_config(args, categories):
    v2.validate_args(args)
    if args.iters < 10 or args.keep_checkpoints < 1 or not math.isfinite(args.depth_max) or args.depth_max <= 0:
        raise ValueError('Invalid course budget/depth range')
    cfg = v2.build_config(args, categories)
    if args.course_large_scale_jitter:
        resizes = [t for t in cfg.train_dataloader.dataset.pipeline if t['type'] == 'RandomResize']
        if len(resizes) != 1 or resizes[0]['resize_type'] != 'RGBDResize':
            raise ValueError('Course scale jitter requires one paired RGB-D resize')
        resizes[0]['ratio_range'] = (0.8, 2.0)
    cfg.custom_imports.imports.append('rgbd.course_data')
    cfg.custom_hooks = [h for h in cfg.custom_hooks if h['type'] != 'ValidationCurveHook']
    cfg.custom_hooks.append(dict(type='CourseCurveHook'))
    # Course annotations cover the whole image. No NYUv2 void clipping is
    # applied to predictions or ground truth.
    cfg.model.panoptic_head.ignore_void = False
    if args.foreground_points:
        raise ValueError('Use --course-foreground-points for fully annotated course data')
    if args.course_foreground_points:
        cfg.custom_imports.imports.append('rgbd.course_head')
        cfg.model.panoptic_head.type = 'CoursePointQuotaMask2FormerHead'
        cfg.model.panoptic_head.course_point_sampling = dict(
            enabled=True, quota=args.foreground_quota, area_threshold=args.small_area_threshold)
    if args.course_small_boundary:
        if args.boundary_loss:
            raise ValueError('Generic and small-only boundary are alternative objectives')
        cfg.custom_imports.imports.append('rgbd.small_boundary_head')
        cfg.model.panoptic_head.type = 'CourseSmallBoundaryMask2FormerHead'
        cfg.model.panoptic_head.small_boundary_loss = dict(
            enabled=True, area_threshold=1024., loss_weight=.1, radius=1, min_target_peak=.5)
    cfg.model.data_preprocessor.depth_max = args.depth_max
    cfg.param_scheduler = scheduler_config(args.schedule, args.iters)
    cfg.default_hooks.checkpoint.max_keep_ckpts = args.keep_checkpoints
    prefix = 'smoke' if args.functional_smoke else 'instances'
    for loader, split in ((cfg.train_dataloader, 'train'), (cfg.val_dataloader, 'val'),
                          (cfg.test_dataloader, 'val')):
        loader.dataset.type = 'CourseRGBDCocoDataset'
        loader.dataset.ann_file = f'annotations/{prefix}_{split}.json'
        loader.dataset.pipeline[0] = dict(type='LoadCourseRGBDImage')
    evaluator = dict(type='CocoMetric', ann_file=str(Path(args.data_root)/'annotations'/f'{prefix}_val.json'),
        metric='segm', classwise=True, prefix='course', proposal_nums=(1, 10, 100))
    cfg.val_evaluator = evaluator
    cfg.test_evaluator = copy.deepcopy(evaluator)
    return cfg


def component_state(state):
    state = dict(state)
    for suffix in ('weight', 'bias'):
        key = 'panoptic_head.cls_embed.' + suffix
        if state[key].shape[0] != 81:
            raise ValueError('Expected official 80-class COCO checkpoint')
        # component has no corresponding COCO category. Initialize its row
        # from the mean foreground row and preserve the no-object row.
        state[key] = torch.stack([state[key][:80].mean(0), state[key][80]])
    state['depth_adapter.weight'] = torch.zeros(3, 2, 1, 1, dtype=state['panoptic_head.cls_embed.weight'].dtype)
    return state


def main():
    args = parser().parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing nonempty experiment output')
    data = Path(args.data_root).resolve()
    marker = ('prepare_complete.json' if (data/'prepare_complete.json').is_file()
              else 'sample_manifest.json' if args.functional_smoke else 'prepare_complete.json')
    if not (data / marker).is_file():
        raise FileNotFoundError('Prepared course data marker missing')
    if marker == 'prepare_complete.json':
        completion=json.loads((data/marker).read_text())
        if completion['manifest_sha256']!=sha256(data/'manifest.json'):
            raise ValueError('Prepared data manifest checksum mismatch')
    prefix = 'smoke' if args.functional_smoke else 'instances'
    train = json.loads((data/'annotations'/f'{prefix}_train.json').read_text())
    val = json.loads((data/'annotations'/f'{prefix}_val.json').read_text())
    if train['categories'] != val['categories'] or train['categories'] != [{'id':1,'name':'component','supercategory':'component'}]:
        raise ValueError('Course categories disagree')
    if {im['scene_id'] for im in train['images']} & {im['scene_id'] for im in val['images']}:
        raise ValueError('Scene leakage')
    if not args.functional_smoke and (len(train['images']),len(val['images'])) != (14432,2584):
        raise ValueError('Full official train/val split required')
    register_all_modules(init_default_scope=True)
    cfg = build_config(args, train['categories'])
    import_modules_from_strings(**cfg.custom_imports)
    origin = Path(__file__).parent
    files = [origin/name for name in ('run_course_experiment.py','run_nyu_v2.py',
        'run_nyu_experiment.py','run_coco_experiment.py','nyu_v4_plan.py')]
    files += list((origin/'rgbd').glob('*.py')) + list((origin/'optimizations').glob('*.py'))
    provenance = dict(train_annotations_sha256=sha256(data/'annotations'/f'{prefix}_train.json'),
        val_annotations_sha256=sha256(data/'annotations'/f'{prefix}_val.json'),
        source_checkpoint_sha256=sha256(args.checkpoint),
        implementation_sha256={str(f.relative_to(origin)):sha256(f) for f in files},
        class_initialization='mean of 80 foreground COCO rows; preserve no-object',
        protocol='component-official-train-val-standard-coco-mask-maxdets100',
        test_annotations_used=False, test_inputs_used=False)
    cfg.experiment_provenance = provenance
    output.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(output/'config.py'))
    report = dict(status='running', args=vars(args), **provenance,
        training_images=len(train['images']), validation_images=len(val['images']),
        mmdetection_commit=repository_commit(Path(args.workspace)/'mmdetection'),
        note='Functional smoke is not a course score or accuracy comparison.' if args.functional_smoke else
             'Exploratory course baseline; fixed final budget primary, no test-set tuning.')
    atomic_json(output/'report.json', report)
    start = time.monotonic()
    try:
        runner = Runner.from_cfg(cfg)
        runner.model.init_weights()
        state = component_state(torch.load(args.checkpoint, map_location='cpu')['state_dict'])
        initial = v2.load_initial(runner.model,state)
        del state
        runner.train()
        latest = Path((output/'last_checkpoint').read_text().strip())
        checkpoint = torch.load(latest, map_location='cpu')
        weights = checkpoint['state_dict']
        def changed(prefix):
            return any(not torch.equal(v,weights[k]) for k,v in initial.items() if k.startswith(prefix))
        updates = dict(classifier_changed=changed('panoptic_head.cls_embed.'),
                       depth_adapter_changed=changed('depth_adapter.'),
                       spatial_projection_changed=changed('spatial_fusion.projections.'),
                       spatial_encoder_changed=changed(('spatial_fusion.stem.','spatial_fusion.stages.')),
                       spatial_gate_changed=changed('spatial_fusion.gates.'))
        expected = dict(classifier_changed=True, depth_adapter_changed=args.depth_fusion,
                        spatial_projection_changed=args.multiscale_depth,
                        spatial_encoder_changed=args.multiscale_depth,
                        spatial_gate_changed=args.reliability_gate)
        if updates != expected:
            raise RuntimeError(f'Weight update mismatch: {updates}')
        if checkpoint['meta']['iter'] != args.iters or optimizer_steps(checkpoint) != [args.iters]:
            raise RuntimeError('Optimizer/final step mismatch')
        if not all(torch.isfinite(v).all() for v in weights.values() if torch.is_tensor(v) and v.is_floating_point()):
            raise RuntimeError('Nonfinite checkpoint')
        scalars = runner.message_hub.log_scalars
        for key, history in scalars.items():
            if ('loss' in key or 'grad_norm' in key) and not all(math.isfinite(float(x)) for x in history.data[0]):
                raise RuntimeError('Nonfinite training scalar')
        metrics = {k.removeprefix('val/'):float(v.current()) for k,v in scalars.items() if k.startswith('val/course/')}
        if any('course/segm_'+key not in metrics for key in ('mAP','mAP_s')):
            raise RuntimeError('Missing Mask AP/APs')
        curve = json.loads((output/'learning_curve.json').read_text())
        if [p['iteration'] for p in curve] != list(range(args.val_interval,args.iters+1,args.val_interval)):
            raise RuntimeError('Incomplete learning curve')
        point_counters = getattr(runner.model.panoptic_head, 'course_point_counters', None)
        if args.course_foreground_points and (not point_counters or point_counters['replaced_points'] <= 0):
            raise RuntimeError('Course foreground switch enabled but never exercised')
        boundary_counters = getattr(runner.model.panoptic_head, 'small_boundary_counters', None)
        if args.course_small_boundary and (not boundary_counters
                or boundary_counters['calls'] != args.iters
                or boundary_counters['nonzero_calls'] <= 0
                or boundary_counters['resolvable'] <= 0):
            raise RuntimeError('Small boundary switch was not exercised correctly')
        report.update(status='passed',**updates, metrics=metrics, learning_curve=curve,
            course_point_counters=point_counters,
            small_boundary_counters=boundary_counters,
            final_iteration=args.iters, final_optimizer_steps=optimizer_steps(checkpoint),
            checkpoint=str(latest), checkpoint_sha256=sha256(latest),
            parameters=sum(p.numel() for p in runner.model.parameters()),
            active_parameters=runner.model.active_parameter_count(),
            seconds_including_evaluation=time.monotonic()-start)
    except Exception as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc))
        atomic_json(output/'report.json',report)
        raise
    atomic_json(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()

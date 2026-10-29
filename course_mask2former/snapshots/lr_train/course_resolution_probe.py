"""Fixed accepted checkpoint, official val only, 1024/1536 inference diagnostic.

No optimizer, training, test input, TTA or score filtering for formal COCO AP.
The larger scale is opt-in. Every invocation requires a new output directory.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time

from course_demo import sha256, verify_run, encode_predictions
from course_batch_predict import VAL_SHA, atomic_json, inside, validate_rows

CHECKPOINT_SHA = 'a833fd0dccee55dc3ccd9a106a4e226d0cc51e4979b5d21b84ea55d80b443790'
FIXED_IDS = list(range(201960001, 201967002, 1000))
DEFAULT_RUN = '/workspace/outputs/course_budget_20261014_v1/budget_diagnostic/runs/gated_depth_long'


def now():
    return datetime.now(timezone.utc).isoformat()


def scaled_pipeline(original, scale=1024):
    if type(scale) is not int or scale not in (1024, 1536):
        raise ValueError('Only predeclared 1024 or 1536 permitted')
    pipeline = copy.deepcopy([dict(t) for t in original if t['type'] != 'LoadAnnotations'])
    if [t['type'] for t in pipeline] != ['LoadCourseRGBDImage', 'RGBDResize', 'PackRGBDDetInputs']:
        raise ValueError('Unexpected inference pipeline')
    if tuple(pipeline[1]['scale']) != (1706, 1024) or not pipeline[1]['keep_ratio']:
        raise ValueError('Unexpected accepted geometry')
    pipeline[1]['scale'] = (1706 * scale // 1024, scale)
    return pipeline


def safe_request(root, image):
    root = Path(root).resolve()
    rgb, depth = inside(root, image['file_name']), inside(root, image['depth_file_name'])
    if ('test' in root.parts or not rgb.is_relative_to(root/'rgb/val')
            or not depth.is_relative_to(root/'depth/val')
            or (image['height'], image['width']) != (1024, 1024)):
        raise ValueError('Only official original-grid val images permitted')
    return dict(img_id=image['id'], img_path=str(rgb), depth_file=str(depth),
                depth_scale_m=image['depth_scale_m'], depth_invalid_code=image['depth_invalid_code'],
                scene=image['scene_id'])


def check_coverage(actual, expected):
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError('Incomplete, reordered or duplicated image coverage')


def check_baseline(metrics, reference):
    keys = ['course/segm_mAP' + suffix for suffix in ('', '_50', '_75', '_s', '_m', '_l')]
    if any(k not in metrics or not math.isfinite(metrics[k])
           or abs(metrics[k] - reference[k]) > .0005 for k in keys):
        raise ValueError('1024 control does not reproduce accepted rounded metrics; stop comparison')


def inspect_prediction(pred, image_id):
    import torch
    if len(pred) > 100 or tuple(pred.masks.shape) != (len(pred), 1024, 1024):
        raise ValueError('Prediction count/original geometry invalid')
    if pred.masks.dtype != torch.bool or not bool((pred.labels == 0).all()):
        raise ValueError('Expected binary component masks')
    if (not bool(torch.isfinite(pred.scores).all()) or not bool(torch.isfinite(pred.bboxes).all())
            or not bool(((pred.scores >= 0) & (pred.scores <= 1)).all())):
        raise ValueError('Nonfinite or invalid prediction values')
    return dict(image_id=image_id, predictions=len(pred), shape=list(pred.masks.shape[-2:]))


def case_diagnostics(pred, coco, image_id):
    import numpy as np
    from pycocotools import mask as mu
    from scipy.ndimage import binary_erosion, binary_dilation
    masks = pred.masks.numpy()
    scores = pred.scores.numpy()
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    gt = [coco.annToRLE(a) for a in anns]
    rles = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    ious = mu.iou(rles, gt, [0] * len(gt)) if rles else np.zeros((0, len(gt)))
    display = np.where(scores >= .3)[0]
    matched = {}
    for p in sorted(display, key=lambda i: -float(scores[i])):
        remaining = [j for j in range(len(anns)) if j not in matched]
        if remaining:
            j = max(remaining, key=lambda j: ious[p, j])
            if ious[p, j] >= .5:
                pe = masks[p] & ~binary_erosion(masks[p], border_value=0)
                gm = mu.decode(gt[j]).astype(bool)
                ge = gm & ~binary_erosion(gm, border_value=0)
                precision = (pe & binary_dilation(ge, iterations=2)).sum()/max(1, pe.sum())
                recall = (ge & binary_dilation(pe, iterations=2)).sum()/max(1, ge.sum())
                f1 = float(2*precision*recall/(precision+recall)) if precision+recall else 0.
                matched[j] = dict(prediction_index=int(p), iou=float(ious[p, j]), boundary_f1=f1)
    def best(j, indices):
        if not len(indices):
            return dict(prediction_index=None, iou=0., score=None)
        p = max(indices, key=lambda p: ious[p, j])
        return dict(prediction_index=int(p), iou=float(ious[p, j]), score=float(scores[p]))
    rows = [dict(annotation_id=a['id'], area_original=a['area'], small=a['area'] <= 1024,
                 best_all=best(j, range(len(scores))), best_display=best(j, display),
                 display_match=matched.get(j)) for j, a in enumerate(anns)]
    return dict(image_id=image_id, instances=rows, predictions=len(scores),
                display_predictions=len(display), display_unmatched_predictions=len(display)-len(matched),
                gt_sent_to_model=False, note='Fixed diagnostic subset, not AP. Raw means model postprocessed output, not all decoder queries.')


def run(args):
    output = Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(Path('/workspace/outputs')):
        raise ValueError('Require new directory under /workspace/outputs')
    report, checkpoint, source = verify_run(args.run)
    if report['checkpoint_sha256'] != CHECKPOINT_SHA or report['final_iteration'] != 30000:
        raise ValueError('Only accepted control30k is authorized')
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.evaluator import Evaluator
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from pycocotools.coco import COCO
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(Path(args.run)/'config.py'))
    import_modules_from_strings(**cfg.custom_imports)
    pipeline = scaled_pipeline(cfg.val_dataloader.dataset.pipeline, args.course_eval_scale)
    ann_file = Path(cfg.val_evaluator.ann_file)
    if ann_file.name != 'instances_val.json' or sha256(ann_file) != VAL_SHA:
        raise ValueError('Official val annotation identity mismatch')
    if (cfg.val_evaluator.type != 'CocoMetric' or cfg.val_evaluator.metric != 'segm'
            or tuple(cfg.val_evaluator.proposal_nums) != (1, 10, 100)):
        raise ValueError('Only accepted standard COCO mask evaluation allowed')
    doc = json.loads(ann_file.read_text())
    images = doc['images']
    if len(images) != 2584 or len({im['id'] for im in images}) != 2584:
        raise ValueError('Wrong official val coverage')
    requests = [safe_request(cfg.val_dataloader.dataset.data_root, im) for im in images]
    by_id = {r['img_id']: r for r in requests}
    if args.mode == 'smoke':
        requests = [by_id[i] for i in FIXED_IDS]
    output.mkdir(parents=True)
    state = dict(status='running', phase='initializing', mode=args.mode,
        input_scale=args.course_eval_scale, started_at_utc=now(), checkpoint_sha256=CHECKPOINT_SHA,
        val_annotations_sha256=VAL_SHA, frozen_source=str(source),
        training_report_sha256=sha256(Path(args.run)/'report.json'),
        config_sha256=sha256(Path(args.run)/'config.py'),
        new_script_sha256=sha256(__file__), training_started=False,
        test_inputs_used=False, test_annotations_used=False, gt_sent_to_model=False,
        expected_images=len(requests), completed_images=0)
    atomic_json(output/'status.json', state)
    try:
        adjusted = copy.deepcopy(cfg)
        adjusted.val_dataloader.dataset.pipeline = pipeline
        adjusted.dump(str(output/'inference_config.py'))
        # Removing LoadAnnotations is common to both arms. Only this Resize
        # scale differs between arms; all model/metric/training values are intact.
        atomic_json(output/'config_delta.json', dict(annotation_free=True, model_changed=False,
            original_scale=list(cfg.val_dataloader.dataset.pipeline[1]['scale']),
            actual_scale=list(pipeline[1]['scale']), evaluator=cfg.val_evaluator.to_dict()))
        saved = torch.load(checkpoint, map_location='cpu')
        trained = Config.fromstring(saved['meta'].get('cfg', saved['meta'].get('config')), '.py')
        if trained.model.to_dict() != cfg.model.to_dict():
            raise ValueError('Checkpoint/model config mismatch')
        if any(not bool(torch.isfinite(t).all()) for t in saved['state_dict'].values() if t.is_floating_point()):
            raise ValueError('Nonfinite checkpoint')
        model = MODELS.build(cfg.model)
        model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        model.cuda().eval()
        torch.backends.cudnn.benchmark = False
        compose = Compose(pipeline)
        def sample(request):
            value = compose(copy.deepcopy(request))
            ds = value['data_samples']
            if 'gt_instances' in ds:
                if len(ds.gt_instances):
                    raise ValueError('GT must not enter inference')
                del ds.gt_instances
            if tuple(value['inputs'].shape) != (5, args.course_eval_scale, args.course_eval_scale):
                raise ValueError('RGB-D packed geometry mismatch')
            if tuple(ds.ori_shape) != (1024, 1024) or not bool(torch.isfinite(value['inputs']).all()):
                raise ValueError('Original shape or numerical values invalid')
            return value
        coco = COCO(str(ann_file))
        evaluator = None
        if args.mode == 'evaluate':
            ev = copy.deepcopy(cfg.val_evaluator)
            ev.outfile_prefix = str(output/'predictions')
            evaluator = Evaluator(ev)
            evaluator.dataset_meta = cfg.val_dataloader.dataset.metainfo.to_dict()
        cases, completed, hashes, item_seconds = [], [], [], []
        with torch.inference_mode():
            fixed = pseudo_collate([sample(by_id[FIXED_IDS[0]])])
            for _ in range(5):
                warm = model.test_step(copy.deepcopy(fixed)); torch.cuda.synchronize(); del warm
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(20):
                batch = copy.deepcopy(fixed)
                torch.cuda.synchronize(); start = time.perf_counter()
                timed = model.test_step(batch); torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                del timed, batch
            timing = dict(warmup=5, repeats=20, image_id=FIXED_IDS[0], all_seconds=times,
                mean_ms=statistics.mean(times)*1000, median_ms=statistics.median(times)*1000,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                scope='FP32 batch1 synchronized model.test_step; includes preprocess/H2D/postprocess; excludes disk/decode/resize/batch clone')
            atomic_json(output/'timing.json', timing)
            del fixed
            state.update(phase='predicting', updated_at_utc=now()); atomic_json(output/'status.json', state)
            for request in requests:
                started = time.monotonic()
                batch = pseudo_collate([sample(request)])
                prediction = model.test_step(batch)[0].cpu()
                identity = request['img_id']
                inspect_prediction(prediction.pred_instances, identity)
                if evaluator is not None:
                    evaluator.process(data_samples=[prediction], data_batch=batch)
                if identity in FIXED_IDS:
                    rows = encode_predictions(prediction.pred_instances.masks.numpy(), prediction.pred_instances.scores.numpy(), identity)
                    validate_rows(rows, identity, [1024, 1024])
                    atomic_json(output/f'case_{identity}_predictions.json', rows)
                    cases.append(case_diagnostics(prediction.pred_instances, coco, identity))
                hashes.append(dict(image_id=identity, rgb_sha256=sha256(request['img_path']), depth_sha256=sha256(request['depth_file'])))
                completed.append(identity)
                item_seconds.append(time.monotonic()-started)
                if len(completed) % 50 == 0 or len(completed) == len(requests):
                    state.update(completed_images=len(completed), last_image_id=identity,
                        recent_seconds_per_image=statistics.mean(item_seconds[-50:]), updated_at_utc=now())
                    atomic_json(output/'status.json', state)
                    print(json.dumps(state), flush=True)
                del prediction, batch
        check_coverage(completed, [r['img_id'] for r in requests])
        atomic_json(output/'input_hashes.json', hashes)
        atomic_json(output/'cases.json', cases)
        metrics = None
        if evaluator is not None:
            state.update(phase='coco_evaluation', updated_at_utc=now()); atomic_json(output/'status.json', state)
            metrics = evaluator.evaluate(len(requests))
            if not metrics or any(not np.isfinite(v) for v in metrics.values()):
                raise ValueError('Invalid complete metrics')
            atomic_json(output/'metrics.json', metrics)
            if args.course_eval_scale == 1024:
                check_baseline(metrics, report['metrics'])
        if sha256(checkpoint) != CHECKPOINT_SHA or sha256(ann_file) != VAL_SHA:
            raise ValueError('Checkpoint/annotations changed during evaluation')
        verify_run(args.run)  # Recheck the original frozen sources at completion.
        state.update(status='passed', phase='complete', finished_at_utc=now(), metrics=metrics,
            input_hashes_sha256=sha256(output/'input_hashes.json'),
            active_parameters=model.active_parameter_count(),
            state_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
            fixed_case_ids=[c['image_id'] for c in cases], timing=timing)
        atomic_json(output/'report.json', state)
        atomic_json(output/'status.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), finished_at_utc=now())
        atomic_json(output/'status.json', state)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default=DEFAULT_RUN)
    p.add_argument('--mode', choices=['smoke', 'evaluate'], required=True)
    p.add_argument('--course-eval-scale', type=int, choices=[1024, 1536], default=1024)
    p.add_argument('--output', required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())

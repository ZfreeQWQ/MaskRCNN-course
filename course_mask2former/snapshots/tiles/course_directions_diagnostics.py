"""Fixed-eight val diagnostics and isolated FP32 timing of accepted 30k arms.

Run once per formal arm in a fresh GPU0 process after training acceptance. No
training, full-val AP evaluation, test inputs, scale search, or checkpoint
selection occurs. Ground truth is used only by the post-prediction diagnostic.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time

from course_batch_predict import VAL_SHA, atomic_json, validate_rows
from course_demo import encode_predictions, sha256, verify_run
from course_resolution_probe import (FIXED_IDS, case_diagnostics, check_coverage,
                                     inspect_prediction, now, safe_request)
from run_course_directions import (BASE_RECIPE, INITIAL_SHA, NEW_FLAGS, OLD_FLAGS,
                                   TRAIN_SHA, jobs, metric_values)


def accepted_identity(report, run_id):
    """Fail closed on smoke, mixed switches, budgets or unaccepted endpoints."""
    choices = {job['id']: job for job in jobs()}
    if run_id not in choices:
        raise ValueError('Only the six predeclared direction arms are supported')
    job = choices[run_id]
    args = report.get('args', {})
    if (report.get('status') != 'passed' or report.get('final_iteration') != 30000
            or report.get('final_optimizer_steps') != [30000]
            or report.get('training_images') != 14432 or report.get('validation_images') != 2584
            or report.get('protocol') != 'component-official-train-val-standard-coco-mask-maxdets100'
            or report.get('test_inputs_used') is not False
            or report.get('test_annotations_used') is not False
            or report.get('finite_checks') != dict(model=True, optimizer=True, loss=True, grad=True)):
        raise ValueError('Require an accepted finite full 30k course endpoint')
    if (report.get('train_annotations_sha256') != TRAIN_SHA
            or report.get('val_annotations_sha256') != VAL_SHA
            or report.get('source_checkpoint_sha256') != INITIAL_SHA):
        raise ValueError('Accepted training data or initialization identity differs')
    if any(args.get(key) != value for key, value in dict(BASE_RECIPE, seed=job['seed']).items()):
        raise ValueError('Unexpected training recipe')
    flags = dict.fromkeys((*OLD_FLAGS, *NEW_FLAGS, 'multiscale_depth', 'reliability_gate',
                           'functional_smoke', 'trained_checkpoint'), False)
    flags.update(job['flags'])
    if any(args.get(name, False) is not value for name, value in flags.items()):
        raise ValueError('Run flags differ from the predeclared single treatment')
    size = 1536 if flags['course_high_resolution'] else 1024
    if report.get('effective_train_size') != size or report.get('effective_eval_size') != size:
        raise ValueError('Effective scale differs from the accepted treatment')
    endpoints = list(range(5000, 30001, 5000))
    curve = report.get('learning_curve', [])
    if [row.get('iteration') for row in curve] != endpoints:
        raise ValueError('Require all six final-budget validation points')
    for row in curve:
        metric_values(row['metrics'])
    if metric_values(report.get('metrics')) != metric_values(curve[-1]['metrics']):
        raise ValueError('Final metrics differ from the predeclared 30k endpoint')
    coverage = report.get('validation_coverage', [])
    if (len(coverage) != 6 or any(row.get('iteration') != step
            or row.get('completed_images') != 2584 or row.get('unique_images') != 2584
            for row, step in zip(coverage, endpoints))):
        raise ValueError('Require six complete official validation coverages')
    return size


def annotation_free_pipeline(original, size):
    if type(size) is not int or size not in (1024, 1536):
        raise ValueError('Only the configured 1024/1536 scale is supported')
    if [part['type'] for part in original] != [
            'LoadCourseRGBDImage', 'RGBDResize', 'LoadAnnotations', 'PackRGBDDetInputs']:
        raise ValueError('Unexpected accepted inference pipeline')
    resize = original[1]
    if tuple(resize['scale']) != (round(size * 1333 / 800), size) or resize.get('keep_ratio') is not True:
        raise ValueError('Configured inference geometry disagrees with the accepted run')
    return copy.deepcopy([dict(part) for part in original if part['type'] != 'LoadAnnotations'])


def select_fixed(doc, data_root):
    images = doc['images']
    if len(images) != 2584 or len({im['id'] for im in images}) != 2584:
        raise ValueError('Require official 2584-image val metadata')
    by_id = {im['id']: im for im in images}
    if any(identity not in by_id for identity in FIXED_IDS):
        raise ValueError('Predetermined fixed diagnostic image missing')
    # Never enumerate another split or fall back to another image selection.
    return [safe_request(data_root, by_id[identity]) for identity in FIXED_IDS]


def input_hashes(requests):
    return [dict(image_id=request['img_id'], rgb_sha256=sha256(request['img_path']),
                 depth_sha256=sha256(request['depth_file'])) for request in requests]


def verify_direction(run_path):
    run_path = Path(run_path).resolve()
    if run_path.parent.name != 'runs' or run_path.parent.parent.name != 'formal':
        raise ValueError('Diagnostics only accepts a formal/runs arm')
    report, checkpoint, source = verify_run(run_path)
    size = accepted_identity(report, run_path.name)
    if checkpoint.name != 'iter_30000.pth':
        raise ValueError('Only the final 30k checkpoint may be diagnosed')
    plan = json.loads((run_path.parents[2] / 'plan.json').read_text())
    implementation = report.get('implementation_sha256', {})
    if not implementation or 'train_course_directions.py' not in implementation:
        raise ValueError('Missing implementation provenance')
    for relative, digest in implementation.items():
        if plan['source_sha256'].get(relative) != digest:
            raise ValueError('Accepted implementation differs from the frozen plan')
    return report, checkpoint, source, size


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument('--run', required=True)
    command.add_argument('--output', required=True)
    return command


def run(args):
    output, run_path = Path(args.output).resolve(), Path(args.run).resolve()
    if output.exists() or not output.is_relative_to(Path('/workspace/outputs')):
        raise ValueError('Require a NEW directory under /workspace/outputs')
    if not run_path.is_relative_to(Path('/workspace/outputs')):
        raise ValueError('Require a course run under /workspace/outputs')
    output.mkdir(parents=True)
    state = dict(status='running', phase='verification', started_at_utc=now(),
                 expected_images=8, completed_images=0, training_started=False,
                 test_inputs_used=False, test_annotations_used=False,
                 gt_sent_to_model=False, ap_reevaluated=False, run=str(run_path))
    atomic_json(output/'status.json', state)
    try:
        report, checkpoint, source, size = verify_direction(run_path)
        if Path(__file__).resolve().parent != source.resolve():
            raise ValueError('Execute the diagnostics from the accepted frozen source directory')
        sys.path.insert(0, str(source))
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose, pseudo_collate
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        from pycocotools.coco import COCO
        from run_coco_experiment import optimizer_steps
        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(str(run_path/'config.py'))
        import_modules_from_strings(**cfg.custom_imports)
        pipeline = annotation_free_pipeline(cfg.val_dataloader.dataset.pipeline, size)
        data_root = Path(cfg.val_dataloader.dataset.data_root).resolve()
        ann = Path(cfg.val_evaluator.ann_file).resolve()
        if (str(data_root) != '/workspace/data/course_blender_20261013/prepared'
                or ann != data_root/'annotations/instances_val.json'
                or sha256(ann) != VAL_SHA):
            raise ValueError('Only the pinned official val annotation/data root is permitted')
        requests = select_fixed(json.loads(ann.read_text()), data_root)
        hashes = input_hashes(requests)
        state.update(input_scale=size, checkpoint_sha256=report['checkpoint_sha256'],
                     val_annotations_sha256=VAL_SHA, frozen_source=str(source),
                     training_report_sha256=sha256(run_path/'report.json'),
                     config_sha256=sha256(run_path/'config.py'),
                     script_sha256=sha256(__file__), training_metrics=report['metrics'])
        atomic_json(output/'status.json', state)
        atomic_json(output/'input_hashes.json', hashes)
        atomic_json(output/'inference_pipeline.json', pipeline)
        saved = torch.load(checkpoint, map_location='cpu')
        trained = Config.fromstring(saved['meta'].get('cfg', saved['meta'].get('config')), '.py')
        if (trained.model.to_dict() != cfg.model.to_dict()
                or trained.val_dataloader.dataset.pipeline != cfg.val_dataloader.dataset.pipeline
                or trained.val_evaluator.to_dict() != cfg.val_evaluator.to_dict()):
            raise ValueError('Checkpoint/model/validation configuration mismatch')
        if saved['meta']['iter'] != 30000 or optimizer_steps(saved) != [30000]:
            raise ValueError('Actual checkpoint is not the accepted 30k optimizer endpoint')
        for value in saved['state_dict'].values():
            if torch.is_tensor(value) and value.is_floating_point():
                if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                    raise ValueError('Require finite FP32 checkpoint weights')
        model = MODELS.build(cfg.model)
        model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        model.cuda().eval()
        torch.backends.cudnn.benchmark = False
        parameters = model.active_parameter_count()
        tensor_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
        if parameters != report['active_parameters'] or tensor_bytes != report['state_tensor_bytes']:
            raise ValueError('Inference model size differs from the accepted training model')
        compose = Compose(pipeline)

        def sample(request):
            value = compose(copy.deepcopy(request))
            ds = value['data_samples']
            if 'gt_instances' in ds:
                if len(ds.gt_instances):
                    raise ValueError('Ground truth must not enter model inference')
                del ds.gt_instances
            if (tuple(value['inputs'].shape) != (5, size, size)
                    or tuple(ds.ori_shape) != (1024, 1024)
                    or not bool(torch.isfinite(value['inputs']).all())):
                raise ValueError('Packed RGB-D/original geometry or finite-value failure')
            return value

        coco = COCO(str(ann))
        state.update(phase='timing')
        atomic_json(output/'status.json', state)
        with torch.inference_mode():
            fixed = pseudo_collate([sample(requests[0])])
            for _ in range(5):
                prediction = model.test_step(copy.deepcopy(fixed))
                torch.cuda.synchronize()
                del prediction
            torch.cuda.reset_peak_memory_stats()
            seconds = []
            for _ in range(20):
                batch = copy.deepcopy(fixed)
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = model.test_step(batch)
                torch.cuda.synchronize()
                seconds.append(time.perf_counter()-started)
                del prediction, batch
            if any(not math.isfinite(value) or value <= 0 for value in seconds):
                raise ValueError('Invalid synchronized timing values')
            timing = dict(warmup=5, repeats=20, image_id=FIXED_IDS[0], all_seconds=seconds,
                          mean_ms=statistics.mean(seconds)*1000,
                          median_ms=statistics.median(seconds)*1000,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          device_name=torch.cuda.get_device_name(0),
                          scope='FP32 batch1 synchronized model.test_step; includes preprocess/H2D/postprocess; excludes disk/decode/resize/batch clone')
            atomic_json(output/'timing.json', timing)
            del fixed
            cases, completed = [], []
            for request in requests:
                prediction = model.test_step(pseudo_collate([sample(request)]))[0].cpu()
                identity = request['img_id']
                inspect_prediction(prediction.pred_instances, identity)
                rows = encode_predictions(prediction.pred_instances.masks.numpy(),
                                          prediction.pred_instances.scores.numpy(), identity)
                validate_rows(rows, identity, [1024, 1024])
                atomic_json(output/f'case_{identity}_predictions.json', rows)
                cases.append(case_diagnostics(prediction.pred_instances, coco, identity))
                completed.append(identity)
                state.update(phase='cases', completed_images=len(completed),
                             last_image_id=identity, updated_at_utc=now())
                atomic_json(output/'status.json', state)
                del prediction
        check_coverage(completed, FIXED_IDS)
        atomic_json(output/'cases.json', cases)
        if input_hashes(requests) != hashes or sha256(ann) != VAL_SHA:
            raise ValueError('Diagnostic inputs/val annotations changed during inference')
        verify_direction(run_path)  # Recheck checkpoint and frozen source hashes.
        if (sha256(run_path/'report.json') != state['training_report_sha256']
                or sha256(run_path/'config.py') != state['config_sha256']):
            raise ValueError('Accepted report/config changed during diagnostics')
        state.update(status='passed', phase='complete', finished_at_utc=now(),
                     fixed_case_ids=completed, original_grid_verified=True,
                     input_hashes_sha256=sha256(output/'input_hashes.json'),
                     cases_sha256=sha256(output/'cases.json'),
                     active_parameters=parameters, state_tensor_bytes=tensor_bytes,
                     timing=timing,
                     note='Fixed eight-image diagnostics, not AP; one fresh process, no optimizer on GPU. Boundary means exclude unmatched GT.')
        atomic_json(output/'report.json', state)
        atomic_json(output/'status.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update(status='failed', error=repr(exc), finished_at_utc=now())
        atomic_json(output/'report.json', state)
        atomic_json(output/'status.json', state)
        raise


if __name__ == '__main__':
    run(parser().parse_args())

"""Frozen, bounded, fail-fast six-job course queue; host-side standard library only.

The archived 30k gated control is reused, never retrained. All six 20-step GPU
smokes must pass before any official job is dispatched. No retries, tuning,
test-set access, or intervention in another user's containers is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid

from launch_nyu_sweep import (CHECKPOINT, IMAGE, SOURCE_FILES, atomic_json,
                              container_path, docker_command, gpu_snapshot, log_progress, sha256)

DATA = 'data/course_blender_20261013/prepared'
CONTROL = 'outputs/course_budget_20261014_v1/budget_diagnostic/runs/gated_depth_long'
CONTROL_REPORT_SHA = 'ef61d1e80d7c94046675b030ba572dcfeedb2488026ca5989aff7986e4e4b128'
CONTROL_CHECKPOINT_SHA = 'a833fd0dccee55dc3ccd9a106a4e226d0cc51e4979b5d21b84ea55d80b443790'
HIGH_RES_REPORT = 'outputs/course_resolution_20261015_v1/evaluate_1536/report.json'
HIGH_RES_REPORT_SHA = 'b32a6badb46fd7e5e3bb9c2e1f228422835a71d173b5d37b3d1a4faf2f546044'
INITIAL_SHA = '41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879'
TRAIN_SHA = '1f72e6564579f9257ab1ef54044207eccaeb452affd33f05b19a6ef00105e882'
VAL_SHA = '71612ebf1d548eb85e146e97de4d211578d2f87d5273f9b0f5e2d48b1af2ea2d'
FIXED_IDS = tuple(201960001 + i * 1000 for i in range(8))
FIXED_INPUTS_SHA = '0715f6ea1aaee444c445a5f65ed979fd7f8bdf0b03f9348b65b0c94a5df0589d'
TOTAL_SECONDS = 96 * 3600
JOB_SECONDS = 12 * 3600
HIGH_RES_SECONDS = 24 * 3600
SMOKE_SECONDS = 1800
CPU_SECONDS = 1800
GPU_WAIT_SECONDS = 6 * 3600
METRICS = tuple('course/segm_' + key for key in
                ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'))
NEW_FLAGS = ('course_detail', 'course_depth_reliability', 'course_copy_paste',
             'course_high_resolution')
OLD_FLAGS = ('depth_fusion', 'small_object_crop', 'small_object_loss', 'boundary_loss',
             'rgb_control', 'foreground_points', 'course_foreground_points',
             'course_large_scale_jitter', 'course_small_boundary')
BASE_RECIPE = dict(iters=30000, val_interval=5000, keep_checkpoints=6,
                   batch_size=2, image_size=1024, workers=2, lr=1e-5,
                   branch_lr_mult=10., schedule='decay', depth_max=10.)
REQUIRED_SOURCES = (*SOURCE_FILES, 'run_nyu_v2.py', 'nyu_v2_plan.py', 'nyu_v4_plan.py',
                    'run_course_experiment.py', 'train_course_directions.py',
                    'run_course_directions.py', 'course_resolution_probe.py',
                    'course_demo.py', 'course_batch_predict.py', 'course_directions_diagnostics.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def jobs():
    shared = dict(multiscale_depth=True, reliability_gate=True)
    variants = [('rgb_long', {}), ('depth_ungated_long', dict(multiscale_depth=True)),
                ('detail_long', dict(shared, course_detail=True)),
                ('reliable_depth_long', dict(shared, course_depth_reliability=True)),
                ('copy_paste_long', dict(shared, course_copy_paste=True)),
                ('high_resolution_long', dict(shared, course_high_resolution=True))]
    return [dict(id=name, seed=20261013, flags=flags, output_rel='runs/' + name,
                 max_job_seconds=HIGH_RES_SECONDS if flags.get('course_high_resolution')
                 else JOB_SECONDS) for name, flags in variants]


def stage_jobs(smoke):
    return [dict(job, flags=dict(job['flags'], functional_smoke=smoke)) for job in jobs()]


def stage_recipe(smoke):
    return dict(BASE_RECIPE, **(dict(iters=20, val_interval=20, keep_checkpoints=1)
                              if smoke else {}))


def metric_values(metrics, allowed_missing=()):
    if not isinstance(metrics, dict):
        raise ValueError('Missing six segmentation metrics')
    values = {key: metrics.get(key) for key in METRICS}
    permitted = set(allowed_missing)
    if not permitted <= {'course/segm_mAP_s', 'course/segm_mAP_m', 'course/segm_mAP_l'}:
        raise ValueError('Only absent area strata may have unavailable smoke metrics')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not (0 <= v <= 1 or (key in permitted and v == -1))
           for key, v in values.items()):
        raise ValueError('Missing, nonfinite, or out-of-range segmentation metrics')
    return values


def missing_area_metrics(document):
    """COCO includes boundary areas in both adjacent ranges; ignore crowd GT."""
    areas = [ann['area'] for ann in document['annotations']
             if not ann.get('iscrowd', False) and not ann.get('ignore', False)]
    if not areas or any(not math.isfinite(area) or area <= 0 for area in areas):
        raise ValueError('Smoke validation requires finite positive non-ignored GT areas')
    return ['course/segm_mAP_' + name for name, low, high in
            (('s', 0, 1024), ('m', 1024, 9216), ('l', 9216, float('inf')))
            if not any(low <= area <= high for area in areas)]


def assert_intact(root, plan):
    for rel, digest in plan['source_sha256'].items():
        if sha256(root / 'source' / rel) != digest:
            raise RuntimeError('Frozen source changed: ' + rel)


def source_paths(origin, cpu_tests):
    required = set(REQUIRED_SOURCES)
    # Existing regression modules reuse sibling runner/config utilities; freeze
    # those Python sources rather than importing a mutable deployment directory.
    required.update(p.name for p in origin.glob('*.py'))
    required.update('tests/' + name + '.py' for name in cpu_tests)
    # Snapshot all local model/augmentation dependencies, including new switches.
    for folder in ('rgbd', 'optimizations', 'tests'):
        required.update(str(p.relative_to(origin)) for p in (origin / folder).glob('*.py'))
    missing = [name for name in sorted(required) if not (origin / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing frozen-source dependency: ' + ', '.join(missing))
    return [origin / name for name in sorted(required)]


def verify_inputs(workspace):
    """Read official train/val metadata only; never inspect a test directory."""
    data = workspace / DATA
    marker = json.loads((data / 'prepare_complete.json').read_text())
    manifest_sha = sha256(data / 'manifest.json')
    if marker['manifest_sha256'] != manifest_sha:
        raise ValueError('Prepared manifest mismatch')
    if sha256(workspace / CHECKPOINT) != INITIAL_SHA:
        raise ValueError('Original COCO initialization changed')
    report_path = workspace / CONTROL / 'report.json'
    if sha256(report_path) != CONTROL_REPORT_SHA:
        raise ValueError('Archived control report changed')
    baseline = json.loads(report_path.read_text())
    if (baseline.get('status') != 'passed' or baseline.get('final_iteration') != 30000 or
            baseline.get('final_optimizer_steps') != [30000] or
            baseline.get('checkpoint_sha256') != CONTROL_CHECKPOINT_SHA or
            baseline.get('source_checkpoint_sha256') != INITIAL_SHA or
            baseline.get('test_inputs_used') is not False or
            baseline.get('test_annotations_used') is not False):
        raise ValueError('Archived control is not the accepted 30k endpoint')
    if sha256(workspace / CONTROL / 'iter_30000.pth') != CONTROL_CHECKPOINT_SHA:
        raise ValueError('Archived control checkpoint changed')
    metric_values(baseline['metrics'])
    if [p['iteration'] for p in baseline['learning_curve']] != list(range(5000, 30001, 5000)):
        raise ValueError('Archived control validation curve incomplete')
    high_res_path = workspace / HIGH_RES_REPORT
    if sha256(high_res_path) != HIGH_RES_REPORT_SHA:
        raise ValueError('Archived1536 evaluation report changed')
    high_res = json.loads(high_res_path.read_text())
    if (high_res.get('status') != 'passed' or high_res.get('input_scale') != 1536 or
            high_res.get('completed_images') != 2584 or
            high_res.get('checkpoint_sha256') != CONTROL_CHECKPOINT_SHA or
            high_res.get('val_annotations_sha256') != VAL_SHA or
            high_res.get('test_inputs_used') is not False or
            high_res.get('test_annotations_used') is not False):
        raise ValueError('High-resolution reference is not the accepted same-weight1536 evaluation')
    metric_values(high_res['metrics'])
    result = dict(dataset_manifest_sha256=manifest_sha, source_checkpoint_sha256=INITIAL_SHA)
    for prefix, expected_counts in (('instances', (14432, 2584)), ('smoke', None)):
        annotations = {}
        for i, split in enumerate(('train', 'val')):
            path = data / 'annotations' / f'{prefix}_{split}.json'
            document = json.loads(path.read_text())
            images = document['images']
            if (not images or len({im['id'] for im in images}) != len(images) or
                    (expected_counts is not None and len(images) != expected_counts[i])):
                raise ValueError('Official split image coverage invalid: ' + str(path))
            if document['categories'] != [dict(id=1, name='component', supercategory='component')]:
                raise ValueError('Unexpected course category metadata')
            annotations[split] = document
            digest = sha256(path)
            result[f'{prefix}_{split}_annotations_sha256'] = digest
            result[f'{prefix}_{split}_images'] = len(images)
            if prefix == 'instances' and digest != (TRAIN_SHA if split == 'train' else VAL_SHA):
                raise ValueError('Official annotations differ from archived control')
        train_scenes = {im['scene_id'] for im in annotations['train']['images']}
        if train_scenes & {im['scene_id'] for im in annotations['val']['images']}:
            raise ValueError('Train/val scene overlap')
        if prefix == 'smoke':
            result['smoke_missing_area_metrics'] = missing_area_metrics(annotations['val'])
    reference = dict(report=str(report_path), report_sha256=CONTROL_REPORT_SHA,
                     checkpoint=str(workspace / CONTROL / 'iter_30000.pth'),
                     checkpoint_sha256=CONTROL_CHECKPOINT_SHA, metrics=baseline['metrics'],
                     high_resolution=dict(report=str(high_res_path), report_sha256=HIGH_RES_REPORT_SHA,
                                          metrics=high_res['metrics']))
    return result, reference


def stage_plan(root, plan, phase):
    smoke = phase == 'smoke'
    return dict(workspace=plan['workspace'], image=plan['image_id'],
                container_output_root=container_path(root / phase, plan['workspace']),
                container_name_prefix=plan['container_name_prefix'] + '-' + phase,
                owner_token=plan['owner_token'],
                runner_script='train_course_directions.py', recipe=stage_recipe(smoke),
                jobs=stage_jobs(smoke))


def training_command(stage, job):
    name, command = docker_command(stage, job, 0)
    command[2:2] = ['--label', 'ai-course-directions-owner=' + stage['owner_token']]
    command[command.index('--data-root') + 1] = '/workspace/' + DATA
    # Existing utility adds no test path. This command intentionally preserves
    # image_size=1024; the high-resolution switch changes only its declared config.
    return name, command


def cpu_command(root, plan):
    workspace = plan['workspace']
    source = container_path(root / 'source', workspace)
    name = plan['container_name_prefix'] + '-cpu'
    command = ['docker', 'run', '--rm', '--pull', 'never', '--name', name,
               '--label', 'ai-course-directions-owner=' + plan['owner_token'],
               '--shm-size=8g', '--user', f'{os.getuid()}:{os.getgid()}',
               '--volume', workspace + ':/workspace', '--volume', workspace + ':' + workspace,
               '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', '--workdir', '/tmp']
    environment = dict(CUDA_VISIBLE_DEVICES='', AI_CLASS_TEST_DEVICE='cpu',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', TMPDIR='/tmp', XDG_CACHE_HOME='/workspace/.cache',
        MPLCONFIGDIR='/workspace/.cache/matplotlib',
        PATH='/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        PYTHONPATH=source + ':' + source + '/tests:/workspace/mmdetection')
    for key, value in environment.items():
        command += ['--env', key + '=' + value]
    command += [plan['image_id'], 'python', '-m', 'unittest', '-v', *plan['cpu_tests']]
    return name, command


def diagnostics_command(root, plan, job):
    stage = stage_plan(root, plan, 'formal')
    name, command = training_command(stage, job)
    name = plan['container_name_prefix'] + '-diagnostics-' + job['id']
    command[command.index('--name') + 1] = name
    command = command[:command.index(plan['image_id']) + 1]
    command += ['python', container_path(root / 'source/course_directions_diagnostics.py', plan['workspace']),
                '--run', container_path(root / 'formal/runs' / job['id'], plan['workspace']),
                '--output', container_path(root / 'diagnostics' / job['id'], plan['workspace'])]
    return name, command


def validate_diagnostics(root, plan, job, trained):
    output = root / 'diagnostics' / job['id']
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'passed' or report.get('completed_images') != 8 or
            report.get('fixed_case_ids') != list(FIXED_IDS) or
            report.get('checkpoint_sha256') != trained['checkpoint_sha256'] or
            report.get('training_report_sha256') != trained['report_sha256']):
        raise ValueError('Diagnostics did not use the accepted final model and predeclared eight images')
    for flag in ('training_started', 'test_inputs_used', 'test_annotations_used', 'gt_sent_to_model', 'ap_reevaluated'):
        if report.get(flag) is not False:
            raise ValueError('Unexpected training/test/AP behavior in diagnostics: ' + flag)
    if (report.get('input_hashes_sha256') != FIXED_INPUTS_SHA or
            sha256(output / 'input_hashes.json') != FIXED_INPUTS_SHA or
            [r['image_id'] for r in json.loads((output / 'input_hashes.json').read_text())] != list(FIXED_IDS)):
        raise ValueError('Diagnostics input files differ from archived fixed eight')
    expected_scale = 1536 if job['flags'].get('course_high_resolution') else 1024
    if report.get('input_scale') != expected_scale:
        raise ValueError('Diagnostic inference scale differs from declared arm')
    if report.get('original_grid_verified') is not True:
        raise ValueError('Diagnostic masks were not verified in original coordinates')
    if report.get('active_parameters') != trained.get('parameters') or report.get('state_tensor_bytes', 0) <= 0:
        raise ValueError('Diagnostic parameter/size record is inconsistent')
    timing = report.get('timing', {})
    seconds = timing.get('all_seconds', [])
    if (timing.get('warmup') != 5 or timing.get('repeats') != 20 or
            timing.get('image_id') != FIXED_IDS[0] or len(seconds) != 20 or
            any(not math.isfinite(x) or x <= 0 for x in seconds) or
            any(not isinstance(timing.get(key), (int, float)) or
                not math.isfinite(timing[key]) or timing[key] <= 0
                for key in ('mean_ms', 'median_ms', 'peak_allocated_mib'))):
        raise ValueError('Incomplete or nonfinite 5-warmup/20-repeat speed measurement')
    if (not math.isclose(timing['mean_ms'], statistics.mean(seconds) * 1000, rel_tol=1e-7) or
            not math.isclose(timing['median_ms'], statistics.median(seconds) * 1000, rel_tol=1e-7)):
        raise ValueError('Timing summary disagrees with its twenty samples')
    if not (output / 'cases.json').is_file() or any(
            not (output / f'case_{identity}_predictions.json').is_file() for identity in FIXED_IDS):
        raise ValueError('Fixed-eight diagnostics artifacts incomplete')
    if [r['image_id'] for r in json.loads((output / 'cases.json').read_text())] != list(FIXED_IDS):
        raise ValueError('Fixed-eight case records incomplete or reordered')
    assert_intact(root, plan)
    return dict(status='passed', report=str(report_path), report_sha256=sha256(report_path),
                timing=timing, active_parameters=report['active_parameters'],
                state_tensor_bytes=report['state_tensor_bytes'], fixed_case_ids=list(FIXED_IDS))


def validate_report(root, plan, phase, job):
    """Validate artifacts without loading unsafe/untrusted checkpoints on host."""
    smoke = phase == 'smoke'
    recipe = stage_recipe(smoke)
    run = root / phase / 'runs' / job['id']
    report_path = run / 'report.json'
    report = json.loads(report_path.read_text())
    iters = recipe['iters']
    if (report.get('status') != 'passed' or report.get('final_iteration') != iters or
            report.get('final_optimizer_steps') != [iters] or
            report.get('test_inputs_used') is not False or
            report.get('test_annotations_used') is not False):
        raise ValueError('Training report did not pass final-step/test-isolation acceptance')
    args = report['args']
    expected_args = dict(recipe, seed=job['seed'])
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise ValueError('Declared training budget/configuration changed')
    expected_flags = dict.fromkeys((*OLD_FLAGS, *NEW_FLAGS, 'multiscale_depth',
                                   'reliability_gate', 'functional_smoke', 'trained_checkpoint'), False)
    expected_flags.update(job['flags'])
    if any(args.get(flag, False) is not expected for flag, expected in expected_flags.items()):
        raise ValueError('Unexpected optimization switch enabled or required switch absent')
    if args.get('data_root') != '/workspace/' + DATA or args.get('checkpoint') != '/workspace/' + CHECKPOINT:
        raise ValueError('Unexpected data root or non-fresh checkpoint initialization')
    size = 1536 if job['flags'].get('course_high_resolution') else 1024
    if report.get('effective_train_size') != size or report.get('effective_eval_size') != size:
        raise ValueError('Effective train/evaluation resolution differs from declared arm')
    prefix = 'smoke' if smoke else 'instances'
    expected = plan['expected_provenance']
    for split in ('train', 'val'):
        if report.get(split + '_annotations_sha256') != expected[f'{prefix}_{split}_annotations_sha256']:
            raise ValueError('Training/validation annotations changed')
    if (report.get('training_images') != expected[f'{prefix}_train_images'] or
            report.get('validation_images') != expected[f'{prefix}_val_images'] or
            report.get('source_checkpoint_sha256') != INITIAL_SHA):
        raise ValueError('Data coverage or original initialization changed')
    implementation = report.get('implementation_sha256', {})
    if not implementation or 'train_course_directions.py' not in implementation:
        raise ValueError('Training entry provenance missing')
    for rel, digest in implementation.items():
        if plan['source_sha256'].get(rel) != digest:
            raise ValueError('Implementation SHA mismatch: ' + rel)
    for rel in plan['source_sha256']:
        if rel.startswith(('rgbd/', 'optimizations/')) and implementation.get(rel) != plan['source_sha256'][rel]:
            raise ValueError('Model or augmentation provenance incomplete: ' + rel)
    if report.get('finite_checks') != dict(model=True, optimizer=True, loss=True, grad=True):
        raise ValueError('Missing finite model/optimizer/loss/gradient acceptance')
    if report.get('finite_train_iterations') != iters:
        raise ValueError('Finite loss/gradient monitoring did not cover every training iteration')
    expected_updates = dict(classifier_changed=True, depth_adapter_changed=False,
        spatial_projection_changed=bool(job['flags'].get('multiscale_depth')),
        spatial_encoder_changed=bool(job['flags'].get('multiscale_depth')),
        spatial_gate_changed=bool(job['flags'].get('reliability_gate')),
        detail_adapter_changed=bool(job['flags'].get('course_detail')))
    if any(report.get(key) is not value for key, value in expected_updates.items()):
        raise ValueError('Expected trainable branch did not update')
    allowed_missing = expected.get('smoke_missing_area_metrics', []) if smoke else []
    metrics = metric_values(report.get('metrics'), allowed_missing)
    curve = report.get('learning_curve', [])
    endpoints = list(range(recipe['val_interval'], iters + 1, recipe['val_interval']))
    if [row.get('iteration') for row in curve] != endpoints:
        raise ValueError('Incomplete or duplicated validation curve')
    if json.loads((run / 'learning_curve.json').read_text()) != curve:
        raise ValueError('Report and standalone validation curves disagree')
    for row in curve:
        metric_values(row['metrics'], allowed_missing)
    if metric_values(curve[-1]['metrics'], allowed_missing) != metrics:
        raise ValueError('Report substituted another validation endpoint')
    coverage = report.get('validation_coverage', [])
    val_images = expected[f'{prefix}_val_images']
    if (len(coverage) != len(endpoints) or
            any(row.get('iteration') != iteration or row.get('completed_images') != val_images or
                row.get('unique_images') != val_images for row, iteration in zip(coverage, endpoints))):
        raise ValueError('Each declared validation must cover the whole official split once')
    checkpoint = run / f'iter_{iters}.pth'
    checkpoints = sorted(p.name for p in run.glob('iter_*.pth'))
    if checkpoints != sorted(f'iter_{step}.pth' for step in endpoints):
        raise ValueError('Declared checkpoint retention incomplete')
    expected_checkpoint = container_path(checkpoint, plan['workspace'])
    if report.get('checkpoint') not in (str(checkpoint), expected_checkpoint):
        raise ValueError('Report checkpoint is not the predeclared final endpoint')
    if report.get('checkpoint_sha256') != sha256(checkpoint):
        raise ValueError('Final checkpoint SHA256 mismatch')
    assert_intact(root, plan)
    return dict(status='passed', report=str(report_path), report_sha256=sha256(report_path),
                final_iteration=iters, metrics=metrics, checkpoint_sha256=report['checkpoint_sha256'],
                parameters=report.get('active_parameters'), functional_smoke=smoke)


class StopRequested(RuntimeError):
    pass


class Supervisor:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.started = time.monotonic()
        self.stop_requested = False
        self.active = None
        self.state = dict(status='running', phase='preflight', pid=os.getpid(),
                          started_at_utc=now(), steps=[], jobs={j['id']: 'pending' for j in jobs()})

    def save(self, **updates):
        self.state.update(updates, updated_at_utc=now())
        atomic_json(self.root / 'status.json', self.state)

    def remaining(self):
        # The absolute deadline includes detached-launch latency and GPU waits.
        return min(self.plan['max_seconds'] - (time.monotonic() - self.started),
                   self.plan['deadline_unix'] - time.time())

    def check_stop(self):
        if self.stop_requested or (self.root / 'STOP').exists():
            raise StopRequested('Stop requested; no further jobs will be dispatched')
        if self.remaining() <= 0:
            raise TimeoutError('96-hour total workflow deadline reached, including GPU waits')

    def wait_idle(self):
        start = time.monotonic()
        while True:
            self.check_stop()
            usage = gpu_snapshot()
            selected = usage.get(0)
            if selected is None:
                raise RuntimeError('GPU0 unavailable')
            if selected['memory_used_mib'] <= 512 and selected['utilization'] <= 10:
                self.save(gpu_before=usage)
                return
            self.save(waiting_for_gpu=True, gpu_last=usage)
            if time.monotonic() - start >= self.plan['max_gpu_wait_seconds']:
                raise TimeoutError('GPU0 remained occupied; no other process was stopped')
            time.sleep(min(20, max(0., self.remaining())))

    def stop_owned(self):
        """Only ever stop the exact container this supervisor just launched."""
        if self.active is None:
            return
        child, name = self.active
        try:
            ownership = subprocess.run(['docker', 'inspect', '--format',
                '{{ index .Config.Labels "ai-course-directions-owner" }}', name],
                check=False, timeout=15, capture_output=True, text=True)
            if ownership.returncode == 0 and ownership.stdout.strip() == self.plan['owner_token']:
                subprocess.run(['docker', 'stop', '--time', '20', name],
                               check=False, timeout=40, capture_output=True)
            elif ownership.returncode == 0:
                self.state.setdefault('cleanup_errors', []).append(
                    'Container ownership label mismatch; refused to stop ' + name)
        except (OSError, subprocess.SubprocessError) as exc:
            self.state.setdefault('cleanup_errors', []).append(repr(exc))
        try:
            child.wait(timeout=25)
        except subprocess.TimeoutExpired:
            # Killing the client alone never authorizes killing any GPU process.
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        self.active = None

    def run_command(self, phase, name, command, timeout):
        self.check_stop()
        assert_intact(self.root, self.plan)
        log = self.root / 'logs' / (phase.replace('/', '__') + '.log')
        self.save(phase=phase, current_command=command, current_log=str(log),
                  container=name, waiting_for_gpu=False)
        begin = time.monotonic()
        last_update = begin
        with log.open('xb', buffering=0) as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            self.active = (child, name)
            self.save(docker_client_pid=child.pid)
            while True:
                self.check_stop()
                code = child.poll()
                if code is not None:
                    break
                if time.monotonic() - begin >= timeout:
                    raise TimeoutError('Per-job deadline exceeded: ' + phase)
                if time.monotonic() - last_update >= 60:
                    self.save(current_elapsed_seconds=time.monotonic() - begin,
                              current_progress=log_progress(log))
                    last_update = time.monotonic()
                time.sleep(min(5, max(0., self.remaining())))
        row = dict(phase=phase, exit_code=code, log=str(log), command=command,
                   container=name, finished_at_utc=now(), seconds=time.monotonic() - begin)
        self.state['steps'].append(row)
        self.save()
        if code != 0:
            raise RuntimeError(f'{phase} exited {code}; first original error retained in {log}')
        self.active = None
        assert_intact(self.root, self.plan)
        return row

    def execute(self):
        self.save()
        try:
            inputs, reference = verify_inputs(Path(self.plan['workspace']))
            if inputs != self.plan['expected_provenance'] or reference != self.plan['reference_baseline']:
                raise ValueError('Data/initialization/control changed after preparation')
            name, command = cpu_command(self.root, self.plan)
            self.run_command('cpu', name, command, CPU_SECONDS)
            for phase in ('smoke', 'formal'):
                stage = stage_plan(self.root, self.plan, phase)
                output = self.root / phase
                output.mkdir()
                (output / 'runs').mkdir()
                (output / 'source').symlink_to(self.root / 'source', target_is_directory=True)
                atomic_json(output / 'plan.json', stage)
                for job in stage['jobs']:
                    self.save(phase=phase + '/' + job['id'])
                    self.check_stop()
                    assert_intact(self.root, self.plan)
                    self.wait_idle()
                    name, command = training_command(stage, job)
                    self.run_command(phase + '/' + job['id'], name, command,
                                     SMOKE_SECONDS if phase == 'smoke' else job['max_job_seconds'])
                    accepted = validate_report(self.root, self.plan, phase, job)
                    trained_step = self.state['steps'][-1]
                    trained_step['acceptance'] = accepted
                    if phase == 'formal':
                        self.state['jobs'][job['id']] = 'trained_accepted'
                        self.save(phase='diagnostics/' + job['id'])
                        (self.root / 'diagnostics').mkdir(exist_ok=True)
                        self.wait_idle()
                        name, command = diagnostics_command(self.root, self.plan, job)
                        self.run_command('diagnostics/' + job['id'], name, command, 1800)
                        diagnosis = validate_diagnostics(self.root, self.plan, job, accepted)
                        self.state['steps'][-1]['acceptance'] = diagnosis
                        trained_step['acceptance']['diagnostics'] = diagnosis
                        self.state['jobs'][job['id']] = 'passed'
                    self.save()
                atomic_json(output / 'summary.json', dict(status='complete',
                    functional_smoke=phase == 'smoke', results=[s['acceptance'] for s in self.state['steps']
                    if s['phase'].startswith(phase + '/')]))
            reference_metrics = self.plan['reference_baseline']['metrics']
            results = {}
            for step in self.state['steps']:
                if not step['phase'].startswith('formal/'):
                    continue
                accepted = step['acceptance']
                identity = step['phase'].split('/')[1]
                paired_reference = (self.plan['reference_baseline']['high_resolution']
                                    if identity == 'high_resolution_long'
                                    else self.plan['reference_baseline'])
                results[step['phase'].split('/')[1]] = dict(accepted,
                    primary_reference_report=paired_reference['report'],
                    delta_vs_primary_reference_points={key: round(100 * (value - paired_reference['metrics'][key]), 6)
                        for key, value in accepted['metrics'].items()},
                    delta_vs_gated_control_points={key: round(100 * (value - reference_metrics[key]), 6)
                        for key, value in accepted['metrics'].items()})
            atomic_json(self.root / 'summary.json', dict(status='complete', results=results,
                reference_baseline=self.plan['reference_baseline'], test_inputs_used=False,
                test_annotations_used=False, note='Final30k endpoints; one seed, no statistical reliability claim. '
                'High-resolution changes training/inference compute; not an equal-compute comparison.'))
            self.save(status='complete', phase='complete', finished_at_utc=now(),
                      docker_client_pid=None, container=None)
        except BaseException as exc:
            self.stop_owned()
            error = dict(type=type(exc).__name__, message=str(exc), phase=self.state['phase'],
                         original_log=self.state.get('current_log'), recorded_at_utc=now())
            error_path = self.root / 'first_error.json'
            if not error_path.exists():
                atomic_json(error_path, error)
            self.save(status='stopped' if isinstance(exc, StopRequested) else 'failed',
                      first_error=error, finished_at_utc=now())
            raise


def worker(root):
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if root != Path(plan['output']).resolve() or not root.is_relative_to(Path(plan['workspace']) / 'outputs'):
        raise ValueError('Worker output does not match frozen plan')
    with (root / '.supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another supervisor owns this queue; no duplicate dispatch') from exc
        if (root / 'status.json').exists():
            raise RuntimeError('Queue already started; no worker replay or resume into old results')
        # Serialize this workflow family even if two fresh output paths are
        # launched together. Other users' processes are still checked read-only.
        with (Path(plan['workspace']) / 'outputs/.course_directions_gpu0.lock').open('a') as gpu_lock:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                error = dict(type='RuntimeError', message='Another directions queue owns personal GPU0 scheduling',
                             phase='preflight', recorded_at_utc=now())
                atomic_json(root / 'first_error.json', error)
                atomic_json(root / 'status.json', dict(status='failed', first_error=error, finished_at_utc=now()))
                raise RuntimeError(error['message']) from exc
            supervisor = Supervisor(root, plan)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            for event in (signal.SIGTERM, signal.SIGINT):
                signal.signal(event, lambda *_: setattr(supervisor, 'stop_requested', True))
            supervisor.execute()


def start(args):
    workspace, root = Path(args.workspace).resolve(), Path(args.output).resolve()
    if root.exists() or root == workspace / 'outputs' or not root.is_relative_to(workspace / 'outputs'):
        raise ValueError('Require a NEW directory strictly below personal workspace/outputs')
    cpu_tests = list(dict.fromkeys(['test_course_directions_workflow', *args.cpu_test]))
    if any(not re.fullmatch(r'test_[A-Za-z0-9_]+', name) for name in cpu_tests):
        raise ValueError('CPU tests must be explicit local unittest module names')
    if len(cpu_tests) < 2:
        raise ValueError('Select algorithm/config regression modules with --cpu-test before launch')
    origin = Path(__file__).resolve().parent
    files = source_paths(origin, cpu_tests)
    if not (workspace / 'envs/mmdet/bin/python').is_file():
        raise FileNotFoundError('Existing MMDetection environment missing')
    if shutil.disk_usage(workspace).free < 60 * 2**30:
        raise RuntimeError('Need at least 60 GiB free for 36 retained checkpoints and reserve')
    provenance, reference = verify_inputs(workspace)
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE], text=True).strip()
    if not image_id.startswith('sha256:'):
        raise ValueError('Cannot pin existing Docker image ID')
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    (root / 'source').mkdir()
    fingerprints = {}
    for path in files:
        rel = path.relative_to(origin)
        target = root / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        fingerprints[str(rel)] = sha256(target)
    created = time.time()
    plan = dict(schema_version=1, protocol='course-six-directions-sequential-v1',
        workspace=str(workspace), output=str(root), created_at_utc=now(),
        deadline_unix=created + TOTAL_SECONDS, max_seconds=TOTAL_SECONDS,
        max_gpu_wait_seconds=GPU_WAIT_SECONDS, gpu=0, image_id=image_id,
        owner_token=uuid.uuid4().hex,
        source_sha256=fingerprints, expected_provenance=provenance,
        reference_baseline=reference, cpu_tests=cpu_tests, jobs=jobs(), recipe=BASE_RECIPE,
        container_name_prefix='ai-course-directions-' + hashlib.sha256(str(root).encode()).hexdigest()[:10],
        failure_policy='Stop first failure; preserve original log; no retries, retuning, or matrix expansion.',
        test_inputs_used=False, test_annotations_used=False,
        note='CPU regressions then ALL six 20-step smokes, then six sequential fresh-COCO30k jobs, '
             'each followed by fixed-eight/paired-speed diagnostics in a fresh container (1800s cap). '
             'GPU0 only; waiting and all tasks share a96h wall deadline; existing30k control is reused.')
    atomic_json(root / 'plan.json', plan)
    command = [sys.executable, '-u', str(root / 'source/run_course_directions.py'), 'worker', '--output', str(root)]
    with (root / 'workflow.log').open('xb', buffering=0) as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    launch = dict(pid=child.pid, output=str(root), started_at_utc=now(), command=command,
                  plan_sha256=sha256(root / 'plan.json'), deadline_unix=plan['deadline_unix'])
    atomic_json(root / 'launch.json', launch)
    print(json.dumps(launch, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    first = sub.add_parser('start')
    first.add_argument('--workspace', default='/mnt/nfs/fenghaibin/AI_Class')
    first.add_argument('--output', required=True)
    first.add_argument('--cpu-test', action='append', default=[], help='Repeat for exact algorithm/config unittest modules')
    second = sub.add_parser('worker')
    second.add_argument('--output', required=True)
    return p


if __name__ == '__main__':
    options = parser().parse_args()
    start(options) if options.mode == 'start' else worker(options.output)

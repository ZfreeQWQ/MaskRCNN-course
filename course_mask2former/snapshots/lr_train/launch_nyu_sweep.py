"""Detached, bounded two-GPU NYUv2 factorial ablation queue (host-side stdlib).

Run on gpu8, not inside Docker. `start` snapshots source, writes a fixed plan,
starts a session-independent supervisor and exits. No SSH connection is needed
after startup. Models always start from the same official COCO checkpoint.
Create OUTPUT/STOP to drain current jobs and stop scheduling new ones.
Failures are preserved, never retried into nonempty experiment directories.
"""
from __future__ import annotations

import argparse
from collections import deque
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
import subprocess
import sys
import threading
import time


FLAGS = ('depth_fusion', 'small_object_crop', 'small_object_loss', 'boundary_loss')
SHORT = ('depth', 'crop', 'small', 'boundary')
IMAGE = 'nvidia_cuda-export:11.8.0-cudnn8-devel-ubuntu22.04'
CHECKPOINT = 'checkpoints/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'
DATA = 'data/nyuv2/development_indoor8_v1'
SOURCE_FILES = ('launch_nyu_sweep.py', 'summarize_nyu_sweep.py', 'run_nyu_experiment.py',
                'run_coco_experiment.py', 'export_nyu_development.py', 'prepare_nyuv2.py',
                'nyuv2_things_v1.json')


class SupervisorAlreadyRunning(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def experiment_jobs(seeds):
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    # Baseline, singles, pairs, triples, all-on. Pair each seed's baseline and
    # single depth early; every requested cell still has an identical budget.
    masks = sorted(range(16), key=lambda m: (bin(m).count('1'), m))
    jobs = []
    for seed in seeds:
        for mask in masks:
            label = '_'.join(name for i, name in enumerate(SHORT) if mask & (1 << i)) or 'rgb'
            identity = f's{seed}_m{mask:02d}_{label}'
            jobs.append(dict(id=identity, seed=seed, mask=mask,
                             flags={name: bool(mask & (1 << i)) for i, name in enumerate(FLAGS)},
                             output_rel='runs/' + identity))
    return jobs


def container_path(path, workspace):
    return str(Path('/workspace') / Path(path).resolve().relative_to(Path(workspace).resolve()))


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    return {int(row[0]): dict(memory_used_mib=int(row[1]), utilization=int(row[2]))
            for line in result.stdout.strip().splitlines()
            for row in [[item.strip() for item in line.split(',')]]}


def docker_command(plan, job, gpu):
    workspace = plan['workspace']
    output = str(Path(plan['container_output_root']) / job['output_rel'])
    source = str(Path(plan['container_output_root']) / 'source')
    name = f"{plan['container_name_prefix']}-g{gpu}-{job['id']}"
    command = ['docker', 'run', '--rm', '--pull', 'never', '--name', name,
               '--gpus', f'device={gpu}', '--shm-size=32g',
               '--user', f'{os.getuid()}:{os.getgid()}']
    environment = {
        'HOME': '/workspace/.home', 'XDG_CACHE_HOME': '/workspace/.cache',
        'TMPDIR': '/workspace/.tmp', 'MPLCONFIGDIR': '/workspace/.cache/matplotlib',
        'TORCH_HOME': '/workspace/.cache/torch', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'PATH': '/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        'PYTHONPATH': source + ':/workspace/mmdetection',
    }
    for key, value in environment.items():
        command += ['--env', f'{key}={value}']
    command += ['--volume', f'{workspace}:/workspace', '--volume', f'{workspace}:{workspace}',
                '--tmpfs', '/tmp:rw,nosuid,nodev,size=1g', '--workdir', '/workspace',
                plan['image'], 'python', str(Path(source) / plan.get('runner_script', 'run_nyu_experiment.py')), 'train',
                '--workspace', '/workspace', '--data-root', '/workspace/' + DATA,
                '--checkpoint', '/workspace/' + CHECKPOINT, '--output', output,
                '--seed', str(job['seed'])]
    recipe = dict(plan['recipe'])
    recipe.update(job.get('recipe_overrides', {}))
    for key, value in recipe.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    command += ['--' + key.replace('_', '-') for key, enabled in job['flags'].items() if enabled]
    return name, command


def log_progress(path):
    """Read a bounded tail: progress is advisory, report.json is authoritative."""
    if not Path(path).is_file():
        return {}
    with Path(path).open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 16384))
        tail = stream.read().decode(errors='replace')
    rows = re.findall(r'Iter\(train\)\s*\[\s*(\d+)/(\d+)\].*?\btime:\s*([0-9.]+)', tail)
    if not rows:
        return {}
    step, total, seconds = rows[-1]
    return dict(iteration=int(step), total_iterations=int(total), seconds_per_iteration=float(seconds))


def prepare(args, jobs=None, extra_source_files=(), extra_train_files=(), plan_extensions=None):
    workspace, output = Path(args.workspace).resolve(), Path(args.output).resolve()
    if output == workspace or not output.is_relative_to(workspace / 'outputs'):
        raise ValueError('Output must be a NEW subdirectory under workspace/outputs')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite or restart an existing sweep: {output}')
    if (args.iters < 1 or args.batch_size < 1 or args.workers < 0 or args.image_size < 32
            or args.image_size % 32 or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('Invalid recipe')
    if not args.gpus or len(args.gpus) != len(set(args.gpus)) or len(args.gpus) > 2:
        raise ValueError('Choose one or two distinct GPUs')
    jobs = experiment_jobs(args.seeds) if jobs is None else jobs
    usage = gpu_snapshot()
    for gpu in args.gpus:
        if gpu not in usage or usage[gpu]['memory_used_mib'] > 512 or usage[gpu]['utilization'] > 10:
            raise RuntimeError(f'GPU {gpu} is not idle; refusing to collide with another job')
    if shutil.disk_usage(workspace).free < (10 + (1.2 if plan_extensions else 0.6) * len(jobs)) * 2**30:
        raise RuntimeError('Insufficient NFS free space for all retained checkpoints plus reserve')
    if not (workspace / 'envs/mmdet/bin/python').is_file():
        raise FileNotFoundError('Prepared mmdet environment is missing')
    image_id = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE],
                              check=True, capture_output=True, text=True).stdout.strip()
    manifest, completion = workspace / DATA / 'manifest.json', workspace / DATA / 'export_complete.json'
    if json.loads(completion.read_text())['manifest_sha256'] != sha256(manifest):
        raise ValueError('Dataset export has an inconsistent completion marker')
    origin = Path(__file__).resolve().parent
    files = [origin / name for name in (*SOURCE_FILES, *extra_source_files)]
    files += sorted((origin / 'optimizations').glob('*.py')) + sorted((origin / 'rgbd').glob('*.py'))
    if any(not p.is_file() for p in files):
        raise FileNotFoundError('Missing supervisor, summarizer or training source')
    source_hash = sha256(workspace / CHECKPOINT)
    if source_hash != '41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879':
        raise ValueError('Official starting checkpoint SHA256 differs from the pinned source')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    (output / 'runs').mkdir()
    snapshot = output / 'source'
    hashes = {}
    for original in files:
        relative = original.relative_to(origin)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[str(relative)] = sha256(target)
    train_files = ['run_nyu_experiment.py', 'run_coco_experiment.py', 'export_nyu_development.py']
    train_files += list(extra_train_files)
    train_files += sorted(k for k in hashes if k.startswith(('optimizations/', 'rgbd/')))
    plan = dict(schema_version=1, created_at_utc=utc_now(), workspace=str(workspace),
                container_output_root=container_path(output, workspace), image=IMAGE, image_id=image_id,
                container_name_prefix='ai-nyu-' + hashlib.sha256(str(output).encode()).hexdigest()[:10],
                gpus=args.gpus, seeds=args.seeds,
                recipe={key: getattr(args, key) for key in
                        ('iters', 'batch_size', 'image_size', 'workers', 'lr')},
                jobs=jobs, snapshot_sha256=hashes,
                expected_provenance=dict(dataset_manifest_sha256=sha256(manifest),
                    source_checkpoint_sha256=source_hash,
                    implementation_sha256={k: hashes[k] for k in train_files}),
                max_job_seconds=3600, failure_policy='Record failure, continue; stop dispatch after 3 consecutive failures',
                note='16 factorial switch combinations per seed; all from identical COCO initialization. '
                     'No official test use. No automatic tuning based on interim validation scores. '
                     'This is not a guarantee of AP/APs improvement. No completion notification is scheduled.')
    if plan_extensions:
        plan.update(plan_extensions)
    atomic_json(output / 'plan.json', plan)
    with (output / 'supervisor.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-u', str(snapshot / 'launch_nyu_sweep.py'),
                                  'worker', '--output', str(output)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 close_fds=True)
    atomic_json(output / 'launch.json', dict(pid=child.pid, output=str(output), launched_at_utc=utc_now()))
    print(json.dumps(dict(pid=child.pid, jobs=len(jobs), output=str(output),
                          status=str(output / 'status.json'), log=str(output / 'supervisor.log')), indent=2))


def run_worker(output):
    output = Path(output).resolve()
    with (output / '.supervisor.lock').open('a') as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorAlreadyRunning('Another supervisor owns this batch; status is unchanged') from exc
        if (output / 'status.json').exists():
            raise SupervisorAlreadyRunning('Batch has already started; use a new output directory, not worker replay')
        return _run_worker_locked(output)


def _run_worker_locked(output):
    import summarize_nyu_sweep

    output = Path(output).resolve()
    plan = json.loads((output / 'plan.json').read_text())
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    stopping = threading.Event()
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda *_: stopping.set())
    state = dict(status='running', pid=os.getpid(), started_at_utc=utc_now(), updated_at_utc=utc_now(),
                 total=len(plan['jobs']), jobs={j['id']: dict(status='pending') for j in plan['jobs']})
    queue = deque(plan['jobs'])
    active, consecutive_failures = {}, 0
    atomic_json(output / 'status.json', state)

    def summary():
        # Separate stdlib child isolates malformed report errors from scheduler.
        result = subprocess.run([sys.executable, str(output / 'source' / plan.get('summary_script', 'summarize_nyu_sweep.py')),
                                 '--root', str(output)], capture_output=True, text=True)
        if result.returncode:
            state['summary_error'] = result.stderr[-4000:] or result.stdout[-4000:]
        else:
            state.pop('summary_error', None)

    def fail_dispatch(message):
        stopping.set()
        state['stop_reason'] = message
        print(message, flush=True)

    def sources_intact():
        try:
            return all(sha256(output / 'source' / rel) == digest for rel, digest in plan['snapshot_sha256'].items())
        except OSError:
            return False

    summary()
    if state.get('summary_error'):
        fail_dispatch('Initial summary validation failed: ' + state['summary_error'])
    while queue or active:
        if (output / 'STOP').exists():
            fail_dispatch('STOP requested: letting active jobs finish; no new jobs will be started')
        if not stopping.is_set() and shutil.disk_usage(output).free < 5 * 2**30:
            fail_dispatch('Less than 5 GiB available; draining queue before disk exhaustion')
        for gpu in plan['gpus']:
            if stopping.is_set() or not queue or gpu in active:
                continue
            if not sources_intact():
                fail_dispatch('Snapshot changed during sweep; refusing mixed-code experiments')
                break
            image_id = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', plan['image']],
                                      capture_output=True, text=True)
            if image_id.returncode or image_id.stdout.strip() != plan['image_id']:
                fail_dispatch('Cached Docker image changed or became unavailable')
                break
            job = queue.popleft()
            name, command = docker_command(plan, job, gpu)
            logfile = output / 'logs' / (job['id'] + '.log')
            with logfile.open('xb', buffering=0) as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu] = dict(job=job, process=process, started=time.monotonic(), name=name, logfile=logfile)
            state['jobs'][job['id']] = dict(status='running', gpu=gpu, pid=process.pid,
                                           container_name=name, started_at_utc=utc_now(), log=str(logfile))
            # Persist ownership before launching anything else: even an
            # exception starting the second slot can stop this exact container.
            atomic_json(output / 'status.json', state)
            print(f'{utc_now()} START GPU {gpu} {job["id"]}', flush=True)
        completed = False
        for gpu, item in list(active.items()):
            process, job = item['process'], item['job']
            record = state['jobs'][job['id']]
            record.update(log_progress(item['logfile']))
            elapsed = time.monotonic() - item['started']
            record['elapsed_seconds'] = round(elapsed, 1)
            if process.poll() is None and elapsed > plan['max_job_seconds']:
                subprocess.run(['docker', 'stop', '--time', '30', item['name']], timeout=45,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                record['error'] = f'Job exceeded explicit {plan["max_job_seconds"]}-second timeout'
                process.wait(timeout=30)
            code = process.poll()
            if code is None:
                continue
            report_path = output / job['output_rel'] / 'report.json'
            try:
                report = json.loads(report_path.read_text())
                passed = code == 0 and report['status'] == 'passed'
            except (OSError, ValueError, KeyError):
                passed = False
            record.update(status='passed' if passed else 'failed', exit_code=code, finished_at_utc=utc_now())
            consecutive_failures = 0 if passed else consecutive_failures + 1
            if not passed:
                record.setdefault('error', 'Nonzero process exit or missing/not-passed report; inspect job log')
            print(f'{utc_now()} {record["status"].upper()} GPU {gpu} {job["id"]} ({elapsed:.1f}s)', flush=True)
            del active[gpu]
            completed = True
            if consecutive_failures >= 3:
                fail_dispatch('Three consecutive job failures; preserving remaining pending jobs')
        if completed:
            summary()
            if state.get('summary_error'):
                fail_dispatch('Summary validation failed: ' + state['summary_error'])
            summary_path = output / 'summary.json'
            if summary_path.exists():
                # Semantic/provenance validation errors must not be ignored.
                result = json.loads(summary_path.read_text())
                if result.get('invalid_count', 0):
                    fail_dispatch('Sweep summary found invalid experiment evidence; inspect summary.json')
        state['updated_at_utc'] = utc_now()
        state['counts'] = {key: sum(item['status'] == key for item in state['jobs'].values())
                           for key in ('pending', 'running', 'passed', 'failed')}
        atomic_json(output / 'status.json', state)
        if stopping.is_set() and not active:
            break
        if queue or active:
            time.sleep(5)
    summary()
    state.update(status=('stopped' if queue else 'completed_with_failures'
                         if state['counts']['failed'] or state.get('summary_error') else 'complete'),
                 finished_at_utc=utc_now(), updated_at_utc=utc_now())
    atomic_json(output / 'status.json', state)
    print(f'{utc_now()} SWEEP {state["status"]}', flush=True)


def stop_owned_after_error(output, state):
    """Best-effort bounded cleanup of only this supervisor's recorded containers."""
    plan = json.loads((Path(output) / 'plan.json').read_text())
    expected_names = {docker_command(plan, job, gpu)[0]
                      for job in plan['jobs'] for gpu in plan['gpus']}
    for record in state.get('jobs', {}).values():
        name = record.get('container_name')
        if record.get('status') != 'running' or name not in expected_names:
            continue
        try:
            result = subprocess.run(['docker', 'stop', '--time', '30', name],
                                     capture_output=True, text=True, timeout=45)
            record.update(status='interrupted', cleanup_exit_code=result.returncode)
            if result.returncode:
                record['cleanup_error'] = result.stderr[-1000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            record['cleanup_error'] = str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    start = sub.add_parser('start')
    start.add_argument('--workspace', default='/mnt/nfs/fenghaibin/AI_Class')
    start.add_argument('--output', required=True)
    start.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    start.add_argument('--seeds', nargs='+', type=int, default=[20260923, 20260924, 20260925])
    start.add_argument('--iters', type=int, default=1000)
    start.add_argument('--batch-size', type=int, default=2)
    start.add_argument('--image-size', type=int, default=512)
    start.add_argument('--workers', type=int, default=2)
    start.add_argument('--lr', type=float, default=1e-5)
    worker = sub.add_parser('worker')
    worker.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'start':
        prepare(args)
    else:
        try:
            run_worker(args.output)
        except SupervisorAlreadyRunning as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2)
        except Exception as exc:
            path = Path(args.output) / 'status.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            try:
                stop_owned_after_error(args.output, state)
            except Exception as cleanup_error:
                state['cleanup_error'] = str(cleanup_error)
            state.update(status='supervisor_failed', error=f'{type(exc).__name__}: {exc}', updated_at_utc=utc_now())
            atomic_json(path, state)
            raise


if __name__ == '__main__':
    main()

"""One-shot, detached v3 workflow: 10 smoke runs, then all 30 confirmation runs.

This file only starts work when explicitly invoked with `start`. Busy GPUs are
an immediate error, not an instruction to wait for or interrupt existing work.
No recurring automation, AP-driven tuning, or automatic retries are created.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from launch_nyu_sweep import SOURCE_FILES, atomic_json, gpu_snapshot, sha256, utc_now
from launch_nyu_v3 import EXTRA_SOURCES, launch, stage_plan
from summarize_nyu_v3 import summarize


def workflow_plan(seeds):
    stages = [stage_plan(stage, seeds) for stage in ('smoke', 'confirm')]
    return dict(schema_version=1, stages=stages, total_jobs=sum(len(s['jobs']) for s in stages),
        policy='All smoke reports must pass strict validation before confirmation. '
               'No automatic selection by score. STOP drains active stage and prevents the next stage.')


def validate_stage(root, stage, seeds, previous_provenance=None):
    root = Path(root)
    status = json.loads((root / 'status.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    expected = stage_plan(stage, seeds)
    for key in ('schema_version', 'stage', 'seeds', 'jobs', 'recipe', 'runner_script', 'summary_script'):
        if plan[key] != expected[key]:
            raise ValueError('Stage plan mismatch: ' + key)
    result = summarize(root)
    if (status['status'] != 'complete' or result['status'] != 'complete'
            or result['counts']['passed'] != len(expected['jobs']) or result['invalid_count']):
        raise RuntimeError('Stage not fully valid; confirmation will not start')
    if previous_provenance is not None and plan['expected_provenance'] != previous_provenance:
        raise RuntimeError('Implementation/data/weights changed between stages')
    return plan['expected_provenance']


def start(args):
    workspace, output = Path(args.workspace).resolve(), Path(args.output).resolve()
    plan = workflow_plan(args.seeds)
    if output == workspace or not output.is_relative_to(workspace / 'outputs'):
        raise ValueError('Use a new subdirectory of workspace/outputs')
    if output.exists():
        raise FileExistsError('Never overwrite/replay an existing workflow')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or len(args.gpus) > 2:
        raise ValueError('Choose one or two distinct GPUs')
    usage = gpu_snapshot()
    for gpu in args.gpus:
        if gpu not in usage or usage[gpu]['memory_used_mib'] > 512 or usage[gpu]['utilization'] > 10:
            raise RuntimeError(f'GPU {gpu} is busy; existing work is left untouched. Run again after it finishes.')
    needed = 10 + len(plan['stages'][0]['jobs']) * .6 + len(plan['stages'][1]['jobs']) * 1.2
    if shutil.disk_usage(workspace).free < needed * 2**30:
        raise RuntimeError(f'Need at least {needed:.1f} GiB free for smoke and retained confirmation weights')
    origin = Path(__file__).resolve().parent
    files = [origin / name for name in sorted(set((*SOURCE_FILES, *EXTRA_SOURCES)))]
    files += sorted((origin / 'rgbd').glob('*.py')) + sorted((origin / 'optimizations').glob('*.py'))
    if not all(path.is_file() for path in files):
        raise FileNotFoundError('Missing workflow implementation')
    output.mkdir(parents=True)
    hashes = {}
    for path in files:
        rel = path.relative_to(origin)
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = sha256(target)
    plan.update(created_at_utc=utc_now(), workspace=str(workspace), output=str(output),
                seeds=args.seeds, gpus=args.gpus, snapshot_sha256=hashes)
    atomic_json(output / 'workflow_plan.json', plan)
    with (output / 'workflow.log').open('xb', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-u', str(output / 'source/run_nyu_v3_all.py'),
            'worker', '--output', str(output)], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    atomic_json(output / 'workflow_launch.json', dict(pid=process.pid, launched_at_utc=utc_now()))
    print(json.dumps(dict(pid=process.pid, jobs=plan['total_jobs'], output=str(output)), indent=2))


def worker(root):
    root = Path(root).resolve()
    with (root / '.workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'workflow_status.json').exists():
            raise RuntimeError('Refusing workflow replay')
        plan = json.loads((root / 'workflow_plan.json').read_text())
        state = dict(status='running', pid=os.getpid(), started_at_utc=utc_now())
        stopping = [False]
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
        previous = None
        active = None
        try:
            for stage in ('smoke', 'confirm'):
                if stopping[0] or (root / 'STOP').exists():
                    state.update(status='stopped', reason='Stopped before next stage')
                    break
                if not all(sha256(root / 'source' / rel) == value for rel, value in plan['snapshot_sha256'].items()):
                    raise RuntimeError('Workflow source snapshot changed')
                active = root / stage
                state.update(stage=stage, stage_root=str(active), updated_at_utc=utc_now())
                atomic_json(root / 'workflow_status.json', state)
                launch(plan['workspace'], str(active), plan['gpus'], stage, plan['seeds'])
                started = time.monotonic()
                while True:
                    if stopping[0] or (root / 'STOP').exists():
                        (active / 'STOP').touch(exist_ok=True)
                    path = active / 'status.json'
                    child = json.loads(path.read_text()) if path.exists() else {}
                    state.update(stage_status=child.get('status', 'starting'), counts=child.get('counts', {}),
                                 updated_at_utc=utc_now())
                    atomic_json(root / 'workflow_status.json', state)
                    if child.get('status') in ('complete', 'completed_with_failures', 'stopped', 'supervisor_failed'):
                        break
                    # Single GPU at max per-job timeout, plus dispatch margin.
                    limit = (len(stage_plan(stage, plan['seeds'])['jobs']) * 10800 / len(plan['gpus'])) + 3600
                    if time.monotonic() - started > limit:
                        raise TimeoutError('Stage exceeded bounded workflow timeout')
                    if not path.exists() and time.monotonic() - started > 120:
                        raise RuntimeError('Stage supervisor did not create status within two minutes')
                    if path.exists() and time.time() - path.stat().st_mtime > 600:
                        raise RuntimeError('Stage supervisor has not updated for 10 minutes')
                    time.sleep(10)
                if stopping[0] or (root / 'STOP').exists():
                    state.update(status='stopped', reason='Active stage drained; no further stage dispatched')
                    break
                previous = validate_stage(active, stage, plan['seeds'], previous)
                state[stage + '_validated'] = True
                active = None
            else:
                state.update(status='complete', summary=str(root / 'confirm/summary.md'))
        except Exception as exc:
            # Do not kill another job or erase evidence. Ask the owned stage
            # supervisor to drain its current jobs and stop further dispatch.
            if active is not None and active.exists():
                (active / 'STOP').touch(exist_ok=True)
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            state.update(finished_at_utc=utc_now(), updated_at_utc=utc_now())
            atomic_json(root / 'workflow_status.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'start', 'worker'])
    p.add_argument('--workspace', default='/mnt/nfs/fenghaibin/AI_Class')
    p.add_argument('--output')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--seeds', type=int, nargs='+', default=[20260923, 20260924, 20260925])
    args = p.parse_args()
    if args.action == 'plan':
        print(json.dumps(workflow_plan(args.seeds), ensure_ascii=False, indent=2))
    elif not args.output:
        p.error('--output is required for start/worker')
    elif args.action == 'start':
        start(args)
    else:
        worker(args.output)


if __name__ == '__main__':
    main()

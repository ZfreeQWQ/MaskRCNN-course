"""New authorized four-hour window; reuse frozen prediction code without changes."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path('/mnt/nfs/fenghaibin/AI_Class')
PREVIOUS = ROOT / 'outputs/course_m2f_tiles_20261018_v1'
OUTPUT = ROOT / 'outputs/course_m2f_tiles_20261019_v1'
START = 1792390445
DEADLINE = START + 14400
NAME = 'run_m2f_tiles_renewal.py'
if (PREVIOUS / 'source').is_dir():
    sys.path.append(str(PREVIOUS / 'source'))
import run_m2f_tiles as old
import m2f_tiles as task


def activate():
    # This file is frozen with the new plan; no old output/source is modified.
    task.PREPARATION_UNIX = old.PREPARATION_UNIX = START
    task.DEADLINE = old.DEADLINE = DEADLINE


def execute_task(mode, root):
    activate()
    if mode == 'binding':
        task.preflight('/workspace')
        task.atomic_json(root / 'binding.json', dict(status='passed', checkpoint_sha256=task.SPEC['weight_sha'],
            reference_report_sha256=task.SPEC['reference_sha'], training_started=False))
    elif mode == 'evaluate':
        task.evaluate(str(root))
    elif mode == 'analyze':
        task.analyze(str(root))
        result = task.read(root / 'analysis/report.json')
        delta = result['delta_points']['course/segm_mAP_s']
        result.update(aps_observed_positive=delta > 0,
            result_classification='observed_positive_candidate' if delta > 0 else 'no_observed_APs_gain' if delta == 0 else 'observed_APs_negative',
            performance_target_is_not_validity_gate=True,
            retention_policy='Keep all outcomes; small APs gains retained, single-seed not statistical confirmation.')
        for name in ('report.json', 'comparison.json'):
            task.atomic_json(root / 'analysis' / name, result)
    else:
        raise ValueError('Unknown phase')


def worker(root):
    activate()
    if root != OUTPUT or (root / 'status.json').exists():
        raise ValueError('No replay or output drift')
    original_docker = old.docker
    def command(r, plan, phase, args, gpu=False):
        if args[:2] == ['python', 'm2f_tiles.py']:
            args = ['python', NAME, 'task', '--phase', args[2], '--output', args[4]]
        return original_docker(r, plan, phase, args, gpu)
    old.docker = command
    plan = task.read(root / 'plan.json')
    if plan['deadline_unix'] != DEADLINE or plan['preparation_unix'] != START:
        raise ValueError('Budget drift')
    supervisor = old.Supervisor(root, plan)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: setattr(supervisor, 'stop_requested', True))
    supervisor.execute()


def start(root, commit):
    if root != OUTPUT or root.exists() or not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('New exact output and committed source required')
    if not START <= time.time() < DEADLINE - 3 * 3600:
        raise TimeoutError('New original4h window insufficient; no extension')
    prior = task.read(PREVIOUS / 'plan.json')
    if task.read(PREVIOUS / 'status.json')['status'] != 'failed' or (PREVIOUS / 'evaluate').exists():
        raise ValueError('Prior blocked experiment must not have evaluated')
    old.queue.assert_intact(PREVIOUS, prior)
    for rel, digest in prior['upstream_sha256'].items():
        if old.sha256(ROOT / 'mmdetection' / rel) != digest:
            raise ValueError('Upstream drift')
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise ValueError('Insufficient output reserve')
    root.mkdir(); (root / 'logs').mkdir()
    shutil.copytree(PREVIOUS / 'source', root / 'source')
    shutil.copy2(Path(__file__), root / 'source' / NAME)
    plan = dict(prior)
    plan.update(protocol='fixed-cp-global4tiles-new-authorized-window', output=str(root),
        source_commit=commit, owner_token=uuid.uuid4().hex, preparation_unix=START,
        deadline_unix=DEADLINE, max_seconds=14400, created_at_utc=old.queue.now(),
        previous_blocked_output=str(PREVIOUS), previous_evidence_sha256={
            n: old.sha256(PREVIOUS / n) for n in ('plan.json', 'status.json', 'first_error.json')},
        frozen_prediction_code_unchanged=True, performance_target_is_not_validity_gate=True)
    plan['source_sha256'] = dict(prior['source_sha256'])
    plan['source_sha256'][NAME] = old.sha256(root / 'source' / NAME)
    old.atomic_json(root / 'plan.json', plan)
    cmd = [sys.executable, '-u', str(root / 'source' / NAME), 'worker', '--output', str(root)]
    with (root / 'workflow.log').open('xb', buffering=0) as log:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    launch = dict(pid=child.pid, started_at_utc=old.queue.now(), deadline_unix=DEADLINE,
        plan_sha256=old.sha256(root / 'plan.json'), command=cmd)
    old.atomic_json(root / 'launch.json', launch)
    print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['start', 'worker', 'task'])
    p.add_argument('--output', required=True); p.add_argument('--source-commit')
    p.add_argument('--phase', choices=['binding', 'evaluate', 'analyze'])
    a = p.parse_args(); root = Path(a.output).resolve()
    if a.mode == 'start': start(root, a.source_commit)
    elif a.mode == 'worker': worker(root)
    else: execute_task(a.phase, root)

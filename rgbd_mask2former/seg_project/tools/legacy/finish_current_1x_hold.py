"""Finish an already running 1x child while its old scheduler is suspended.

The suspended parent retains the child exit status. Only that scheduler PID is
terminated after the child's successful exit; no training process is signalled.
New training is forbidden by the persistent user hold. The final invocation
reuses verified completed runs and only finishes benchmarks and the 1x report.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_supplemental import completed_run

PROJECT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT / 'outputs'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def proc(pid):
    # /proc/stat: fields after the comm closing parenthesis start at field 3.
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return {'state': fields[0], 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'exit_code': int(fields[49])}


def main(args):
    hold_path = OUTPUTS / 'user_training_hold.json'
    if not hold_path.exists():
        raise RuntimeError('User hold must exist; refusing to supervise without it')
    status_path = OUTPUTS / 'finish_current_1x_status.json'

    def status(value, **extra):
        write_json(status_path, {'pid': os.getpid(), 'updated_utc': utc(), 'status': value,
                                'scheduler_pid': args.scheduler, 'child_pid': args.child, **extra})
        print(utc(), value, extra, flush=True)

    parent = proc(args.scheduler)
    child = proc(args.child)
    if parent['state'] not in ('T', 't') or child['ppid'] != args.scheduler:
        raise RuntimeError('Expected suspended scheduler and its existing child')
    if parent['start_ticks'] != args.scheduler_start or child['start_ticks'] != args.child_start:
        raise RuntimeError('PID identity changed')
    for name in ('b0', 'b1', 'a1'):
        completed_run(OUTPUTS / ('1x_' + name), '1x', 1)
    try:
        while True:
            parent = proc(args.scheduler)
            child = proc(args.child)
            if parent['start_ticks'] != args.scheduler_start or child['start_ticks'] != args.child_start:
                raise RuntimeError('PID identity changed while waiting')
            if parent['state'] not in ('T', 't') or child['ppid'] != args.scheduler:
                raise RuntimeError('Scheduler was resumed or child ownership changed')
            if child['state'] == 'Z':
                if child['exit_code'] != 0:
                    raise RuntimeError(f'Current training exited unsuccessfully: {child}')
                break
            status('waiting_for_current_m1_final_evaluation', child_state=child['state'])
            time.sleep(30)

        directory = OUTPUTS / '1x_m1'
        marker = {'stage': '1x', 'config': str(PROJECT / 'configs/m1_rgbd_small.yaml'),
                  'num_gpus': 1, 'completed_utc': utc(),
                  'completion_evidence': {'pid': args.child, 'start_ticks': args.child_start,
                                          'exit_code': child['exit_code']}}
        completed_run(directory, '1x', 1, marker_info=marker)
        write_json(directory / 'training_complete.json', marker)
        status('current_four_runs_verified_complete')
        # The old parent must never resume into its in-memory final-DDP launch.
        os.kill(args.scheduler, signal.SIGKILL)
        for _ in range(100):
            try:
                p = proc(args.scheduler)
            except FileNotFoundError:
                break
            if p['state'] == 'Z':
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Old scheduler did not terminate')
        status('finishing_benchmarks_and_report_only')
        result = subprocess.run([sys.executable, '-u', str(PROJECT / 'tools/run_experiments.py'),
                                 '--stages', '1x', '--final-stage', 'none'], cwd=PROJECT.parent)
        if result.returncode:
            raise RuntimeError(f'Report finalization failed: {result.returncode}; no new training permitted')
        pipeline = json.loads((OUTPUTS / 'pipeline_status.json').read_text())
        if pipeline['jobs']['pipeline']['status'] != 'paused_by_user_after_1x':
            raise RuntimeError('Final user pause was not recorded')
        hold = json.loads(hold_path.read_text())
        hold.update(current_1x_completed=True, finalization_completed_utc=utc(),
                    scheduler_6897='terminated after current child exited successfully',
                    supplemental_8517='cancelled before any supplemental GPU job launched')
        write_json(hold_path, hold)
        status('paused_by_user_after_1x', report=str(PROJECT / 'reports/report_1x.md'),
               final_ddp_launched=False, supplemental_training_launched=False)
    except BaseException as error:
        status('needs_review_training_remains_disabled', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheduler', type=int, required=True)
    parser.add_argument('--child', type=int, required=True)
    parser.add_argument('--scheduler-start', type=int, required=True)
    parser.add_argument('--child-start', type=int, required=True)
    main(parser.parse_args())

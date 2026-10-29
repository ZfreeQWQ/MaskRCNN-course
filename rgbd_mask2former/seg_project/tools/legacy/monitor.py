"""Observe actual pipeline processes and artifacts; never restart a live job."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]


def read_tail(path, byte_limit=65536):
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - byte_limit))
        return stream.read().decode('utf-8', errors='replace').splitlines()


def process(pid):
    if not isinstance(pid, int):
        return {'pid': pid, 'live': False, 'reason': 'no_pid'}
    try:
        proc = Path('/proc') / str(pid)
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'live': fields[0] not in ('Z', 'X'), 'state': fields[0],
                'start_ticks': int(fields[19]), 'cpu_ticks': int(fields[11]) + int(fields[12]),
                'command': command}
    except (FileNotFoundError, ProcessLookupError):
        return {'pid': pid, 'live': False, 'reason': 'process_missing'}


def observe():
    state_path = PROJECT / 'outputs/pipeline_status.json'
    state = json.loads(state_path.read_text())
    result = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'pipeline': process(state.get('pid')), 'declared': state['jobs'].get('pipeline'), 'jobs': {}}
    for name, job in state['jobs'].items():
        if name == 'pipeline':
            continue
        item = {'status': job.get('status')}
        if job.get('status') == 'running':
            item['process'] = process(job.get('pid'))
        directory = PROJECT / 'outputs' / name
        rows = []
        for line in read_tail(directory / 'metrics.json'):
            try:
                row = json.loads(line)
                if 'total_loss' in row:
                    rows.append(row)
            except json.JSONDecodeError:
                pass
        if rows:
            item['training'] = {key: rows[-1].get(key) for key in ('iteration', 'total_loss', 'time', 'lr')}
        best = directory / 'best_metrics.json'
        if best.is_file():
            data = json.loads(best.read_text())
            item['best'] = {'completed_iterations': data['completed_iterations'], 'segm': data['metrics']['segm']}
        log = Path(job.get('log', str(directory / 'console.log')))
        lines = read_tail(log)
        if lines:
            item['last_log_line'] = lines[-1]
            item['log_bytes'] = log.stat().st_size
            # Validation can retain the preceding training iteration for several
            # minutes. Show the current operation rather than treating it as stalled.
            last = lines[-1]
            if 'd2.utils.events' in last and 'iter:' in last:
                item['phase'] = 'training'
            elif 'Inference done ' in last:
                item['phase'] = 'validation_inference'
            elif any(token in last for token in ('Evaluate annotation type', 'Running per image evaluation',
                                                  'Accumulating evaluation results', 'DONE (t=',
                                                  'Average Precision', 'Average Recall')):
                item['phase'] = 'validation_cpu_evaluation'
            elif 'Saving checkpoint' in last:
                item['phase'] = 'saving_checkpoint'
            else:
                item['phase'] = 'see_last_log_line'
            for line in reversed(lines):
                match = re.search(r'Inference done (\d+)/(\d+).*ETA=([^ ]+)', line)
                if match:
                    item['validation_inference'] = {'images': int(match[1]), 'total': int(match[2]),
                                                    'eta': match[3],
                                                    'current_phase': item['phase'] == 'validation_inference'}
                    break
            item['recent_fatal_lines'] = [line for line in lines if any(
                token in line for token in ('Traceback (most recent call last)', 'RuntimeError:', 'Error during training'))][-5:]
        result['jobs'][name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='Persist this observation and append monitoring history')
    args = parser.parse_args()
    observation = observe()
    if args.record:
        folder = PROJECT / 'reports/monitoring'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'latest.json'
        if target.is_file():
            previous = json.loads(target.read_text())
            observation['previous_observed_utc'] = previous.get('observed_utc')
            for name, item in observation['jobs'].items():
                before = previous.get('jobs', {}).get(name, {})
                proc, old_proc = item.get('process', {}), before.get('process', {})
                if (proc.get('live') and proc.get('pid') == old_proc.get('pid')
                        and proc.get('start_ticks') == old_proc.get('start_ticks')):
                    item['cpu_ticks_since_previous_observation'] = proc['cpu_ticks'] - old_proc['cpu_ticks']
                if 'log_bytes' in item and 'log_bytes' in before:
                    item['log_bytes_since_previous_observation'] = item['log_bytes'] - before['log_bytes']
        temporary = folder / f'.latest.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(observation, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(target)
        with (folder / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
    print(json.dumps(observation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

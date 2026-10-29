"""Wait for the main pipeline, then run matched 1x ablations and one matched-budget DDP control.

No model is constructed here. Every GPU task holds the same pipeline.lock as the
main scheduler. --dry-run prints commands and artifact gates without taking locks,
writing status, waiting, or launching a subprocess.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import yaml

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
OUTPUTS = PROJECT / 'outputs'
ANNOTATIONS = PROJECT / 'datasets/blender/annotations'
ABLATIONS = {
    'no_small_anchors': ('ablate_no_small_anchors.yaml', {'MODEL.ANCHOR_GENERATOR.SIZES'}),
    'no_extra_proposals': ('ablate_no_extra_proposals.yaml', {
        'MODEL.RPN.PRE_NMS_TOPK_TRAIN', 'MODEL.RPN.PRE_NMS_TOPK_TEST',
        'MODEL.RPN.POST_NMS_TOPK_TRAIN', 'MODEL.RPN.POST_NMS_TOPK_TEST'}),
    'no_high_maskpool': ('ablate_no_high_maskpool.yaml', {'MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION'}),
}
ARCHITECTURES = {'B1': 'b1_rgbd.yaml', 'M1': 'm1_rgbd_small.yaml'}
TERMINAL_PRIMARY = {'training_complete_test_blocked', 'completed_computation'}
STOP = threading.Event()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def flatten(mapping, prefix=''):
    result = {}
    for key, value in mapping.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def finite_score(best):
    result = best['metrics']['segm']
    score = tuple(result[key] for key in ('APs', 'AP'))
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in score):
        raise ValueError('Best checkpoint needs finite nonnegative full-val segm/APs and AP')
    return score


def completed_run(directory, stage, num_gpus, marker_info=None):
    """Check marker AND actual final checkpoint, config, logged iterations and selected artifacts."""
    directory = Path(directory)
    marker = marker_info if marker_info is not None else read_json(directory / 'training_complete.json')
    if marker['stage'] != stage or marker['num_gpus'] != num_gpus:
        raise ValueError(f'{directory}: wrong completed stage/GPU count')
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    batch = config['SOLVER']['IMS_PER_BATCH']
    budget = 14432 * {'1x': 10, '2x': 20}[stage]
    expected = round(budget / batch)
    if batch != num_gpus or config['SOLVER']['MAX_ITER'] != expected:
        raise ValueError(f'{directory}: expected one image/card and {budget} training-image budget')
    project = config['PROJECT']
    if project.get('EVAL_LIMIT', 0) or project.get('SKIP_FINAL_EVAL', False):
        raise ValueError(f'{directory}: smoke configuration cannot qualify')
    if config['DATASETS']['TEST'] != ['blender_val']:
        raise ValueError(f'{directory}: requires fixed full blender_val')
    best = read_json(directory / 'best_metrics.json')
    finite_score(best)
    if best.get('selection_metric') != 'segm/APs' or not 0 <= best['iteration'] < expected:
        raise ValueError(f'{directory}: invalid best iteration/selection')
    checkpoint = best['checkpoint']
    if Path(checkpoint).name != checkpoint:
        raise ValueError('Checkpoint must be a local filename')
    for name in ('model_final.pth', checkpoint, 'best_val_predictions.json'):
        if (directory / name).stat().st_size <= 2:
            raise ValueError(f'{directory}: missing/nonexistent trained artifact {name}')
    # Event JSONL, not the marker alone, must reach the final optimizer update.
    last_iteration = -1
    with (directory / 'metrics.json').open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        length = stream.tell()
        stream.seek(max(0, length - 131072))
        lines = stream.read().decode().splitlines()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # first line may be a partial JSON record after seek
        last_iteration = max(last_iteration, row.get('iteration', -1))
    if last_iteration < expected - 1:
        raise ValueError(f'{directory}: event log has not reached {expected - 1}')
    return {'directory': str(directory), 'config': config, 'best': best,
            'completed_iterations': expected, 'training_image_budget': expected * batch}


def prerequisites(outputs=OUTPUTS):
    """Refuse a failed/crashed/partial primary run even if stale artifacts happen to exist."""
    outputs = Path(outputs)
    status = read_json(outputs / 'pipeline_status.json')
    terminal = status.get('jobs', {}).get('pipeline', {}).get('status')
    if terminal not in TERMINAL_PRIMARY:
        raise ValueError(f'Primary pipeline not successfully terminal: {terminal!r}; no supplemental GPU work')
    runs = {name: completed_run(outputs / f'1x_{name}', '1x', 1)
            for name in ('b0', 'b1', 'm1', 'a1')}
    selection = read_json(outputs / 'final_selection.json')
    if selection.get('test_labels_used') is not False:
        raise ValueError('Selection must explicitly certify test_labels_used=false')
    selected = Path(selection['selected_directory']).resolve()
    allowed_root = outputs.resolve()
    if selected.parent != allowed_root:
        raise ValueError('Final selection points outside outputs directory')
    if not math.isclose(selection['inference_score_threshold'], .05):
        raise ValueError('Final inference threshold differs from the fixed fair reference .05')
    ddp = []
    for candidate in selection['candidates']:
        directory = Path(candidate).resolve()
        if directory.parent != allowed_root:
            raise ValueError('Selection candidate points outside outputs directory')
        if directory.name.startswith('final_ddp_'):
            marker = read_json(directory / 'training_complete.json')
            if marker['stage'] not in ('1x', '2x'):
                raise ValueError('Formal DDP stage must be 1x or 2x')
            ddp.append(completed_run(directory, marker['stage'], 4))
    if not ddp:
        raise ValueError('No completed four-card formal DDP candidate in final_selection.json')
    selected_config = yaml.safe_load((selected / 'config.yaml').read_text())
    architecture = selected_config['PROJECT']['EXPERIMENT'].upper()
    if architecture not in ARCHITECTURES or not selected_config['PROJECT']['USE_DEPTH']:
        raise ValueError('Supplemental DDP control requires selected B1/M1 RGB-D architecture')
    if str(selected) not in selection['candidates']:
        raise ValueError('Selected model is absent from candidate list')
    # A selected single-card model must be one of the already verified formal 1x runs.
    verified = {Path(item['directory']).resolve() for item in [*runs.values(), *ddp]}
    if selected not in verified:
        raise ValueError('Selected directory is not a verified formal candidate')
    return {'primary_terminal_status': terminal, 'runs': runs, 'ddp_runs': ddp,
            'architecture': architecture, 'selected_directory': str(selected)}


def train_command(config, directory, num_gpus=1):
    command = [sys.executable, '-u', str(PROJECT / 'tools/train.py'), '--num-gpus', str(num_gpus),
               '--config-file', str(config), '--stage', '1x']
    if (Path(directory) / 'last_checkpoint').exists():
        command.append('--resume')
    return command + ['OUTPUT_DIR', str(directory), 'DATALOADER.NUM_WORKERS', '2']


def jobs(outputs=OUTPUTS, architecture=None):
    result = []
    for gpu, (name, (filename, _)) in enumerate(ABLATIONS.items()):
        directory = Path(outputs) / f'1x_ablate_{name}'
        result.append({'name': f'ablate_{name}', 'phase': 'parallel_single_card', 'gpu': str(gpu),
                       'directory': str(directory), 'config': str(PROJECT / 'configs' / filename),
                       'num_gpus': 1, 'training_image_budget': 144320, 'optimizer_steps': 144320,
                       'global_batch': 1, 'base_lr': .0025,
                       'command': train_command(PROJECT / 'configs' / filename, directory),
                       'benchmark_command': benchmark_command(directory)})
    for name in ([architecture] if architecture else ['B1', 'M1']):
        directory = Path(outputs) / f'supplemental_ddp_1x_{name.lower()}'
        result.append({'name': f'ddp_1x_{name.lower()}', 'phase': 'after_all_three_ablations',
                       'conditional_on_selected_architecture': name, 'gpu': '0,1,2,3',
                       'directory': str(directory), 'config': str(PROJECT / 'configs' / ARCHITECTURES[name]),
                       'num_gpus': 4, 'training_image_budget': 144320, 'optimizer_steps': 36080,
                       'global_batch': 4, 'base_lr': .01,
                       'command': train_command(PROJECT / 'configs' / ARCHITECTURES[name], directory, 4),
                       'benchmark_command': benchmark_command(directory)})
    return result


def benchmark_command(directory):
    directory = Path(directory)
    best = read_json(directory / 'best_metrics.json') if (directory / 'best_metrics.json').exists() else {'checkpoint': 'best_segm_APs.pth'}
    return [sys.executable, '-u', str(PROJECT / 'tools/benchmark.py'),
            '--config', str(directory / 'config.yaml'), '--weights', str(directory / best['checkpoint']),
            '--metadata', str(ANNOTATIONS / 'instances_val.json'),
            '--output', str(directory / 'report_artifacts/benchmark.json')]


def report_command(directories):
    command = [sys.executable, '-u', str(PROJECT / 'tools/build_report.py'), '--title',
               'RGB-D：同预算逐项消融与单卡/DDP 1x对照', '--output', str(PROJECT / 'reports/report_supplemental.md')]
    for name, directory in directories.items():
        command += ['--experiment', f'{name}={Path(directory) / "report_artifacts"}']
    return command


def config_contract(config_file, directory, reference, allowed):
    # Configuration-only import: no build_model / tensor creation / GPU queries.
    for path in (str(PROJECT), str(ROOT / 'detectron2')):
        if path not in sys.path:
            sys.path.insert(0, path)
    from rgbd_seg.config import setup_cfg
    desired = setup_cfg(config_file, ['OUTPUT_DIR', str(directory), 'DATALOADER.NUM_WORKERS', '2'])
    desired = yaml.safe_load(desired.dump())
    left, right = flatten(reference), flatten(desired)
    permitted = set(allowed) | {'OUTPUT_DIR', 'PROJECT.EXPERIMENT'}
    changes = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    unexpected = changes - permitted
    if unexpected:
        raise ValueError(f'Configuration drift versus formal reference: {sorted(unexpected)}')
    if set(allowed) - changes:
        raise ValueError('Requested remove-one configuration does not change the intended module')
    return desired, sorted(changes)


class Progress:
    def __init__(self, outputs):
        self.outputs = Path(outputs)
        self.path = self.outputs / 'supplemental_status.json'
        self.log_path = self.outputs / 'supplemental.log'
        self.lock = threading.Lock()
        self.state = read_json(self.path) if self.path.exists() else {}
        self.state.update(pid=os.getpid(), started_utc=utc(),
                          note='Three leave-one-out 1x runs, then selected architecture four-card 1x; test inputs untouched')

    def update(self, name, **fields):
        with self.lock:
            self.state.setdefault('jobs', {}).setdefault(name, {}).update(fields)
            self.state['updated_utc'] = utc()
            atomic_json(self.path, self.state)
            message = f"[{self.state['updated_utc']}] {name}: " + json.dumps(fields, ensure_ascii=False)
            with self.log_path.open('a') as stream:
                stream.write(message + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            print(message, flush=True)

    def run(self, name, command, logfile, gpu):
        if STOP.is_set():
            raise InterruptedError('Supplemental stop requested; no additional process launched')
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        with logfile.open('a') as stream:
            stream.write(f'\n[{utc()}] COMMAND ' + json.dumps(command) + '\n')
            stream.flush()
            # One process group includes all Detectron2 DDP workers, so a stop
            # cannot release pipeline.lock while orphan GPU workers keep running.
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self.update(name, status='running', command=command, gpu=gpu, pid=process.pid, log=str(logfile))
                while True:
                    try:
                        returncode = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if STOP.is_set():
                            raise InterruptedError('Supplemental stop requested')
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    except ProcessLookupError:
                        process.wait()
        if returncode:
            self.update(name, status='failed', returncode=returncode)
            self.update('supplemental', status='job_failed_waiting_workers', failed_job=name)
            raise RuntimeError(f'{name} failed ({returncode}); inspect {logfile}')
        self.update(name, status='completed', returncode=0)


def benchmark_valid(path, weights):
    try:
        data = read_json(path)
        return (data.get('smoke') is False and data['warmup'] >= 20 and data['iterations'] >= 200
                and data['input_shape'][-2:] == [1024, 1024]
                and data.get('tta') is False and data.get('tiling') is False
                and Path(data['weights']).resolve() == Path(weights).resolve()
                and len(data['model_ms_samples']) >= data['iterations']
                and len(data['read_preprocess_ms_samples']) >= data['iterations']
                and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                        for value in data['model_ms_samples']))
    except (OSError, KeyError, ValueError, TypeError):
        return False


def artifacts(progress, directory, gpu):
    directory = Path(directory)
    target = directory / 'report_artifacts'
    best = read_json(directory / 'best_metrics.json')
    atomic_json(target / 'metrics.json', {
        'metrics': best['metrics'], 'units': 'percent (0-100)', 'primary_metric': 'segm/APs',
        'num_images': 2584, 'smoke': False, 'selection': best,
        'source': str(directory / 'best_metrics.json'),
        'predictions': str(directory / 'best_val_predictions.json'),
        'evaluation_scope': 'complete fixed blender_val; official COCOEvaluator; maxDets=100'})
    weights = directory / best['checkpoint']
    if not benchmark_valid(target / 'benchmark.json', weights):
        progress.run(directory.name + '_benchmark', benchmark_command(directory), target / 'benchmark.log', gpu)
    if not benchmark_valid(target / 'benchmark.json', weights):
        raise ValueError(f'{directory}: standard benchmark verification failed')
    progress.update(directory.name + '_artifacts', status='completed', directory=str(target))
    return target


def train_job(progress, job, reference, allowed):
    directory = Path(job['directory'])
    desired, changes = config_contract(job['config'], directory, reference, allowed)
    contract = {'stage': '1x', 'config': job['config'], 'num_gpus': job['num_gpus'],
                'changes_vs_single_card_reference': changes,
                'reference_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
                'desired_config_sha256': hashlib.sha256(json.dumps(desired, sort_keys=True).encode()).hexdigest()}
    marker = directory / 'training_complete.json'
    if marker.exists():
        completed_run(directory, '1x', job['num_gpus'])
        saved = read_json(directory / 'supplemental_contract.json')
        if saved != contract:
            raise ValueError(f'{directory}: completed run contract changed; use a new directory')
        progress.update(job['name'], status='completed_previously_verified')
    else:
        contract_path = directory / 'supplemental_contract.json'
        if contract_path.exists() and read_json(contract_path) != contract:
            raise ValueError(f'{directory}: resume contract differs from original invocation')
        atomic_json(contract_path, contract)
        progress.run(job['name'], job['command'], directory / 'console.log', job['gpu'])
        completion = {'stage': '1x', 'config': job['config'], 'num_gpus': job['num_gpus'], 'completed_utc': utc()}
        completed_run(directory, '1x', job['num_gpus'], marker_info=completion)
        atomic_json(marker, completion)
    artifacts(progress, directory, job['gpu'].split(',')[0])
    return directory


def loss_and_throughput(directory):
    rows = []
    with (Path(directory) / 'metrics.json').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('iteration', 0) >= 20 and isinstance(row.get('time'), (int, float)) and row['time'] > 0:
                rows.append(row)
    config = yaml.safe_load((Path(directory) / 'config.yaml').read_text())
    batch = config['SOLVER']['IMS_PER_BATCH']
    losses = [row['total_loss'] for row in rows if isinstance(row.get('total_loss'), (int, float)) and math.isfinite(row['total_loss'])]
    return {'global_batch': batch, 'base_lr': config['SOLVER']['BASE_LR'],
            'max_iter': config['SOLVER']['MAX_ITER'], 'training_image_budget': batch * config['SOLVER']['MAX_ITER'],
            'median_step_s': statistics.median([row['time'] for row in rows]) if rows else None,
            'global_training_images_per_s_excluding_eval_hooks': batch / statistics.median([row['time'] for row in rows]) if rows else None,
            'first_50_logged_loss_median': statistics.median(losses[:50]) if losses else None,
            'last_50_logged_loss_median': statistics.median(losses[-50:]) if losses else None,
            'note': 'Logged iteration timing excludes validation hooks; loss curves differ with batch, LR and optimization updates; not a communication-only causal estimate'}


def write_findings(directories, architecture):
    reference = read_json(directories['M1'] / 'best_metrics.json')['metrics']['segm']
    lines = ['# 同预算逐项消融与 DDP 1x 补充结果', '',
             '所有结果来自固定完整 val，AP 单位为百分点；三个 remove-one 相对正式单卡 M1，其他训练配置不变。P2 为共同结构，不作为新增贡献。', '',
             '| 实验 | APs | AP | 相对 M1 ΔAPs | 相对 M1 ΔAP | batch | LR | optimizer steps | 训练图次 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    details = {}
    for name, directory in directories.items():
        metric = read_json(directory / 'best_metrics.json')['metrics']['segm']
        stats = loss_and_throughput(directory)
        details[name] = stats
        lines.append(f"| {name} | {metric['APs']:.4f} | {metric['AP']:.4f} | {metric['APs']-reference['APs']:+.4f} | {metric['AP']-reference['AP']:+.4f} | {stats['global_batch']} | {stats['base_lr']} | {stats['max_iter']} | {stats['training_image_budget']} |")
    single = details[architecture]
    ddp = details['DDP_1X_' + architecture]
    lines += ['', f'所选架构 {architecture} 的单卡/DDP图次预算分别为 {single["training_image_budget"]}/{ddp["training_image_budget"]}。',
              '四卡 batch=4、LR=0.01，单卡 batch=1、LR=0.0025；图次预算相同但更新次数与优化轨迹不同。不得将AP差异归因于通信，也不得称各进程四份未同步权重为DDP。', '',
              '每组纯模型推理、读取/预处理、参数量、checkpoint/纯模型容量与峰值显存见 report_supplemental.md。训练吞吐和首尾损失摘要见本文件同名JSON；原始loss曲线保留在各目录metrics.json。', '',
              '逐项消融表示其他M1模块存在时的条件贡献；具体错误原因仍需结合已复核实例，不把候选标签当作确定原因。']
    path = PROJECT / 'reports/report_supplemental_findings.md'
    path.write_text('\n'.join(lines) + '\n')
    atomic_json(path.with_suffix('.json'), details)


def main(args):
    if args.dry_run:
        try:
            gate = prerequisites()
            preflight = {'ready': True, 'architecture': gate['architecture'], 'terminal_status': gate['primary_terminal_status']}
        except (OSError, KeyError, TypeError, ValueError) as error:
            gate = None
            preflight = {'ready': False, 'reason': str(error)}
        planned = jobs(architecture=gate['architecture'] if gate else None)
        # Even before formal artifacts exist, validate intended remove-one changes
        # against current source configurations. Actual saved 1x configs are checked
        # again under the shared lock before GPU work can start.
        for path in (str(PROJECT), str(ROOT / 'detectron2')):
            if path not in sys.path:
                sys.path.insert(0, path)
        from rgbd_seg.config import setup_cfg
        references = {name: yaml.safe_load(setup_cfg(PROJECT / 'configs' / filename,
                      ['DATALOADER.NUM_WORKERS', '2']).dump()) for name, filename in ARCHITECTURES.items()}
        for index, job in enumerate(planned):
            allowed = list(ABLATIONS.values())[index][1] if index < 3 else set()
            reference_name = 'M1' if index < 3 else job['conditional_on_selected_architecture']
            _, job['validated_config_changes'] = config_contract(job['config'], job['directory'], references[reference_name], allowed)
        report_plans = {}
        for architecture in ([gate['architecture']] if gate else ['B1', 'M1']):
            directories = {job['name'].upper(): Path(job['directory']) for job in planned[:3]}
            directories['M1'] = OUTPUTS / '1x_m1'
            directories[architecture] = OUTPUTS / ('1x_' + architecture.lower())
            directories['DDP_1X_' + architecture] = OUTPUTS / ('supplemental_ddp_1x_' + architecture.lower())
            report_plans[architecture] = report_command(directories)
        print(json.dumps({'dry_run': True, 'gpu_or_model_operations': False,
                          'prerequisites': preflight, 'wait_lock': str(OUTPUTS / 'pipeline.lock'),
                          'jobs': planned, 'conditional_report_commands': report_plans,
                          'additional_internal_report': str(PROJECT / 'reports/report_supplemental_findings.md'),
                          'note': 'When selection is unresolved, B1/M1 DDP commands are alternatives; only the selected one will run.'}, indent=2))
        return
    if (OUTPUTS / 'user_training_hold.json').exists():
        raise SystemExit('Supplemental training is disabled by outputs/user_training_hold.json.')
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with (OUTPUTS / 'supplemental.lock').open('a+') as own_lock:
        try:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another supplemental scheduler owns supplemental.lock; no new work started.')
        progress = Progress(OUTPUTS)
        (OUTPUTS / 'supplemental.pid').write_text(str(os.getpid()) + '\n')
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        try:
            started = time.monotonic()
            progress.update('supplemental', status='waiting_for_pipeline_lock', pipeline_lock=str(OUTPUTS / 'pipeline.lock'))
            with (OUTPUTS / 'pipeline.lock').open('a+') as shared_lock:
                while True:
                    if STOP.is_set():
                        raise InterruptedError('Stopped while waiting; no GPU job launched')
                    try:
                        fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        progress.update('supplemental', status='waiting_for_pipeline_lock', elapsed_seconds=round(time.monotonic()-started), next_check_seconds=args.poll_seconds)
                        STOP.wait(args.poll_seconds)
                progress.update('supplemental', status='checking_primary_artifacts', waited_seconds=round(time.monotonic()-started))
                gate = prerequisites()
                architecture = gate['architecture']
                progress.update('supplemental', status='primary_artifacts_verified', selected_architecture=architecture,
                                note='pipeline.lock remains held for all supplemental training and benchmark jobs')
                planned = jobs(architecture=architecture)
                # All config checks happen before starting the first GPU task.
                reference = gate['runs']['m1']['config']
                for job, (_, allowed) in zip(planned[:3], ABLATIONS.values()):
                    config_contract(job['config'], job['directory'], reference, allowed)
                config_contract(planned[3]['config'], planned[3]['directory'], gate['runs'][architecture.lower()]['config'], set())
                progress.update('supplemental', status='running_parallel_ablations', gpu_assignments=[0, 1, 2], gpu3='idle')
                def work(item):
                    job, (_, allowed) = item
                    return job['name'].upper(), train_job(progress, job, reference, allowed)
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    results = dict(pool.map(work, zip(planned[:3], ABLATIONS.values())))
                # The pool has joined all training and benchmark subprocesses; no owned GPU job remains.
                progress.update('supplemental', status='starting_matched_budget_ddp_1x', architecture=architecture, gpu='0,1,2,3')
                comparator = train_job(progress, planned[3], gate['runs'][architecture.lower()]['config'], set())
                results['M1'] = Path(gate['runs']['m1']['directory'])
                results[architecture] = Path(gate['runs'][architecture.lower()]['directory'])
                results['DDP_1X_' + architecture] = comparator
                for name in {'M1', architecture}:
                    artifacts(progress, results[name], '0')
                progress.update('supplemental', status='building_reports')
                progress.run('supplemental_report', report_command(results), PROJECT / 'reports/supplemental_reports.log', '0')
                write_findings(results, architecture)
                progress.update('supplemental', status='completed', reports=[str(PROJECT / 'reports/report_supplemental.md'),
                                str(PROJECT / 'reports/report_supplemental_findings.md')], test_data_read=False)
        except BaseException as error:
            progress.update('supplemental', status='failed', error=repr(error), note='No pipeline_status.json writes; inspect supplemental/job logs before resuming')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='CPU-only gate and command plan; no locks, writes, waiting or subprocesses')
    parser.add_argument('--poll-seconds', type=int, default=60, help='Persist a waiting heartbeat at this interval')
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('--poll-seconds must be >=5')
    main(args)

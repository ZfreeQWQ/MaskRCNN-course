"""Durable, resumable experiment sequence. Run via seg_project/run.sh.

Four independent equal-budget experiments, then four-GPU training of the
validation-selected RGB-D architecture. No package installation or test labels.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
CONFIGS = {'b0': 'b0_rgb.yaml', 'b1': 'b1_rgbd.yaml',
           'm1': 'm1_rgbd_small.yaml', 'a1': 'a1_rgb_small.yaml'}
ANNOTATIONS = PROJECT / 'datasets/blender/annotations'
LOCK = threading.Lock()
STATE = {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def update(name, **fields):
    with LOCK:
        STATE.setdefault('jobs', {}).setdefault(name, {}).update(fields)
        STATE['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(PROJECT / 'outputs/pipeline_status.json', STATE)


def run(name, argv, output, gpu):
    if (PROJECT / 'outputs/user_training_hold.json').exists() and Path(argv[0]).name == 'train.py':
        raise RuntimeError('New training is disabled by outputs/user_training_hold.json; current runs may finish.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    command = [sys.executable, '-u', *map(str, argv)]
    with output.open('a') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        update(name, status='running', pid=proc.pid, command=command, gpu=gpu, log=str(output))
        code = proc.wait()
    if code:
        update(name, status='failed', returncode=code)
        raise RuntimeError(f'{name} exited {code}; see {output}')
    update(name, status='completed', returncode=code)


def train(name, config, stage, gpu, num_gpus=1):
    directory = PROJECT / 'outputs' / name
    marker = directory / 'training_complete.json'
    if not marker.exists():
        argv = [PROJECT / 'tools/train.py', '--num-gpus', num_gpus, '--config-file', config, '--stage', stage]
        if (directory / 'last_checkpoint').exists():
            argv.append('--resume')
        argv += ['OUTPUT_DIR', directory, 'DATALOADER.NUM_WORKERS', 2]
        if stage == 'pilot':
            argv += ['SOLVER.CHECKPOINT_PERIOD', 1000, 'TEST.EVAL_PERIOD', 1000]
        run(name, argv, directory / 'console.log', gpu)
        if not (directory / 'best_metrics.json').exists():
            raise RuntimeError(f'{name}: no valid full-val segm/APs checkpoint')
        write_json(marker, {'stage': stage, 'config': str(config), 'num_gpus': num_gpus})
    else:
        update(name, status='completed_previously')
    return directory


def artifacts(directory, gpu):
    target = directory / 'report_artifacts'
    best = json.loads((directory / 'best_metrics.json').read_text())
    predictions = directory / 'best_val_predictions.json'
    if not predictions.exists():
        raise RuntimeError(f'Missing predictions from selected checkpoint: {predictions}')
    write_json(target / 'metrics.json', {
        'metrics': best['metrics'], 'units': 'percent (0-100)', 'primary_metric': 'segm/APs',
        'num_images': 2584, 'smoke': False, 'selection': best,
        'source': str(directory / 'best_metrics.json'), 'predictions': str(predictions),
        'evaluation_scope': 'complete fixed blender_val, official COCOEvaluator, maxDets=100'})
    if not (target / 'benchmark.json').exists():
        run(directory.name + '_benchmark', [PROJECT / 'tools/benchmark.py',
            '--config', directory / 'config.yaml', '--weights', directory / best['checkpoint'],
            '--metadata', ANNOTATIONS / 'instances_val.json', '--output', target / 'benchmark.json'],
            target / 'benchmark.log', gpu)
    return target


def report(directories, title, filename):
    argv = [PROJECT / 'tools/build_report.py', '--title', title, '--output', PROJECT / 'reports' / filename]
    for name, directory in directories.items():
        argv += ['--experiment', f'{name.upper()}={directory / "report_artifacts"}']
    run('report_' + filename, argv, PROJECT / 'reports/pipeline_reports.log', '0')


def score(directory):
    result = json.loads((directory / 'best_metrics.json').read_text())['metrics']['segm']
    value = (result['APs'], result['AP'])
    if not all(math.isfinite(x) and x >= 0 for x in value):
        raise ValueError(f'Invalid selection metrics in {directory}')
    return value


def main(args):
    os.chdir(ROOT)
    outputs = PROJECT / 'outputs'
    outputs.mkdir(parents=True, exist_ok=True)
    with (outputs / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STATE.update(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     stages=args.stages, final_stage=args.final_stage,
                     note='Pilot is preliminary; formal comparisons require all 1x experiments. A2 equals B1 by construction.')
        update('pipeline', status='running')
        stage_directories = {}
        for stage in args.stages:
            def work(item):
                idx, (experiment, filename) = item
                directory = train(f'{stage}_{experiment}', PROJECT / 'configs' / filename, stage, str(idx))
                artifacts(directory, str(idx))
                return experiment, directory
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                stage_directories = dict(pool.map(work, enumerate(CONFIGS.items())))
            report(stage_directories, f'RGB-D 实例分割：{stage} 等预算实验（真实完整验证）', f'report_{stage}.md')
        if (outputs / 'user_training_hold.json').exists():
            update('pipeline', status='paused_by_user_after_1x',
                   reason='Current four 1x runs finished; final DDP and supplemental training disabled pending a new direction.')
            return
        if args.final_stage == 'none':
            update('pipeline', status='completed_requested_stages')
            return
        # The task requires depth fusion. RGB-only controls remain fully reported.
        architecture = max(('b1', 'm1'), key=lambda name: score(stage_directories[name]))
        formal = train('final_ddp_' + args.final_stage + '_' + architecture, PROJECT / 'configs' / CONFIGS[architecture],
                       args.final_stage, '0,1,2,3', num_gpus=4)
        artifacts(formal, '0')
        candidates = [stage_directories['b1'], stage_directories['m1'], formal]
        selected = max(candidates, key=score)
        selection = {'selected_directory': str(selected), 'selected_score_APs_AP': score(selected),
                     'criterion': 'highest full-val segm/APs, tie-break segm/AP; RGB-D candidates only',
                     'candidates': {str(path): score(path) for path in candidates},
                     'test_labels_used': False, 'inference_score_threshold': .05}
        write_json(outputs / 'final_selection.json', selection)
        stage_directories['final_ddp'] = formal
        report(stage_directories, 'RGB-D 元器件实例分割：正式实验与最终模型', 'report.md')
        error_dir = PROJECT / 'reports/final_error_analysis'
        if not (error_dir / 'analysis.md').exists():
            run('error_analysis', [PROJECT / 'tools/error_analysis.py', '--annotations',
                ANNOTATIONS / 'instances_val.json', '--predictions', selected / 'best_val_predictions.json',
                '--baseline-predictions', stage_directories['b0'] / 'best_val_predictions.json',
                '--output-dir', error_dir, '--render'], error_dir / 'console.log', '0')
        metadata = ANNOTATIONS / 'test_images.json'
        obj = json.loads(metadata.read_text())
        images = obj['images'] if isinstance(obj, dict) else obj
        if isinstance(obj, dict) and 'annotations' in obj:
            raise ValueError('Test metadata must not contain annotations')
        if len(images) != 2112 or any(im.get('depth_scale_m') is None or im.get('depth_invalid_code') is None for im in images):
            update('pipeline', status='training_complete_test_blocked',
                   reason='Supply official per-image test depth_scale_m/depth_invalid_code in test_images.json; no inference or test label access performed.')
            return
        best = json.loads((selected / 'best_metrics.json').read_text())
        submission = PROJECT / 'submissions'
        run('test_inference', [PROJECT / 'tools/infer.py', '--config', selected / 'config.yaml',
            '--weights', selected / best['checkpoint'], '--metadata', metadata, '--split', 'test',
            '--output', submission / 'masks.json', '--csv', submission / 'boxes.csv'], submission / 'inference.log', '0')
        run('test_validation', [PROJECT / 'tools/validate_submission.py', '--metadata', metadata,
            '--predictions', submission / 'masks.json', '--csv', submission / 'boxes.csv',
            '--manifest', submission / 'masks.json.manifest.json', '--output', submission / 'validation.json'],
            submission / 'validation.log', '0')
        update('pipeline', status='completed_computation',
               note='Review real error-analysis cases and write supported scientific conclusions before submission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', choices=('pilot', '1x', '2x'), default=['pilot', '1x'])
    parser.add_argument('--final-stage', choices=('none', '1x', '2x'), default='2x')
    options = parser.parse_args()
    try:
        main(options)
    except BlockingIOError:
        raise SystemExit('Another pipeline already holds outputs/pipeline.lock; not starting a duplicate.')
    except Exception as error:
        update('pipeline', status='failed', error=repr(error))
        raise

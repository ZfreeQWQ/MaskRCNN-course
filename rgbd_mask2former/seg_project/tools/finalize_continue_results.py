"""Wait for the existing training job, compare full-val predictions, export the winner.

Never starts/stops training or reads test labels. Run from /private/rgzndzy via run.sh.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from common import read_json, write_json
from run_round1 import ANNOTATIONS, TEST_METADATA, merge, evaluate
from run_continue_training import parse_gpu_memory

TRAIN = Path('seg_project/experiments/mask2former/d_continue6_20261026')
BASELINE = Path('seg_project/experiments/mask2former/optimization_round1_20261026')
ROOT = Path('seg_project/experiments/mask2former/d_continue6_results_20261026')
SHARDS = 8
SOURCE_NAMES = ['tools/finalize_continue_results.py', 'tools/infer_round1.py',
                'tools/run_round1.py', 'rgbd_seg/m2f/experimental_inference.py',
                'tools/export_submission.py', 'tools/validate_submission.py', 'tools/common.py']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def training_complete(root=TRAIN):
    state = read_json(root / 'manager_state.json')
    if state['status'] in ('failed', 'manager_error'):
        raise RuntimeError(f'Training requires inspection: {state}')
    exit_path = root / 'manager.exit_code'
    if exit_path.exists():
        if exit_path.read_text().strip() != '0' or state['status'] != 'completed' or state.get('exit_code') != 0:
            raise RuntimeError('Training did not complete successfully')
        if alive(state.get('training_pid')):
            raise RuntimeError('Training launcher unexpectedly still alive after completion')
        if not (root / 'model_final.pth').is_file():
            raise RuntimeError('Missing final training checkpoint')
        if (root / 'last_checkpoint').read_text().strip() != 'model_final.pth':
            raise RuntimeError('Training did not publish its final checkpoint')
        return True
    if not alive(state.get('manager_pid')):
        raise RuntimeError('Training manager died; refusing to run inference or restart training')
    return False


def choose(rows):
    for row in rows:
        if row['num_images'] != 2584:
            raise ValueError('Only full validation results may select the submission')
        for key in ('APs', 'AP'):
            value = row['metrics'][key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Expected finite COCO metrics in 0-1 units')
    # Baseline comes first, so an exact tie preserves the existing artifact.
    return max(rows, key=lambda row: (row['metrics']['APs'], row['metrics']['AP']))


def check_reproduction(result, best):
    differences = {key: result['metrics'][key] - best['metrics']['segm'][key] / 100
                   for key in ('APs', 'AP', 'AP50', 'AP75')}
    if max(abs(value) for value in differences.values()) > .001:
        raise ValueError(f'Trainer / experimental legacy validation drift: {differences}')
    return differences


def state(stage, **details):
    value = dict(stage=stage, pid=os.getpid(), updated=time.time(), **details)
    temporary = ROOT / 'workflow_state.json.partial'
    write_json(temporary, value)
    temporary.replace(ROOT / 'workflow_state.json')
    print(json.dumps(value), flush=True)


def wait_for_idle_gpus():
    state('waiting_for_idle_gpus')
    for _ in range(40):
        result = subprocess.run(['ixsmi'], capture_output=True, text=True, timeout=15, check=True)
        memory = parse_gpu_memory(result.stdout)
        if all(index in memory and memory[index] < 512 for index in range(4)):
            return
        time.sleep(30)
    raise RuntimeError('GPUs not idle after training; no existing process was interrupted')


def infer(split, audit):
    wait_for_idle_gpus()
    state('inference_' + split)
    def job(index):
        output = ROOT / split / f'shard{index}'
        if (output / 'complete.json').exists():
            summary = read_json(output / 'complete.json')
            expected = dict(split=split, smoke=False, num_shards=SHARDS, shard_index=index,
                            weights=str(ROOT / 'best_segm_APs.pth'), config=str(ROOT / 'config.yaml'),
                            code_sha256=audit['source_sha256']['rgbd_seg/m2f/experimental_inference.py'])
            if any(summary.get(key) != value for key, value in expected.items()) or summary['options']['size'] != 1536:
                raise ValueError('Existing shard has incompatible provenance')
            return
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'launch.json').exists() or (output / 'job.log').exists():
            raise RuntimeError(f'Unfinished shard must be inspected before retry: {output}')
        command = ['./seg_project/run.sh', 'python', '-u', 'seg_project/tools/infer_round1.py',
                   '--config', str(ROOT / 'config.yaml'), '--weights', str(ROOT / 'best_segm_APs.pth'),
                   '--metadata', ANNOTATIONS if split == 'val' else TEST_METADATA,
                   '--split', split, '--size', '1536', '--output-dir', str(output),
                   '--num-shards', str(SHARDS), '--shard-index', str(index)]
        with (output / 'job.log').open('x') as log:
            process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index % 4)),
                                       stdout=log, stderr=subprocess.STDOUT)
            write_json(output / 'launch.json', dict(pid=process.pid, command=command, started=time.time()))
            code = process.wait()
        (output / 'exit_code').write_text(str(code))
        if code or not (output / 'complete.json').exists():
            raise RuntimeError(f'Inference failed: {output}')
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(job, range(SHARDS)))
    check_sources(audit)


def check_sources(audit):
    if digest(ANNOTATIONS) != audit['annotations_sha256'] or digest(TEST_METADATA) != audit['test_metadata_sha256']:
        raise ValueError('Dataset metadata changed during workflow')
    if digest(ROOT / 'best_segm_APs.pth') != audit['weights_sha256']:
        raise ValueError('Frozen checkpoint changed')
    if digest(ROOT / 'config.yaml') != audit['config_sha256']:
        raise ValueError('Frozen config changed')
    for name, expected in audit['source_sha256'].items():
        if digest(Path('seg_project') / name) != expected:
            raise ValueError(f'Source changed during inference: {name}')


def finalize():
    while not training_complete():
        state('waiting_for_training', training=read_json(TRAIN / 'manager_state.json'))
        time.sleep(30)
    state('snapshot')
    # Source is stable: the owning manager and torchrun have completed successfully.
    audit_path = ROOT / 'execution_audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        check_sources(audit)
    else:
        for name in ('best_segm_APs.pth', 'config.yaml', 'best_metrics.json'):
            target = ROOT / name
            if target.exists():
                raise FileExistsError(f'Incomplete snapshot requires inspection: {target}')
            shutil.copyfile(TRAIN / name, target)
        audit = dict(training=str(TRAIN), training_state=read_json(TRAIN / 'manager_state.json'),
                     weights_sha256=digest(ROOT / 'best_segm_APs.pth'),
                     config_sha256=digest(ROOT / 'config.yaml'),
                     best_metrics=read_json(ROOT / 'best_metrics.json'),
                     annotations_sha256=digest(ANNOTATIONS),
                     test_metadata_sha256=digest(TEST_METADATA),
                     source_sha256={name: digest(Path('seg_project') / name) for name in SOURCE_NAMES},
                     baseline_results_sha256=digest(BASELINE / 'val_results.json'))
        write_json(audit_path, audit)
    baseline = read_json(BASELINE / 'val_results.json')
    if audit['annotations_sha256'] != read_json(BASELINE / 'execution_audit.json')['annotations_sha256']:
        raise ValueError('Validation labels differ from the baseline experiment')
    if digest(BASELINE / 'val_results.json') != audit['baseline_results_sha256']:
        raise ValueError('Baseline report changed')
    control = next(row for row in baseline['results'] if row['name'] == baseline['selected'])
    infer('val', audit)
    evaluations = []
    for variant in ('legacy', 'mask_first'):
        predictions = ROOT / ('val_' + variant + '.json')
        manifest = merge([ROOT / 'val' / f'shard{i}' / (variant + '.json') for i in range(SHARDS)],
                         ANNOTATIONS, 'val', predictions)
        evaluations.append(evaluate(('continued1536_' + variant, str(predictions), manifest,
                                     ROOT / ('eval_' + variant + '.json'))))
    drift = check_reproduction(evaluations[0], audit['best_metrics'])
    winner = choose([control] + evaluations)
    write_json(ROOT / 'val_results.json', dict(split='val', num_images=2584, units='0-1',
               selected=winner['name'], results=[control] + evaluations, trainer_legacy_drift=drift,
               selection='Highest full-val mask APs; overall AP breaks ties; exact ties preserve baseline. No test labels.'))
    output = ROOT / 'submission.csv'
    if winner['name'] == control['name']:
        prior = read_json(BASELINE / 'test_complete.json')
        if prior['selected'] != winner['name'] or digest(BASELINE / 'submission.csv') != prior['sha256']:
            raise ValueError('Baseline submission provenance mismatch')
        if not output.exists():
            shutil.copyfile(BASELINE / 'submission.csv', output)
        if digest(output) != prior['sha256']:
            raise ValueError('Copied baseline hash mismatch')
        validation = dict(prior['validation'], output=str(output), reused_from=str(BASELINE))
        write_json(str(output) + '.validation.json', validation)
    else:
        infer('test', audit)
        variant = winner['name'].removeprefix('continued1536_')
        predictions = ROOT / 'selected_test.json'
        merge([ROOT / 'test' / f'shard{i}' / (variant + '.json') for i in range(SHARDS)],
              TEST_METADATA, 'test', predictions)
        if output.exists():
            # Revalidate existing output rather than overwrite it after an interrupted workflow.
            from validate_submission import validate
            validation = validate(TEST_METADATA, str(predictions), str(output), manifest=str(predictions) + '.manifest.json')
            write_json(str(output) + '.validation.json', validation)
        else:
            subprocess.run(['./seg_project/run.sh', 'python', 'seg_project/tools/export_submission.py',
                            '--metadata', TEST_METADATA, '--predictions', str(predictions),
                            '--manifest', str(predictions) + '.manifest.json', '--template', 'blender/submission.csv',
                            '--output', str(output)], check=True)
            validation = read_json(str(output) + '.validation.json')
    if not validation['valid'] or not validation['submission_ready'] or validation['num_images'] != 2112:
        raise ValueError('Submission validation failed')
    check_sources(audit)
    completed = dict(selected=winner['name'], submission=str(output), sha256=digest(output), validation=validation,
                     val_APs=winner['metrics']['APs'], val_AP=winner['metrics']['AP'],
                     gain_APs_vs_round1=winner['metrics']['APs'] - control['metrics']['APs'])
    write_json(ROOT / 'test_complete.json', completed)
    text = (f"# 追加训练最终结果\n\n完整 val 选择：`{winner['name']}`，Mask APs={completed['val_APs']:.6f}，"
            f"Mask AP={completed['val_AP']:.6f}；相对第一轮 APs 变化 {completed['gain_APs_vs_round1']:+.6f}。\n\n"
            f"提交：`submission.csv`，2,112 张图片，{validation['num_instances']:,} 个实例。"
            "全量 RLE 解码及源预测一致性校验通过。未使用 test 标注，Kaggle 分数尚未测得。\n\n"
            f"SHA256：`{completed['sha256']}`\n")
    (ROOT / 'README.md').write_text(text, encoding='utf-8')
    state('completed', **completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true', help='Read training readiness without starting work')
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(dict(training_complete=training_complete(), training=read_json(TRAIN / 'manager_state.json'))))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (ROOT / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'test_complete.json').exists():
            prior = read_json(ROOT / 'test_complete.json')
            if digest(ROOT / 'submission.csv') != prior['sha256']:
                raise ValueError('Completed submission hash mismatch')
            print(json.dumps(prior))
            return
        try:
            finalize()
        except BaseException as error:
            state('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()

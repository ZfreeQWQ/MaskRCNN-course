"""Validate fixed NYUv2 indoor8 development evidence and write a new summary.

Standard-library only; never starts training or contacts the network. A summary
is rejected if runs, provenance, budgets or optional paired benchmarks disagree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


PROTOCOL = 'nyuv2-official-train-scene-dev-indoor8-v1'
NAMES = ('rgb_100', 'depth_100', 'all_on_smoke')
FLAGS = ('depth_fusion', 'small_object_crop', 'small_object_loss', 'boundary_loss')
EXPECTED_NAMES = ['book', 'bottle', 'chair', 'bowl', 'cup', 'vase', 'sofa', 'bed']
EXPECTED_COCO = [73, 39, 56, 45, 41, 75, 57, 59]
METRICS = ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l')
BENCHMARK_NAME = 'inference_benchmark.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value, label):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            f'{label} is not a SHA256 string')


def finite_number(value, label, minimum=0, strict=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > minimum if strict else value >= minimum),
            f'{label} is not an allowed finite number')
    return float(value)


def metric_points(report, key):
    name = f'nyuv2_void/segm_{key}'
    value = report['metrics'][name]
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value == -1 or 0 <= value <= 1),
            f'{name} must be [0,1], or -1 for an unrepresented GT area range')
    return None if value == -1 else round(value * 100, 6)


def validate_reports(reports):
    baseline = reports['rgb_100']
    shared = ('protocol', 'dataset_manifest_sha256', 'source_checkpoint_sha256',
              'implementation_sha256', 'categories', 'parameters',
              'mmdetection_commit', 'evaluation_policy', 'training_images')
    expected_flags = {'rgb_100': (False, False, False, False),
                      'depth_100': (True, False, False, False),
                      'all_on_smoke': (True, True, True, True)}
    rows = {}
    for name in NAMES:
        report = reports[name]
        require(report['status'] == 'passed', f'{name}: status is not passed')
        require(report['protocol'] == PROTOCOL, f'{name}: unexpected development protocol')
        args = report['args']
        require(args['mode'] == 'train' and args['trained_checkpoint'] is False,
                f'{name}: not a fresh independent training run')
        require(all(isinstance(args[key], bool) for key in FLAGS)
                and tuple(args[key] for key in FLAGS) == expected_flags[name],
                f'{name}: optimization switches do not match its label')
        for key in ('dataset_manifest_sha256', 'source_checkpoint_sha256', 'checkpoint_sha256'):
            valid_sha(report[key], f'{name}.{key}')
        implementation = report['implementation_sha256']
        require(isinstance(implementation, dict) and implementation,
                f'{name}: missing implementation hashes')
        for key, value in implementation.items():
            valid_sha(value, f'{name}.implementation_sha256.{key}')
        for key in shared:
            require(report[key] == baseline[key], f'{name}: differs from RGB baseline in {key}')
        categories = report['categories']
        require([row['id'] for row in categories] == list(range(1, 9))
                and [row['name'] for row in categories] == EXPECTED_NAMES
                and [row['coco80_class_index'] for row in categories] == EXPECTED_COCO,
                f'{name}: category order or COCO mapping differs from indoor8 v1')
        require(report['training_images'] == 640, f'{name}: expected all 640 development training frames')
        require(isinstance(report['parameters'], int) and report['parameters'] > 0,
                f'{name}: invalid parameter count')
        iterations = args['iters']
        require(isinstance(iterations, int) and not isinstance(iterations, bool) and iterations > 0
                and report['final_iteration'] == iterations,
                f'{name}: final iteration does not match requested iterations')
        steps = report['final_optimizer_steps']
        require(isinstance(steps, list) and steps and all(
            isinstance(step, int) and not isinstance(step, bool) and 0 < step <= iterations
            for step in steps) and max(steps) == iterations,
            f'{name}: invalid/nonadvancing optimizer step counts')
        require(report['classifier_changed'] is True, f'{name}: classifier was not updated')
        require(report['depth_adapter_changed'] is args['depth_fusion'],
                f'{name}: depth adapter update does not match its switch')
        adapter_norm = finite_number(report['depth_adapter_l2'], f'{name}.depth_adapter_l2')
        require((adapter_norm > 0) == args['depth_fusion'],
                f'{name}: zero-initialized depth adapter norm contradicts its switch')
        finite_number(report['first_loss'], f'{name}.first_loss')
        finite_number(report['last_loss'], f'{name}.last_loss')
        finite_number(report['seconds_including_evaluation'], f'{name}.seconds_including_evaluation', strict=True)
        finite_number(report['whole_run_peak_allocated_mib'], f'{name}.whole_run_peak_allocated_mib', strict=True)
        points = {key: metric_points(report, key) for key in METRICS}
        rows[name] = dict(name=name, switches={key: args[key] for key in FLAGS},
                          iterations=iterations, batch_size=args['batch_size'],
                          image_size=args['image_size'], seed=args['seed'],
                          mask_AP_points=points['mAP'], mask_APs_points=points['mAP_s'],
                          metrics_AP_points=points, checkpoint=report['checkpoint'],
                          checkpoint_sha256=report['checkpoint_sha256'],
                          depth_adapter_changed=report['depth_adapter_changed'],
                          depth_adapter_l2=adapter_norm,
                          seconds_including_evaluation=report['seconds_including_evaluation'],
                          whole_run_peak_allocated_mib=report['whole_run_peak_allocated_mib'])
    rgb_args = {k: v for k, v in baseline['args'].items() if k not in ('output', 'depth_fusion')}
    depth_args = {k: v for k, v in reports['depth_100']['args'].items() if k not in ('output', 'depth_fusion')}
    require(rgb_args == depth_args, 'RGB/depth args differ beyond output and depth_fusion')
    require(baseline['args']['iters'] == 100 and baseline['args']['seed'] == 20260923,
            'This summary expects the fixed 100-iteration, seed-20260923 development comparison')
    require(baseline['args']['image_size'] == 512 and baseline['args']['batch_size'] == 2,
            'This comparison expects image_size=512 and batch_size=2')
    smoke_args = reports['all_on_smoke']['args']
    require(smoke_args['iters'] == 2 and smoke_args['image_size'] == 256,
            'all_on_smoke must be the separate 2-iteration, 256-size integration check')
    for name in ('rgb_100', 'depth_100'):
        for suffix in ('AP', 'APs'):
            value, reference = rows[name][f'mask_{suffix}_points'], rows['rgb_100'][f'mask_{suffix}_points']
            rows[name][f'delta_{suffix}_vs_RGB_percentage_points'] = (
                None if value is None or reference is None else round(value - reference, 6))
    rows['all_on_smoke']['comparable_for_AP_ranking'] = False
    return rows, {key: baseline[key] for key in shared}, rgb_args


def validate_benchmarks(benchmarks, reports):
    rgb = benchmarks['rgb_100']
    equal_keys = ('protocol', 'script_sha256', 'seed', 'batch_size', 'warmup', 'repeats',
                  'input', 'environment', 'parameters', 'parameter_storage_bytes',
                  'parameter_dtypes', 'timing_scope', 'memory_scope')
    rows = {}
    for name, bench in benchmarks.items():
        require(bench['status'] == 'complete', f'{name}: benchmark status is not complete')
        require(bench['protocol'] == 'fixed-first-validation-frame-cpu-to-prediction-batch1-v1',
                f'{name}: unknown benchmark protocol')
        for key in equal_keys:
            require(bench[key] == rgb[key], f'{name}: benchmark differs in {key}')
        for key in ('script_sha256', 'checkpoint_sha256', 'config_sha256'):
            valid_sha(bench[key], f'{name}.benchmark.{key}')
        valid_sha(bench['input']['cpu_tensor_sha256'], f'{name}.benchmark.input.cpu_tensor_sha256')
        require(bench['checkpoint_sha256'] == reports[name]['checkpoint_sha256']
                and bench['checkpoint_iteration'] == reports[name]['final_iteration'],
                f'{name}: benchmark did not use the verified trained checkpoint')
        require(bench['parameters'] == reports[name]['parameters'],
                f'{name}: benchmark parameter count does not match training')
        require(bench['batch_size'] == 1, f'{name}: benchmark must use batch size one')
        for key in ('warmup', 'repeats'):
            require(isinstance(bench[key], int) and not isinstance(bench[key], bool) and bench[key] >= 1,
                    f'{name}: invalid benchmark {key}')
        require(isinstance(bench['environment'], dict) and bench['environment'],
                f'{name}: missing benchmark environment')
        flags = bench['model_flags']
        require(flags['depth_fusion']['enabled'] is reports[name]['args']['depth_fusion']
                and flags['small_object_loss']['enabled'] is False
                and flags['boundary_loss']['enabled'] is False
                and flags['ignore_void'] is True,
                f'{name}: benchmark model switches do not match the compared model')
        latency = bench['latency_ms']
        require(len(latency['samples']) == bench['repeats'], f'{name}: benchmark sample count mismatch')
        values = [finite_number(x, f'{name}.latency sample', strict=True) for x in latency['samples']]
        for key in ('mean', 'median', 'p90', 'p95', 'minimum', 'maximum'):
            finite_number(latency[key], f'{name}.latency_ms.{key}', strict=True)
        finite_number(latency['population_std'], f'{name}.latency_ms.population_std')
        require(math.isclose(latency['mean'], statistics.mean(values), rel_tol=1e-9),
                f'{name}: reported latency mean does not match samples')
        rate = finite_number(bench['images_per_second_reciprocal_mean_latency'],
                             f'{name}.reciprocal_latency', strict=True)
        require(math.isclose(rate, 1000 / latency['mean'], rel_tol=1e-9),
                f'{name}: reciprocal-latency rate is inconsistent')
        memory = bench['cuda_memory_mib']
        for key, value in memory.items():
            finite_number(value, f'{name}.cuda_memory_mib.{key}')
        rows[name] = dict(mean_ms=latency['mean'], median_ms=latency['median'],
                          p95_ms=latency['p95'], warmup=bench['warmup'], repeats=bench['repeats'],
                          reciprocal_mean_latency_images_per_second=rate,
                          peak_allocated_mib=memory['peak_allocated'],
                          peak_reserved_mib=memory['peak_reserved'],
                          checkpoint_sha256=bench['checkpoint_sha256'])
    return dict(status='complete', protocol=rgb['protocol'], rows=rows,
                shared_conditions={key: rgb[key] for key in equal_keys},
                delta_depth_minus_RGB_mean_ms=rows['depth_100']['mean_ms'] - rows['rgb_100']['mean_ms'],
                caveat='Single fixed validation frame, batch one; synchronized preprocessing + CPU-to-GPU '
                       'transfer + prediction/postprocessing. Excludes disk/decode/resize and void-aware AP. '
                       'Not dataset-wide throughput, kernel-only latency, or proof of a stable speedup. '
                       'Recorded environment equality does not prove absence of shared GPU/host load.')


def load_record(path):
    value = json.loads(path.read_text())
    return value, dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def summarize(root):
    root = Path(root).resolve()
    reports, sources = {}, {}
    for name in NAMES:
        reports[name], sources[name] = load_record(root / name / 'report.json')
    rows, shared, args = validate_reports(reports)
    paths = {name: root / name / BENCHMARK_NAME for name in ('rgb_100', 'depth_100')}
    found = {name: path.is_file() for name, path in paths.items()}
    benchmark_sources = {}
    if any(found.values()):
        require(all(found.values()), 'Only one paired benchmark exists; require both or neither')
        benches = {}
        for name, path in paths.items():
            benches[name], benchmark_sources[name] = load_record(path)
        benchmark = validate_benchmarks(benches, reports)
    else:
        benchmark = dict(status='not_measured', note='No paired inference benchmarks found; no speed/latency is invented.')
    return dict(status='passed', protocol=PROTOCOL, root=str(root),
                shared_training_conditions=shared, shared_RGB_depth_args=args,
                comparison=[rows['rgb_100'], rows['depth_100']],
                integration_smoke=rows['all_on_smoke'], benchmark=benchmark,
                source_report_files=sources, source_benchmark_files=benchmark_sources,
                metric_units='AP values are original [0,1] metrics multiplied by 100; deltas are '
                             'percentage points, not relative percentage improvement. COCO -1 -> null (no GT).',
                interpretation='One seed, 100 iterations, public indoor8 development protocol. Differences are '
                               'exploratory and do not establish stable algorithm improvement, official NYUv2 '
                               'scores or course-dataset results. The 2-step/256 smoke test is not ranked.',
                timing_note='seconds_including_evaluation is train+validation time, not FPS. '
                            'whole_run_peak_allocated_mib is PyTorch allocator training/evaluation peak, '
                            'not NVML total memory or inference-only peak.')


def display(value, signed=False):
    return '不可用（无对应 GT）' if value is None else (f'{value:+.2f}' if signed else f'{value:.2f}')


def markdown(result):
    lines = ['# NYUv2 indoor-8：已校验开发实验汇总', '',
             '仅为公开开发数据上的单种子短程实验，不是课程正式结果、官方 NYUv2 指标或稳定提分证明。', '',
             '## 同预算对照', '',
             '两组均为 100 步、batch 2、512 分辨率、seed 20260923；相同数据、起始权重、实现哈希、类别与参数数目。'
             '只有深度开关与输出目录不同。', '',
             '| 实验 | Mask AP | APs | AP 差（百分点） | APs 差（百分点） |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in result['comparison']:
        lines.append(f'| {row["name"]} | {display(row["mask_AP_points"])} | '
                     f'{display(row["mask_APs_points"])} | '
                     f'{display(row["delta_AP_vs_RGB_percentage_points"], True)} | '
                     f'{display(row["delta_APs_vs_RGB_percentage_points"], True)} |')
    lines.extend(['', 'AP/APs 已乘 100；差值为百分点，不是相对百分比。评测使用相同的 label-valid mask 裁剪协议，'
                  '不能与普通 COCO 或官方 NYUv2 表格直接比较。100 步与一个 seed 不能证明稳定提升。', '',
                  '## 全开集成检查', '',
                  '`all_on_smoke` 已通过：四开关全开，2 步、256 分辨率；分类器与深度适配器均更新。'
                  '由于预算和分辨率不同，不纳入上表排名。', '', '## 独立推理计时', ''])
    benchmark = result['benchmark']
    if benchmark['status'] == 'not_measured':
        lines.append('未测：没有成对的独立推理 benchmark 文件。不从训练耗时推算 FPS，也不编造显存/速度收益。')
    else:
        lines.extend(['两份 benchmark 均完成，固定输入张量 hash、测量协议、warmup/repeats、记录环境、参数与计时代码 hash 一致。', '',
                      '| 模型 | 均值 ms | P95 ms | 峰值 allocated MiB | 峰值 reserved MiB |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for name, row in benchmark['rows'].items():
            lines.append(f'| {name} | {row["mean_ms"]:.2f} | {row["p95_ms"]:.2f} | '
                         f'{row["peak_allocated_mib"]:.2f} | {row["peak_reserved_mib"]:.2f} |')
        lines.extend(['', f'深度减 RGB 的均值差：{benchmark["delta_depth_minus_RGB_mean_ms"]:+.2f} ms。', '',
                      '仅固定一张验证图、batch 1：包含预处理、CPU→GPU、模型预测/后处理与同步；不含读取、解码、resize、'
                      '输入 clone 和 AP 评测。它不是整个数据集的速度，也不是纯 GPU kernel 延迟。环境记录相同不证明无共享负载干扰。'])
    lines.extend(['', '## 证据与解释边界', '',
                  f'- 数据 manifest SHA256：`{result["shared_training_conditions"]["dataset_manifest_sha256"]}`。',
                  f'- 起始 checkpoint SHA256：`{result["shared_training_conditions"]["source_checkpoint_sha256"]}`。',
                  '- 三个原始 report 及可选 benchmark 的路径、SHA256、完整共同条件见 `summary.json`。',
                  '- 训练总耗时包含评测，不是 FPS；训练/验证峰值显存不能代替独立推理峰值。',
                  '- 官方 test 不用于本次训练和调参；历史全量准备检查曾访问 test，不宣称它从未被检查。',
                  '- mask 监督忽略 void，不等于解决 query 级别的 no-object 漏标监督。',
                  '- 下一步需更充分训练、多 seed 消融，并在课程数据到位后重做正式基线。', ''])
    return '\n'.join(lines)


def write_summary(result, output):
    output = Path(output)
    # Reserve a new directory only after all validations pass; never overwrite
    # even a pre-existing empty directory or a previous identical summary.
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with (output / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/outputs/nyu_indoor8_v1_20260922'))
    parser.add_argument('--output', type=Path, help='New output directory; default ROOT/summary; never overwrite')
    args = parser.parse_args()
    output = args.output or args.root / 'summary'
    if output.exists():
        parser.error('Output directory already exists; choose a new --output directory')
    try:
        result = summarize(args.root)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        parser.error(f'Incomplete or incompatible experiment evidence: {exc}')
    write_summary(result, output)
    print(json.dumps({'status': 'passed', 'output': str(output),
                      'benchmark': result['benchmark']['status'],
                      'comparison': result['comparison']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

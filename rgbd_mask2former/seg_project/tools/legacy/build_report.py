"""Build an evidence-based report from saved run artifacts; never opens test labels."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import yaml
from common import PROJECT, ROOT, read_json, write_json


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fmt(value, digits=2, signed=False):
    if not finite(value):
        return '待完成'
    return format(value, ('+' if signed else '') + f'.{digits}f')


def metric_fmt(metric, path):
    absent = object()
    value = get(metric, path, absent)
    if value is absent:
        return '待完成'
    return fmt(value) if finite(value) else '未定义'


def cell(value):
    if value is None:
        return '未记录'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def get(obj, path, default=None):
    for key in path.split('.'):
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def optional_json(path):
    return read_json(path) if Path(path).is_file() else {}


def source_link(path, output):
    import os
    return f'[{Path(path).name}]({os.path.relpath(Path(path).resolve(), Path(output).resolve().parent)})'


def logged_iterations(path):
    """Read only a JSONL tail. A live log can end with an incomplete record."""
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    import json
    iterations = []
    for line in lines:
        try:
            value = json.loads(line).get('iteration')
            if finite(value):
                iterations.append(int(value) + 1)
        except (ValueError, AttributeError):
            continue
    return max(iterations, default=None)


def load_experiment(name, directory, train_images=14432):
    directory = Path(directory).resolve()
    run = directory.parent if directory.name == 'report_artifacts' else directory
    # Training metrics.json is JSONL; only read a single metrics JSON from the artifact directory.
    metric_path = directory / 'metrics.json'
    if directory == run and (run / 'config.yaml').is_file():
        metric_path = run / 'report_artifacts/metrics.json'
    if not metric_path.is_file() and metric_path.with_name('metrics_smoke.json').is_file():
        metric_path = metric_path.with_name('metrics_smoke.json')
    metric = optional_json(metric_path)
    best = optional_json(run / 'best_metrics.json')
    snapshots = sorted((run / 'snapshots').glob('*/resolved_config.yaml'))
    config_path = snapshots[-1] if snapshots else run / 'config.yaml'
    config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
    config = config or {}
    manifest_path = config_path.parent / 'manifest.json' if snapshots else run / 'manifest.json'
    manifest = optional_json(manifest_path)
    if metric and not metric.get('units', '').startswith('percent'):
        raise ValueError(f'{metric_path}: run tools/evaluate.py to normalize units; training metrics.json is JSONL')
    if not metric and best:
        limit = get(config, 'PROJECT.EVAL_LIMIT')
        metric = dict(metrics=best.get('metrics', {}), units='percent (0-100)',
                      num_images=limit if finite(limit) and limit > 0 else (2584 if limit == 0 else None),
                      smoke=bool(limit) if limit is not None else None,
                      source=str(run / 'best_metrics.json'), selection=best)
        metric_path = run / 'best_metrics.json'
    speed_path = directory / 'benchmark.json'
    speed = optional_json(speed_path)
    if directory == run and not speed:
        speed_path = run / 'report_artifacts/benchmark.json'
        speed = optional_json(speed_path)
    if not speed and speed_path.with_name('benchmark_smoke.json').is_file():
        speed_path = speed_path.with_name('benchmark_smoke.json')
        speed = optional_json(speed_path)
    marker = optional_json(run / 'training_complete.json')
    stage = get(manifest, 'arguments.stage') or marker.get('stage')
    if not stage:
        stage = next((value for value in ('pilot', '1x', '2x')
                      if run.name.startswith(value + '_') or f'_ddp_{value}_' in run.name), '未记录/smoke')
    steps, batch = get(config, 'SOLVER.MAX_ITER'), get(config, 'SOLVER.IMS_PER_BATCH')
    updates_complete = (run / 'model_final.pth').is_file()
    completed = steps if marker or updates_complete else logged_iterations(run / 'metrics.json')
    best_selection = metric.get('selection') or best
    best_iteration = best_selection.get('completed_iterations')
    if best_iteration is None and finite(best_selection.get('iteration')):
        best_iteration = int(best_selection['iteration']) + 1
    epochs = steps * batch / train_images if finite(steps) and finite(batch) else None
    best_epochs = best_iteration * batch / train_images if finite(best_iteration) and finite(batch) else None
    full_val = metric.get('num_images') == 2584 and metric.get('smoke') is False
    shape = speed.get('input_shape', [])
    standard_speed = (bool(speed) and speed.get('smoke') is False and
                      speed.get('warmup', 0) >= 20 and speed.get('iterations', 0) >= 200 and
                      len(shape) == 4 and shape[0] == 1 and shape[-2:] == [1024, 1024])
    return dict(name=name, directory=str(directory), run=str(run), metrics=metric, benchmark=speed,
                config=config, manifest=manifest, config_path=str(config_path), manifest_path=str(manifest_path),
                metric_path=str(metric_path), benchmark_path=str(speed_path), stage=stage,
                role=str(get(config, 'PROJECT.EXPERIMENT', name)).upper(),
                max_iter=steps, batch=batch, completed_iterations=completed,
                best_completed_iterations=best_iteration, epochs=epochs, best_epochs=best_epochs,
                seed=get(config, 'SEED'), complete=bool(marker), optimizer_updates_complete=updates_complete,
                full_val=full_val, standard_speed=standard_speed)


def delta(a, b, section, path):
    left, right = get(a[section], path), get(b[section], path)
    return right - left if finite(left) and finite(right) else None


# These are controlled settings, excluding intentional modality/anchor/proposal/mask-head differences.
CONTROLS = ('SEED', 'SOLVER.MAX_ITER', 'SOLVER.IMS_PER_BATCH', 'SOLVER.BASE_LR',
            'SOLVER.STEPS', 'SOLVER.WARMUP_ITERS', 'SOLVER.MOMENTUM', 'SOLVER.WEIGHT_DECAY',
            'SOLVER.AMP.ENABLED', 'INPUT.MIN_SIZE_TRAIN', 'INPUT.MAX_SIZE_TRAIN',
            'INPUT.RANDOM_FLIP', 'INPUT.MIN_SIZE_TEST', 'INPUT.MAX_SIZE_TEST',
            'MODEL.BACKBONE.FREEZE_AT', 'MODEL.WEIGHTS', 'MODEL.ROI_HEADS.SCORE_THRESH_TEST',
            'MODEL.ROI_HEADS.NMS_THRESH_TEST', 'TEST.DETECTIONS_PER_IMAGE', 'TEST.EVAL_PERIOD')


def compare(left, right):
    changed = [path for path in CONTROLS if get(left['config'], path) != get(right['config'], path)]
    missing = [path for path in CONTROLS if get(left['config'], path) is None or get(right['config'], path) is None]
    notes = []
    if changed:
        notes.append('控制项不同：' + '、'.join(changed))
    if missing:
        notes.append('控制项缺记录：' + '、'.join(missing))
    for path in ('rgbd_seg/data.py', 'rgbd_seg/trainer.py', 'rgbd_seg/checkpoint.py'):
        a, b = left['manifest'].get('source_sha256', {}).get(path), right['manifest'].get('source_sha256', {}).get(path)
        if a and b and a != b:
            notes.append(f'快照代码不同：{path}')
    if not (left['full_val'] and right['full_val']):
        notes.append('尚非两组完整val结果（smoke/缺失项不构成正式消融）')
    if not (left['complete'] and right['complete']):
        notes.append('训练仍在进行，当前best不是等预算训练完成后的正式对照')
    if left['stage'] == 'pilot' or right['stage'] == 'pilot':
        notes.append('pilot为约0.208epoch先导运行，不能替代1x正式消融')
    comparable_accuracy = not notes
    speed_notes = []
    if not (left['standard_speed'] and right['standard_speed']):
        speed_notes.append('尚非两组标准200张计时')
    for key in ('device_name', 'torch_version', 'score_threshold', 'tta', 'tiling'):
        a, b = left['benchmark'].get(key), right['benchmark'].get(key)
        if a is None or b is None:
            speed_notes.append(f'速度条件未记录：{key}')
        elif a != b:
            speed_notes.append(f'速度条件不同：{key}')
    for key in ('MODEL.ROI_HEADS.NMS_THRESH_TEST', 'TEST.DETECTIONS_PER_IMAGE', 'INPUT.MIN_SIZE_TEST', 'INPUT.MAX_SIZE_TEST'):
        if get(left['config'], key) != get(right['config'], key):
            speed_notes.append(f'速度后处理/输入条件不同：{key}')
    return dict(left=left['name'], right=right['name'],
                delta_APs=delta(left, right, 'metrics', 'metrics.segm.APs'),
                delta_AP=delta(left, right, 'metrics', 'metrics.segm.AP'),
                delta_latency_ms=delta(left, right, 'benchmark', 'latency_model_mean_ms'),
                delta_fps=delta(left, right, 'benchmark', 'fps_model'),
                delta_peak_MiB=delta(left, right, 'benchmark', 'peak_allocated_MiB'),
                comparable_accuracy=comparable_accuracy, comparable_speed=not speed_notes,
                accuracy_notes=notes, speed_notes=speed_notes)


def evidence_sections(project, output):
    validation_path = project / 'reports/data_validation.json'
    depth_path = project / 'datasets/blender/annotations/depth_stats.json'
    env_path = project.parent / 'runs/env_verified.json'
    ddp_path = project.parent / 'runs/ddp_smoke.json'
    validation, depth, env, ddp = map(optional_json, (validation_path, depth_path, env_path, ddp_path))
    lines = ['## 数据与运行环境证据', '', '| split | 图像 | 实例 | 已解码mask | 场景 | <1024px²实例 |',
             '|---|---:|---:|---:|---:|---:|']
    for split in ('train', 'val'):
        info = validation.get('splits', {}).get(split, {})
        small = get(info, 'coco_area_counts.small_lt_1024')
        count = info.get('annotations')
        small_text = f'{small:,} ({small/count:.2%})' if finite(small) and finite(count) and count else '待检查'
        lines.append('| ' + ' | '.join([split] + [cell(info.get(key)) for key in ('images', 'annotations', 'all_masks_decoded', 'scenes')] + [small_text]) + ' |')
    if validation:
        lines += ['', f"全量验证记录：场景互斥={validation.get('scene_disjoint')}，合并文件SHA256匹配={validation.get('merged_files_match_preparation_sha256')}；{validation.get('mask_validation', '详细检查未记录')}。来源：{source_link(validation_path, output)}。面积计数使用<1024的互斥描述区间，正式APs以COCO evaluator边界实现为准。"]
    else:
        lines += ['', '数据验证产物缺失，以上计数尚待核实。']
    if depth:
        invalid_fraction = depth.get('invalid_fraction')
        lines += ['', f"训练深度统计：覆盖{cell(depth.get('images'))}张、{cell(depth.get('valid_pixels'))}个有效像素；无效率{fmt(invalid_fraction*100 if finite(invalid_fraction) else None, 4)}%。p1={fmt(depth.get('p1'), 4)}m、p99={fmt(depth.get('p99'), 4)}m、中位数{fmt(depth.get('median'), 4)}m，有效范围[{fmt(depth.get('minimum_m'), 4)}, {fmt(depth.get('maximum_m'), 4)}]m。split={cell(depth.get('split'))}、train_only={cell(depth.get('train_only'))}。",
                  f"统计方法：{depth.get('method', '未记录')}；来源：{source_link(depth_path, output)}。训练中出现的scale值{cell(depth.get('depth_scale_values'))}不能外推为test标定。"]
    else:
        lines += ['', '训练深度统计未完成，不推断深度范围或test系数。']
    if env:
        devices = env.get('devices', [])
        names = sorted({device.get('name', '未记录') for device in devices})
        passed = sum(device.get('status') == 'passed' for device in devices)
        totals = [get(device, 'free_total_bytes') for device in devices]
        gib = sorted({round(item[1]/2**30, 2) for item in totals if isinstance(item, list) and len(item) == 2})
        lines += ['', f"环境实测：{len(devices)}张 {'/'.join(names)}，每卡容量{cell(gib)}GiB，设备算子检查{passed}/{len(devices)}通过；PyTorch包版本{cell(get(env, 'packages.torch'))}、torchvision={cell(get(env, 'packages.torchvision'))}、pycocotools={cell(get(env, 'packages.pycocotools'))}。ixsmi返回码={cell(get(env, 'ixsmi.returncode'))}。",
                  f"该CoreX PyTorch通过torch.cuda兼容API暴露设备；CUDA API版本字段{cell(env.get('cuda_api_version'))}不代表NVIDIA硬件。run.sh仅为子进程配置厂商库路径；没有安装或更改GPU底层依赖。来源：{source_link(env_path, output)}。"]
    else:
        lines += ['', '环境验证JSON缺失；不凭配置推断GPU测试已通过。']
    if ddp:
        lines += ['', f"DDP通信记录：status={cell(ddp.get('status'))}，world_size={cell(ddp.get('world_size'))}，backend={cell(ddp.get('backend'))}，steps={cell(ddp.get('steps'))}，权重一致={cell(ddp.get('identical_weights'))}；来源：{source_link(ddp_path, output)}。通信smoke只证明链路，不替代正式训练结果。"]
    return lines, validation


def final_sections(project, output, experiments, validation):
    selection_path = project / 'outputs/final_selection.json'
    error_path = project / 'reports/final_error_analysis/error_candidates.json'
    pipeline_path = project / 'outputs/pipeline_status.json'
    submission_path = project / 'submissions/validation.json'
    selection, errors, pipeline, submission = map(optional_json, (selection_path, error_path, pipeline_path, submission_path))
    lines = ['## 最终模型选择与提交状态', '']
    if selection:
        selected = selection.get('selected_directory')
        lines += [f"验证集选择：`{cell(selected)}`；准则：{cell(selection.get('criterion'))}。选中[APs, AP]={cell(selection.get('selected_score_APs_AP'))}，推理阈值={cell(selection.get('inference_score_threshold'))}；test_labels_used={cell(selection.get('test_labels_used'))}。来源：{source_link(selection_path, output)}。", '',
                  '| 候选运行 | segm/APs (%) | segm/AP (%) |', '|---|---:|---:|']
        for path, scores in selection.get('candidates', {}).items():
            lines.append(f'| {cell(path)} | {fmt(scores[0])} | {fmt(scores[1])} |')
        selected_row = next((item for item in experiments if Path(item['run']) == Path(selected).resolve()), None) if selected else None
        if selected_row:
            lines += ['', f"选中模型预算为{fmt(selected_row['epochs'], 3)}epoch，最佳checkpoint处于{fmt(selected_row['best_epochs'], 3)}epoch。按val选择只表示候选中的观察最优，不证明架构改动的因果效果。"]
    else:
        lines += ['最终选择文件尚未生成，不能宣布最终模型或填写最终test结果。']
    state = get(pipeline, 'jobs.pipeline', {})
    if state:
        lines += ['', f"pipeline记录状态：`{cell(state.get('status'))}`（更新于{cell(pipeline.get('updated_utc'))}）。{cell(state.get('reason', state.get('note', '')))} 来源：{source_link(pipeline_path, output)}。"]
    if submission.get('valid') is True and submission.get('submission_ready') is not False and submission.get('num_images') == 2112:
        lines += ['', f"提交格式校验通过：{submission['num_images']}图、{cell(submission.get('num_instances'))}实例、{cell(submission.get('csv_rows'))}条CSV数据行；来源：{source_link(submission_path, output)}。这仅为格式与覆盖校验，没有test精度结果。"]
    else:
        calibrated = get(validation, 'test_metadata.depth_calibration_available')
        lines += ['', '尚无通过的2,112图最终提交校验。' + ('数据检查明确记录test深度标定不可用，RGB-D test推理需补齐原始允许的逐图depth_scale_m/depth_invalid_code，不能猜测。' if calibrated is False else 'test输入元数据与提交格式仍需最终核验。')]
    lines += ['', 'test仅允许RGB-D输入及原始输入元数据；本报告生成器读取数据验证摘要和提交校验摘要，不打开任何test标注。JSON为compressed RLE，CSV为score/xmin/ymin/xmax/ymax，含零检测图在内共2,112行数据。', '', '## 验证集错误候选与证据边界', '']
    if errors:
        lines += [f"候选分析阈值={cell(errors.get('score_threshold'))}，小目标口径={cell(errors.get('small_area_definition'))}。来源：{source_link(error_path, output)}。", '',
                  '| 候选类型 | 实例候选数 | 保存案例数 |', '|---|---:|---:|']
        cases_by_kind = errors.get('stratified_cases') or errors.get('cases', {})
        counts = errors.get('all_area_counts') or errors.get('counts', {})
        for kind, count in counts.items():
            lines.append(f"| {cell(kind)} | {cell(count)} | {len(cases_by_kind.get(kind, []))} |")
        sources = errors.get('sources', {})
        if sources:
            lines += ['', '候选来源（Selected指当前分析模型，不能据此默认是M1）：', '']
            for label, path in sources.items():
                lines.append(f'- {cell(label)}：`{cell(path)}`。')
            if not sources.get('M1_predictions'):
                lines.append('- 独立M1预测来源未提供，不能把Selected列当成M1列。')
        if errors.get('counts_by_area'):
            lines += ['', '| 候选类型 | small | medium | large |', '|---|---:|---:|---:|']
            for kind, areas in errors['counts_by_area'].items():
                lines.append('| ' + ' | '.join([cell(kind)] + [cell(areas.get(group)) for group in ('S', 'M', 'L')]) + ' |')
            lines += ['', 'S/M/L采用COCO面积边界，1024与9216px²边界可归入相邻两组，不能简单相加为互斥总数；案例主area_group为确定展示层。人口分母与多标签重叠见候选JSON。']
        lines += ['', '以下ID来自真实候选记录，B0/最终模型对照图见同目录analysis.md；同一实例可以命中多个候选标签，因此各类数量不能相加作为独立错误总数。', '',
                  '| 候选类型 | case_id | area | image_id | annotation_id | semantic_type | GT面积(px²) | best IoU | 复核状态 |', '|---|---|---|---:|---:|---|---:|---:|---|']
        for kind, cases in cases_by_kind.items():
            for case in cases[:3]:
                lines.append('| ' + ' | '.join([cell(kind), cell(case.get('case_id')), cell(case.get('area_group')),
                             cell(case.get('image_id')), cell(case.get('annotation_id')), cell(case.get('semantic_type')),
                             fmt(case.get('gt_area'), 0), fmt(case.get('best_iou'), 3),
                             cell(case.get('review_status', 'pending_human_review'))]) + ' |')
        analysis_path = error_path.with_name('analysis.md')
        if analysis_path.is_file():
            lines += ['', f'完整候选及RGB/深度/GT/B0/最终模型展示：{source_link(analysis_path, output)}。']
        lines += ['', '候选是IoU/覆盖率启发式筛选，尚不能自动确认为边界偏移、粘连或特定成因。遮挡、深度失效、目标像素或anchor不匹配等解释需人工观察和专门消融支持；本报告不把这些假设写成已证实结论。']
        if not sources.get('selected_predictions') and not errors.get('predictions') and not errors.get('prediction_source'):
            lines += ['该候选JSON未内嵌预测文件/模型来源，计数仅归属于所引用候选产物；应结合pipeline命令日志核对是否对应当前选中模型，不能仅凭同目录位置推定来源一致。']
    else:
        lines += ['真实错误候选尚未生成，不能填写错误数量、典型案例或原因。正式产物就绪后由error_analysis.py生成并在下次汇总时引用。']
    demo = project / 'reports/figures/demo_smoke_m1_val.png'
    if demo.is_file():
        lines += ['', f'已有链路Demo：{source_link(demo, output)}，来自20iter smoke模型；仅演示RGB/深度/GT/预测布局，不代表最终模型。']
    return lines


def build_report(specifications, output, title, project=PROJECT):
    project, output = Path(project).resolve(), Path(output)
    validation = optional_json(project / 'reports/data_validation.json')
    train_images = get(validation, 'splits.train.images', 14432)
    experiments = [load_experiment(*specification.split('=', 1), train_images=train_images) for specification in specifications]
    lines = [f'# {title}', '',
             '主指标为官方COCO segm/APs，所有AP值以0–100百分数报告，差值用百分点(pp)。未定义项与缺失产物不填分数。原train/val场景划分固定；test不用于训练、阈值、案例选择或本报告的性能比较。', '',
             '## 实验方法与可复现预算', '',
             'B0为RGB Mask R-CNN R50-FPN；B1为4通道RGB-D早期融合；M1在B1上联合改变小尺度anchors、proposal数量及mask ROI分辨率；A1是M1移除深度。B1同时充当移除小目标组合改进的A2。标准FPN原本已有P2，不能把保留P2单独当作新增模块。以下实际配置优先于方法名称。', '',
             '| 实验 | stage | 全局batch | 计划iter | 计划epoch | 日志/完成iter | seed | 最佳完成iter (epoch) | 状态 |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    for item in experiments:
        status = ('训练及最终验证已完成（完成标记）' if item['complete'] else
                  '训练更新完成，最终验证/退出待确认' if item['optimizer_updates_complete'] else
                  '训练未完成/状态待核实')
        current = cell(item['completed_iterations']) + ('' if item['complete'] else '（日志下界）')
        best = f"{cell(item['best_completed_iterations'])} ({fmt(item['best_epochs'], 3)})"
        lines.append('| ' + ' | '.join([cell(item['name']), cell(item['stage']), cell(item['batch']), cell(item['max_iter']),
                     fmt(item['epochs'], 3), current, cell(item['seed']), best, status]) + ' |')
    lines += ['', f'epoch按MAX_ITER×全局batch/{train_images}计算，是训练样本呈现次数口径，不把iteration数当epoch。pilot预算3,000图次约0.208epoch；1x为10epoch；最终DDP 2x为20epoch（四卡batch=4通常为72,160iter）。不同训练预算、batch与学习率不能作为公平架构消融；2x相对1x的变化同时受额外训练影响。各行预算以上述resolved config实值为准。', '',
              '| 实验 | 深度 | 深度mean/std | anchors(px) | RPN pre/post test | mask ROI | 训练尺度 | LR / steps | score / NMS |',
              '|---|---|---|---|---|---:|---|---|---|']
    for item in experiments:
        cfg = item['config']
        means, stds = get(cfg, 'MODEL.PIXEL_MEAN', []), get(cfg, 'MODEL.PIXEL_STD', [])
        depth_norm = f'{means[3]}/{stds[3]}' if len(means) == len(stds) == 4 else '—'
        values = [item['name'], get(cfg, 'PROJECT.USE_DEPTH'), depth_norm, get(cfg, 'MODEL.ANCHOR_GENERATOR.SIZES'),
                  f"{cell(get(cfg, 'MODEL.RPN.PRE_NMS_TOPK_TEST'))}/{cell(get(cfg, 'MODEL.RPN.POST_NMS_TOPK_TEST'))}",
                  get(cfg, 'MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION'), get(cfg, 'INPUT.MIN_SIZE_TRAIN'),
                  f"{cell(get(cfg, 'SOLVER.BASE_LR'))} / {cell(get(cfg, 'SOLVER.STEPS'))}",
                  f"{cell(get(cfg, 'MODEL.ROI_HEADS.SCORE_THRESH_TEST'))} / {cell(get(cfg, 'MODEL.ROI_HEADS.NMS_THRESH_TEST'))}"]
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    lines += ['', '深度按逐图系数转米，valid由原始无效编码定义。实际Mapper对有效值按训练p1/p99裁剪归一化，对无效值填深度均值0.5，再按std=0.25标准化，使其进入backbone时为0；独立valid mask用于同步几何变换与无效值重置，没有作为第五通道输入。val复用训练统计。', '',
              '## 验证集主结果', '', '| 实验 | segm/APs ↑ | AP | AP50 | AP75 | APm | APl | bbox AP | 评价范围 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for item in experiments:
        metric = item['metrics']
        status = '完整val 2,584图' if item['full_val'] else ('smoke/子集，仅链路验证' if metric else '待完成')
        lines.append('| ' + ' | '.join([cell(item['name'])] + [metric_fmt(metric, 'metrics.segm.' + key)
                     for key in ('APs', 'AP', 'AP50', 'AP75', 'APm', 'APl')] + [metric_fmt(metric, 'metrics.bbox.AP'), status]) + ' |')
    lines += ['', '完整val指标仍需结合训练状态理解：训练进行中的best仅是已评价checkpoint中的当前最佳，不能宣称训练完成。无GT对应面积组的指标显示未定义，缺失产物显示待完成，均不按0填充。pipeline的bbox AP来自Detectron2检测框；独立evaluate.py的bbox AP来自导出JSON框，不能混用来源比较。segm评估按mask面积分组。', '',
              '## 模型大小与标准推理速度', '',
              '| 实验 | 参数(M) | 纯model_state(MiB) | 完整checkpoint(MiB) | 模型ms | 读/预处理ms | FPS | 峰值allocated(MiB) | 测量状态 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for item in experiments:
        speed = item['benchmark']
        parameters = speed.get('parameters')
        status = '标准1024，20+≥200' if item['standard_speed'] else ('smoke/标准条件未满足' if speed else '待测')
        values = [cell(item['name']), fmt(parameters / 1e6 if finite(parameters) else None)] + [fmt(speed.get(key)) for key in
                  ('model_state_MiB', 'checkpoint_MiB', 'latency_model_mean_ms', 'read_preprocess_mean_ms', 'fps_model', 'peak_allocated_MiB')] + [status]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '标准测量要求batch=1、1024×1024、至少20次预热与200张计时，并在模型前后按CoreX同步。模型时间包含设备传输/forward/D2后处理，不含RLE编码；读图与深度预处理单列。纯model_state统计参数及buffer张量字节，完整checkpoint还可能包含优化器/调度器状态，二者不能互代。峰值显存为单进程模型推理allocated值，不是DDP训练峰值；TTA/裁剪、设备、原始时间样本见benchmark JSON。', '',
              '## 真实训练与验证曲线', '',
              '曲线只读取各run的Detectron2 metrics.json事件日志。主横轴为completed iteration×全局batch/训练图数，上轴为completed iteration；每组单独标明stage和计划epoch，不把pilot、1x、DDP2x混为相同预算。仅绘制真实有限数值散点，不插补、不把没有验证的迭代填0。loss/time是原生日志的窗口统计点，不是逐step原始值；恢复后同一iteration同一键取最后记录。', '',
              '| 实验 | loss点 | LR点 | full-val APs/AP点 | time日志点 | 训练step中位秒 | 推算全局图/秒 |',
              '|---|---:|---:|---|---:|---:|---:|']
    from plot_training_curves import plot_experiment
    for item in experiments:
        curve = plot_experiment(item, project / 'reports/figures', output.stem, train_images)
        item['training_curves'] = curve
        counts = curve.get('points', {})
        lines.append('| ' + ' | '.join([cell(item['name']), cell(counts.get('total_loss')), cell(counts.get('lr')),
                     f"{cell(counts.get('segm/APs'))}/{cell(counts.get('segm/AP'))}", cell(curve.get('time_logged_samples')),
                     fmt(curve.get('train_step_median_seconds'), 4), fmt(curve.get('global_train_images_per_second'))]) + ' |')
    lines += ['', '训练吞吐仅为全局batch除以日志time点的中位数；JSONWriter保存窗口中位数而非全部迭代，故这是采样训练step速率估计。IterationTimer包含当步数据等待、前后向/优化器及DDP通信，不含val/checkpoint等hooks与进程初始化/首次加载边界；不能当作含验证和落盘的全程吞吐，也不能替代同步推理benchmark。单卡与DDP若全局batch、学习率、训练预算或设备数不同，图/秒差值不能直接解释为线性扩展效率。']
    for item in experiments:
        curve = item['training_curves']
        lines += ['', f"**{item['name']}训练日志：**`{curve['log']}`；验证范围判据：{curve.get('validation_scope')}。"]
        if curve.get('status') == 'created':
            lines += ['', f"![{item['name']}真实训练/验证事件曲线]({os.path.relpath(curve['figure'], output.resolve().parent)})",
                      f"原始绘图点与时间统计：{source_link(curve['event_data'], output)}。"]
        else:
            lines.append(f"未生成曲线：{cell(curve.get('reason'))}。")
        if curve.get('excluded_unverified_val_points'):
            lines.append(f"已排除{curve['excluded_unverified_val_points']}个部分/范围未证实的val标量事件，不将其标为完整val。")
        if curve.get('malformed_or_partial_lines'):
            lines.append(f"本次读取跳过{curve['malformed_or_partial_lines']}条不完整/无效JSON行；日志仍在追加时，以本次图数据快照为准。")
    lines += ['', '## 消融差值与可比性', '',
              '所有差值均为右侧减左侧；APS/AP正值表示观察分数更高，延迟/显存正值表示开销增加。联合改动的M1只支持组合对照，未分别跑anchor/proposal/mask分辨率消融时不能给其中某项单独归因。', '',
              '| 对照 | ΔAPS(pp) | ΔAP(pp) | Δ模型ms | ΔFPS | Δ显存MiB | 精度对照状态 | 速度对照状态 |',
              '|---|---:|---:|---:|---:|---:|---|---|']
    component_roles = ('M1_NO_SMALL_ANCHORS', 'M1_NO_EXTRA_PROPOSALS', 'M1_NO_HIGH_MASKPOOL')
    allowed_roles = ('B0', 'B1', 'M1', 'A1', *component_roles)
    roles = {}
    for item in experiments:
        if item['name'].upper() in allowed_roles:
            roles[item['name'].upper()] = item
        elif item['role'] in allowed_roles and 'final' not in item['name'].lower():
            roles.setdefault(item['role'], item)
    comparisons = []
    pairs = [('B0', 'B1'), ('B1', 'M1'), ('A1', 'M1')]
    pairs += [(role, 'M1') for role in component_roles if role in roles]
    for left, right in pairs:
        if left not in roles or right not in roles:
            lines.append(f'| {left}→{right} | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 缺实验 | 缺实验 |')
            continue
        comparison = compare(roles[left], roles[right])
        comparisons.append(comparison)
        lines.append('| ' + ' | '.join([f'{left}→{right}'] + [fmt(comparison[key], signed=True) for key in
                     ('delta_APs', 'delta_AP', 'delta_latency_ms', 'delta_fps', 'delta_peak_MiB')] +
                     ['已记录控制项一致' if comparison['comparable_accuracy'] else '受限，见下文',
                      '标准条件一致' if comparison['comparable_speed'] else '受限，见下文']) + ' |')
    for comparison in comparisons:
        label = f"{comparison['left']}→{comparison['right']}"
        aps, ap = comparison['delta_APs'], comparison['delta_AP']
        if finite(aps) and finite(ap):
            description = f'观察到APS变化{fmt(aps, signed=True)}pp、AP变化{fmt(ap, signed=True)}pp。'
            if aps > 0 and ap < 0:
                description += '小目标分数增加同时整体AP下降，需要按预先确定的容忍度评估权衡，不能只报告APS。'
        else:
            description = '至少一侧精度缺失，尚不能计算精度变化。'
        lines += ['', f"**{label}：**{description}"]
        for kind, notes in (('精度限制', comparison['accuracy_notes']), ('速度限制', comparison['speed_notes'])):
            if notes:
                lines.append(f"{kind}：{'；'.join(notes)}。")
        if comparison['comparable_accuracy']:
            lines.append('在已记录控制项和完整val口径下可作该配置组合的对照；当前单种子观察没有置信区间，仍不能把差值当作已排除随机波动的因果证据。')
    if any(role in roles for role in component_roles):
        lines += ['', '组件行按“M1减去移除该组件的对照”计算：NO_SMALL_ANCHORS恢复标准anchors，NO_EXTRA_PROPOSALS恢复基线proposal数量，NO_HIGH_MASKPOOL恢复mask ROI=14。实际配置仍须以上表核实；只有完整val、等预算与其他控制项一致时，差值才可用于该联合配置下对应组件的局部对照，不能外推为各组件普遍独立的收益。']
    lines += ['', 'pilot与1x、1x与最终DDP 2x仅能分别展示运行结果；最终预算更长和同步batch不同，不能据其更高AP宣称架构优越。最终checkpoint由val选择，报告不使用test精度支持任何结论。']
    extra, validation = evidence_sections(project, output)
    lines += ['', *extra, '', *final_sections(project, output, experiments, validation), '', '## 可追溯产物', '']
    for item in experiments:
        links = []
        for key in ('config_path', 'manifest_path', 'metric_path', 'benchmark_path'):
            path = Path(item[key])
            links.append(source_link(path, output) if path.is_file() else f'`{path}`（未生成）')
        pretrained = get(item['config'], 'MODEL.WEIGHTS')
        lines += [f"- {cell(item['name'])}: {'；'.join(links)}。预训练/初始权重配置：`{cell(pretrained)}`。resolved config取最近一次启动快照，历史resume配置仍保留于snapshots；manifest保存源码SHA256、启动参数与设备映射；最佳iter为0-based iteration+1。"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output.with_suffix('.artifacts.json'), experiments)
    return experiments, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', action='append', required=True, metavar='NAME=DIRECTORY')
    parser.add_argument('--output', default='seg_project/reports/report.md')
    parser.add_argument('--title', default='RGB-D 元器件实例分割实验报告（真实产物汇总）')
    args = parser.parse_args()
    build_report(args.experiment, args.output, args.title)
    print(args.output)


if __name__ == '__main__':
    main()

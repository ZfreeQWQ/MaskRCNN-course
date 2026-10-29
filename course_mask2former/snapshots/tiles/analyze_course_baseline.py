"""Paired validation-only failure illustrations and synchronized inference timing.

Uses the completed baseline's frozen modules and strict checkpoint/config checks.
The eight images are fixed first views of the first eight validation scenes,
selected before prediction. Their thresholded diagnostic recall is not COCO AP.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from scipy.ndimage import binary_erosion, binary_dilation

from launch_nyu_sweep import atomic_json, sha256
from course_checkpoint import resolve_final_checkpoint


def overlay(rgb, masks):
    output = np.asarray(rgb).astype(np.float32).copy()
    rng = np.random.default_rng(53)
    for mask in masks:
        color = rng.uniform(40, 255, size=3)
        output[mask] = .5 * output[mask] + .5 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def boundary_f1(prediction, target, tolerance=2):
    """Diagnostic contour F1 with a Manhattan-distance pixel tolerance, not AP.

    Computed only for IoU>=0.5 matched objects; missed objects are separately
    counted and not hidden inside this conditional boundary-quality metric.
    """
    pred_edge = prediction & ~binary_erosion(prediction, border_value=0)
    gt_edge = target & ~binary_erosion(target, border_value=0)
    if not pred_edge.any() or not gt_edge.any():
        return 0.0
    precision = (pred_edge & binary_dilation(gt_edge, iterations=tolerance)).sum()/pred_edge.sum()
    recall = (gt_edge & binary_dilation(pred_edge, iterations=tolerance)).sum()/gt_edge.sum()
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', nargs='+', default=['rgb', 'gated_depth'],
                        help='Completed run names sharing the same frozen workflow source')
    args = parser.parse_args()
    root, output = Path(args.baseline_root).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Diagnostic output must be new')
    if len(args.models) != len(set(args.models)) or any(not n.isidentifier() for n in args.models):
        raise ValueError('Unique, simple run names required')
    source = root.parent / 'source'
    plan = json.loads((root.parent/'plan.json').read_text())
    for rel, digest in plan['source_sha256'].items():
        if sha256(source/rel) != digest:
            raise RuntimeError('Frozen baseline source changed: '+rel)
    # Training modules must come from the original baseline snapshot.
    sys.path.insert(0, str(source))
    register_all_modules(init_default_scope=True)
    output.mkdir(parents=True)
    records = {}
    panels = {}
    sample_ids = None
    for name in args.models:
        run = root/'runs'/name
        report = json.loads((run/'report.json').read_text())
        checkpoint = resolve_final_checkpoint(run, report)
        if report['status'] != 'passed' or sha256(checkpoint) != report['checkpoint_sha256']:
            raise ValueError('Baseline checkpoint integrity failed')
        cfg = Config.fromfile(str(run/'config.py'))
        import_modules_from_strings(**cfg.custom_imports)
        dataset = DATASETS.build(copy.deepcopy(cfg.val_dataloader.dataset))
        if len(dataset) != 2584 or 'instances_val.json' not in dataset.ann_file:
            raise ValueError('Only official validation is permitted')
        # MMDetection releases its temporary COCO handle after loading records.
        coco = COCO(dataset.ann_file)
        annotation_hash = sha256(dataset.ann_file)
        if annotation_hash != report['val_annotations_sha256']:
            raise ValueError('Validation annotations changed')
        scene_seen, indices = set(), []
        for i in range(len(dataset)):
            info = dataset.get_data_info(i)
            scene = info['scene']
            if scene not in scene_seen:
                scene_seen.add(scene); indices.append(i)
            if len(indices) == 8:
                break
        ids = [dataset.get_data_info(i)['img_id'] for i in indices]
        if sample_ids is not None and ids != sample_ids:
            raise ValueError('Paired diagnostic images disagree')
        sample_ids = ids
        saved = torch.load(checkpoint, map_location='cpu')
        previous = Config.fromstring(saved['meta'].get('cfg', saved['meta'].get('config')), '.py')
        if previous.model.to_dict() != cfg.model.to_dict():
            raise ValueError('Model config/checkpoint mismatch')
        model = MODELS.build(cfg.model)
        model.load_state_dict(saved['state_dict'], strict=True)
        state_bytes = sum(t.numel()*t.element_size() for t in saved['state_dict'].values())
        del saved
        model.cuda().eval()
        torch.backends.cudnn.benchmark = cfg.get('env_cfg', {}).get('cudnn_benchmark', False)
        fixed = pseudo_collate([dataset[indices[0]]])
        times, cases = [], []
        with torch.inference_mode():
            for _ in range(5):
                batch = copy.deepcopy(fixed)
                prediction = model.test_step(batch)
                torch.cuda.synchronize()
                del prediction, batch
            gc.collect(); torch.cuda.reset_peak_memory_stats()
            for _ in range(20):
                batch = copy.deepcopy(fixed)
                torch.cuda.synchronize(); started = time.perf_counter()
                prediction = model.test_step(batch)
                torch.cuda.synchronize(); times.append(time.perf_counter()-started)
                del prediction, batch
            peak = torch.cuda.max_memory_allocated()/2**20
            for i in indices:
                sample = dataset[i]
                info = dataset.get_data_info(i)
                pred = model.test_step(pseudo_collate([sample]))[0].pred_instances.to('cpu')
                keep = pred.scores >= .3
                predicted_masks = pred.masks[keep].numpy().astype(bool)
                scores = pred.scores[keep].numpy()
                order = np.argsort(-scores, kind='stable')
                predicted_masks = predicted_masks[order]
                anns = coco.loadAnns(coco.getAnnIds(imgIds=[info['img_id']]))
                gt_rles = [coco.annToRLE(a) for a in anns]
                pred_rles = [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in predicted_masks]
                ious = (mask_utils.iou(pred_rles, gt_rles, [0]*len(anns))
                        if len(pred_rles) and len(anns) else np.zeros((len(pred_rles),len(anns))))
                used, matched_pairs = set(), []
                for pred_index, row in enumerate(ious):
                    available = [j for j in range(len(anns)) if j not in used]
                    if available:
                        j = max(available, key=lambda j: row[j])
                        if row[j] >= .5:
                            used.add(j)
                            matched_pairs.append((pred_index, j))
                small = [j for j,a in enumerate(anns) if a['area'] <= 1024]
                best = ious.max(0) if len(pred_rles) else np.zeros(len(anns))
                gt_masks = [mask_utils.decode(r).astype(bool) for r in gt_rles]
                boundary = [dict(annotation_id=anns[j]['id'], iou=float(ious[p,j]),
                    small=j in small, boundary_f1=boundary_f1(predicted_masks[p], gt_masks[j]))
                    for p,j in matched_pairs]
                cases.append(dict(image_id=info['img_id'], scene_id=info['scene'],
                    instances=len(anns), small_instances=len(small), predictions=len(pred_rles),
                    matched_iou50=len(used), unmatched_predictions=len(pred_rles)-len(used),
                    small_matched=sum(j in used for j in small),
                    small_best_iou=[float(best[j]) for j in small],
                    small_missed_annotation_ids=[anns[j]['id'] for j in small if j not in used],
                    matched_boundary_diagnostics=boundary))
                rgb = np.array(Image.open(info['img_path']).convert('RGB'))
                depth_path = Path(info['depth_file'])
                depth = np.array(Image.open(depth_path)).astype(float)*info['depth_scale_m']
                panels.setdefault(info['img_id'], dict(rgb=rgb, depth=depth, gt=overlay(rgb, gt_masks)))
                panels[info['img_id']][name+'_prediction'] = overlay(rgb, predicted_masks)
                del pred, predicted_masks
        records[name] = dict(checkpoint_sha256=report['checkpoint_sha256'],
            validation_annotations_sha256=annotation_hash, metrics=report['metrics'],
            learning_curve=report['learning_curve'],
            active_parameters=model.active_parameter_count(),
            stored_parameters=sum(p.numel() for p in model.parameters()), state_tensor_bytes=state_bytes,
            timing=dict(warmup=5,repeats=20,image_id=ids[0],input_shape=list(fixed['inputs'][0].shape),
                median_ms=statistics.median(times)*1000,mean_ms=statistics.mean(times)*1000,
                p90_ms=float(np.percentile(times,90))*1000,fps=1/statistics.mean(times),
                peak_allocated_mib=peak,all_seconds=times,
                scope='Synchronized batch1 model.test_step including preprocess/H2D/postprocess; excludes disk/decode/resize and batch cloning; FP32, no autocast; one fixed validation image'),
            cases=cases, small_instances=sum(c['small_instances'] for c in cases),
            small_matched=sum(c['small_matched'] for c in cases))
        atomic_json(output/(name+'.json'),records[name])
        print(name, json.dumps({k:records[name][k] for k in ('timing','small_instances','small_matched')}),flush=True)
        del model, dataset, coco, fixed, sample
        gc.collect(); torch.cuda.empty_cache()
    for number, (image_id, content) in enumerate(panels.items()):
        columns = [('rgb','RGB'),('depth','Depth (m)'),('gt','Ground truth')]
        columns += [(name+'_prediction', name) for name in args.models]
        fig, axes = plt.subplots(1,len(columns),figsize=(4*len(columns),4.4))
        for ax, (key, title) in zip(axes, columns):
            value = content[key]
            ax.imshow(value, cmap='viridis' if key == 'depth' else None)
            ax.set_title(title); ax.axis('off')
        fig.suptitle(f'Validation image {image_id}; prediction score >= 0.3')
        fig.tight_layout(); fig.savefig(output/f'case_{number:02d}_{image_id}.png',dpi=120); plt.close(fig)
    atomic_json(output/'summary.json',dict(status='complete',test_inputs_used=False,test_annotations_used=False,
        sample_selection='First view of first eight distinct validation scenes, before prediction',
        diagnostic_thresholds=dict(score=.3,mask_iou=.5),results=records,
        boundary_protocol='Matched IoU>=0.5 only; contour F1, Manhattan tolerance 2px, original 1024 image; not Boundary AP',
        note='Eight-image failure illustrations are not full validation AP or a confidence interval. '
             'Timing is a one-image sequential microbenchmark; stored state includes inactive modules.'))


if __name__ == '__main__':
    main()

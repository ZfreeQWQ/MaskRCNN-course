"""Validation-only, instance-addressable error candidates; human review is required."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_util
from common import load_images, read_json, write_json
from demo import depth_view, overlay

KINDS = ('missed', 'false_positive', 'boundary_candidate', 'adjacent_merge_candidate')
GROUPS = ('S', 'M', 'L')
LEGACY = {'missed': 'small_missed', 'false_positive': 'small_false_positive',
          'boundary_candidate': 'small_boundary_candidate',
          'adjacent_merge_candidate': 'adjacent_merge_candidate'}


def area_groups(area):
    """COCOeval areaRng is inclusive at both ends; boundary instances overlap bins."""
    return [name for name, low, high in [('S', 0, 1024), ('M', 1024, 9216), ('L', 9216, 1e10)]
            if low <= area <= high]


def reference(row, index=None, gt=False):
    area = float(mask_util.area(row['segmentation']))
    bbox = [float(x) for x in mask_util.toBbox(row['segmentation'])]
    fields = dict(area=area, area_groups=area_groups(area), mask_bbox=bbox,
                  bbox=[float(x) for x in row.get('bbox', bbox)])
    if gt:
        fields.update(annotation_id=row['id'], semantic_type=row.get('semantic_type'),
                      source_area=row.get('area'))
    else:
        fields.update(prediction_index=index, prediction_id=row.get('id'), score=float(row['score']))
    return fields


def select_stratified(pools, maximum):
    # Up to half small and a quarter each medium/large; fill unused quota deterministically.
    quotas = {'S': (maximum + 1) // 2, 'M': maximum // 4, 'L': maximum // 4}
    selected, seen = [], set()
    for group in GROUPS:
        for row in pools[group][:quotas[group]]:
            selected.append(row); seen.add(row['case_id'])
    for group in GROUPS:
        for row in pools[group]:
            if len(selected) >= maximum:
                return selected
            if row['case_id'] not in seen:
                selected.append(row); seen.add(row['case_id'])
    return selected


def analyze(data, predictions, threshold=0.5, max_cases=20):
    """Existing call signature/legacy small fields retained; new stratified fields cover all sizes.

    Prediction indices always refer to the ORIGINAL JSON list, before filtering/sorting.
    Counts cover every candidate; only the stored examples are capped at <=20 per kind.
    """
    if not 0 <= threshold <= 1 or not 1 <= max_cases <= 20:
        raise ValueError('threshold must be in [0,1] and max_cases in [1,20]')
    image_ids = {row['id'] for row in data['images']}
    if len(image_ids) != len(data['images']):
        raise ValueError('Duplicate image IDs')
    if any(row.get('split', 'val') != 'val' or '/test/' in row['file_name'] for row in data['images']):
        raise ValueError('Error analysis is restricted to val')
    gt_groups, pred_groups = defaultdict(list), defaultdict(list)
    for row in data['annotations']:
        if row['image_id'] not in image_ids:
            raise ValueError('GT outside validation images')
        if not row.get('iscrowd', 0):
            gt_groups[row['image_id']].append(row)
    for index, row in enumerate(predictions):
        if row['image_id'] not in image_ids:
            raise ValueError('Predictions outside validation images')
        if row.get('category_id') != 1 or not math.isfinite(float(row['score'])):
            raise ValueError('Expected finite-score component predictions')
        if row['score'] >= threshold:
            pred_groups[row['image_id']].append((index, row))
    counts = {kind: Counter({group: 0 for group in GROUPS}) for kind in KINDS}
    totals = Counter(); legacy_counts = Counter(); semantics = Counter()
    pools = {kind: {group: [] for group in GROUPS} for kind in KINDS}
    population = {'gt': Counter({g: 0 for g in GROUPS}), 'predictions_above_threshold': Counter({g: 0 for g in GROUPS})}

    def record(kind, image_id, gt=None, pred=None, original_index=None, **evidence):
        gr = reference(gt, gt=True) if gt is not None else None
        pr = reference(pred, original_index) if pred is not None else None
        area_ref = gr if kind in ('missed', 'boundary_candidate') else pr
        groups = area_ref['area_groups']; primary = groups[0]
        totals[kind] += 1
        for group in groups:
            counts[kind][group] += 1
        small = 'S' in groups
        if kind == 'adjacent_merge_candidate':
            small = any('S' in item['area_groups'] for item in evidence['covered_annotations'])
        if small:
            legacy_counts[LEGACY[kind]] += 1
        if gr:
            semantics[(kind, str(gr.get('semantic_type', 'unknown')))] += 1
        if len(pools[kind][primary]) >= max_cases:
            return
        case_id = f'{kind}_im{image_id}_gt{gr["annotation_id"] if gr else "none"}_pred{original_index if pr else "none"}'
        pools[kind][primary].append(dict(
            case_id=case_id, candidate_id=case_id, kind=kind, image_id=image_id, candidate_model='Selected', review_status='pending_human_review',
            area_basis='GT mask' if gr is not None and kind in ('missed', 'boundary_candidate') else 'prediction mask',
            area_group=primary, area_groups=groups, area=area_ref['area'],
            annotation_id=gr['annotation_id'] if gr else None,
            semantic_type=gr.get('semantic_type') if gr else None,
            gt_area=gr['area'] if gr else None, gt_bbox=gr['bbox'] if gr else None,
            prediction_index=original_index if pr else None,
            prediction_id=pr.get('prediction_id') if pr else None,
            prediction_bbox=pr['bbox'] if pr else None,
            prediction_score=pr['score'] if pr else None,
            gt=gr, prediction=pr, legacy_small_candidate=small, **evidence))

    for info in sorted(data['images'], key=lambda image: image['id']):
        image_id = info['id']
        truth = sorted(gt_groups[image_id], key=lambda row: row['id'])
        indexed_preds = pred_groups[image_id]
        preds = [row for _, row in indexed_preds]
        gt_areas = np.array([float(mask_util.area(row['segmentation'])) for row in truth])
        pred_areas = np.array([float(mask_util.area(row['segmentation'])) for row in preds])
        for area in gt_areas:
            population['gt'].update(area_groups(area))
        for area in pred_areas:
            population['predictions_above_threshold'].update(area_groups(area))
        ious = mask_util.iou([row['segmentation'] for row in preds],
                            [row['segmentation'] for row in truth], [0] * len(truth)) if preds and truth else np.zeros((len(preds), len(truth)))
        intersection = ious * (pred_areas[:, None] + gt_areas[None, :]) / (1 + ious)
        coverage = intersection / np.maximum(gt_areas[None, :], 1)
        used, matched_pred = set(), set()
        for pi in sorted(range(len(preds)), key=lambda i: (-preds[i]['score'], indexed_preds[i][0])):
            candidates = [gi for gi in range(len(truth)) if gi not in used and ious[pi, gi] >= .5]
            if candidates:
                gi = max(candidates, key=lambda i: (ious[pi, i], -truth[i]['id']))
                used.add(gi); matched_pred.add(pi)
        for gi, gt in enumerate(truth):
            pi = int(ious[:, gi].argmax()) if preds else None
            best = float(ious[pi, gi]) if pi is not None else 0.0
            evidence = dict(best_iou=best, gt_coverage=float(coverage[pi, gi]) if pi is not None else 0.0,
                            gt_matched_at_iou_05=gi in used)
            if gi not in used:
                record('missed', image_id, gt, preds[pi] if best > 0 else None,
                       indexed_preds[pi][0] if best > 0 else None, **evidence)
            if .1 <= best < .75:
                record('boundary_candidate', image_id, gt, preds[pi], indexed_preds[pi][0], **evidence)
        for pi, pred in enumerate(preds):
            gi = int(ious[pi].argmax()) if truth else None
            best = float(ious[pi, gi]) if gi is not None else 0.0
            evidence = dict(best_iou=best, best_gt_annotation_id=truth[gi]['id'] if best > 0 else None,
                            best_gt_coverage=float(coverage[pi, gi]) if gi is not None else 0.0)
            if pi not in matched_pred:
                record('false_positive', image_id, pred=pred, original_index=indexed_preds[pi][0], **evidence)
            covered = np.flatnonzero(coverage[pi] >= .5)
            if len(covered) >= 2:
                refs = [dict(reference(truth[int(index)], gt=True),
                             coverage=float(coverage[pi, index]), iou=float(ious[pi, index])) for index in covered]
                gaps = []
                for a,b in combinations(refs, 2):
                    x,y,w,h=a['mask_bbox'];xx,yy,ww,hh=b['mask_bbox']
                    gaps.append(math.hypot(max(x-xx-ww,xx-x-w,0),max(y-yy-hh,yy-y-h,0)))
                record('adjacent_merge_candidate', image_id, pred=pred, original_index=indexed_preds[pi][0],
                       covered_annotation_ids=[item['annotation_id'] for item in refs], covered_annotations=refs,
                       minimum_covered_gt_bbox_gap_px=min(gaps), **evidence)
    stratified = {kind: select_stratified(pools[kind], max_cases) for kind in KINDS}
    legacy_cases = {LEGACY[kind]: [row for row in stratified[kind] if row['legacy_small_candidate']] for kind in KINDS}
    return dict(score_threshold=threshold, diagnostic_iou_threshold=.5,
        prediction_index_convention='0-based position in original --predictions JSON, before score filtering',
        small_area_definition='COCO mask area S=[0,1024], M=[1024,9216], L=[9216,1e10]; inclusive shared boundaries',
        area_group_policy='Counts include both bins at 1024/9216; one primary (first matching S,M,L) group used for deterministic example selection.',
        counts=dict(legacy_counts), cases=legacy_cases, all_area_counts=dict(totals),
        counts_by_area={kind: dict(value) for kind,value in counts.items()},
        population_by_area={key: dict(value) for key,value in population.items()}, stratified_cases=stratified,
        selection_summary={kind: {'available_unique_candidates': int(totals[kind]),
            'stored_unique_candidates': len(stratified[kind]),
            'stored_primary_area_groups': dict(Counter(row['area_group'] for row in stratified[kind])),
            'human_confirmed_errors': None} for kind in KINDS},
        max_cases_per_kind=max_cases, selection='Deterministic image ID / GT ID / original prediction index; quotas approximately S:M:L=2:1:1, unused quotas filled S then M then L.',
        semantic_counts=[dict(kind=k, semantic_type=s, count=n) for (k,s),n in sorted(semantics.items())],
        note='Candidates only, not confirmed errors or causes. One-to-one IoU=.5 diagnostics are not COCO AP. Low IoU alone does not establish a boundary error; coverage of multiple GT masks alone does not prove merging. Depth/area/overlap fields are observations, not causal conclusions. No human review has been performed by this tool.')


def enrich_depth(cases, images, annotations, predictions, data_root):
    """Read only selected val image inputs; mask-level and surrounding-ring observations."""
    lookup = {row['id']: row for row in annotations}
    grouped = defaultdict(list)
    for rows in cases.values():
        for case in rows:
            grouped[case['image_id']].append(case)
    for image_id, rows in grouped.items():
        info = images[image_id]
        if info.get('depth_scale_m') is None or info.get('depth_invalid_code') is None or not info.get('depth_file_name'):
            for case in rows:
                case['depth_evidence'] = {'available': False, 'reason': 'Missing explicit image depth metadata'}
            continue
        with Image.open(Path(data_root) / info['depth_file_name']) as stream:
            raw = np.asarray(stream)
        if raw.dtype != np.uint16 or raw.shape != (info['height'], info['width']):
            raise ValueError('Depth must be aligned uint16 PNG')
        valid = raw != info['depth_invalid_code']; meters = raw.astype(np.float64) * info['depth_scale_m']
        for case in rows:
            basis = lookup[case['annotation_id']] if case['area_basis'] == 'GT mask' else predictions[case['prediction_index']]
            mask = mask_util.decode(basis['segmentation']).astype(bool)
            padded = np.pad(mask, 1); ring = np.zeros_like(mask)
            for dy in range(3):
                for dx in range(3):
                    ring |= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
            ring &= ~mask
            inside = meters[mask & valid]; outside = meters[ring & valid]
            case['depth_evidence'] = dict(available=True, support=case['area_basis'], mask_pixels=int(mask.sum()),
                invalid_fraction=float((mask & ~valid).sum()/max(mask.sum(),1)),
                valid_pixels=int(inside.size), ring_pixels=int(ring.sum()),
                median_m=float(np.median(inside)) if inside.size else None,
                p10_m=float(np.percentile(inside,10)) if inside.size else None,
                p90_m=float(np.percentile(inside,90)) if inside.size else None,
                ring_median_m=float(np.median(outside)) if outside.size else None,
                absolute_median_ring_difference_m=float(abs(np.median(inside)-np.median(outside))) if inside.size and outside.size else None,
                interpretation='Observed mask/ring depth statistics; does not identify occlusion or establish a cause.')


def mark(panel, boxes, labels=True):
    panel = panel.copy(); draw = ImageDraw.Draw(panel)
    for index,(box,label) in enumerate(boxes):
        x,y,w,h=box
        draw.rectangle((x,y,x+w,y+h), outline='#ff3030', width=3)
        if labels:
            text_y = max(0,y-12) if index%2 == 0 else min(panel.height-12,y+h+3)
            draw.text((max(0,x),text_y),label,fill='#ff3030',stroke_width=1,stroke_fill='white')
    return panel


def render_case(case, info, data_root, gt, sources, output, threshold):
    """An instance sheet has full image and same-window enlarged ROI in every column."""
    with Image.open(Path(data_root)/info['file_name']) as stream: rgb=stream.convert('RGB')
    if info.get('depth_scale_m') is None or info.get('depth_invalid_code') is None:
        depth=Image.new('RGB',rgb.size,'black'); caption='Depth metadata missing'
    else: depth,caption=depth_view(Path(data_root)/info['depth_file_name'], info)
    boxes=[]
    if case['gt'] is not None: boxes.append((case['gt']['mask_bbox'],f'GT {case["annotation_id"]}'))
    if case['prediction'] is not None: boxes.append((case['prediction']['mask_bbox'],f'Selected pred[{case["prediction_index"]}]'))
    for row in case.get('covered_annotations',[]):boxes.append((row['mask_bbox'],f'GT {row["annotation_id"]}'))
    panels=[('RGB',rgb),('Depth (m)',depth),('GT (val only)',overlay(rgb,gt))]
    for label, rows in sources:
        panels.append((f'{label}; score >= {threshold:g}' + (' [NOT PROVIDED]' if rows is None else ''), overlay(rgb,rows,threshold) if rows is not None else Image.new('RGB',rgb.size,'#dddddd')))
    x1=min(box[0][0] for box in boxes);y1=min(box[0][1] for box in boxes)
    x2=max(box[0][0]+box[0][2] for box in boxes);y2=max(box[0][1]+box[0][3] for box in boxes)
    margin=max(16,.15*max(x2-x1,y2-y1));crop=(max(0,int(x1-margin)),max(0,int(y1-margin)),min(rgb.width,int(math.ceil(x2+margin))),min(rgb.height,int(math.ceil(y2+margin))))
    tile=384; header=96;sheet=Image.new('RGB',(tile*len(panels),header+tile*2+24),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,4),f'{case["case_id"]} | area {case["area_group"]}={case["area"]:g}px | IoU={case["best_iou"]:.3f} | PENDING HUMAN REVIEW',fill='black')
    evidence=case.get('depth_evidence',{});draw.text((8,22),f'Depth invalid fraction={evidence.get("invalid_fraction", "unavailable")} | ROI={crop}; red boxes identify this candidate',fill='black')
    draw.text((8,40),'Diagnostic candidate labels are not established causes. Missing model columns remain explicitly unavailable.',fill='black')
    for column,(label,panel) in enumerate(panels):
        draw.text((column*tile+6,64),label,fill='black')
        marked=mark(panel,boxes);sheet.paste(marked.resize((tile,tile)),(column*tile,header))
        # Crop the clean panel, then add fixed-width ROI outlines. Never magnify text.
        roi=panel.crop(crop);scale=tile/max(roi.size)
        roi=roi.resize((max(1,round(roi.width*scale)),max(1,round(roi.height*scale))),Image.Resampling.NEAREST)
        sx,sy=roi.width/(crop[2]-crop[0]),roi.height/(crop[3]-crop[1])
        roi_boxes=[([(box[0]-crop[0])*sx,(box[1]-crop[1])*sy,box[2]*sx,box[3]*sy],label) for box,label in boxes]
        roi=mark(roi,roi_boxes,labels=False)
        sheet.paste(roi,(column*tile+(tile-roi.width)//2,header+tile+24+(tile-roi.height)//2))
    Path(output).parent.mkdir(parents=True,exist_ok=True);sheet.save(output)
    return list(crop)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True,help='Validation COCO only')
    p.add_argument('--predictions',required=True,help='Selected model predictions; never implicitly labelled M1')
    p.add_argument('--baseline-predictions')
    p.add_argument('--m1-predictions',help='Explicit M1 predictions for required B0/M1 comparison')
    p.add_argument('--data-root',default='/private/rgzndzy/blender')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--max-cases',type=int,default=20)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--render',action='store_true',help='Render per-instance RGB/depth/GT/B0/M1/Selected full+ROI sheets')
    args=p.parse_args()
    annotation_path=Path(args.annotations)
    if 'test' in annotation_path.parts or annotation_path.name.startswith(('instances_test','test_')):
        p.error('Test annotation paths are forbidden')
    images=load_images(args.annotations,'val');data=read_json(args.annotations);predictions=read_json(args.predictions)
    result=analyze(data,predictions,args.threshold,args.max_cases)
    result['sources']={'annotations':args.annotations,'selected_predictions':args.predictions,
                       'B0_predictions':args.baseline_predictions,'M1_predictions':args.m1_predictions}
    image_lookup={row['id']:row for row in images}
    enrich_depth(result['stratified_cases'],image_lookup,data['annotations'],predictions,args.data_root)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    text=['# Validation instance error candidates','',result['note'],'',
          f'Diagnostic/display score threshold: {args.threshold}. This does not alter AP or submission thresholds.',
          '','Counts use inclusive COCO area bins; exact 1024/9216 areas appear in both neighboring bins.',
          '','| Candidate | S | M | L | All unique | Stored |','|---|---:|---:|---:|---:|---:|']
    for kind in KINDS:
        c=result['counts_by_area'][kind];text.append(f'| {kind} | {c["S"]} | {c["M"]} | {c["L"]} | {result["all_area_counts"].get(kind,0)} | {len(result["stratified_cases"][kind])} |')
    text+=['','Each kind stores at most 20 distinct instances, with S/M/L selection quotas and small-target priority.','',
           'Review status is pending. Reviewers must distinguish observations, hypotheses, counterexamples and actual ablation support; insufficient cases must be disclosed.','',
           '| Case | Area/basis | GT ID | Original selected prediction index | Evidence | Sheet |','|---|---|---|---|---|---|']
    groups=defaultdict(list)
    for row in data['annotations']:groups[row['image_id']].append(row)
    sources=[]
    if args.render:
        for label,path in [('B0',args.baseline_predictions),('M1',args.m1_predictions),('Selected',args.predictions)]:
            by_image=defaultdict(list)
            if path:
                rows=predictions if label=='Selected' else read_json(path)
                for index,row in enumerate(rows):
                    if row['image_id'] not in image_lookup:raise ValueError(f'{label} prediction outside val')
                    if row['score']>=args.threshold:by_image[row['image_id']].append(row)
            sources.append((label,by_image if path else None))
    for kind in KINDS:
        for case in result['stratified_cases'][kind]:
            filename=case['case_id']+'.png'
            if args.render:
                local_sources=[(label,rows[case['image_id']] if rows is not None else None) for label,rows in sources]
                case['roi_xyxy']=render_case(case,image_lookup[case['image_id']],args.data_root,
                    groups[case['image_id']],local_sources,out/filename,args.threshold)
                case['sheet']=filename
            ev=case.get('depth_evidence',{})
            evidence=f'IoU={case["best_iou"]:.3f}; invalid={ev.get("invalid_fraction","unavailable")}'
            link=f'[{filename}]({filename})' if args.render else 'not rendered'
            text.append(f'| {case["case_id"]} | {case["area_group"]} / {case["area_basis"]} | {case["annotation_id"]} | {case["prediction_index"]} | {evidence} | {link} |')
    write_json(out/'error_candidates.json',result)
    write_json(out/'human_review_template.json',{'status':'pending_human_review','instruction':'Populate only after inspecting the per-instance sheets; candidate tags alone are not confirmed errors or causes.',
        'cases':[{'case_id':row['case_id'],'confirmed_error':None,'reviewer':None,'observations':[],
                  'cause_hypotheses':[],'counterexamples_or_limits':[],'ablation_support':[], 'review_status':'pending_human_review'}
                 for rows in result['stratified_cases'].values() for row in rows]})
    (out/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':main()

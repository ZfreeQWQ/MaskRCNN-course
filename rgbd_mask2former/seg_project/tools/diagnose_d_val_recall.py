"""Full-val candidate coverage diagnostics; best-IoU coverage is NOT COCO AP/AR.

One prediction may cover multiple GTs in this diagnostic. No one-to-one matching,
no test annotations and no model/score modifications are involved.
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_util

root=Path('/private/rgzndzy/seg_project')
gt=json.loads((root/'datasets/blender/annotations/instances_val.json').read_text())
if any(im.get('split')!='val' for im in gt['images']):
    raise ValueError('Only explicitly labelled val data is permitted')
pred=json.loads((root/'experiments/mask2former/d_screen/best_val_predictions.json').read_text())
gts=defaultdict(list)
dts=defaultdict(list)
for row in gt['annotations']:
    if not row.get('iscrowd',0):
        gts[row['image_id']].append(row)
for row in pred: dts[row['image_id']].append(row)
groups=defaultdict(list)
score_groups=defaultdict(list)
examples=[]
for im in gt['images']:
    anns=gts[im['id']]
    detections=dts[im['id']]
    if not anns: continue
    matrix=mask_util.iou([r['segmentation'] for r in detections],
                         [r['segmentation'] for r in anns],[0]*len(anns)) if detections else np.zeros((0,len(anns)))
    best=matrix.max(axis=0) if detections else np.zeros(len(anns))
    for index,(ann,iou) in enumerate(zip(anns,best)):
        area=float(mask_util.area(ann['segmentation']))
        labels=['all']
        if area<=1024:
            labels+=['small','tiny_le256' if area<=256 else 'small_256_to1024']
            width,height=ann['bbox'][2:]
            fill=area/max(width*height,1)
            labels+=['small_fill_lt025' if fill<.25 else 'small_fill_ge025']
            if len(anns)>=60: labels+=['small_dense_ge60']
            if iou<.5 and len(examples)<20:
                examples.append(dict(image_id=im['id'],annotation_id=ann['id'],area=area,best_iou=float(iou)))
        elif area<=9216: labels+=['medium']
        else: labels+=['large']
        for label in labels:
            groups[label].append(float(iou))
            # Diagnostic score of the highest-IoU candidate; not an AP-calibrated target.
            if detections and iou>=.5:
                score_groups[label].append(float(detections[int(np.argmax(matrix[:,index]))]['score']))
results={}
for label,values in groups.items():
    values=np.asarray(values)
    scores=np.asarray(score_groups[label])
    results[label]=dict(num_gt=len(values),best_iou_mean=float(values.mean()),
                       coverage_at_iou={str(t):float((values>=t).mean()) for t in [.1,.5,.75,.9]},
                       share_without_iou05_candidate=float((values<.5).mean()),
                       best_candidate_score_quantiles_given_iou05=np.quantile(scores,[.1,.5,.9]).tolist() if len(scores) else [])
report=dict(split='val',num_images=len(gt['images']),num_predictions=len(pred),groups=results,
            definition='Fraction of GT masks with any one of the saved <=100 predictions reaching each IoU. Not COCO AR: no one-to-one or score ordering.',
            small_miss_examples=examples)
out=root/'reports/analysis/d_val_candidate_coverage_20261025.json'
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

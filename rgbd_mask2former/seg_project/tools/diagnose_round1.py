"""Explain the selected validation result using GT-size candidate coverage (not AP)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_util
from run_round1 import ROOT,ANNOTATIONS
from common import read_json,write_json


def coverage(predictions,gt):
    gts=defaultdict(list); dts=defaultdict(list); groups=defaultdict(list)
    for row in gt['annotations']:
        if not row.get('iscrowd',0): gts[row['image_id']].append(row)
    for row in read_json(predictions): dts[row['image_id']].append(row)
    for image in gt['images']:
        anns=gts[image['id']]; detections=dts[image['id']]
        if not anns: continue
        matrix=mask_util.iou([d['segmentation'] for d in detections],
                             [g['segmentation'] for g in anns],[0]*len(anns)) if detections else None
        best=matrix.max(axis=0) if detections else np.zeros(len(anns))
        for ann,iou in zip(anns,best):
            area=float(mask_util.area(ann['segmentation']))
            if area>1024: continue
            names=['small','tiny_le256' if area<=256 else 'small_256_to1024']
            width,height=ann['bbox'][2:]
            names.append('small_fill_lt025' if area/max(width*height,1)<.25 else 'small_fill_ge025')
            for name in names: groups[name].append(float(iou))
    return {name:dict(num_gt=len(values),mean_best_iou=float(np.mean(values)),
                     coverage_at_iou={str(t):float(np.mean(np.array(values)>=t)) for t in [.5,.75,.9]})
            for name,values in groups.items()}


if __name__=='__main__':
    report=read_json(ROOT/'val_results.json')
    gt=read_json(ANNOTATIONS)
    if any(im.get('split')!='val' for im in gt['images']): raise ValueError('Only val data permitted')
    names=list(dict.fromkeys(['d1024_legacy',report['selected']]))
    output=dict(split='val',num_images=len(gt['images']),selected=report['selected'],
                definition='GT fraction with any saved prediction reaching each IoU; no one-to-one matching, not COCO AP/AR.',
                results={name:coverage(ROOT/'merged_val'/(name+'.json'),gt) for name in names})
    write_json(ROOT/'candidate_coverage.json',output)
    print(json.dumps(output,indent=2))

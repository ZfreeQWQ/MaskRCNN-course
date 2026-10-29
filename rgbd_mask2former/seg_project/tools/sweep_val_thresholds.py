"""Compare score cutoffs with independent official COCO mask evaluations on full val.

Only val annotations are accepted. Report units are 0..1. The unfiltered baseline
checks consistency with the checkpoint's recorded validation metrics.
"""
import argparse
import contextlib
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def evaluate_one(job):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    threshold, annotations, predictions, output = job
    started = time.time()
    log = Path(output)/f'val_t{threshold:g}.log'
    with log.open('w', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        gt = COCO(annotations)
        if any(im.get('split') != 'val' for im in gt.dataset['images']):
            raise ValueError('Only an explicitly labelled full val split may be used')
        gt.dataset.setdefault('info', {})
        rows = json.loads(Path(predictions).read_text())
        ids=set(gt.getImgIds())
        if any(row['image_id'] not in ids for row in rows):
            raise ValueError('Predictions contain non-val image ids')
        chosen = [{key:value for key,value in row.items() if key!='bbox'}
                  for row in rows if row['score']>=threshold]
        if not chosen:
            raise ValueError('No predictions survive this threshold')
        dt = gt.loadRes(chosen)
        evaluator=COCOeval(gt,dt,'segm')
        evaluator.params.imgIds=sorted(ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
    result=dict(threshold=threshold, num_images=len(ids),num_instances=len(chosen),
                images_with_predictions=len({row['image_id'] for row in chosen}),
                metrics={name:float(value) if value>=0 else None for name,value in zip(names,evaluator.stats)},
                wall_seconds=time.time()-started, log=str(log))
    Path(output,f'val_t{threshold:g}.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--checkpoint-metrics',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--thresholds',nargs='+',type=float,default=[0,.05,.1,.3,.5])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    thresholds=sorted(set(args.thresholds))
    if 0 not in thresholds or any(not 0<=t<=1 for t in thresholds):
        parser.error('Thresholds must include the unfiltered 0 baseline and lie in [0,1]')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'val_thresholds.json').exists():
        parser.error('Results already exist; select a new output directory')
    rows=[]
    jobs=[(t,args.annotations,args.predictions,str(output)) for t in thresholds]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(evaluate_one,job) for job in jobs]):
            result=future.result()
            rows.append(result)
            print(json.dumps(result),flush=True)
    rows.sort(key=lambda r:r['threshold'])
    expected=json.loads(Path(args.checkpoint_metrics).read_text())['metrics']['segm']
    baseline=next(row for row in rows if row['threshold']==0)
    differences={metric:abs(baseline['metrics'][metric]-expected[metric]/100)
                 for metric in ['AP','AP50','AP75','APs','APm','APl']}
    if max(differences.values())>1e-8:
        raise ValueError(f'Unfiltered val does not reproduce checkpoint metrics: {differences}')
    best=max(rows,key=lambda row:(row['metrics']['APs'],row['metrics']['AP'],-row['threshold']))
    report=dict(split='val',num_images=baseline['num_images'],units='0-1',iou_type='segm',
                small_area='COCO mask area <= 32^2, not bbox area',max_detections=100,
                annotations=args.annotations,predictions=args.predictions,
                predictions_sha256=hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
                checkpoint_metrics=args.checkpoint_metrics,
                baseline_matches_checkpoint=True,baseline_differences=differences,
                selected_threshold=best['threshold'],results=rows,
                selection='Highest full-val Mask APs; tie break overall Mask AP. No test labels used.')
    (output/'val_thresholds.json').write_text(json.dumps(report,indent=2))
    lines=['# D model validation score thresholds','',
           'Full validation set; official COCO mask AP, units 0–1. No test labels used.','',
           '| threshold | instances | Mask APs | Mask AP | small AR |',
           '|---:|---:|---:|---:|---:|']
    for row in rows:
        metrics=row['metrics']
        lines.append(f"| {row['threshold']:g} | {row['num_instances']} | {metrics['APs']:.6f} | {metrics['AP']:.6f} | {metrics['ARs']:.6f} |")
    lines+=['',f"Selected threshold: **{best['threshold']:g}**. Unfiltered baseline reproduces checkpoint metrics."]
    (output/'val_thresholds.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SELECTED_THRESHOLD',best['threshold'],flush=True)


if __name__=='__main__':
    main()

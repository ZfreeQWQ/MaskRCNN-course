"""Export and fully validate test submissions for the completed val threshold sweep.

All cutoffs share one unfiltered test inference. Each output gets a matching
filtered COCO JSON, manifest and full RLE/CSV validation report. No test labels.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import load_images, read_json, write_csv, write_json
from validate_submission import validate


def export_one(job):
    threshold, metadata, predictions, manifest_path, template, output = job
    started=time.time()
    folder=Path(output)/f't{threshold:g}'
    folder.mkdir(parents=True,exist_ok=True)
    csv_path=folder/f'submission_t{threshold:g}.csv'
    if csv_path.exists():
        raise ValueError(f'Output already exists: {csv_path}')
    images=load_images(metadata,'test')
    by_id={im['id']:im for im in images}
    with open(template,newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['image_id','PredictionString']:
            raise ValueError('Unexpected template header')
        ids=[int(row['image_id']) for row in reader]
    if len(ids)!=len(set(ids)) or set(ids)!=set(by_id):
        raise ValueError('Template coverage mismatch')
    manifest=read_json(manifest_path)
    if manifest.get('score_threshold')!=0 or manifest.get('smoke') or manifest.get('split')!='test':
        raise ValueError('Master predictions must be complete unfiltered test inference')
    chosen=[row for row in read_json(predictions) if row['score']>=threshold]
    filtered=folder/'predictions.json'
    write_json(filtered,chosen)
    manifest.update(score_threshold=threshold,num_instances=len(chosen),
                    source_predictions=predictions,source_manifest=manifest_path)
    output_manifest=str(filtered)+'.manifest.json'
    write_json(output_manifest,manifest)
    temporary=csv_path.with_suffix('.csv.partial')
    write_csv(temporary,[by_id[image_id] for image_id in ids],chosen)
    report=validate(metadata,filtered,temporary,manifest=output_manifest)
    # Confirm the CSV can also be consumed by the common Kaggle/pandas reader.
    import pandas as pd
    frame=pd.read_csv(temporary,dtype={'image_id':str,'PredictionString':str})
    if frame.columns.tolist()!=['image_id','PredictionString'] or frame.isna().any().any():
        raise ValueError('CSV contains missing values or unexpected columns')
    if frame.image_id.tolist()!=[str(image_id) for image_id in ids]:
        raise ValueError('CSV order differs from template')
    os.rename(temporary,csv_path)
    report.update(threshold=threshold,csv=str(csv_path),source_predictions=str(filtered),
                  sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  bytes=csv_path.stat().st_size,null_counts=frame.isna().sum().to_dict(),
                  template_order_matches=True,wall_seconds=time.time()-started)
    write_json(str(csv_path)+'.validation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--template',required=True)
    parser.add_argument('--val-report',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    val=read_json(args.val_report)
    if val.get('split')!='val' or not val.get('baseline_matches_checkpoint'):
        raise ValueError('A verified full-val threshold report is required')
    thresholds=[row['threshold'] for row in val['results']]
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'submissions.json').exists():
        raise ValueError('Final submission report already exists')
    jobs=[(t,args.metadata,args.predictions,args.manifest,args.template,str(output)) for t in thresholds]
    reports=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as pool:
        for future in as_completed([pool.submit(export_one,job) for job in jobs]):
            report=future.result()
            reports.append(report)
            print(json.dumps(report),flush=True)
    reports.sort(key=lambda row:row['threshold'])
    recommended=next(row for row in reports if row['threshold']==val['selected_threshold'])
    result=dict(selected_on='full val Mask APs; no test labels',
                recommended_threshold=val['selected_threshold'],recommended_csv=recommended['csv'],
                val_report=args.val_report,master_predictions=args.predictions,
                submissions=reports)
    write_json(output/'submissions.json',result)
    print('RECOMMENDED',recommended['csv'],flush=True)


if __name__=='__main__':
    main()

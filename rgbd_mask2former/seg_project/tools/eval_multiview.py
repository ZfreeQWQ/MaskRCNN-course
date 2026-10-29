#!/usr/bin/env python
"""Four independent GPU shards; val-only design/confirmation, metadata-only test."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import load_images, image_record, require_depth_metadata, write_json, read_json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'blender'
ANNOTATIONS=ROOT/'datasets/blender/annotations/instances_val.json'
TEST_META=ROOT/'datasets/blender/annotations/test_images_depth.json'
VARIANTS=('full2048','tiles090','tiles100')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def pilot_ids(dataset):
    scenes=sorted({r['scene_id'] for r in dataset['images']},
                  key=lambda s:hashlib.sha256(('multiview-design-20261028:'+s).encode()).hexdigest())[:64]
    return {r['id'] for r in dataset['images'] if r['scene_id'] in scenes}


def shard(args):
    import train_m2f
    import torch
    from detectron2.modeling import build_model
    from rgbd_seg.m2f.generalization import setup_cfg
    import rgbd_seg.m2f.frequency_fusion
    from rgbd_seg.m2f.checkpoint import M2FCheckpointer
    from rgbd_seg.m2f.multiview_inference import MultiViewPredictor
    class StrictCheckpointer(M2FCheckpointer):
        def _load_model(self,checkpoint):
            incompatible=super()._load_model(checkpoint)
            if incompatible.missing_keys or incompatible.unexpected_keys or incompatible.incorrect_shapes:
                raise RuntimeError('Checkpoint mismatch: '+str(incompatible))
            return incompatible
    out=Path(args.output); source=Path(args.source);meta=read_json(out/'experiment.json')
    weights=source/'best_segm_APs.pth'
    if digest(weights)!=meta['weights_sha256']:
        raise RuntimeError('Selected checkpoint changed')
    cfg=setup_cfg(source/'config.yaml')
    if cfg.INPUT.TEST_IMAGE_SIZE!=1536 or cfg.PROJECT.HIRES_MASK.ENABLED:
        raise RuntimeError('Expected selected mask-first 1536 no-H model')
    torch.set_num_threads(cfg.PROJECT.CPU_THREADS)
    model=build_model(cfg).eval();StrictCheckpointer(model).load(str(weights))
    predictor=MultiViewPredictor(cfg,model)
    images=load_images(out/'images.json','test' if args.phase=='test' else 'val')[args.shard::4]
    require_depth_metadata(images)
    folder=out/f'shard{args.shard}';folder.mkdir(exist_ok=True)
    names=VARIANTS if args.variant=='all' else (args.variant,)
    results={name:[] for name in names}; records=[];started=time.time()
    with torch.inference_mode():
        for index,info in enumerate(images,1):
            outputs,diag=predictor(image_record(info,DATA),args.variant,verify_reference=args.phase=='smoke')
            for name,rows in outputs.items():
                if len(rows)>100 or any(r['segmentation']['size']!=[info['height'],info['width']] for r in rows):
                    raise RuntimeError('Invalid count or native-coordinate mask size')
                results[name].extend(rows)
            records.append(diag)
            if index%10==0 or index==len(images):
                write_json(folder/'progress.json',dict(processed=index,total=len(images),elapsed_seconds=time.time()-started))
                print(f'{index}/{len(images)}',flush=True)
    for name,rows in results.items():
        prediction=folder/(name+'.json');write_json(prediction,rows)
        manifest=dict(split='test' if args.phase=='test' else 'val',metadata=str(out/'images.json'),
            image_ids=[r['id'] for r in images],num_images=len(images),num_instances=len(rows),
            weights=str(weights),weights_sha256=meta['weights_sha256'],config=str(source/'config.yaml'),
            score_threshold=0,tta=False,tiling=name.startswith('tiles'),variant=name,
            size=2048 if name=='full2048' else 1536,smoke=args.phase=='smoke',
            num_shards=4,shard_index=args.shard,strict_checkpoint_load=True,
            wall_seconds=time.time()-started)
        write_json(str(prediction)+'.manifest.json',manifest)
    write_json(folder/'complete.json',dict(valid=True,images=len(images),records=records,
               peak_gpu_bytes=torch.cuda.max_memory_allocated(),phase=args.phase))


def annotation_subset(dataset,ids,path):
    obj={k:v for k,v in dataset.items() if k not in ('images','annotations')}
    obj['images']=[r for r in dataset['images'] if r['id'] in ids]
    obj['annotations']=[r for r in dataset['annotations'] if r['image_id'] in ids]
    write_json(path,obj)


def score(annotations,predictions,ids,folder):
    from report_small_generalization import report
    folder.mkdir(exist_ok=False,parents=True)
    annotation_subset(annotations,ids,folder/'instances_val.json')
    write_json(folder/'predictions.json',[r for r in predictions if r['image_id'] in ids])
    result=report(folder/'instances_val.json',folder/'predictions.json')
    write_json(folder/'metrics.json',result)
    return result


def manager(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source)
    if args.phase=='test':
        if args.variant=='all' or not args.confirmation:
            raise ValueError('Test export requires a single fully confirmed inference variant')
        confirmation=read_json(args.confirmation)
        if not confirmation.get('full_confirmed') or confirmation['selected_variant']!=args.variant:
            raise ValueError('Full/held-out validation did not confirm this variant')
        if confirmation['weights_sha256']!=digest(source/'best_segm_APs.pth'):
            raise ValueError('Confirmation weights changed')
        images=load_images(TEST_META,'test');dataset=None
    else:
        dataset=read_json(ANNOTATIONS); design=pilot_ids(dataset)
        images=load_images(ANNOTATIONS,'val')
        if args.phase=='pilot':
            images=[r for r in images if r['id'] in design]
        elif args.phase=='smoke':
            images=[r for r in images if r['id'] in design][:4]
    require_depth_metadata(images)
    meta=dict(phase=args.phase,variant=args.variant,source=str(source),
        weights_sha256=digest(source/'best_segm_APs.pth'),config_sha256=digest(source/'config.yaml'),
        images=len(images),pilot_rule='64 scenes ordered by fixed SHA256; all views per scene',
        test_annotations_accessed=False)
    write_json(out/'experiment.json',meta);write_json(out/'images.json',{'images':images})
    def launch(index):
        folder=out/f'shard{index}';folder.mkdir()
        cmd=[str(ROOT/'run.sh'),'python','-u',str(Path(__file__).resolve()),'--source',str(source),
             '--output',str(out),'--phase',args.phase,'--variant',args.variant,'--shard',str(index)]
        with (folder/'job.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT.parent,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index)),
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write_json(folder/'launch.json',dict(pid=proc.pid,command=cmd))
            code=proc.wait()
        if code or not (folder/'complete.json').exists():
            raise RuntimeError(f'Shard {index} failed with code {code}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch,range(4)))
    completed=[read_json(out/f'shard{i}/complete.json') for i in range(4)]
    if sum(r['images'] for r in completed)!=len(images):
        raise RuntimeError('Missing validation images')
    if args.phase=='smoke':
        if not all(r['full_view_matches_existing_inference'] for s in completed for r in s['records']):
            raise RuntimeError('Full-view baseline path not reproduced')
        write_json(out/'summary.json',dict(meta,valid=True,subset_used_for_selection=False,
             four_gpu_reference_check=True,max_peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in completed)))
        return
    names=VARIANTS if args.variant=='all' else (args.variant,)
    predictions={name:sum([read_json(out/f'shard{i}/{name}.json') for i in range(4)],[]) for name in names}
    if args.phase=='test':
        prediction=out/'predictions.json'
        subprocess.run([str(ROOT/'run.sh'),'python',str(ROOT/'tools/merge_predictions.py'),
            '--metadata',str(TEST_META),'--split','test','--shards',
            *[str(out/f'shard{i}/{args.variant}.json') for i in range(4)],'--output',str(prediction)],check=True)
        csv=out/'submission.csv'
        subprocess.run([str(ROOT/'run.sh'),'python',str(ROOT/'tools/export_submission.py'),
            '--metadata',str(TEST_META),'--predictions',str(prediction),'--manifest',str(prediction)+'.manifest.json',
            '--template',str(DATA/'submission.csv'),'--output',str(csv)],check=True)
        validation=read_json(str(csv)+'.validation.json')
        if not validation['valid'] or not validation['submission_ready'] or validation['csv_rows']!=2112:
            raise RuntimeError('Submission validation failed')
        write_json(out/'summary.json',dict(meta,valid=True,submission=str(csv),sha256=digest(csv),
            validation=validation,hidden_test_score=None))
        return
    ids={r['id'] for r in images}; baseline=read_json(source/'best_val_predictions.json')
    base=score(dataset,baseline,ids,out/'scores/baseline');results={}
    for name,rows in predictions.items():
        measured=score(dataset,rows,ids,out/'scores'/name)
        results[name]=dict(APs=measured['APs'],AP=measured['AP'],
                          delta_APs=measured['APs']-base['APs'],delta_AP=measured['AP']-base['AP'])
    summary=dict(meta,valid=True,baseline=dict(APs=base['APs'],AP=base['AP']),results=results)
    if args.phase=='full':
        name=args.variant; holdout=ids-pilot_ids(dataset)
        before=score(dataset,baseline,holdout,out/'holdout/baseline')
        after=score(dataset,predictions[name],holdout,out/'holdout'/name)
        delta=after['APs']-before['APs']
        summary.update(selected_variant=name,holdout_images=len(holdout),holdout_gain_APs=delta,
            full_confirmed=results[name]['delta_APs']>=1. and results[name]['delta_AP']>=-.3 and
                           delta>=.8 and after['AP']>=before['AP']-.3)
    write_json(out/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=('smoke','pilot','full','test'),required=True)
    p.add_argument('--variant',choices=('all',)+VARIANTS,default='all');p.add_argument('--shard',type=int,choices=range(4))
    p.add_argument('--confirmation');args=p.parse_args()
    if args.phase in ('full','test') and args.variant=='all':
        p.error('Full/test phase needs the preselected variant')
    if args.shard is None: manager(args)
    else: shard(args)

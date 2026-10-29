"""Persistent, restartable paired full-val experiments using the installed runtime."""
import argparse
import collections
import concurrent.futures as futures
import contextlib
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from common import load_images,read_json,write_json

ROOT=Path('seg_project/experiments/mask2former/optimization_round1_20261026')
ANNOTATIONS='seg_project/datasets/blender/annotations/instances_val.json'
TEST_METADATA='seg_project/datasets/blender/annotations/test_images_depth.json'
SETTINGS=[dict(name='d1024',model='d',size=1024,refine=False),
          dict(name='d1280',model='d',size=1280,refine=False),
          dict(name='d1536',model='d',size=1536,refine=False),
          dict(name='dsh1024',model='dsh',size=1024,refine=True)]
SHARDS=8


def merge(shards,metadata,split,output):
    expected={im['id'] for im in load_images(metadata,split)}
    covered=set(); rows=[]; manifests=[]
    for source in shards:
        m=read_json(str(source)+'.manifest.json')
        ids=set(m['image_ids'])
        if m['split']!=split or m['smoke'] or len(ids)!=len(m['image_ids']) or covered&ids or not ids<=expected:
            raise ValueError(f'Invalid shard coverage: {source}')
        if m['num_shards']!=len(shards) or m['shard_index']!=len(manifests):
            raise ValueError('Unexpected shard index/count')
        if manifests:
            for key in ('metadata','weights','config','score_threshold','tta','tiling','options','variant','code_sha256'):
                if m[key]!=manifests[0][key]: raise ValueError(f'Incompatible shard: {key}')
        data=read_json(source)
        if len(data)!=m['num_instances'] or any(r['image_id'] not in ids for r in data):
            raise ValueError('Prediction coverage/count disagrees with manifest')
        if max(collections.Counter(r['image_id'] for r in data).values(),default=0)>100:
            raise ValueError('Instance cap exceeded')
        rows.extend(data); covered|=ids; manifests.append(m)
    if covered!=expected: raise ValueError('Incomplete image coverage')
    rows.sort(key=lambda r:(r['image_id'],-r['score']))
    write_json(output,rows)
    m=dict(manifests[0],image_ids=sorted(covered),num_images=len(covered),num_instances=len(rows),
           num_shards=1,shard_index=0,shards=[str(p) for p in shards],
           wall_seconds=sum(m['wall_seconds'] for m in manifests),
           peak_allocated_MiB=max(m['peak_allocated_MiB'] for m in manifests))
    write_json(str(output)+'.manifest.json',m)
    return m


def infer_job(setting,split,shard,gpu):
    out=ROOT/split/setting['name']/f'shard{shard}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        print(f'REUSE {split}/{setting["name"]}/{shard}',flush=True)
        return
    command=['./seg_project/run.sh','python','-u','seg_project/tools/infer_round1.py',
             '--config',f'seg_project/experiments/mask2former/{setting["model"]}_screen/config.yaml',
             '--weights',f'seg_project/experiments/mask2former/{setting["model"]}_screen/best_segm_APs.pth',
             '--metadata',ANNOTATIONS if split=='val' else TEST_METADATA,'--split',split,
             '--size',str(setting['size']),'--output-dir',str(out),'--num-shards',str(SHARDS),
             '--shard-index',str(shard)]
    if setting['refine']: command.append('--refine-small')
    with (out/'job.log').open('w') as log:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        write_json(out/'launch.json',dict(pid=process.pid,gpu=gpu,command=command,started=time.time()))
        print(f'START {split}/{setting["name"]}/{shard} gpu={gpu} pid={process.pid}',flush=True)
        code=process.wait()
    (out/'exit_code').write_text(str(code))
    if code or not (out/'complete.json').exists():
        raise RuntimeError(f'Inference failed code={code}: {out}/job.log')
    print(f'DONE {split}/{setting["name"]}/{shard}',flush=True)


def infer_all(settings,split):
    gpu_ids=[int(value) for value in os.environ.get('ROUND1_GPUS','0,1,2,3').split(',')]
    if not gpu_ids or len(set(gpu_ids))!=len(gpu_ids) or any(g<0 or g>3 for g in gpu_ids):
        raise ValueError('ROUND1_GPUS must contain distinct physical GPU indices from 0 to 3')
    jobs=queue.Queue()
    for setting in settings:
        for shard in range(SHARDS): jobs.put((setting,shard))
    stopped=threading.Event()
    def worker(gpu):
        while not stopped.is_set():
            try: setting,shard=jobs.get_nowait()
            except queue.Empty: return
            try: infer_job(setting,split,shard,gpu)
            except BaseException:
                stopped.set()
                raise
    # Two independent processes per GPU overlap RGB-D preparation / RLE encoding
    # with GPU work. Smoke peaks <=7.1 GiB; the server has 32 GiB per device.
    print('INFERENCE_GPUS '+str(gpu_ids),flush=True)
    with futures.ThreadPoolExecutor(max_workers=2*len(gpu_ids)) as pool:
        pending=[pool.submit(worker,gpu) for gpu in gpu_ids for _ in range(2)]
        for future in futures.as_completed(pending): future.result()


def evaluate(job):
    name,predictions,manifest,output=job
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    started=time.time()
    output=Path(output)
    if output.exists(): return read_json(output)
    with output.with_suffix('.log').open('w') as log, contextlib.redirect_stdout(log):
        gt=COCO(ANNOTATIONS)
        if any(im.get('split')!='val' for im in gt.dataset['images']): raise ValueError('Not val')
        gt.dataset.setdefault('info',{})
        if manifest['split']!='val' or manifest['smoke'] or set(manifest['image_ids'])!=set(gt.getImgIds()):
            raise ValueError('Full validation coverage required')
        # Without bbox, COCO.loadRes uses segmentation area for small-object AP.
        rows=[{k:v for k,v in r.items() if k!='bbox'} for r in read_json(predictions)]
        dt=gt.loadRes(rows)
        evaluator=COCOeval(gt,dt,'segm')
        evaluator.params.imgIds=sorted(gt.getImgIds())
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    names=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
    result=dict(name=name,predictions=str(predictions),num_images=manifest['num_images'],
                num_instances=len(rows),metrics={k:float(v) for k,v in zip(names,evaluator.stats)},
                options=manifest['options'],variant=manifest['variant'],
                peak_allocated_MiB=manifest['peak_allocated_MiB'],
                inference_seconds=manifest['wall_seconds'],evaluation_seconds=time.time()-started,
                predictions_sha256=hashlib.sha256(Path(predictions).read_bytes()).hexdigest())
    write_json(output,result)
    print(f'EVAL {name} APs={result["metrics"]["APs"]:.9f} AP={result["metrics"]["AP"]:.9f}',flush=True)
    return result


def val():
    for setting in SETTINGS:
        smoke=ROOT/'smoke'/setting['name']
        if (smoke/'exit_code').read_text().strip()!='0': raise ValueError('Smoke failed')
        diagnostic=read_json(smoke/'diagnostics.json')
        if not diagnostic[0]['legacy_matches_upstream']: raise ValueError('No reference match')
        if setting['refine'] and read_json(smoke/'complete.json')['total_refined_changed_pixels']<=0:
            raise ValueError('Refinement inactive')
    infer_all(SETTINGS,'val')
    jobs=[]
    for setting in SETTINGS:
        names=['legacy','mask_first']+(['refined_small'] if setting['refine'] else [])
        for variant in names:
            name=setting['name']+'_'+variant
            out=ROOT/'merged_val'/(name+'.json')
            shards=[ROOT/'val'/setting['name']/f'shard{i}'/(variant+'.json') for i in range(SHARDS)]
            manifest=merge(shards,ANNOTATIONS,'val',out)
            evaluation=ROOT/'evaluation'/(name+'.json')
            evaluation.parent.mkdir(exist_ok=True)
            jobs.append((name,str(out),manifest,str(evaluation)))
    with futures.ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(evaluate,jobs))
    results.sort(key=lambda r:(r['metrics']['APs'],r['metrics']['AP']),reverse=True)
    controls={}
    for name,model in [('d1024_legacy','d'),('dsh1024_legacy','dsh')]:
        actual=next(r for r in results if r['name']==name)
        old=read_json(f'seg_project/experiments/mask2former/{model}_screen/best_metrics.json')['metrics']['segm']
        differences={k:actual['metrics'][k]-old[k]/100 for k in ('AP','APs','AP50','AP75')}
        controls[name]=differences
        # GPU batch/kernel floating point can cause tiny differences; large drift
        # blocks automatic selection and must be investigated explicitly.
        if max(abs(v) for v in differences.values())>0.001:
            write_json(ROOT/'control_failure.json',controls)
            raise ValueError(f'Unexpected baseline drift: {controls}')
    report=dict(split='val',units='0-1',metric='COCO mask APs@[0.50:0.95], maxDets=100',
                num_images=2584,results=results,controls_vs_historical=controls,
                selected=results[0]['name'],selection='Highest full-val APs; tie break overall AP. No test labels used.')
    write_json(ROOT/'val_results.json',report)
    print('SELECTED '+report['selected'],flush=True)


def test():
    report=read_json(ROOT/'val_results.json')
    name=report['selected']
    setting=next(s for s in SETTINGS if name.startswith(s['name']+'_'))
    variant=name[len(setting['name'])+1:]
    infer_all([setting],'test')
    output=ROOT/'selected_test.json'
    shards=[ROOT/'test'/setting['name']/f'shard{i}'/(variant+'.json') for i in range(SHARDS)]
    merge(shards,TEST_METADATA,'test',output)
    command=['./seg_project/run.sh','python','seg_project/tools/export_submission.py',
             '--metadata',TEST_METADATA,'--predictions',str(output),'--manifest',str(output)+'.manifest.json',
             '--template','blender/submission.csv','--output',str(ROOT/'submission.csv')]
    subprocess.run(command,check=True)
    write_json(ROOT/'test_complete.json',dict(selected=name,submission=str(ROOT/'submission.csv'),
               sha256=hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest(),
               validation=read_json(str(ROOT/'submission.csv')+'.validation.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['val','test'])
    args=parser.parse_args()
    val() if args.stage=='val' else test()

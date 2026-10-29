"""Round-one paired inference for ranking, resolution, and trained DSH refinement."""
import argparse
import hashlib
import statistics
import time
from pathlib import Path

from common import encode_instances,image_record,load_images,require_depth_metadata,write_json,read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--metadata',required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--data-root',default='/private/rgzndzy/blender')
    p.add_argument('--size',type=int,default=1024)
    p.add_argument('--refine-small',action='store_true')
    p.add_argument('--chunk-size',type=int,default=8)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--image-ids',help='Explicit diagnostic subset; always marked smoke')
    p.add_argument('--check-reference',action='store_true')
    args=p.parse_args()
    if args.num_shards<1 or not 0<=args.shard_index<args.num_shards:
        p.error('Invalid shard settings')
    images=load_images(args.metadata,args.split)
    if args.image_ids:
        selected=set(read_json(args.image_ids))
        if not selected<={row['id'] for row in images}: p.error('Unknown subset ids')
        images=[row for row in images if row['id'] in selected]
    images=images[args.shard_index::args.num_shards]
    if not images: p.error('Empty image shard')
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): p.error('Completed output exists; refusing overwrite')
    from rgbd_seg.m2f.config import setup_cfg
    from rgbd_seg.m2f.experimental_inference import ExperimentalPredictor
    import torch
    cfg=setup_cfg(args.config,('INPUT.TEST_IMAGE_SIZE',str(args.size)))
    (out/'resolved_config.yaml').write_text(cfg.dump(),encoding='utf-8')
    if cfg.PROJECT.USE_DEPTH: require_depth_metadata(images)
    predictor=ExperimentalPredictor(cfg,args.weights,args.refine_small,args.chunk_size)
    names=['legacy','mask_first']+(['refined_small'] if args.refine_small else [])
    results={name:[] for name in names}
    diagnostics=[]
    timings=[]
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    for index,info in enumerate(images):
        before=time.time()
        variants,stats=predictor(image_record(info,args.data_root),check_reference=args.check_reference and index==0)
        for name,instances in variants.items():
            encoded=encode_instances(instances,info['id'])
            if len(encoded)>100: raise ValueError('More than 100 instances per image')
            results[name].extend(encoded)
        timings.append(time.time()-before)
        stats['image_id']=info['id']
        diagnostics.append(stats)
        if (index+1)%25==0 or index+1==len(images):
            print(f'{index+1}/{len(images)} images; {sum(timings)/len(timings):.3f} s/image',flush=True)
    options=dict(size=args.size,refine_small=args.refine_small,chunk_size=args.chunk_size,
                 score_threshold=0,postprocess='original-size logits, mask>0, final max100')
    module=Path(__file__).resolve().parents[1]/'rgbd_seg/m2f/experimental_inference.py'
    summary=dict(split=args.split,metadata=args.metadata,config=args.config,weights=args.weights,
                 image_ids=[row['id'] for row in images],num_images=len(images),
                 num_shards=args.num_shards,shard_index=args.shard_index,smoke=bool(args.image_ids),
                 score_threshold=0,tta=False,tiling=False,options=options,
                 code_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                 wall_seconds=time.time()-started,
                 seconds_per_image_mean=statistics.mean(timings),
                 seconds_per_image_after_warmup=statistics.mean(timings[5:] or timings),
                 peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                 total_selection_replacements=sum(d['selection_replacements'] for d in diagnostics),
                 total_refined_queries=sum(d['refined_queries'] for d in diagnostics),
                 total_refined_changed_pixels=sum(d['refined_changed_pixels'] for d in diagnostics),
                 head_residual_norm=predictor.head_residual_norm)
    if args.check_reference and not diagnostics[0].get('legacy_matches_upstream'):
        raise AssertionError('Reference check was not performed')
    for name,rows in results.items():
        path=out/(name+'.json')
        rows.sort(key=lambda row:(row['image_id'],-row['score']))
        write_json(path,rows)
        write_json(str(path)+'.manifest.json',dict(summary,variant=name,num_instances=len(rows)))
    write_json(out/'diagnostics.json',diagnostics)
    write_json(out/'complete.json',summary)
    print(summary,flush=True)


if __name__=='__main__': main()

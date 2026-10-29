"""One fixed-weight global+4-tile val experiment; no training or GT-guided tiles."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time

from course_demo import sha256, encode_predictions
from course_batch_predict import VAL_SHA, atomic_json, validate_rows
from course_resolution_probe import safe_request, scaled_pipeline, check_coverage, inspect_prediction, case_diagnostics, FIXED_IDS, now
from m2f_decoder_probe import ARMS, MANIFEST, MANIFEST_SHA, manifest, read

SPEC = ARMS['copy_paste']
DECODER = 'outputs/course_m2f_decoder_20261018_v1'
DECODER_REPORT_SHA = '3465b294cfb1f8b093fca2d49b590bae36ab3eac57e8b2594e59e4f3608424ee'
DECODER_GT_SHA = 'e9452e47365a4ad0ef26402a3c1e10081a533ee9166995d432c3d2b342d0eb4f'
PREPARATION_UNIX = 1792334136.0  # 2026-10-18 22:35:36 Beijing
DEADLINE = PREPARATION_UNIX + 14400
WINDOWS = ((0,0),(384,0),(0,384),(384,384))  # (x,y), original1024
TILE_SIZE = 640
MAX_TILE_AREA = 1024
NMS_IOU = .5
MAX_INSTANCES = 100
METRICS = tuple('course/segm_mAP'+s for s in ('','_50','_75','_s','_m','_l'))


def check_time():
    if time.time() >= DEADLINE:
        raise TimeoutError('Original4h preparation-inclusive deadline; no extension')


def crop_loaded(loaded, origin):
    """Synchronized, GT-independent crop before accepted depth-aware resize."""
    import numpy as np
    if any(k in loaded for k in ('instances','gt_bboxes','gt_masks','gt_instances')):
        raise ValueError('No annotation fields in inference views')
    keys = ('img','rgbd_depth','rgbd_raw_valid','rgbd_label_valid')
    if any(tuple(loaded[k].shape[:2]) != (1024,1024) for k in keys):
        raise ValueError('Original RGB-D planes must be aligned1024')
    if origin is not None and origin not in WINDOWS:
        raise ValueError('Only four predeclared windows')
    out = copy.deepcopy(loaded)
    if origin is not None:
        x,y = origin
        for k in keys:out[k] = np.ascontiguousarray(loaded[k][y:y+640,x:x+640]).copy()
        out['ori_shape'] = (640,640)
        out['img_shape'] = (640,640)
    return out


def pack_views(request, loader, tail):
    import torch
    raw = loader(copy.deepcopy(request))
    values=[]
    for origin in (None,*WINDOWS):
        value=tail(crop_loaded(raw,origin));ds=value['data_samples']
        if 'gt_instances' in ds:
            if len(ds.gt_instances):raise ValueError('GT enters model')
            del ds.gt_instances
        expected=(1024,1024) if origin is None else (640,640)
        if tuple(value['inputs'].shape)!=(5,1536,1536) or tuple(ds.ori_shape)!=expected:
            raise ValueError('View geometry mismatch')
        if not torch.isfinite(value['inputs']).all():raise ValueError('Nonfinite RGB-D input')
        values.append(value)
    return values


def candidates(pred, view):
    """CPU original-grid RLEs; tile area filter is computed BEFORE stitching."""
    import numpy as np
    import torch
    from pycocotools import mask as mu
    shape=(1024,1024) if view==0 else (640,640)
    if view not in range(5) or tuple(pred.masks.shape)!=(len(pred),*shape) or len(pred)>100:
        raise ValueError('Invalid prediction/view geometry')
    if pred.masks.dtype!=torch.bool or not (pred.labels==0).all():raise ValueError('Mask/label contract')
    if not torch.isfinite(pred.scores).all() or not torch.isfinite(pred.bboxes).all() or not ((pred.scores>=0)&(pred.scores<=1)).all():
        raise ValueError('Invalid scores')
    result=[]
    for idx,(mask,score) in enumerate(zip(pred.masks.numpy(),pred.scores.tolist())):
        area=int(mask.sum())
        if not area or (view and area>MAX_TILE_AREA):continue
        edge=False
        if view:
            x,y=WINDOWS[view-1]
            edge=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
            full=np.zeros((1024,1024),np.uint8);full[y:y+640,x:x+640]=mask
        else:full=mask.astype(np.uint8)
        rle=mu.encode(np.asfortranarray(full))
        if not np.array_equal(mu.decode(rle).astype(bool),full.astype(bool)):
            raise ValueError('RLE original-grid roundtrip')
        result.append(dict(rle=rle,score=float(score),view=view,index=idx,area=area,tile_edge=edge))
    return result


def merge_candidates(pool):
    """One score-sorted mask-IoU NMS; global-first stable ties, no score cutoff."""
    from pycocotools import mask as mu
    if any(not math.isfinite(c['score']) or not 0<=c['score']<=1 for c in pool):
        raise ValueError('Bad merged score')
    ordered=sorted(pool,key=lambda c:(-c['score'],c['view'],c['index']))
    kept=[]
    for c in ordered:
        if kept and (mu.iou([c['rle']],[k['rle'] for k in kept],[0]*len(kept))>NMS_IOU).any():continue
        kept.append(c)
        if len(kept)==MAX_INSTANCES:break
    return kept


def as_instances(kept):
    import numpy as np
    import torch
    from mmengine.structures import InstanceData
    from pycocotools import mask as mu
    masks=np.stack([mu.decode(c['rle']).astype(bool) for c in kept]) if kept else np.zeros((0,1024,1024),bool)
    boxes=np.array([mu.toBbox(c['rle']) for c in kept],dtype=np.float32).reshape(-1,4)
    boxes[:,2:]+=boxes[:,:2]
    return InstanceData(masks=torch.from_numpy(masks),scores=torch.tensor([c['score'] for c in kept],dtype=torch.float32),
        labels=torch.zeros(len(kept),dtype=torch.long),bboxes=torch.from_numpy(boxes))


def multiview(model, batches, counters):
    pool=[];prototype=None;counts=[]
    for view,batch in enumerate(batches):
        check_time()
        prediction=model.test_step(batch)[0].cpu();counters['eval_calls']+=1
        if view==0:prototype=prediction
        new=candidates(prediction.pred_instances,view);counts.append(len(new));pool.extend(new)
        if view:del prediction
    kept=merge_candidates(pool)
    prototype.pred_instances=as_instances(kept)
    inspect_prediction(prototype.pred_instances,int(prototype.img_id))
    return prototype,dict(candidate_counts=counts,retained=len(kept),origins=[
        {k:c[k] for k in ('view','index','area','tile_edge')} for c in kept])


def preflight(workspace):
    from mmengine.config import Config
    from course_directions_diagnostics import verify_direction
    workspace=Path(workspace);run=workspace/SPEC['run'];ref=workspace/SPEC['reference']
    report,weight,source,size=verify_direction(run)
    if (sha256(run/'report.json')!=SPEC['report_sha'] or sha256(weight)!=SPEC['weight_sha'] or size!=1024
        or sha256(ref/'report.json')!=SPEC['reference_sha'] or sha256(workspace/DECODER/'analysis/report.json')!=DECODER_REPORT_SHA):
        raise ValueError('Accepted references changed')
    cfg=Config.fromfile(str(run/'config.py'));old=Config.fromfile(str(ref/'inference_config.py'))
    pipeline=scaled_pipeline(cfg.val_dataloader.dataset.pipeline,1536)
    if cfg.model.to_dict()!=old.model.to_dict() or pipeline!=old.val_dataloader.dataset.pipeline:
        raise ValueError('Model/postprocessing/global1536 geometry changed')
    if (cfg.val_evaluator.type!='CocoMetric' or cfg.val_evaluator.metric!='segm'
        or tuple(cfg.val_evaluator.proposal_nums)!=(1,10,100) or sha256(cfg.val_evaluator.ann_file)!=VAL_SHA):
        raise ValueError('Official COCO segmentation protocol drift')
    images=read(cfg.val_evaluator.ann_file)['images']
    if len(images)!=2584 or len({i['id'] for i in images})!=2584:raise ValueError('Full official val required')
    m=manifest(workspace,images)
    old_report=read(ref/'report.json')
    if sha256(ref/'input_hashes.json')!=old_report['input_hashes_sha256']:raise ValueError('Reference input binding')
    if [x['image_id'] for x in read(ref/'input_hashes.json')]!=[x['id'] for x in images]:raise ValueError('Reference val order')
    return cfg,weight,pipeline,images,m


def evaluate(root):
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.evaluator import Evaluator
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from pycocotools.coco import COCO
    root=Path(root);out=root/'evaluate'
    if out.exists():raise ValueError('No evaluation replay')
    check_time();cfg,weight,pipeline,images,m=preflight('/workspace')
    out.mkdir();(out/'subset').mkdir()
    state=dict(status='running',phase='initializing',started_at_utc=now(),training_started=False,
        test_inputs_used=False,test_annotations_used=False,gt_sent_to_model=False,completed_images=0,
        checkpoint_sha256=SPEC['weight_sha'],training_report_sha256=SPEC['report_sha'],reference_report_sha256=SPEC['reference_sha'],
        val_annotations_sha256=VAL_SHA,manifest_sha256=MANIFEST_SHA,input_scale=1536,original_shape=[1024,1024],
        tile_origins=WINDOWS,tile_size=640,tile_max_area=1024,mask_nms_iou=.5,max_instances=100)
    atomic_json(out/'status.json',state)
    try:
        register_all_modules(init_default_scope=True);import_modules_from_strings(**cfg.custom_imports)
        saved=torch.load(weight,map_location='cpu')
        trained=Config.fromstring(saved['meta'].get('cfg',saved['meta'].get('config')),'.py')
        if trained.model.to_dict()!=cfg.model.to_dict():raise ValueError('Checkpoint model binding')
        if any(not torch.isfinite(t).all() for t in saved['state_dict'].values() if t.is_floating_point()):raise ValueError('Nonfinite weight')
        model=MODELS.build(cfg.model);model.load_state_dict(saved['state_dict'],strict=True);del saved
        model.cuda().eval();torch.backends.cudnn.benchmark=False
        loader=Compose(pipeline[:1]);tail=Compose(pipeline[1:])
        adjusted=copy.deepcopy(cfg);adjusted.val_dataloader.dataset.pipeline=pipeline;adjusted.dump(str(out/'global_config.py'))
        requests=[safe_request(cfg.val_dataloader.dataset.data_root,i) for i in images];by_id={r['img_id']:r for r in requests}
        subset=set(m['image_ids']);capture=subset|set(FIXED_IDS)
        coco=COCO(cfg.val_evaluator.ann_file)
        ev=copy.deepcopy(cfg.val_evaluator);ev.outfile_prefix=str(out/'predictions')
        evaluator=Evaluator(ev);evaluator.dataset_meta=cfg.val_dataloader.dataset.metainfo.to_dict()
        counter=dict(eval_calls=0,train_calls=0);times=[];hashes=[];completed=[];cases=[];recent=[];capture_sha={}
        with torch.inference_mode():
            fixed=pack_views(by_id[FIXED_IDS[0]],loader,tail)
            # Exact accepted global-view pixels/depth/meta; no extra model forward.
            direct=Compose(pipeline)(copy.deepcopy(by_id[FIXED_IDS[0]]))
            if not torch.equal(fixed[0]['inputs'],direct['inputs']) or fixed[0]['data_samples'].metainfo!=direct['data_samples'].metainfo:
                raise ValueError('Global input differs from accepted pipeline')
            del direct
            for _ in range(5):
                batches=[pseudo_collate([copy.deepcopy(v)]) for v in fixed]
                warm,_=multiview(model,batches,counter);torch.cuda.synchronize();del warm,batches
            torch.cuda.reset_peak_memory_stats()
            for _ in range(20):
                batches=[pseudo_collate([copy.deepcopy(v)]) for v in fixed]
                torch.cuda.synchronize();start=time.perf_counter()
                timed,_=multiview(model,batches,counter);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                del timed,batches
            timing=dict(warmup=5,repeats=20,views_per_call=5,image_id=FIXED_IDS[0],all_seconds=times,
                mean_ms=statistics.mean(times)*1000,median_ms=statistics.median(times)*1000,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                scope='FP32 sequential5 model.test_step + D2H/RLE/merge/original-grid assembly, synchronized; excludes disk/decode/resize/batch clone. Baseline excludes merge because single view.')
            atomic_json(out/'timing.json',timing);del fixed
            state.update(phase='predicting',timing=timing);atomic_json(out/'status.json',state)
            for request in requests:
                check_time();start=time.monotonic();views=pack_views(request,loader,tail)
                batches=[pseudo_collate([v]) for v in views]
                prediction,info=multiview(model,batches,counter);identity=request['img_id']
                evaluator.process(data_samples=[prediction],data_batch=batches[0])
                if identity in capture:
                    rows=encode_predictions(prediction.pred_instances.masks.numpy(),prediction.pred_instances.scores.numpy(),identity)
                    validate_rows(rows,identity,[1024,1024])
                    path=out/'subset'/f'{identity}.json'
                    atomic_json(path,dict(image_id=identity,predictions=rows,**info));capture_sha[str(path.relative_to(out))]=sha256(path)
                if identity in FIXED_IDS:
                    cases.append(case_diagnostics(prediction.pred_instances,coco,identity))
                    atomic_json(out/f'case_{identity}_predictions.json',rows)
                hashes.append(dict(image_id=identity,rgb_sha256=sha256(request['img_path']),depth_sha256=sha256(request['depth_file'])))
                completed.append(identity);recent.append(time.monotonic()-start)
                if len(completed)%50==0 or len(completed)==2584:
                    state.update(completed_images=len(completed),recent_seconds_per_image=statistics.mean(recent[-50:]),updated_at_utc=now(),**counter)
                    atomic_json(out/'status.json',state);print(json.dumps(state),flush=True)
                del prediction,batches,views
        check_coverage(completed,[x['id'] for x in images])
        if hashes!=read(Path('/workspace')/SPEC['reference']/'input_hashes.json'):raise ValueError('Full input IDs/order/SHA mismatch')
        atomic_json(out/'input_hashes.json',hashes);atomic_json(out/'cases.json',cases)
        state.update(phase='coco_evaluation');atomic_json(out/'status.json',state);check_time()
        metrics=evaluator.evaluate(2584)
        if any(k not in metrics or not math.isfinite(metrics[k]) or not 0<=metrics[k]<=1 for k in METRICS):raise ValueError('Invalid complete metrics')
        atomic_json(out/'metrics.json',metrics)
        preflight('/workspace');check_time()
        state.update(status='passed',phase='complete',finished_at_utc=now(),completed_images=2584,image_ids=completed,
            metrics=metrics,input_hashes_sha256=sha256(out/'input_hashes.json'),fixed_case_ids=[x['image_id'] for x in cases],
            subset_ids=m['image_ids'],captured_ids=sorted(capture),capture_sha256=capture_sha,
            active_parameters=model.active_parameter_count(),state_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
            global_input_equivalent=True,rle_roundtrip_verified=True,**counter)
        if counter['eval_calls']!=5*(2584+25):raise ValueError('Unexpected inference calls')
        atomic_json(out/'report.json',state);atomic_json(out/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),finished_at_utc=now());atomic_json(out/'status.json',state);raise


def analyze(root,workspace='/workspace'):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools import mask as mu
    check_time();root=Path(root);workspace=Path(workspace);out=root/'analysis'
    if out.exists():raise ValueError('No analysis replay')
    out.mkdir();r=read(root/'evaluate/report.json');old=read(workspace/SPEC['reference']/'report.json')
    if sha256(workspace/DECODER/'analysis/report.json')!=DECODER_REPORT_SHA:raise ValueError('Decoder baseline changed')
    previous=read(workspace/DECODER/'analysis/copy_paste_per_gt.json')
    if sha256(workspace/DECODER/'analysis/copy_paste_per_gt.json')!=DECODER_GT_SHA:raise ValueError('Original per-GT cohort changed')
    prev={x['annotation_id']:x for x in previous if x['area']<=1024}
    if len(prev)!=1187:raise ValueError('Original small cohort differs')
    coco=COCO(str(workspace/'data/course_blender_20261013/prepared/annotations/instances_val.json'))
    records=[];image_stats=[]
    for identity in r['subset_ids']:
        check_time();entry=read(root/'evaluate/subset'/f'{identity}.json');rows=entry['predictions']
        anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[identity])) if a['area']<=1024]
        ious=mu.iou([x['segmentation'] for x in rows],[coco.annToRLE(a) for a in anns],[0]*len(anns)) if rows and anns else np.zeros((len(rows),len(anns)))
        for j,a in enumerate(anns):
            best=int(ious[:,j].argmax()) if rows else None;iou=float(ious[best,j]) if best is not None else 0.
            before=prev[a['id']]
            records.append(dict(annotation_id=a['id'],image_id=identity,area=a['area'],old_final_iou=before['layers'][-1]['best_iou'],
                old_never_covered=before['never_covered'],old_early_lost=before['early_covered_final_lost'],old_covered=before['final_covered'],
                new_iou=iou,new_covered=iou>=.5,new_score=rows[best]['score'] if best is not None else None,
                new_origin=entry['origins'][best] if best is not None else None))
        # Best-IoU geometric FP diagnostic, all sizes, not precision/AP.
        allanns=coco.loadAnns(coco.getAnnIds(imgIds=[identity]))
        allious=mu.iou([x['segmentation'] for x in rows],[coco.annToRLE(a) for a in allanns],[0]*len(allanns)) if rows and allanns else np.zeros((len(rows),len(allanns)))
        unmatched=[i for i,x in enumerate(rows) if x['score']>=.3 and (not allanns or float(allious[i].max())<.5)]
        image_stats.append(dict(image_id=identity,display_no_gt_overlap=len(unmatched),
            tile_edge_unmatched=sum(entry['origins'][i]['tile_edge'] for i in unmatched),**{k:entry[k] for k in ['candidate_counts','retained']}))
    def group(xs):
        return dict(total=len(xs),old_covered=sum(x['old_covered'] for x in xs),new_covered=sum(x['new_covered'] for x in xs),
            gained=sum(not x['old_covered'] and x['new_covered'] for x in xs),lost=sum(x['old_covered'] and not x['new_covered'] for x in xs))
    never=[x for x in records if x['old_never_covered']];early=[x for x in records if x['old_early_lost']]
    if len(never)!=230 or len(early)!=51:raise ValueError('Original diagnostic cohort not bound')
    ratio=r['timing']['mean_ms']/old['timing']['mean_ms'];memory=r['timing']['peak_allocated_mib']/old['timing']['peak_allocated_mib']
    m=r['metrics'];b=old['metrics']
    result=dict(status='passed',finished_at_utc=now(),report_sha256=sha256(root/'evaluate/report.json'),reference_report_sha256=SPEC['reference_sha'],
        metrics=m,reference_metrics=b,delta_points={k:round(100*(m[k]-b[k]),5) for k in METRICS},latency_ratio=ratio,peak_memory_ratio=memory,
        engineering_targets_passed=(m[METRICS[3]]>=.408 and m[METRICS[0]]>=.712 and m[METRICS[5]]>=b[METRICS[5]]-.005 and ratio<=6 and memory<=1.2),
        small=group(records),tiny=group([x for x in records if x['area']<=256]),small_257_1024=group([x for x in records if x['area']>256]),
        original_never230=group(never),original_early_lost51=group(early),training_started=False,test_inputs_used=False,
        limitations='Single fixed-weight, added5-view compute; fixed256 IoU.5 coverage is not AP or proof of architectural causality. GT only after inference.')
    atomic_json(out/'per_gt.json',records);atomic_json(out/'image_stats.json',image_stats)
    atomic_json(out/'comparison.json',result);atomic_json(out/'report.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['evaluate','analyze','binding']);p.add_argument('--root',required=True);a=p.parse_args()
    if a.mode=='evaluate':evaluate(a.root)
    elif a.mode=='analyze':analyze(a.root)
    else:
        preflight('/workspace');atomic_json(Path(a.root)/'binding.json',dict(status='passed',checkpoint_sha256=SPEC['weight_sha'],reference_report_sha256=SPEC['reference_sha'],training_started=False))

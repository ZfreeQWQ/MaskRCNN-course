"""Frozen best system + four 512px views on 128 fresh train scenes only."""
import argparse
import collections
import copy
import math
import os
from pathlib import Path
import random
import signal
import statistics
import time
import unittest
import run_m2f_tiny_positive as w

START=1792526847
PREP_END,CAPTURE_END,ANALYSIS_END,DEADLINE=(START+n for n in (1200,3000,4200,5400))
NAME='course_m2f_fine512_20261021_v1'
OWNER='0b7a37c9fce2424389f3b8579b9d9fa9'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
BEST=ROOT/'outputs/course_m2f_tiny_hflip_20261021_v1'
SCENE=ROOT/'outputs/course_m2f_tiny_scene_20261021_v1'
OLD=ROOT/'outputs/course_m2f_boundary_refine_20261020_v1'
BEST_SHA='8a73c0241f429c23daad53a2f9cf5674bada77b2dae114fb38e26f586aefb9e6'
HEAD_SHA='8e5e87770706e4a3a8b6c645b8c51ee545c285f9c01e99fbdd8343c18700fedb'
BEST_APS=.41399673960953054
WINDOWS=((0,0),(512,0),(0,512),(512,512))
read,save,sha=w.read,w.save,w.sha

def cutoff(end):
    assert time.time()<min(end,DEADLINE),'Immutable deadline'
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Immutable phase deadline')))
    signal.alarm(max(1,math.ceil(min(end,DEADLINE)-time.time())))

def mark(phase,**kw):
    save(OUT/'progress.json',dict(phase=phase,unix=time.time(),**kw))

def select_images(images,excluded,seed=20261021):
    """Only image metadata; neither annotations nor predictions select samples."""
    groups=collections.defaultdict(list)
    for im in images:
        if str(im['scene_id']) not in excluded:groups[str(im['scene_id'])].append(im['id'])
    scenes=sorted(groups);rng=random.Random(seed);rng.shuffle(scenes)
    assert len(scenes)>=128,'Insufficient fresh scenes'
    chosen=[(s,rng.choice(sorted(groups[s]))) for s in scenes[:128]]
    return dict(ids=[i for s,i in chosen],scenes={str(i):s for s,i in chosen},
        available_scenes=len(scenes),excluded_scenes=sorted(excluded),seed=seed,
        gt_selected=False,one_per_scene=True)

def crop512(raw,origin):
    import numpy as np
    assert origin in WINDOWS
    assert not any(k in raw for k in ('instances','gt_instances','gt_bboxes','gt_masks'))
    out=copy.deepcopy(raw);x,y=origin
    for key in ('img','rgbd_depth','rgbd_raw_valid','rgbd_label_valid'):
        assert raw[key].shape[:2]==(1024,1024)
        out[key]=np.ascontiguousarray(raw[key][y:y+512,x:x+512]).copy()
    out['ori_shape']=out['img_shape']=(512,512)
    return out

def encode512(pred,origin,identity,view):
    """Keep ALL native outputs before any area/entry/NMS filtering."""
    import numpy as np
    import torch
    from pycocotools import mask as mu
    assert origin in WINDOWS and view in range(5,9)
    assert len(pred)<=100 and tuple(pred.masks.shape)==(len(pred),512,512)
    assert pred.masks.dtype==torch.bool and bool((pred.labels==0).all())
    assert bool(torch.isfinite(pred.scores).all()) and bool(torch.isfinite(pred.bboxes).all())
    assert bool(((pred.scores>=0)&(pred.scores<=1)).all())
    x,y=origin;rows=[];sources=[]
    full=np.zeros((1024,1024),dtype=np.uint8,order='F')
    for idx,(mask,score) in enumerate(zip(pred.masks.numpy(),pred.scores.tolist())):
        full.fill(0);full[y:y+512,x:x+512]=mask
        rle=mu.encode(full);assert np.array_equal(mu.decode(rle),full)
        box=mu.toBbox(rle).astype(float).tolist();rle['counts']=rle['counts'].decode()
        area=int(mu.area(rle))
        rows.append(dict(image_id=identity,category_id=1,score=float(score),bbox=box,segmentation=rle))
        sources.append(dict(view=view,index=idx,origin=list(origin),area=area,
            boundary=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())))
    return rows,sources

class Contracts(unittest.TestCase):
    def test_scene_split(self):
        ims=[dict(id=i,scene_id=str(i//2)) for i in range(600)]
        a=select_images(ims,{'0','1'})
        self.assertEqual(a,select_images(list(reversed(ims)),{'0','1'}))
        self.assertEqual(len(set(a['scenes'].values())),128)
        self.assertFalse({'0','1'}&set(a['scenes'].values()))
    def test_shortage(self):
        with self.assertRaises(AssertionError):select_images([dict(id=i,scene_id='x') for i in range(200)],set())
    def test_aligned_crop_and_copy(self):
        import numpy as np
        a=np.arange(1024*1024).reshape(1024,1024)
        raw={k:a.copy() for k in ('img','rgbd_depth','rgbd_raw_valid','rgbd_label_valid')}
        for origin in WINDOWS:
            out=crop512(raw,origin);x,y=origin
            for k in raw:self.assertTrue(np.array_equal(out[k],a[y:y+512,x:x+512]));self.assertFalse(np.shares_memory(out[k],raw[k]))
    def test_gt_reject(self):
        with self.assertRaises(AssertionError):crop512({'gt_instances':[]},WINDOWS[0])
    def test_raw_mapping(self):
        import numpy as np
        import torch
        from mmengine.structures import InstanceData
        from pycocotools import mask as mu
        a=np.zeros((3,512,512),bool);a[0,0,0]=a[0,-1,-1]=True;a[1,1:33,1:33]=True
        p=InstanceData(masks=torch.from_numpy(a),scores=torch.tensor([.9,.8,0.]),labels=torch.zeros(3,dtype=torch.long),bboxes=torch.zeros(3,4))
        for v,xy in enumerate(WINDOWS,5):
            rows,sources=encode512(p,xy,1,v);x,y=xy
            self.assertEqual(len(rows),3);self.assertEqual([r['area'] for r in sources],[2,1024,0])
            for i,r in enumerate(rows):self.assertTrue(np.array_equal(mu.decode(r['segmentation'])[y:y+512,x:x+512],a[i]))
    def test_unchanged_no_additions(self):
        import m2f_tiny_verifier as tiny
        from test_m2f_tiny_verifier import row
        base=[row()];merged=[row(300)]
        j=tiny.eligible(base,merged);a=tiny.merge(base,merged,j,[.8])
        self.assertEqual(a,tiny.merge(base,copy.deepcopy(merged),j,[.8]))
    def test_old_prefix(self):
        import m2f_tiny_verifier as tiny
        from test_m2f_tiny_verifier import row
        base=[row()];old=[row(300)];new=[row(500)]
        j=tiny.eligible(base,old);joint=tiny.eligible(base,old+new)
        self.assertEqual(j,joint[:len(j)]);self.assertEqual(joint[-1]['index'],1)
    def test_budget(self):
        self.assertEqual(DEADLINE-START,5400);self.assertEqual(PREP_END-START,1200)

def setup_pipeline():
    import run_m2f_quality_select as original
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    from mmdet.utils import register_all_modules
    original.START,original.DEADLINE=START,DEADLINE
    accepted,earlier,tiles,_=original.modules()
    compact=read(w.PARENT/'compact.json');cfg=Config.fromfile(str(w.PARENT/'live_config.py'))
    register_all_modules(init_default_scope=True);import_modules_from_strings(**cfg.custom_imports)
    pipeline=earlier.typed_pipeline(cfg,compact,tiles)
    return original,accepted,tiles,compact,cfg,Compose(pipeline[:1]),Compose(pipeline[1:])

def prep():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    cutoff(PREP_END);mark('cpu_prep')
    suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(Contracts),
        unittest.defaultTestLoader.loadTestsFromName('test_m2f_tiny_verifier')))
    tests=unittest.TextTestRunner(verbosity=2).run(suite);assert tests.wasSuccessful() and not tests.skipped
    inputs={}
    def bind(p,expected=None):
        p=Path(p);digest=sha(p)
        if expected is not None:assert digest==expected,str(p)
        inputs[str(p.relative_to(ROOT))]=digest;return digest
    bind(BEST/'report.json',BEST_SHA);best=read(BEST/'report.json');assert best['new_aps']==BEST_APS
    bind(BEST/'binding.json',best['artifacts_sha256']['binding.json'])
    bind(BEST/'verifier_2000.pth',HEAD_SHA)
    assert read(BEST/'status.json')['status']=='complete'
    for n,h in read(BEST/'plan.json')['source_sha256'].items():bind(OUT/'source'/n,h)
    expected=read(BEST/'binding.json')['input_sha256']
    bind(SCENE/'prepared_manifest.json',expected[str((SCENE/'prepared_manifest.json').relative_to(ROOT))])
    bind(OLD/'prepared_manifest.json','f33dc02cff69ca6fd6f63cadd8332f4ad62aa0392fc9bf23f913054e1532e243')
    sm=read(SCENE/'prepared_manifest.json');old=read(OLD/'prepared_manifest.json')
    excluded={str(x['request']['scene']) for x in old['images'].values()}|set(map(str,sm['scenes'].values()))
    assert len(old['images'])==512 and len(sm['scenes'])==384
    data=ROOT/w.base.DATA/'annotations/instances_train.json';bind(data,w.base.TRAIN_SHA)
    images=read(data)['images'];assert len(images)==14432
    m=select_images(images,excluded);im={r['id']:r for r in images}
    m['requests']={str(i):w.base.image_request(im[i],'train') for i in m['ids']}
    m['input_hashes']={str(i):dict(image_id=i,rgb_sha256=bind(Path(q['img_path'])),depth_sha256=bind(Path(q['depth_file']))) for i,q in ((int(k),q) for k,q in m['requests'].items())}
    # Exact previously accepted inference config/compact are copied unchanged upstream.
    parent=ROOT/'outputs/course_m2f_candidate_reuse_20261020_v1'
    cm=read(parent/'compact_manifest.json')
    for n in ('compact.json','live_config.py'):bind(w.PARENT/n,cm['files'][n])
    bind(w.PARENT/'selector.json',read(BEST/'binding.json')['selector_sha256'])
    bind(w.BEST/'refiner_3000.pth',w.HEAD_SHA)
    _,accepted,tiles,compact,cfg,loader,tail=setup_pipeline()
    bind(Path(compact['weight']),tiles.SPEC['weight_sha'])
    frozen=read(ROOT/'outputs/course_m2f_tiles_20261019_v1/plan.json')
    for n,h in frozen['upstream_sha256'].items():bind(ROOT/'mmdetection'/n,h)
    # Check actual transformed RGB-D views on one selected train image, no model.
    import torch
    request=m['requests'][str(m['ids'][0])];oldviews=tiles.pack_views(request,loader,tail)
    raw=loader(copy.deepcopy(request))
    for xy in WINDOWS:
        v=tail(crop512(raw,xy));ds=v['data_samples']
        if 'gt_instances' in ds:assert len(ds.gt_instances)==0;del ds.gt_instances
        assert tuple(v['inputs'].shape)==(5,1536,1536) and tuple(ds.ori_shape)==(512,512)
        assert bool(torch.isfinite(v['inputs']).all())
    save(OUT/'prepared_manifest.json',m)
    save(OUT/'binding.json',dict(input_sha256=inputs,best_report_sha256=BEST_SHA,best_head_sha256=HEAD_SHA))
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=tests.testsRun,skipped=0,images=128,
        fresh_scenes=128,excluded_scenes=len(excluded),scene_disjoint=True,actual_views_checked=9,
        annotations_select_samples=False,gpu_used=False,finished_unix=time.time()))

def probabilities(jobs,rows,rgbd,head):
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    ps=[]
    for first in range(0,len(jobs),32):
        group=jobs[first:first+32];x=np.zeros((32,6,128,128),np.float32)
        for j,job in enumerate(group):x[j]=tiny.patch(rows[job['index']],job['xy'],rgbd)
        p=torch.sigmoid(head(torch.from_numpy(x).cuda())).cpu().numpy()[:len(group)]
        ps.extend(map(float,p))
    return ps

def capture():
    import numpy as np
    import torch
    import m2f_boundary_refiner as ref
    import m2f_quality_selector as quality
    import m2f_tiny_verifier as tiny
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    end=read(OUT/'phase_deadlines.json')['capture_end'];cutoff(end)
    original,accepted,tiles,compact,cfg,loader,tail=setup_pipeline()
    b=read(OUT/'binding.json')['input_sha256']
    for p in (Path(compact['weight']),w.BEST/'refiner_3000.pth',BEST/'verifier_2000.pth',w.PARENT/'selector.json'):
        assert sha(p)==b[str(p.relative_to(ROOT))]
    saved=torch.load(compact['weight'],map_location='cpu')
    model=MODELS.build(cfg.model);model.load_state_dict(saved['state_dict'],strict=True);del saved
    model.cuda().eval().requires_grad_(False);correction=w.load_current().cuda().requires_grad_(False)
    head=tiny.Verifier();head.load_state_dict(torch.load(BEST/'verifier_2000.pth',map_location='cpu')['state_dict'],strict=True)
    head.cuda().eval().requires_grad_(False)
    models={'m2f':model,'correction':correction,'verifier':head}
    state={k:original.tensor_digest(v.state_dict()) for k,v in models.items()}
    calls={'eval':0,'train':0}
    def hook(module,args):calls['train' if module.training else 'eval']+=1
    # The original detector's test_step invokes forward; record actual module calls.
    handle=model.register_forward_pre_hook(hook)
    selector=read(w.PARENT/'selector.json');m=read(OUT/'prepared_manifest.json')
    (OUT/'captures').mkdir();files=[];times=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for n,ident in enumerate(m['ids']):
            assert time.time()<end;request=m['requests'][str(ident)]
            actual=dict(image_id=ident,rgb_sha256=sha(request['img_path']),depth_sha256=sha(request['depth_file']))
            assert actual==m['input_hashes'][str(ident)]
            torch.cuda.synchronize();begin=time.perf_counter()
            torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False
            views=tiles.pack_views(request,loader,tail);encoder=accepted.CandidateEncoder(tiles)
            pool=[];global_rows=[]
            for idx,v in enumerate(views):
                pred=model.test_step(pseudo_collate([v]))[0].cpu()
                rows,candidates=encoder.encode(pred.pred_instances,idx,ident)
                if idx==0:global_rows=rows
                pool.extend(candidates)
                del pred
            merged=accepted.base.direct_rows(tiles.merge_candidates(pool),ident)
            chosen=quality.choose(global_rows,merged,selector)[0]
            torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
            rgbd=ref.load_rgbd(request);current=ref.infer(chosen,rgbd,correction,'cuda')[0]
            jobs=tiny.eligible(current,merged);ps=probabilities(jobs,merged,rgbd,head)
            before,ba=tiny.merge(current,merged,jobs,ps)
            torch.cuda.synchronize();old_seconds=time.perf_counter()-begin
            del views,pool
            # New views have their own explicit size, never patch old640 constants.
            begin=time.perf_counter();raw=loader(copy.deepcopy(request));new=[];origins=[]
            torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False
            for view,xy in enumerate(WINDOWS,5):
                v=tail(crop512(raw,xy));ds=v['data_samples']
                if 'gt_instances' in ds:assert len(ds.gt_instances)==0;del ds.gt_instances
                assert tuple(v['inputs'].shape)==(5,1536,1536) and tuple(ds.ori_shape)==(512,512)
                assert bool(torch.isfinite(v['inputs']).all())
                pred=model.test_step(pseudo_collate([v]))[0].cpu()
                rows,sources=encode512(pred.pred_instances,xy,ident,view);new.extend(rows);origins.extend(sources)
                del v,pred
            torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
            joint=merged+new;alljobs=tiny.eligible(current,joint)
            assert alljobs[:len(jobs)]==jobs
            newjobs=alljobs[len(jobs):];newps=probabilities(newjobs,joint,rgbd,head)
            after,aa=tiny.merge(current,joint,alljobs,ps+newps)
            # Same cached old decisions remain byte-for-byte reproducible.
            assert tiny.merge(current,joint[:len(merged)],jobs,ps)==(before,ba)
            torch.cuda.synchronize();extra_seconds=time.perf_counter()-begin
            assert encoder.encoded==encoder.roundtrips
            assert len(before)<=100 and len(after)<=100
            detail=dict(image_id=ident,input_hashes=actual,global_rows=global_rows,merged=merged,
                current=current,new_raw=new,new_origins=origins,before=before,after=after,
                before_activity=ba,after_activity=aa,
                before_candidates=[dict(j,probability=p) for j,p in zip(jobs,ps)],
                after_candidates=[dict(j,probability=p) for j,p in zip(alljobs,ps+newps)],
                timing=dict(baseline_seconds=old_seconds,added_seconds=extra_seconds,
                    total_seconds=old_seconds+extra_seconds,includes_decode_resize_d2h_rle_heads_merge=True))
            path=OUT/'captures'/f'{ident}.json';save(path,detail);files.append(dict(image_id=ident,sha256=sha(path)))
            times.append(detail['timing']);mark('capture',images=n+1,expected=128,eval_calls=calls['eval'],train_calls=calls['train'])
            if n==15:
                projected=(time.time()-started)/16*112*1.25+60
                save(OUT/'throughput_gate.json',dict(status='passed' if time.time()+projected<end else 'failed',
                    completed=16,remaining=112,projected_seconds=projected,deadline=end,unix=time.time()))
                assert time.time()+projected<end,'Insufficient original GPU budget for full128; do not shrink'
    handle.remove();assert calls=={'eval':1152,'train':0},calls
    assert all(original.tensor_digest(v.state_dict())==state[k] for k,v in models.items())
    save(OUT/'capture_report.json',dict(status='passed',images=128,image_ids=m['ids'],files=files,
        calls=calls,state_before=state,state_after=state,all_weights_unchanged=True,gt_to_model=False,
        new_views=4,old_views=5,new_view_size=512,input_size=1536,original_size=1024,
        new_raw_rows=sum(len(read(OUT/'captures'/f'{i}.json')['new_raw']) for i in m['ids']),
        timing=dict(per_image=times,baseline_mean_ms=1000*statistics.mean(x['baseline_seconds'] for x in times),
            added_mean_ms=1000*statistics.mean(x['added_seconds'] for x in times),
            combined_mean_ms=1000*statistics.mean(x['total_seconds'] for x in times),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,distribution_measurement=True,
            warmup=0,repeats_per_image=1,scope='128 distinct train images, baseline5views+heads then4newviews+new-head/merge; excludes output JSON/disk hash, includes decode/resize; NOT standardized5warm20repeat paired benchmark'),
        finished_unix=time.time()))

def finish():
    from pycocotools import mask as mu
    cutoff(ANALYSIS_END);assert not Path('/dev/nvidiactl').exists()
    m=read(OUT/'prepared_manifest.json');capture=read(OUT/'capture_report.json');assert capture['status']=='passed'
    coco=w.coco_load('train');before=[];after=[];opportunities=[];cases=[]
    for file in capture['files']:
        p=OUT/'captures'/f"{file['image_id']}.json";assert sha(p)==file['sha256']
        d=read(p);before.extend(d['before']);after.extend(d['after'])
        anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[file['image_id']])) if 0<a['area']<=256 and not a.get('iscrowd',0) and not a.get('ignore',0)]
        def best(rows,ann):
            masks=[r['segmentation'] for r in rows if r['category_id']==ann['category_id']]
            return float(mu.iou(masks,[coco.annToRLE(ann)],[0]).max()) if masks else 0.
        for ann in anns:
            old_iou=best(d['before'],ann);new_iou=best(d['new_raw'],ann);final_iou=best(d['after'],ann)
            entry=dict(gt_id=ann['id'],image_id=ann['image_id'],scene=m['scenes'][str(ann['image_id'])],area=ann['area'],old_iou=old_iou,new_raw_iou=new_iou,final_iou=final_iou)
            cases.append(entry)
            if old_iou<.5 and new_iou>=.5:opportunities.append(entry)
    bm=w.metrics(coco,m['ids'],before);am=w.metrics(coco,m['ids'],after)
    bp=w.matched(coco,m['ids'],before);ap=w.matched(coco,m['ids'],after)
    gained=sorted(set(ap['tiny'])-set(bp['tiny']));lost=sorted(set(bp['tiny'])-set(ap['tiny']))
    scenes=len({r['scene'] for r in opportunities})
    gate=len(opportunities)>=8 and scenes>=4 and am['APs']>bm['APs'] and len(gained)>=len(lost)
    errors=dict(before=bp,after=ap,tiny_gained=gained,tiny_lost=lost,
        area_bins={name:dict(gained=[i for i in gained if lo<=coco.anns[i]['area']<=hi],lost=[i for i in lost if lo<=coco.anns[i]['area']<=hi]) for name,lo,hi in [('1_16',1,16),('17_64',17,64),('65_256',65,256)]})
    save(OUT/'paired_matches.json',errors);save(OUT/'geometry_cases.json',cases);save(OUT/'opportunities.json',opportunities)
    save(OUT/'comparison.json',dict(status='passed',baseline_metrics=bm,metrics=am,
        delta_points={k:100*(am[k]-bm[k]) for k in bm},images=128,official_val=False,
        opportunity_gt=len(opportunities),opportunity_scenes=scenes,tiny_gained=gained,tiny_lost=lost,
        tiny_net=len(gained)-len(lost),fp_before=len(bp['false_positives']),fp_after=len(ap['false_positives']),
        gate_passed=gate,rule='opportunity>=8GT/4scenes AND paired APs strictly positive AND tiny net>=0',
        geometry_not_one_to_one_or_ap=True,heads_unseen_scenes_but_m2f_used_official_train=True))
    for p,h in read(OUT/'binding.json')['input_sha256'].items():assert sha(ROOT/p)==h,p
    artifacts=['prepared_manifest.json','binding.json','cpu_acceptance.json','gpu_gate.json','gpu_released.json',
        'throughput_gate.json','capture_report.json','paired_matches.json','geometry_cases.json','opportunities.json','comparison.json']
    save(OUT/'report.json',dict(status='passed',outcome='pilot_passed' if gate else 'stopped_failed_pilot',
        start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),new_official_aps=None,
        official_best_aps=BEST_APS,full_val_evaluations=0,paired_train_evaluations=2,weights_updated=False,
        maskrcnn_paused=True,artifacts_sha256={n:sha(OUT/n) for n in artifacts},comparison=read(OUT/'comparison.json')))

def main():
    stage=argparse.ArgumentParser();stage.add_argument('stage',choices=['prep','capture','finish']);args=stage.parse_args()
    w.START,w.DEADLINE=START,DEADLINE
    globals()[args.stage]()
if __name__=='__main__':main()

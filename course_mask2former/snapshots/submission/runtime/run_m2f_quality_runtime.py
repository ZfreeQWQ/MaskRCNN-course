"""Eight-image live parity and complete nine-view timing; no AP or training."""
import copy,json,os,statistics,time,unittest
from pathlib import Path
import run_m2f_fine512_fullval as v

START=1792574220
PREP_END,GPU_END,ANALYSIS_END,DEADLINE=(START+n for n in (600,1200,1500,1800))
NAME='course_m2f_quality_runtime_20261021_v1'
OWNER='7ce77bfe14a743e28d6821866ff64883'
ROOT=Path('/workspace');OUT=ROOT/'outputs'/NAME
REF=ROOT/'outputs/course_m2f_fine512_quality_fullval_20261021_v1'
FULL=ROOT/'outputs/course_m2f_fine512_fullval_20261021_v2'
PILOT=ROOT/'outputs/course_m2f_fine512_quality_20261021_v1'
read,save,sha=v.read,v.save,v.sha
w,old=v.w,v.old
old.START,old.DEADLINE=START,DEADLINE
def cutoff(end):old.cutoff(end)
def mark(phase,**kw):save(OUT/'progress.json',dict(phase=phase,unix=time.time(),**kw))

def prep():
    import m2f_fine512_quality as f
    cutoff(PREP_END)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ('test_m2f_fine512_geometry','test_m2f_fine512_quality','test_m2f_quality_runtime'))
    r=unittest.TextTestRunner(verbosity=2).run(suite);assert r.wasSuccessful() and not r.skipped
    inputs={}
    def bound(p,h=None):
        p=Path(p);digest=sha(p);assert h is None or digest==h,str(p)
        inputs[str(p.relative_to(ROOT))]=digest;return read(p) if p.suffix=='.json' else digest
    report=bound(REF/'report.json','cf78ffd816bd06662b126f552f665979a343fb084b11ea03432a131e4fb79d41')
    assert report['status']=='passed' and report['full_val_evaluations']==1 and report['selector_fits']==0
    selector=bound(PILOT/'selector.json','e5f3ef4e39d7f74a965a6d93301e279bbfda4d00f8964a89c5055f9b1ee41af8')
    fr=bound(FULL/'report.json','89b80f5704d4328b41d53804024ffab68d78309fdad2b2f31edebbfd4116f8fa')
    manifest=bound(FULL/'prepared_manifest.json',fr['artifacts_sha256']['prepared_manifest.json'])
    frozen=bound(FULL/'predictions_frozen.json',fr['artifacts_sha256']['predictions_frozen.json'])
    index={x['image_id']:x['sha256'] for x in manifest['files']}
    captures={x['image_id']:x['capture_sha256'] for x in frozen['files']}
    # Read the big final cache once on CPU, never during the GPU lease.
    allrows=bound(REF/'predictions.segm.json',report['artifacts_sha256']['predictions.segm.json'])
    ids=list(w.base.FIXED8);expected={i:[] for i in ids}
    for row in allrows:
        if row['image_id'] in expected:expected[row['image_id']].append(row)
    del allrows
    refs=[]
    for i in ids:
        cutoff(PREP_END);b=bound(FULL/'base_cache'/f'{i}.json',index[i]);c=bound(FULL/'captures'/f'{i}.json',captures[i])
        assert b['input_hashes']==c['input_hashes'] and b['image_id']==c['image_id']==i
        for k,p in (('rgb_sha256',b['request']['img_path']),('depth_sha256',b['request']['depth_file'])):bound(p,b['input_hashes'][k])
        rows,activity=f.apply(b['current'],c['new_raw'],b['before'],b['before_activity']['kept'],selector)
        assert rows==expected[i] and len(rows)<=100
        refs.append(dict(base=b,new_raw=c['new_raw'],new_origins=c['new_origins'],rows=rows,activity=activity))
    binding=bound(FULL/'binding.json',fr['artifacts_sha256']['binding.json'])
    compact=bound(w.PARENT/'compact.json',binding['input_sha256'][str((w.PARENT/'compact.json').relative_to(ROOT))])
    for p in (Path(compact['weight']),w.BEST/'refiner_3000.pth',old.BEST/'verifier_2000.pth',w.PARENT/'live_config.py',w.PARENT/'selector.json'):
        bound(p,binding['input_sha256'][str(p.relative_to(ROOT))])
    save(OUT/'reference8.json',refs);save(OUT/'selector.json',selector)
    save(OUT/'binding.json',dict(input_sha256=inputs,reference_report_sha256=sha(REF/'report.json'),reference_APs=report['comparison']['metrics']['APs'],images=ids,gt_used=False))
    save(OUT/'cpu_acceptance.json',dict(status='passed',tests=r.testsRun,skipped=0,reference8_sha256=sha(OUT/'reference8.json'),selector_sha256=sha(OUT/'selector.json'),unix=time.time()))

def assert_parity(actual,ref):
    for key in ('global_rows','current','merged','jobs','probabilities','before','before_activity'):
        assert actual[key]==ref['base'][key],(ref['base']['image_id'],key)
    for key in ('new_raw','new_origins','rows','activity'):
        assert actual[key]==ref[key],(ref['base']['image_id'],key)

def gpu():
    import m2f_fine512_quality as f
    import numpy as np
    import torch
    import m2f_boundary_refiner as correction_lib
    import m2f_quality_selector as quality
    import m2f_tiny_verifier as tiny
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS
    end=read(OUT/'phase_deadlines.json')['gpu_end'];cutoff(end)
    accepted_cpu=read(OUT/'cpu_acceptance.json');assert accepted_cpu['status']=='passed'
    assert sha(OUT/'reference8.json')==accepted_cpu['reference8_sha256'] and sha(OUT/'selector.json')==accepted_cpu['selector_sha256']
    refs=read(OUT/'reference8.json');quality_selector=read(OUT/'selector.json')
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    original,accepted,tiles,compact,cfg,loader,tail=old.setup_pipeline()
    ck=torch.load(compact['weight'],map_location='cpu');m2f=MODELS.build(cfg.model);m2f.load_state_dict(ck['state_dict'],strict=True);del ck
    m2f.cuda().eval().requires_grad_(False);correction=w.load_current().cuda().eval().requires_grad_(False)
    oldhead=tiny.Verifier();oldhead.load_state_dict(torch.load(old.BEST/'verifier_2000.pth',map_location='cpu')['state_dict'],strict=True)
    oldhead.cuda().eval().requires_grad_(False)
    models=dict(m2f=m2f,correction=correction,old_verifier=oldhead)
    state={k:original.tensor_digest(m.state_dict()) for k,m in models.items()};calls=dict(eval=0,train=0)
    def hook(m,args):calls['train' if m.training else 'eval']+=1
    handle=m2f.register_forward_pre_hook(hook);selector=read(w.PARENT/'selector.json')
    def m2f_flags():
        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.benchmark
    def head_flags():torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    def new_views(request,i):
        raw=loader(copy.deepcopy(request));new=[];origins=[];m2f_flags()
        for view,xy in enumerate(old.WINDOWS,5):
            assert time.time()<end
            viewdata=tail(old.crop512(raw,xy));ds=viewdata['data_samples']
            if 'gt_instances' in ds:assert len(ds.gt_instances)==0;del ds.gt_instances
            assert tuple(viewdata['inputs'].shape)==(5,1536,1536) and tuple(ds.ori_shape)==(512,512)
            assert bool(torch.isfinite(viewdata['inputs']).all())
            pred=m2f.test_step(pseudo_collate([viewdata]))[0].cpu()
            rows,sources=old.encode512(pred.pred_instances,xy,i,view);new.extend(rows);origins.extend(sources)
            del pred,viewdata
        return new,origins
    def actual_full(request,i):
        assert time.time()<end;torch.cuda.synchronize();t=time.perf_counter()
        m2f_flags();views=tiles.pack_views(request,loader,tail);encoder=accepted.CandidateEncoder(tiles);pool=[];globals_=[]
        for k,view in enumerate(views):
            pred=m2f.test_step(pseudo_collate([view]))[0].cpu();rows,cs=encoder.encode(pred.pred_instances,k,i)
            if k==0:globals_=rows
            pool.extend(cs);del pred
        merged=accepted.base.direct_rows(tiles.merge_candidates(pool),i);chosen=quality.choose(globals_,merged,selector)[0]
        head_flags();rgbd=correction_lib.load_rgbd(request);current=correction_lib.infer(chosen,rgbd,correction,'cuda')[0]
        jobs=tiny.eligible(current,merged);ps=old.probabilities(jobs,merged,rgbd,oldhead)
        before,before_activity=tiny.merge(current,merged,jobs,ps)
        new,origins=new_views(request,i)
        final,activity=f.apply(current,new,before,before_activity['kept'],quality_selector)
        json.dumps(final,separators=(',',':'),allow_nan=False);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
        assert encoder.encoded==encoder.roundtrips
        return dict(global_rows=globals_,current=current,merged=merged,jobs=jobs,probabilities=ps,before=before,before_activity=before_activity,new_raw=new,new_origins=origins,rows=final,activity=activity),ms
    parity=[];times=[];peaks=[];reserved=[]
    with torch.inference_mode():
        for ref in refs:
            b=ref['base'];actual,ms=actual_full(b['request'],b['image_id']);assert_parity(actual,ref)
            for row in actual['rows']:
                from pycocotools import mask as mu
                assert np.array_equal(mu.decode(row['segmentation']),mu.decode(mu.encode(np.asfortranarray(mu.decode(row['segmentation'])))))
            save(OUT/'actual8'/f"{b['image_id']}.json",actual)
            parity.append(dict(image_id=b['image_id'],all_intermediates_exact=True,final_masks_scores_order_exact=True,instances=len(actual['rows'])))
            mark('parity',images=len(parity),expected=8,eval_calls=calls['eval'],train_calls=calls['train'])
        save(OUT/'parity.json',parity)
        for n in range(25):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            b=refs[0]['base'];actual,ms=actual_full(b['request'],b['image_id']);assert_parity(actual,refs[0])
            if n>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20);reserved.append(torch.cuda.max_memory_reserved()/2**20)
            mark('timing',completed=n+1,expected=25,eval_calls=calls['eval'],train_calls=calls['train'])
    handle.remove();assert calls==dict(eval=297,train=0)
    after={k:original.tensor_digest(m.state_dict()) for k,m in models.items()};assert state==after
    assert sha(OUT/'selector.json')==accepted_cpu['selector_sha256']
    save(OUT/'timing.json',dict(warmup=5,repeats=20,milliseconds=times,mean_ms=statistics.mean(times),median_ms=statistics.median(times),peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved),scope='RGB-D file read/decode,full+four640+four512 views at1536,M2F,D2H,RLE,mapping/merge,old selector,4px correction,old tiny verifier,NMS/top100,quality selector/replacement,final JSON; excludes output writes/hash/parity comparisons',paired_baseline=False))
    save(OUT/'gpu_report.json',dict(status='passed',calls=calls,state_before=state,state_after=after,weights_unchanged=True,training_updates=0,full_val_evaluations=0,parity_images=8,active_parameters=m2f.active_parameter_count()+sum(p.numel() for k,m in models.items() if k!='m2f' for p in m.parameters()),model_state_bytes=sum(t.numel()*t.element_size() for m in models.values() for t in m.state_dict().values()),quality_selector_parameters=105,unix=time.time()))

def finish():
    cutoff(ANALYSIS_END);assert not Path('/dev/nvidiactl').exists()
    gr=read(OUT/'gpu_report.json');assert gr['status']=='passed' and gr['calls']==dict(eval=297,train=0)
    assert len(read(OUT/'parity.json'))==8 and read(OUT/'gpu_released.json')['seconds']<=600
    assert all(sha(ROOT/p)==h for p,h in read(OUT/'binding.json')['input_sha256'].items())
    plan=read(OUT/'plan.json');assert all(sha(OUT/'source'/p)==h for p,h in plan['source_sha256'].items())
    artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.json') if 'source' not in p.parts and p.name not in ('status.json','progress.json','plan.json','launch.json','report.json','terminal_acceptance.json','phases.json')}
    save(OUT/'report.json',dict(status='passed',outcome='runtime_verified',APs_reused=.4176790163964242,AP_evaluations=0,training_updates=0,CNN_paused=True,strict_parity=True,gpu=gr,timing=read(OUT/'timing.json'),gpu_lease=read(OUT/'gpu_released.json'),artifacts_sha256=artifacts,start_unix=START,finished_unix=time.time(),deadline_unix=DEADLINE,buffer_used=False))

if __name__=='__main__':
    import sys
    {'prep':prep,'gpu':gpu,'finish':finish}[sys.argv[1]]()

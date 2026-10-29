"""One frozen checkpoint substitution, actual nine-view fullval. No training."""
import ast,copy,gc,hashlib,json,os,statistics,sys,textwrap,time,unittest
from pathlib import Path
from types import SimpleNamespace
import abc_spec as s
ROOT=Path('/workspace');BATCH=ROOT/'outputs'/s.NAME;OUT=BATCH/'arms/A'
sys.path.insert(0,str(OUT/'source/runtime'))
import run_m2f_quality_runtime as r
w,old,read,sha=r.w,r.old,r.read,r.sha

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2));tmp.replace(path)
def deadline(end):
    if time.time()>=min(end,s.A_END,s.DEADLINE):raise TimeoutError('A finite deadline')
def mark(phase,**kw):save(OUT/'progress.json',dict(phase=phase,unix=time.time(),**kw))
def bound(path,expected,inputs):
    h=sha(path);assert expected is None or h==expected,str(path)
    inputs[str(Path(path).relative_to(ROOT))]=h
    return read(path)

def numeric_functions():
    source=(OUT/'source/runtime/run_m2f_quality_runtime.py').read_text();tree=ast.parse(source)
    gpu=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu')
    names=('m2f_flags','head_flags','new_views','actual_full')
    return {n.name:ast.get_source_segment(source,n) for n in gpu.body if isinstance(n,ast.FunctionDef) and n.name in names}

def prep():
    deadline(s.A_CPU_END)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(
        ['test_abc_a','test_m2f_quality_runtime','test_m2f_fine512_quality','test_m2f_fine512_geometry']))
    assert result.wasSuccessful() and not result.skipped
    prior=ROOT/s.RUNTIME;inputs={}
    p=bound(prior/'plan.json',None,inputs)
    for n,h in p['source_sha256'].items():assert sha(OUT/'source/runtime'/n)==h
    accepted=bound(prior/'report.json','34751510d1ba51ba465e6b1a88826fd965995d152a4bf3ab70190f8d3f6d7e7f',inputs)
    assert accepted['status']=='passed' and accepted['strict_parity']
    binding=bound(prior/'binding.json',None,inputs)
    # Small-head/selector/model/data references are explicitly bound once before GPU.
    for n,h in binding['input_sha256'].items():assert sha(ROOT/n)==h,n
    inputs.update(binding['input_sha256'])
    refs=bound(prior/'reference8.json',read(prior/'cpu_acceptance.json')['reference8_sha256'],inputs)
    selector=bound(prior/'selector.json',read(prior/'cpu_acceptance.json')['selector_sha256'],inputs)
    reference=bound(ROOT/s.REFERENCE/'report.json',s.REF_SHA,inputs)
    for n in ('predictions.segm.json','val_input_hashes.json','full_val_metrics.json'):
        path=ROOT/s.REFERENCE/n;assert sha(path)==reference['artifacts_sha256'][n];inputs[str(path.relative_to(ROOT))]=sha(path)
    metrics=read(ROOT/s.REFERENCE/'full_val_metrics.json')
    assert abs(metrics['metrics']['APs']*100-41.76790163964242)<1e-10
    hashes=read(ROOT/s.REFERENCE/'val_input_hashes.json');coco=w.coco_load('val')
    ids=[im['id'] for im in coco.dataset['images']]
    assert ids==metrics['image_ids']==[x['image_id'] for x in hashes] and len(ids)==len(set(ids))==2584
    assert [x['base']['image_id'] for x in refs]==list(w.base.FIXED8)
    point=bound(ROOT/s.NOISE/'evaluate_1536/report.json','2ab5e154a7ccad7f6ec714b382c9c0803bb4a613c0f3bf28c4174745f754c659',inputs)
    assert point['status']=='passed' and sha(ROOT/s.WEIGHT)==s.WEIGHT_SHA
    inputs[s.WEIGHT]=s.WEIGHT_SHA
    import torch
    compact=read(w.PARENT/'compact.json');a=torch.load(compact['weight'],map_location='cpu')['state_dict'];b=torch.load(ROOT/s.WEIGHT,map_location='cpu')['state_dict']
    assert set(a)==set(b) and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a)
    assert all(bool(torch.isfinite(v).all()) for v in b.values() if v.is_floating_point())
    del a,b
    save(OUT/'reference8.json',refs);save(OUT/'selector.json',selector)
    save(OUT/'manifest.json',dict(ids=ids,images=[dict(image_id=i,request=w.base.image_request(coco.imgs[i],'val'),input_hashes=h) for i,h in zip(ids,hashes)]))
    save(OUT/'binding.json',dict(input_sha256=inputs,reference_metrics=metrics['metrics'],new_weight_sha256=s.WEIGHT_SHA,only_model_checkpoint_changed=True))
    save(OUT/'numerical_functions.json',{k:hashlib.sha256(v.encode()).hexdigest() for k,v in numeric_functions().items()})
    deadline(s.A_CPU_END);save(OUT/'cpu_acceptance.json',dict(status='passed',tests=result.testsRun,skipped=0,strict_state_shapes=True,gt_to_model=False,unix=time.time()))

def runtime(weight,end):
    import numpy as np,torch
    import m2f_fine512_quality as f
    import m2f_boundary_refiner as correction_lib
    import m2f_quality_selector as quality
    import m2f_tiny_verifier as tiny
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import MODELS
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    old.START,old.DEADLINE=s.START,s.A_END
    original,accepted,tiles,compact,cfg,loader,tail=old.setup_pipeline()
    ck=torch.load(weight,map_location='cpu');m2f=MODELS.build(cfg.model);m2f.load_state_dict(ck['state_dict'],strict=True);del ck
    m2f.cuda().eval().requires_grad_(False);correction=w.load_current().cuda().eval().requires_grad_(False)
    oldhead=tiny.Verifier();oldhead.load_state_dict(torch.load(old.BEST/'verifier_2000.pth',map_location='cpu')['state_dict'],strict=True)
    oldhead.cuda().eval().requires_grad_(False)
    models=dict(m2f=m2f,correction=correction,old_verifier=oldhead)
    state={k:original.tensor_digest(m.state_dict()) for k,m in models.items()};calls=dict(eval=0,train=0)
    def hook(m,args):calls['train' if m.training else 'eval']+=1
    handle=m2f.register_forward_pre_hook(hook);selector=read(w.PARENT/'selector.json');quality_selector=read(OUT/'selector.json')
    # Reuse each accepted numerical function verbatim, not a rewritten approximation.
    ns=dict(globals(),**locals());funcs=numeric_functions()
    assert set(funcs)=={'m2f_flags','head_flags','new_views','actual_full'}
    for name in ('m2f_flags','head_flags','new_views','actual_full'):exec(compile(textwrap.dedent(funcs[name]),'<frozen '+name+'>','exec'),ns)
    def close():
        handle.remove();assert state=={k:original.tensor_digest(m.state_dict()) for k,m in models.items()}
        return dict(calls=calls,weights_unchanged=True,state_before=state,state_after=state,
            active_parameters=m2f.active_parameter_count()+sum(p.numel() for k,m in models.items() if k!='m2f' for p in m.parameters()),
            model_state_bytes=sum(t.numel()*t.element_size() for m in models.values() for t in m.state_dict().values()))
    return ns['actual_full'],close

def validate_rows(rows,ident):
    import numpy as np
    from pycocotools import mask as mu
    assert len(rows)<=100
    for row in rows:
        assert row['image_id']==ident and row['category_id']==1 and 0<=row['score']<=1
        assert row['segmentation']['size']==[1024,1024]
        d=mu.decode(row['segmentation']);assert np.array_equal(d,mu.decode(mu.encode(np.asfortranarray(d))))

def gpu():
    import torch
    import m2f_fine512_quality as f
    end=read(OUT/'phase_deadlines.json')['worker_end'];deadline(end)
    assert read(OUT/'cpu_acceptance.json')['status']=='passed'
    refs=read(OUT/'reference8.json');manifest=read(OUT/'manifest.json');binding=read(OUT/'binding.json')
    timing={};parity=[];gpu_states={};selected=read(OUT/'selector.json')
    compact=read(w.PARENT/'compact.json');mark('old_parity')
    for arm,weight in (('baseline',compact['weight']),('A',ROOT/s.WEIGHT)):
        predict,close=runtime(weight,end);actual_refs={};elapsed=[]
        with torch.inference_mode():
            for ref in refs:
                deadline(end);b=ref['base'];begin=time.time();actual,ms=predict(b['request'],b['image_id'])
                if arm=='baseline':r.assert_parity(actual,ref)
                else:
                    rows,activity=f.apply(actual['current'],actual['new_raw'],actual['before'],actual['before_activity']['kept'],selected)
                    assert rows==actual['rows'] and activity==actual['activity']
                    p=OUT/'captures'/f"{b['image_id']}.json";save(p,actual)
                    actual_refs[b['image_id']]=actual
                    save(OUT/'capture_seconds'/f"{b['image_id']}.json",dict(seconds=time.time()-begin))
                validate_rows(actual['rows'],b['image_id']);parity.append(dict(arm=arm,image_id=b['image_id'],exact=True))
                mark('old_parity' if arm=='baseline' else 'new_cache_parity',images=len(parity))
            if arm=='A':
                for ref in refs:
                    b=ref['base'];actual,_=predict(b['request'],b['image_id']);assert actual==read(OUT/'captures'/f"{b['image_id']}.json")
                    mark('new_actual_cache_parity',image_id=b['image_id'])
            times=[];peaks=[];reserved=[]
            for n in range(25):
                deadline(end);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                b=refs[0]['base'];actual,ms=predict(b['request'],b['image_id'])
                if arm=='baseline':r.assert_parity(actual,refs[0])
                else:assert actual==actual_refs[b['image_id']]
                if n>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20);reserved.append(torch.cuda.max_memory_reserved()/2**20)
                mark('paired_timing',arm=arm,completed=n+1)
            timing[arm]=dict(warmup=5,repeats=20,mean_ms=statistics.mean(times),milliseconds=times,peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved),image_id=refs[0]['base']['image_id'],scope='Complete accepted nine-view RGB-D IO through final JSON; excludes model load/output writes/hash/parity comparison')
            if arm=='A':
                files=[];allrows=[];input_hashes=[];capture_times=[]
                for n,item in enumerate(manifest['images']):
                    deadline(end);i=item['image_id'];request=item['request'];expected=item['input_hashes']
                    for key,path in (('rgb_sha256',request['img_path']),('depth_sha256',request['depth_file'])):assert sha(Path(path))==expected[key]
                    p=OUT/'captures'/f'{i}.json'
                    if i in actual_refs:actual=actual_refs[i];duration=read(OUT/'capture_seconds'/f'{i}.json')['seconds']
                    else:
                        begin=time.time();actual,_=predict(request,i);save(p,actual);duration=time.time()-begin
                        save(OUT/'capture_seconds'/f'{i}.json',dict(seconds=duration))
                    validate_rows(actual['rows'],i);capture_times.append(duration)
                    allrows+=actual['rows'];input_hashes.append(expected);files.append(dict(image_id=i,capture_sha256=sha(p)))
                    mark('capture',images=n+1,expected=2584,eval_calls=(n+1)*9,training_updates=0)
                    if n==31:
                        mean=statistics.mean(capture_times[4:]);estimate=(2584-32)*mean*1.15+300
                        gate=dict(status='passed' if time.time()+estimate<end else 'failed',images=32,steady28_seconds=mean,remaining_estimate=estimate,remaining_seconds=end-time.time(),factor=1.15)
                        save(OUT/'throughput_gate.json',gate);assert gate['status']=='passed',gate
                save(OUT/'evaluate_1536/predictions.segm.json',allrows);save(OUT/'input_hashes.json',input_hashes)
                save(OUT/'predictions_frozen.json',dict(ids=manifest['ids'],files=files))
        gpu_states[arm]=close()
        assert gpu_states[arm]['calls']==dict(eval=297 if arm=='baseline' else (2584+8+25)*9,train=0)
        assert gpu_states[arm]['active_parameters']==44544128 and gpu_states[arm]['model_state_bytes']==178389440
        del predict,close;gc.collect();torch.cuda.empty_cache();save(OUT/'paired_timing.json',timing)
    save(OUT/'parity.json',parity);save(OUT/'gpu_report.json',dict(status='passed',models=gpu_states,new_evals=(2584+8+25)*9,baseline_evals=297,model_training=0,unix=time.time()))

def finish():
    deadline(s.A_END-60);assert not Path('/dev/nvidiactl').exists()
    assert read(OUT/'gpu_report.json')['status']=='passed'
    manifest=read(OUT/'manifest.json');frozen=read(OUT/'predictions_frozen.json');assert frozen['ids']==manifest['ids']
    for item in frozen['files']:assert sha(OUT/'captures'/f"{item['image_id']}.json")==item['capture_sha256']
    assert read(OUT/'input_hashes.json')==read(ROOT/s.REFERENCE/'val_input_hashes.json')
    binding=read(OUT/'binding.json');path=OUT/'evaluate_1536/predictions.segm.json'
    assert not (OUT/'new_exact.json').exists();mark('one_fullval_AP')
    vals=w.metrics(w.coco_load('val'),manifest['ids'],read(path));save(OUT/'new_exact.json',dict(status='passed',metrics=vals,AP_runs=1,cache_sha256=sha(path),unix=time.time()))
    # Accepted post-hoc matching implementation, no new AP or model invocation.
    ns=dict(ROOT=ROOT,OUT=OUT,s=SimpleNamespace(REFERENCE=s.REFERENCE,ARM='A',DEADLINE=s.A_END),read=read,sha=sha,save=save,deadline=deadline,mark=mark)
    exec(compile((OUT/'source/accepted_error_analysis.py').read_text(),'<accepted error analysis>','exec'),ns);ns['error_analysis']()
    from collections import Counter
    from pycocotools import mask as mu
    coco=w.coco_load('val');counts=Counter();cases=[]
    for item in manifest['images']:
        deadline(s.A_END-60);i=item['image_id'];capture=read(OUT/'captures'/f'{i}.json')
        anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[i])) if not a.get('iscrowd',0)]
        gts=[coco.annToRLE(a) for a in anns]
        for decision in capture['activity']['decisions']:
            if not decision['accepted']:continue
            original=capture['current'][decision['index']];new=capture['new_raw'][decision['new_index']]
            if gts:
                values=mu.iou([original['segmentation'],new['segmentation']],gts,[0]*len(gts));j=int(values[0].argmax());oi,ni=map(float,values[:,j]);gt=anns[j]
            else:oi=ni=0.;gt=None
            counts['accepted']+=1;counts['same_GT_improved']+=ni>oi;counts['same_GT_degraded']+=ni<oi
            counts['crossed_below_0.5']+=oi>=.5 and ni<.5;counts['crossed_above_0.5']+=oi<.5 and ni>=.5
            if i in w.base.FIXED8 or (oi>=.5 and ni<.5) or (oi<.5 and ni>=.5):cases.append(dict(image_id=i,decision=decision,GT_id=gt['id'] if gt else None,GT_area=gt['area'] if gt else None,old_iou=oi,new_iou=ni))
    save(OUT/'replacement_audit.json',dict(summary=dict(counts),cases=cases,scope='Post-hoc same-GT geometry before/after512 replacement within new system; not AP or unique rescue'))
    for n,h in binding['input_sha256'].items():assert sha(ROOT/n)==h,n
    metrics=binding['reference_metrics'];save(OUT/'comparison.json',dict(status='passed',metrics=vals,baseline=metrics,delta_points={k:100*(vals[k]-metrics[k]) for k in vals},positive_retained=vals['APs']>metrics['APs'],single_seed_development=True,baseline_AP_runs=0,new_AP_runs=1,reference_best_unchanged=True))
    save(OUT/'report.json',dict(status='passed',checkpoint_only=True,training_updates=0,full_val_evaluations=1,images=2584,comparison=read(OUT/'comparison.json'),timing=read(OUT/'paired_timing.json'),gpu=read(OUT/'gpu_report.json'),gpu_lease=read(OUT/'gpu_released.json'),start_unix=s.START,finish_unix=time.time(),deadline=s.A_END,CNN_paused=True))
if __name__=='__main__':globals()[sys.argv[1]]()

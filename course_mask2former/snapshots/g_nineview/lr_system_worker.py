"""LR checkpoint substitution with frozen accepted nine-view numerical functions."""
import copy,gc,hashlib,inspect,json,os,shutil,statistics,sys,time,unittest,textwrap
from pathlib import Path
from types import SimpleNamespace
import lr_system_spec as p
import abc_spec as s
for k,v in dict(START=p.START,DEADLINE=p.END,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.GPU_END,A_END=p.END,A_GPU_MAX=p.GPU_MAX,GPU_MAX=p.GPU_MAX,WEIGHT=p.WEIGHT,WEIGHT_SHA=p.WEIGHT_SHA,REFERENCE='outputs/'+p.NAME+'/arms/'+p.ARM+'/reference').items():setattr(s,k,v)
import abc_a_worker as a
ROOT=Path('/workspace');OUT=ROOT/'outputs'/p.NAME/'arms'/p.ARM;PRIOR=ROOT/p.PRIOR;LR=ROOT/p.LR
a.OUT=OUT
# Restore the accepted LR source search root for its existing auxiliary modules.
sys.path.append(str(LR/'source'))
read,save,sha,mark,deadline=a.read,a.save,a.sha,a.mark,a.deadline

def frozen_m2f_flags():
    import torch
    ns={'torch':torch}
    exec(compile(textwrap.dedent(a.numeric_functions()['m2f_flags']),'<accepted M2F flags>','exec'),ns)
    ns['m2f_flags']()
    assert torch.backends.cudnn.allow_tf32 and not torch.backends.cudnn.deterministic
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.benchmark


def compare_global(actual,ident):
    from pycocotools import mask as mu
    rows=sorted([x for x in actual['global_rows'] if int(mu.area(x['segmentation']))>0],key=lambda x:-x['score'])
    assert rows==read(LR/'evaluate_1536'/f'case_{ident}_predictions.json'),('LR global exact mismatch',ident)

def finish_source():
    source=inspect.getsource(a.finish)
    assert "ARM='A'" in source
    return source.replace("ARM='A'","ARM='LR_nineview'")

def prep():
    import torch
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    import rgbd
    deadline(p.CPU_END);assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not Path('/dev/nvidiactl').exists()
    prior=ROOT/p.PREVIOUS;accepted=read(prior/'cpu_acceptance.json');assert accepted['status']=='passed'
    assert read(prior/'status.json')['status']=='failed' and read(prior/'first_error.json')['phase']=='gpu'
    assert read(prior/'gpu_released.json')['original_lock_released'] and not (prior/'captures').exists() and not (prior/'new_exact.json').exists()
    plan=read(OUT/'plan.json')
    for n,h in plan['inherited_sha256'].items():assert sha(OUT/'source'/n)==h,n
    for n,h in plan['external_source_sha256'].items():assert sha(ROOT/n)==h,n
    for n in ('reference8.json','selector.json','manifest.json','numerical_functions.json','binding.json'):
        shutil.copy2(prior/n,OUT/n)
    shutil.copytree(prior/'reference',OUT/'reference',symlinks=True)
    # Same sibling depth, reference symlinks keep the exact accepted targets.
    assert sha(OUT/'reference/predictions.segm.json')==p.REF_CACHE_SHA
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_lr_system'));assert result.wasSuccessful() and not result.skipped
    extra=str(LR/'source/rgbd')
    if extra not in rgbd.__path__:rgbd.__path__.append(extra)
    register_all_modules(init_default_scope=True);cfg=Config.fromfile(str(LR/'formal/runs/lr3e5_long/config.py'));import_modules_from_strings(**cfg.custom_imports)
    saved=torch.load(ROOT/p.WEIGHT,map_location='cpu');model=MODELS.build(cfg.model);model.load_state_dict(saved['state_dict'],strict=True)
    assert model.active_parameter_count()==44400414 and sum(t.numel()*t.element_size() for t in model.state_dict().values())==177814584
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point());del model,saved
    a.w.base.configure_torch();frozen_m2f_flags()
    a.old.START,a.old.DEADLINE=p.START,p.END
    original,accepted_runtime,tiles,compact,cfg,loader,tail=a.old.setup_pipeline();assert callable(loader) and callable(tail)
    assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in a.numeric_functions().items()}==read(OUT/'numerical_functions.json')
    save(OUT/'adapter_sources.json',dict(finish=finish_source()))
    save(OUT/'cpu_reuse.json',dict(status='passed',prior=p.PREVIOUS,prior_acceptance_sha256=sha(prior/'cpu_acceptance.json'),prior_binding_sha256=sha(prior/'binding.json'),prior_first_error_sha256=sha(prior/'first_error.json'),prior_gpu_seconds=p.PRIOR_GPU_SECONDS,actual_LR_config_and_state_loaded=True,accepted_m2f_flags_reused=True,model_updates=0,input_rehash_repeated=False))
    deadline(p.CPU_END);save(OUT/'cpu_acceptance.json',dict(accepted,tests=result.testsRun,skipped=0,reused_inputs=True,actual_single_config=True,unix=time.time()))

def single_runtime(end):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from course_demo import encode_predictions
    from course_resolution_probe import scaled_pipeline
    import rgbd
    extra=str(LR/'source/rgbd')
    if extra not in rgbd.__path__:rgbd.__path__.append(extra)
    torch=a.w.base.configure_torch();register_all_modules(init_default_scope=True)
    cfg=Config.fromfile(str(LR/'formal/runs/lr3e5_long/config.py'));import_modules_from_strings(**cfg.custom_imports)
    model=MODELS.build(cfg.model);ck=torch.load(ROOT/p.WEIGHT,map_location='cpu');model.load_state_dict(ck['state_dict'],strict=True);del ck
    model.cuda().eval().requires_grad_(False);before=a.old.setup_pipeline()[0].tensor_digest(model.state_dict());calls=dict(eval=0,train=0)
    def hook(m,args):calls['train' if m.training else 'eval']+=1
    handle=model.register_forward_pre_hook(hook);compose=Compose(scaled_pipeline(cfg.val_dataloader.dataset.pipeline,1536))
    def predict(request,ident):
        deadline(end);frozen_m2f_flags();torch.cuda.synchronize();start=time.perf_counter();sample=compose(copy.deepcopy(request));ds=sample['data_samples']
        if 'gt_instances' in ds:assert len(ds.gt_instances)==0;del ds.gt_instances
        pred=model.test_step(pseudo_collate([sample]))[0].cpu().pred_instances
        rows=encode_predictions(pred.masks.numpy(),pred.scores.numpy(),ident);json.dumps(rows,allow_nan=False);torch.cuda.synchronize()
        return rows,(time.perf_counter()-start)*1000
    def close():
        handle.remove();after=a.old.setup_pipeline()[0].tensor_digest(model.state_dict());assert before==after
        return dict(calls=calls,weights_unchanged=True,state_before=before,state_after=after,active_parameters=model.active_parameter_count(),model_state_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()))
    return predict,close

def gpu():
    import torch
    import m2f_fine512_quality as f
    end=read(OUT/'phase_deadlines.json')['worker_end'];deadline(end);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
    refs=read(OUT/'reference8.json');manifest=read(OUT/'manifest.json');parity=[];timing={};states={};selected=read(OUT/'selector.json')
    # Fair full IO-to-JSON single-scale timing, using the accepted LR model and pipeline.
    predict,close=single_runtime(end);times=[];peaks=[];reserved=[]
    with torch.inference_mode():
        for ref in refs:
            b=ref['base'];rows,_=predict(b['request'],b['image_id']);expected=read(LR/'evaluate_1536'/f"case_{b['image_id']}_predictions.json")
            if rows!=expected:
                save(OUT/'failed_single_parity.json',dict(image_id=b['image_id'],actual=rows,expected=expected,exact=False));save(OUT/'partial_gpu_state.json',close());raise AssertionError(('single_parity',b['image_id']))
            parity.append(dict(arm='LR_single1536',image_id=b['image_id'],exact=True));mark('single_parity',images=len(parity))
        for n in range(25):
            torch.cuda.reset_peak_memory_stats();b=refs[0]['base'];rows,ms=predict(b['request'],b['image_id']);assert rows==read(LR/'evaluate_1536'/f"case_{b['image_id']}_predictions.json")
            if n>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20);reserved.append(torch.cuda.max_memory_reserved()/2**20)
            mark('paired_timing',arm='baseline',completed=n+1)
    timing['baseline']=dict(warmup=5,repeats=20,mean_ms=statistics.mean(times),milliseconds=times,peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved),image_id=refs[0]['base']['image_id'],scope='LR single1536 complete RGB-D IO/resize/pack/H2D/forward/D2H/RLE/JSON; excludes construction/hash/parity')
    states['LR_single1536']=close();assert states['LR_single1536']['calls']==dict(eval=33,train=0);assert states['LR_single1536']['active_parameters']==44400414 and states['LR_single1536']['model_state_bytes']==177814584
    del predict,close;gc.collect();torch.cuda.empty_cache();save(OUT/'paired_timing.json',timing)
    for arm,weight in (('old_system',ROOT/p.OLD_WEIGHT),('LR_nineview',ROOT/p.WEIGHT)):
        predict,close=a.runtime(weight,end);actual_refs={}
        with torch.inference_mode():
            for ref in refs:
                deadline(end);b=ref['base'];start=time.time();actual,ms=predict(b['request'],b['image_id'])
                if arm=='old_system':assert actual==read(PRIOR/'captures'/f"{b['image_id']}.json"),('old_system_parity',b['image_id'])
                else:
                    compare_global(actual,b['image_id']);rows,activity=f.apply(actual['current'],actual['new_raw'],actual['before'],actual['before_activity']['kept'],selected)
                    assert rows==actual['rows'] and activity==actual['activity'];save(OUT/'captures'/f"{b['image_id']}.json",actual);actual_refs[b['image_id']]=actual
                    save(OUT/'capture_seconds'/f"{b['image_id']}.json",dict(seconds=time.time()-start))
                a.validate_rows(actual['rows'],b['image_id']);parity.append(dict(arm=arm,image_id=b['image_id'],exact=True));mark('strict_parity',arm=arm,images=len(parity))
            if arm=='LR_nineview':
                for ref in refs:
                    b=ref['base'];actual,_=predict(b['request'],b['image_id']);assert actual==read(OUT/'captures'/f"{b['image_id']}.json");mark('new_actual_cache_parity',image_id=b['image_id'])
                times=[];peaks=[];reserved=[]
                for n in range(25):
                    deadline(end);torch.cuda.reset_peak_memory_stats();b=refs[0]['base'];actual,ms=predict(b['request'],b['image_id']);assert actual==actual_refs[b['image_id']]
                    if n>=5:times.append(ms);peaks.append(torch.cuda.max_memory_allocated()/2**20);reserved.append(torch.cuda.max_memory_reserved()/2**20)
                    mark('paired_timing',arm=arm,completed=n+1)
                timing[arm]=dict(warmup=5,repeats=20,mean_ms=statistics.mean(times),milliseconds=times,peak_allocated_mib=max(peaks),peak_reserved_mib=max(reserved),image_id=refs[0]['base']['image_id'],scope='Frozen complete nine-view RGB-D IO through four heads and final JSON; excludes construction/hash/parity')
                files=[];allrows=[];hashes=[];durations=[]
                for n,item in enumerate(manifest['images']):
                    deadline(end);i=item['image_id'];request=item['request'];expected=item['input_hashes']
                    for key,path in (('rgb_sha256',request['img_path']),('depth_sha256',request['depth_file'])):assert sha(path)==expected[key]
                    path=OUT/'captures'/f'{i}.json'
                    if i in actual_refs:actual=actual_refs[i];duration=read(OUT/'capture_seconds'/f'{i}.json')['seconds']
                    else:
                        start=time.time();actual,_=predict(request,i);save(path,actual);duration=time.time()-start;save(OUT/'capture_seconds'/f'{i}.json',dict(seconds=duration))
                    a.validate_rows(actual['rows'],i);durations.append(duration);allrows+=actual['rows'];hashes.append(expected);files.append(dict(image_id=i,capture_sha256=sha(path)))
                    mark('capture',images=n+1,expected=2584,eval_calls=(n+1)*9,training_updates=0)
                    if n==31:
                        mean=statistics.mean(durations[4:]);estimate=(2584-32)*mean*1.15+300;gate=dict(status='passed' if time.time()+estimate<end else 'failed',images=32,steady28_seconds=mean,remaining_estimate=estimate,remaining_seconds=end-time.time(),factor=1.15)
                        save(OUT/'throughput_gate.json',gate);assert gate['status']=='passed',gate
                save(OUT/'evaluate_1536/predictions.segm.json',allrows);save(OUT/'input_hashes.json',hashes);save(OUT/'predictions_frozen.json',dict(ids=manifest['ids'],files=files))
        states[arm]=close();assert states[arm]['calls']==dict(eval=72 if arm=='old_system' else 23553,train=0)
        assert states[arm]['active_parameters']==44544128 and states[arm]['model_state_bytes']==178389440
        del predict,close;gc.collect();torch.cuda.empty_cache();save(OUT/'paired_timing.json',timing)
    save(OUT/'parity.json',parity);save(OUT/'gpu_report.json',dict(status='passed',models=states,new_evals=23553,old_system_evals=72,baseline_evals=33,total_evals=23658,model_training=0,unix=time.time()))

def finish():
    namespace=dict(a.__dict__);exec(compile(finish_source(),'<LR finish>','exec'),namespace);namespace['finish']()
    # Independent secondary historical comparison, reusing the same once-evaluated new AP.
    old=read(PRIOR/'new_exact.json')['metrics'];new=read(OUT/'new_exact.json')['metrics'];save(OUT/'secondary_comparison.json',dict(status='passed',baseline=old,metrics=new,delta_points={k:(new[k]-old[k])*100 for k in new},new_AP_runs=0,primary_retention_threshold_APs=45.03649817971514))
    report=read(OUT/'report.json');report.update(prior_partial_evals=p.PRIOR_PARTIAL_EVALS,prior_GPU_seconds=p.PRIOR_GPU_SECONDS,protocol='LR3e5_frozen_nineview_gpu7',primary='LR_single1536_45.03649818',secondary='historical_B_into_A_42.46308704',source_plan_sha256=sha(OUT/'plan.json'));save(OUT/'report.json',report)
if __name__=='__main__':globals()[sys.argv[1]]()

"""One authorized test inference. No labels, training, or AP evaluation."""
import copy,csv,gc,hashlib,json,os,shutil,statistics,sys,time,unittest
from pathlib import Path
import g_system_worker as g
import g_submit_spec as p
import abc_spec as s
for k,v in dict(START=p.START,DEADLINE=p.DEADLINE,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.GPU_END,A_END=p.DEADLINE,A_GPU_MAX=p.GPU_MAX,GPU_MAX=p.GPU_MAX,WEIGHT=p.WEIGHT,WEIGHT_SHA=p.WEIGHT_SHA).items():setattr(s,k,v)
import abc_a_worker as a
import g_submit_codec as codec
ROOT=Path('/workspace');OUT=ROOT/'outputs'/p.NAME/'arms/test';SYSTEM=ROOT/p.SYSTEM;a.OUT=OUT
read,save,sha=a.read,a.save,a.sha

def deadline(end):
    if time.time()>=min(end,p.DEADLINE):raise TimeoutError('Authorized one-test hard deadline')
def mark(phase,**kw):a.mark(phase,**kw)
def safe_images(document):
    allowed={'id','file_name','depth_file_name','width','height','scene_id','split','depth_scale_m','depth_invalid_code','depth_semantics'}
    assert isinstance(document,list) and len(document)==2112
    assert all(set(im)<=allowed and im['split']=='test' and im['width']==im['height']==1024 for im in document)
    codec.ids_checked([im['id'] for im in document])
    return {im['id']:im for im in document}

def prep():
    deadline(p.CPU_END);assert not Path('/dev/nvidiactl').exists() and os.environ['CUDA_VISIBLE_DEVICES']==''
    test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_g_submit'));assert test.wasSuccessful() and not test.skipped
    prior=read(SYSTEM/'plan.json');plan=read(OUT/'plan.json');inputs={}
    for n,h in prior['source_sha256'].items():assert sha(OUT/'source'/n)==sha(SYSTEM/'source'/n)==h
    for n,h in prior['external_source_sha256'].items():assert sha(ROOT/n)==h
    assert read(SYSTEM/'status.json')['status']=='complete' and sha(SYSTEM/'report.json')==p.SYSTEM_REPORT_SHA
    assert sha(SYSTEM/'evaluate_1536/predictions.segm.json')==p.SYSTEM_CACHE_SHA and sha(ROOT/p.WEIGHT)==p.WEIGHT_SHA
    for n,h in read(SYSTEM/'binding.json')['input_sha256'].items():assert sha(ROOT/n)==h;inputs[n]=h
    for n in (p.INPUT,p.TEMPLATE,p.K,p.SYSTEM+'/report.json',p.SYSTEM+'/predictions_frozen.json'):inputs[n]=sha(ROOT/n)
    assert sha(ROOT/p.INPUT)==p.INPUT_SHA and sha(ROOT/p.K)==p.K_SHA
    images=safe_images(read(ROOT/p.INPUT));calibration=read(ROOT/p.K);assert len(calibration)==2112 and [x['id'] for x in calibration]==list(images)
    kvals={x['id']:x['K'] for x in calibration};assert all(set(x)=={'id','K'} for x in calibration)
    with (ROOT/p.TEMPLATE).open(newline='') as f:
        r=csv.DictReader(f);assert r.fieldnames==['image_id','PredictionString'];rows=list(r)
    ids=[int(x['image_id']) for x in rows];assert len(ids)==2112 and len(set(ids))==2112 and set(ids)==set(images);assert all(json.loads(x['PredictionString'])==[] for x in rows)
    manifest=[]
    for j,i in enumerate(ids):
        deadline(p.CPU_END);request=a.w.base.image_request(images[i],'test')
        for path in ('img_path','depth_file'):
            q=Path(request[path]);assert '/test/' in str(q.resolve()) and q.suffix=='.png'
        hashes={k:sha(Path(request[path])) for k,path in [('rgb_sha256','img_path'),('depth_sha256','depth_file')]}
        manifest.append(dict(image_id=i,request=request,input_hashes=hashes))
        if j%100==0:mark('safe_test_index',images=j+1)
    index=make_geometry_index(manifest,kvals,read(ROOT/p.TRAIN/'geometry_index.json'),read(SYSTEM/'reference8.json'))
    save(OUT/'inference_geometry_index.json',index);inputs[str((OUT/'inference_geometry_index.json').relative_to(ROOT))]=sha(OUT/'inference_geometry_index.json')
    for n in ('selector.json','reference8.json','numerical_functions.json'):shutil.copy2(SYSTEM/n,OUT/n)
    captured={x['image_id']:x['capture_sha256'] for x in read(SYSTEM/'predictions_frozen.json')['files']}
    for ref in read(OUT/'reference8.json'):
        i=ref['base']['image_id'];q=SYSTEM/'captures'/f'{i}.json';assert sha(q)==captured[i];inputs[str(q.relative_to(ROOT))]=captured[i]
    assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in a.numeric_functions().items()}==read(SYSTEM/'numerical_functions.json')
    a.old.START,a.old.DEADLINE=p.START,p.DEADLINE
    original,accepted,tiles,compact,cfg,loader,tail=a.old.setup_pipeline();assert callable(loader) and callable(tail)
    save(OUT/'input_manifest.json',dict(scope='authorized_test_once',ids=ids,images=manifest,labels=False));save(OUT/'binding.json',dict(input_sha256=inputs,system_report=p.SYSTEM_REPORT_SHA,system_cache=p.SYSTEM_CACHE_SHA,weight=p.WEIGHT_SHA,test_labels_read=False))
    deadline(p.CPU_END);save(OUT/'cpu_acceptance.json',dict(status='passed',tests=test.testsRun,skipped=0,images=2112,labels_read=False,old_numeric_functions_unchanged=True,empty_production_masks_preserved_until_CPU_export=True,derived_K_sha256=p.K_SHA,unix=time.time()))

def gpu():
    import torch
    end=read(OUT/'phase_deadlines.json')['worker_end'];deadline(end);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
    manifest=read(OUT/'input_manifest.json');predict,close=geometry_runtime(ROOT/p.WEIGHT,end);parity=[];files=[];times=[];peak=0
    with torch.inference_mode():
        for ref in read(OUT/'reference8.json'):
            deadline(end);b=ref['base'];actual,_=predict(b['request'],b['image_id']);assert actual==read(SYSTEM/'captures'/f"{b['image_id']}.json");parity.append(dict(image_id=b['image_id'],exact=True));mark('parity',images=len(parity),expected=8)
        save(OUT/'parity.json',parity)
        for n,item in enumerate(manifest['images']):
            deadline(end);i=item['image_id'];request=item['request']
            assert all(sha(Path(request[path]))==item['input_hashes'][key] for key,path in [('rgb_sha256','img_path'),('depth_sha256','depth_file')])
            torch.cuda.reset_peak_memory_stats();begin=time.time();actual,ms=predict(request,i);rows=actual['rows'];a.validate_rows(rows,i)
            path=OUT/'predictions'/f'{i}.json';assert not path.exists();save(path,rows);duration=time.time()-begin;times.append(duration);peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
            files.append(dict(image_id=i,predictions=str(path.relative_to(OUT)),sha256=sha(path),instances=len(rows),seconds=duration,complete_pipeline_ms=ms))
            mark('test_prediction',images=n+1,expected=2112,eval_calls=(8+n+1)*9,train_calls=0)
            if n==31:
                mean=statistics.mean(times[4:]);estimate=(2112-32)*mean*1.15+300;gate=dict(status='passed' if time.time()+estimate<end else 'failed',steady28_seconds=mean,remaining_estimate=estimate,remaining_seconds=end-time.time(),images=32,factor=1.15);save(OUT/'throughput_gate.json',gate);assert gate['status']=='passed',gate
    state=close();assert state['calls']==dict(eval=(8+2112)*9,train=0) and state['active_parameters']==46527188 and state['model_state_bytes']==186321680
    save(OUT/'prediction_manifest.json',dict(scope=codec.SCOPE,test_label_access=False,images=files,input_manifest_sha256=sha(OUT/'input_manifest.json'),weight_sha256=p.WEIGHT_SHA))
    save(OUT/'gpu_report.json',dict(status='passed',images=2112,parity_images=8,model=state,peak_allocated_mib=peak,mean_complete_pipeline_ms=statistics.mean(x['complete_pipeline_ms'] for x in files),labels_read=False,new_AP_runs=0,unix=time.time()));del predict,close;gc.collect();torch.cuda.empty_cache()

def finish():
    deadline(p.DEADLINE-60);assert not Path('/dev/nvidiactl').exists();assert read(OUT/'gpu_report.json')['status']=='passed'
    counts=[0]
    def tick():
        deadline(p.DEADLINE-60);counts[0]+=1
        if counts[0]%20==0:mark('CPU_export_validation',validated_image_calls=counts[0])
    codec.callback=tick
    result=codec.export_manifest(OUT/'prediction_manifest.json',OUT/'input_manifest.json',OUT/'submission.csv')
    for n,h in read(OUT/'binding.json')['input_sha256'].items():assert sha(ROOT/n)==h
    result.update(status='passed',real_test_images=2112,test_labels_read=False,model_updates=0,new_AP_runs=0,uploaded=False,upload_attempts=0,selected_system=p.SYSTEM,selected_system_APs=56.68493022152318,weight_sha256=p.WEIGHT_SHA,prediction_manifest_sha256=sha(OUT/'prediction_manifest.json'),source_plan_sha256=sha(OUT/'plan.json'),gpu_lease=read(OUT/'gpu_released.json'),start_unix=p.START,finish_unix=time.time(),deadline=p.DEADLINE)
    save(OUT/'report.json',result);save(OUT/'comparison.json',dict(status='passed',purpose='exact accepted G nine-view parity; no test accuracy available',new_AP_runs=0,labels_read=False));mark('ready_for_user_file_delivery',images=2112,csv_sha256=result['csv_sha256'])

def make_geometry_index(manifest,kvals,original,refs):
    import math
    assert len(manifest)==len(kvals)==2112
    entries={}
    for item in manifest:
        i=item['image_id'];r=item['request'];K=kvals[i]
        assert len(K)==3 and all(len(x)==3 for x in K) and all(math.isfinite(float(v)) for x in K for v in x)
        assert K[0][0]>0 and K[1][1]>0 and K[2]==[0,0,1]
        entries[str(i)]=dict(id=i,K=K,depth_scale_m=r['depth_scale_m'],depth_invalid_code=r['depth_invalid_code'],depth_file=r['depth_file'],img_path=r['img_path'],height=1024,width=1024,split='test')
    for ref in refs:
        i=str(ref['base']['image_id']);assert i not in entries;entries[i]=copy.deepcopy(original['images'][i]);assert entries[i]['split']=='val'
    assert len(entries)==2120
    return dict(schema='train_val_K_depth_only_v1',schema_note='Inherited storage contract only; this inference-only index contains 2112 authorized test inputs plus 8 fixed validation parity inputs, never training',images=entries,test_K_sha256=p.K_SHA,labels=False)

def geometry_runtime(weight,end):
    import inspect
    src=inspect.getsource(g.geometry_runtime)
    old="pipe[1]['scale']=(2559,1536)"
    assert src.count(old)==1
    src=src.replace(old,"pipe[0]['geometry_index']=str(submission_index);"+old)
    ns=dict(g.__dict__,submission_index=OUT/'inference_geometry_index.json')
    exec(compile(src,'<accepted G runtime with safe inference index>','exec'),ns)
    return ns['geometry_runtime'](weight,end)

if __name__=='__main__':globals()[sys.argv[1]]()

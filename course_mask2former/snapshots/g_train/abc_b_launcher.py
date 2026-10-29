"""Serial B launcher reuses the accepted bounded point-noise supervisor."""
import inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import abc_b_spec as spec
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/spec.NAME;BATCH=ROOT/'outputs'/spec.BATCH
sys.path.insert(0,str(ROOT/spec.PARENT/'source'))
import launch_m2f_pointnoise as prior
from launch_nyu_sweep import atomic_json as save,sha256 as sha
import run_course_directions as q

def read(p):return json.loads(Path(p).read_text())
def worker_source():
    source=inspect.getsource(prior.worker)
    changes={
        "selected=plan.get('selected_gpu');child=None":"selected=0;child=None",
        "for card in ([selected] if selected is not None else [0,1]):":"for card in (0,):",
        "'ai-m2f-noise-'":"'ai-m2f-match-'",
        "'m2f_pointnoise_worker.py',mode":"'abc_b_worker.py',mode",
        "assert r['config_checks']['point_noise_only']":"assert r['config_checks']['tiny_matching_only']\n        ma=read(run/'matching_audit.json');assert ma['status']=='passed' and ma['iterations']==n and ma['activity']['calls']==n*20 and ma['activity']['replaced_points']>0",
        "update();execute('cpu',s.CPU_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'":"""update();execute('cpu',s.CPU_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
        if not read(OUT/'space_gate.json')['gate_passed']:
            state['jobs'][s.JOB]='stopped_space_gate'
            save(OUT/'comparison.json',dict(status='passed',outcome='space_gate_rejected',new_AP_runs=0,training_updates=0,space_gate_sha256=sha(OUT/'space_gate.json')))
            save(OUT/'terminal_acceptance.json',dict(status='passed',stopped_gate=True,GPU_seconds=0,comparison_sha256=sha(OUT/'comparison.json'),unix=time.time()))
            update(status='complete',phase='stopped_space_gate');return""",
    }
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source

def start(commit):
    assert time.time()<spec.CPU_END and not OUT.exists()
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    a=BATCH/'arms/A_recovery1';assert read(a/'terminal_acceptance.json')['status']=='passed'
    assert read(a/'delivery_verification.json')['status']=='passed'
    ledger=read(BATCH/'gpu_ledger.json');assert ledger['selected_gpu']==0 and ledger['total_seconds']==spec.PRIOR_GPU
    assert read(a/'gpu_released.json')['original_lock_released']
    for owner in ('7f728d93869445c6bd924367569fe2431',spec.OWNER):
        assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+owner,'--format','{{.ID}}'],text=True).strip()
    parent=ROOT/spec.PARENT;p=read(parent/'plan.json');assert read(parent/'status.json')['status']=='complete'
    assert all(sha(parent/'source'/n)==h for n,h in p['source_sha256'].items())
    provenance,reference=q.verify_inputs(ROOT);assert shutil.disk_usage(ROOT).free>=18*2**30
    worker_source();OUT.mkdir();(OUT/'logs').mkdir();shutil.copytree(parent/'source',OUT/'source')
    for n in spec.NEW:
        dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__).parent/n,dst)
    files={n:sha(OUT/'source'/n) for n in (*p['source_sha256'],*spec.NEW)};assert all(files[n]==h for n,h in p['source_sha256'].items())
    exact=read(ROOT/spec.EXACT);assert exact['status']=='passed' and exact['AP_runs']==1
    assert sha(ROOT/spec.EXACT)==spec.EXACT_SHA
    assert spec.EXACT_CACHE_SHA==sha(ROOT/spec.REFERENCE/'predictions.segm.json')
    image=subprocess.check_output(['docker','image','inspect',q.IMAGE,'--format','{{.Id}}'],text=True).strip()
    plan=dict(protocol='B_tiny_Hungarian_shared_points',source_commit=commit,source_sha256=files,inherited_sha256=p['source_sha256'],workspace=str(ROOT),output=str(OUT),image_id=image,expected_provenance=provenance,reference_baseline=reference,owner_token=spec.OWNER,start_unix=spec.START,cpu_end=spec.CPU_END,smoke_end=spec.SMOKE_END,formal_end=spec.FORMAL_END,eval_end=spec.EVAL_END,deadline_unix=spec.DEADLINE,gpu_max_seconds=spec.GPU_MAX,prior_ABC_GPU=spec.PRIOR_GPU,round_gpu_max=spec.ROUND_GPU_MAX,round_deadline=spec.ROUND_END,selected_gpu=0,lock_wait_seconds=0,steps=30000,maximum_formal_runs=1,CNN_paused=True,CPU_cores=8,OMP_MKL_threads=2,primary_reference=spec.REFERENCE,primary_reference_sha256=spec.REFERENCE_SHA,exact_cache_sha256=spec.EXACT_CACHE_SHA,space_thresholds={'old_miss_min':.10,'new_miss_max':.01,'quota':1024,'shared_points':12544})
    save(OUT/'plan.json',plan)
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/abc_b_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')))
    b=read(BATCH/'status.json');b['current']='B';b['arms']['B']='running';b['B_active_root']=str(OUT);save(BATCH/'status.json',b)
    print(json.dumps(read(OUT/'launch.json')))

def worker():
    ns=dict(prior.__dict__,spec=spec,ROOT=ROOT,OUT=OUT,s=None)
    exec(compile(worker_source(),'<bounded matching supervisor>','exec'),ns)
    try:ns['worker']()
    finally:
        if (OUT/'gpu_released.json').exists():used=read(OUT/'gpu_released.json')['total_seconds']
        else:used=0
        assert used<=spec.GPU_MAX and used+spec.PRIOR_GPU<=spec.ROUND_GPU_MAX
        save(BATCH/'gpu_ledger.json',dict(selected_gpu=0,total_seconds=spec.PRIOR_GPU+used,arms={'A':spec.PRIOR_GPU,'B':used},max_seconds=spec.ROUND_GPU_MAX))
        b=read(BATCH/'status.json');b['arms']['B']=read(OUT/'status.json')['status'];b['current']='B_delivery_pending';save(BATCH/'status.json',b)
if __name__=='__main__':start(sys.argv[2]) if sys.argv[1]=='start' else worker()

"""gpu7 single-card bounded supervisor, preserving accepted trainer protections."""
import fcntl,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import geometry_spec as spec
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/spec.NAME
sys.path.append(str(ROOT/spec.PARENT/'source'))
import launch_lr as prior
from launch_nyu_sweep import atomic_json as save,sha256 as sha
q=prior.q

def read(p):return json.loads(Path(p).read_text())
def private_holders():
    path=Path(spec.PRIVATE_LOCK)
    if not path.exists():return []
    st=path.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
    return [line for line in Path('/proc/locks').read_text().splitlines() if key in line]
def verify_uuid():
    assert subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=8).strip()==spec.UUID
class Locks:
    def __init__(self):
        self.files=[]
        try:
            assert os.uname().nodename==spec.HOSTNAME and not private_holders()
            for name,mode in [(spec.LOCKS[0],'r+'),(spec.UUID_LOCK,'a+')]:
                f=(ROOT/'outputs'/name).open(mode);self.files.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not private_holders();verify_uuid()
        except BaseException:self.close();raise
    def fileno(self):return self.files[0].fileno()
    def close(self):
        for f in reversed(self.files):f.close()
        self.files=[]
def worker_source():
    source=prior.worker_source()
    a=source.index("        if not read(OUT/'space_gate.json')['gate_passed']:")
    b=source.index("        stage='smoke'",a)
    source=source[:a]+"        binding=read(OUT/'cpu_model_binding.json');s.ACTIVE=binding['active'];s.STATE=binding['state_bytes']\n"+source[b:]
    changes={
      "f=(ROOT/'outputs'/s.LOCKS[card]).open('r+')":"f=Locks()",
      "'ai-m2f-lr-'":"'ai-m2f-G-'",
      "'lr_worker.py',mode":"'geometry_worker.py',mode",
      "OPENBLAS_NUM_THREADS='2'":"OPENBLAS_NUM_THREADS='2',OPENBLAS_CORETYPE='Prescott'",
      "'12g'":"'24g'",
      "assert r['config_checks']['learning_rate_only']":"assert r['config_checks']['geometry_only']\n        ga=read(run/'geometry_audit.json');assert ga['status']=='passed' and ga['iterations']==n and all(ga['updated'].values()) and ga['activity']['train_calls']==n",
      "acquire(14*3600,s.FORMAL_END)":"acquire(18*3600,s.FORMAL_END)",
      "acquire(2700,s.EVAL_END)":"acquire(3600,s.EVAL_END)",
      "execute('finish',s.DEADLINE-60)":"execute('finish',s.SINGLE_END-60)",
      "update(status='complete',phase='complete',container=None)":"save(OUT/'system_queue.json',dict(status='authorized_pending_implementation_and_single_host_acceptance',start_unix=s.START,deadline=s.DEADLINE,system_end=s.SYSTEM_END,max_system_GPU=14400,prior_GPU_seconds=used,baseline_APs=47.70413206494912,training_runs_remaining=0))\n        update(status='complete',phase='single_complete_system_pending',container=None)",
      "inspected=False":"inspected=False;last_resource_check=begin",
      "used=0.;leases=[]":"used=plan['prior_GPU_seconds'];leases=read(ROOT/spec.PREVIOUS/'gpu_released.json')['leases'].copy()",
      "original_lock_acquired=True,lock=":"gpu_uuid=spec.UUID,uuid_guard=spec.UUID_LOCK,original_lock_acquired=True,lock="
    }
    for old,new in changes.items():assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    old='                time.sleep(2)'
    new="""                if gpu and inspected and now-last_resource_check>=30:
                    verify_uuid();assert not private_holders(),'existing external benchmark lock became held'
                    own={x.strip() for x in subprocess.check_output(['docker','top',name,'-eo','pid'],text=True,timeout=8).splitlines()[1:]}
                    actual=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=8).strip().splitlines()
                    assert set(actual)<=own,('external compute conflict',actual);last_resource_check=now
                time.sleep(2)"""
    assert source.count(old)==1;return source.replace(old,new)
def start():
    assert os.uname().nodename==spec.HOSTNAME and time.time()<spec.CPU_END and not OUT.exists()
    assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+spec.OWNER,'--format','{{.ID}}'],text=True).strip()
    previous=ROOT/spec.PREVIOUS
    assert read(previous/'status.json')['status']=='failed' and read(previous/'first_error.json')['phase']=='formal'
    failed=read(previous/'formal/runs/geometry_long/report.json')
    assert 'steady_mean_seconds' in failed['error'] and "'step': 200" in failed['error']
    assert not list((previous/'formal/runs/geometry_long').glob('iter_*.pth')) and not (previous/'evaluate_1536').exists()
    assert read(previous/'gpu_released.json')['original_lock_released']
    proof=read(ROOT/spec.ROI_PROOF)
    assert proof['status']=='passed' and proof['GPU_calls']==0 and proof['all_full_geometry_and_160_crops_bitwise_equal'] and proof['all_16_actual_augmented_RGBD_geometry_GT_RNG_equal']
    assert proof['mean_pipeline']['new']<proof['mean_pipeline']['old'] and proof['mean_donor']['new_region']<proof['mean_donor']['old_full']*.2
    # CPU estimate only admits one repaired launch; the unchanged actual200 gate still decides.
    projected=1.6636832936604817*proof['mean_pipeline']['new']/proof['mean_pipeline']['old']
    assert 29800*projected*1.15+6*2400+600<64800
    origin=Path(__file__).parent
    for n,path in [('geometry_signal.py',origin/'geometry_signal.py'),('course_geometry_data.py',origin/'rgbd/course_geometry_data.py')]:assert sha(path)==proof['candidate_sha'][n]
    oldplan=read(previous/'plan.json');assert all(sha(previous/'source'/n)==h for n,h in oldplan['source_sha256'].items())
    assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+oldplan['owner_token'],'--format','{{.ID}}'],text=True).strip()
    parent=ROOT/spec.PARENT;p=read(parent/'plan.json');assert read(parent/'status.json')['status']=='complete'
    assert len(p['source_sha256'])==122 and all(sha(parent/'source'/n)==h for n,h in p['source_sha256'].items())
    provenance,reference=q.verify_inputs(ROOT);assert shutil.disk_usage(ROOT).free>=35*2**30
    assert subprocess.check_output(['docker','image','inspect',spec.IMAGE,'--format','{{.Id}}'],text=True).strip()==spec.IMAGE
    assert sha(ROOT/spec.EXACT)==spec.EXACT_SHA and sha(ROOT/spec.REFERENCE/'predictions.segm.json')==spec.EXACT_CACHE_SHA
    compile(worker_source(),'<G bounded supervisor>','exec');OUT.mkdir();(OUT/'logs').mkdir();shutil.copytree(parent/'source',OUT/'source')
    origin=Path(__file__).parent
    for n in spec.NEW:
        assert n not in p['source_sha256'];dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin/n,dst)
    files={n:sha(OUT/'source'/n) for n in (*p['source_sha256'],*spec.NEW)}
    bundle=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    plan=dict(original_CPU_attempt=spec.ORIGINAL,previous_attempt=spec.PREVIOUS,previous_first_error_sha256=sha(previous/'first_error.json'),roi_proof=spec.ROI_PROOF,roi_proof_sha256=sha(ROOT/spec.ROI_PROOF),discarded_prior_formal_steps=200,completed_prior_30k_runs=0,prior_GPU_seconds=read(previous/'gpu_released.json')['total_seconds'],protocol='G_7geometry_4scale_spatial_channel_LR3e5_30k_single_then_system',source_commit='',source_bundle_sha256=bundle,source_sha256=files,inherited_sha256=p['source_sha256'],workspace=str(ROOT),output=str(OUT),image_id=spec.IMAGE,expected_provenance=provenance,reference_baseline=reference,owner_token=spec.OWNER,start_unix=spec.START,cpu_end=spec.CPU_END,smoke_end=spec.SMOKE_END,formal_end=spec.FORMAL_END,eval_end=spec.EVAL_END,single_end=spec.SINGLE_END,system_end=spec.SYSTEM_END,deadline_unix=spec.DEADLINE,gpu_max_seconds=spec.GPU_MAX,phase_GPU_max=dict(smoke=900,formal=64800,evaluate=3600,system=14400),selected_gpu=0,gpu_uuid=spec.UUID,host='gpu7',hostname=spec.HOSTNAME,resource_guards=[spec.LOCKS[0],spec.UUID_LOCK],lock_wait_seconds=0,steps=30000,maximum_formal_runs=1,CNN_stopped=True,CPU_cores=8,OMP_MKL_threads=2,primary_reference=spec.REFERENCE,primary_reference_sha256=spec.REFERENCE_SHA,exact_cache_sha256=spec.EXACT_CACHE_SHA,no_test=True,no_submission=True,system_reference_APs=47.70413206494912)
    save(OUT/'plan.json',plan)
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/geometry_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')));print(json.dumps(read(OUT/'launch.json')))
def worker():
    q.BASE_RECIPE=dict(q.BASE_RECIPE,lr=spec.LR)
    ns=dict(prior.parent.prior.__dict__,spec=spec,ROOT=ROOT,OUT=OUT,s=None,q=q,Locks=Locks,verify_uuid=verify_uuid,private_holders=private_holders)
    exec(compile(worker_source(),'<G bounded supervisor>','exec'),ns);ns['worker']()
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()

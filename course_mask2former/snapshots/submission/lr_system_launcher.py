"""Single gpu7/card0 launch; original project plus host-UUID coordination guards."""
import fcntl,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import lr_system_spec as p
import abc_spec as s
for k,v in dict(START=p.START,DEADLINE=p.END,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.GPU_END,A_END=p.END,A_GPU_MAX=p.GPU_MAX,GPU_MAX=p.GPU_MAX,IMAGE=p.IMAGE,LOCKS={0:p.PROJECT_LOCK}).items():setattr(s,k,v)
import abc_a_launcher as l
ROOT=l.ROOT;BATCH=ROOT/'outputs'/p.NAME;OUT=BATCH/'arms'/p.ARM;PRIOR=ROOT/p.PRIOR

def observed_private_holders():
    path=Path(p.PRIVATE_OBSERVED_LOCK)
    if not path.exists():return []
    st=path.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
    return [line for line in Path('/proc/locks').read_text().splitlines() if key in line]

class LockBundle:
    def __init__(self):
        self.files=[]
        try:
            assert os.uname().nodename==p.HOSTNAME
            assert not observed_private_holders(),'existing private GPU benchmark lock held; no access/permission change'
            # Existing project coordination is retained, not replaced by the UUID guard.
            self.project=(ROOT/'outputs'/p.PROJECT_LOCK).open('r+');self.files.append(self.project);fcntl.flock(self.project,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.host=(ROOT/'outputs'/p.UUID_LOCK).open('a+');self.files.append(self.host);fcntl.flock(self.host,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not observed_private_holders()
        except BaseException:self.close();raise
    def fileno(self):return self.project.fileno()
    def close(self):
        for f in reversed(self.files):f.close()
        self.files=[]

def verify_uuid():
    raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=8).strip();assert raw==p.UUID,raw

def transformed():
    source=inspect.getsource(l.worker)
    changes={"jobs={'A':'pending'}":"jobs={'"+p.ARM+"':'pending'}","for card in (0,1):":"for card in (0,):","f=(ROOT/'outputs'/s.LOCKS[card]).open('r+')":"f=LockBundle()","selected=card;lock=f":"verify_uuid();selected=card;lock=f","original_lock_acquired=True,lock=":"host=p.HOST,gpu_uuid=p.UUID,uuid_guard=p.UUID_LOCK,private_kernel_holders=observed_private_holders(),original_lock_acquired=True,lock=","rel='/workspace/outputs/'+s.NAME+'/arms/A'":"rel='/workspace/outputs/'+s.NAME+'/arms/"+p.ARM+"'","rel+'/source/runtime:/workspace/mmdetection'":"rel+'/source/runtime:/workspace/outputs/course_m2f_tiles_20261019_v1/source:/workspace/outputs/course_m2f_geometry_20261020_v1/source:/workspace/mmdetection'","'abc_a_worker.py',mode":"'lr_system_worker.py',mode","state['jobs']['A']":"state['jobs']['"+p.ARM+"']","batch['arms']['A']":"batch['arms']['"+p.ARM+"']","'A_delivery_pending'":"'LR_delivery_pending'","arms={'A':used}":"arms={'"+p.ARM+"':used}","'ai-m2f-abc-'":"'ai-m2f-lrsys-'"}
    for old,new in changes.items():assert old in source,old;source=source.replace(old,new)
    # Check for external GPU compute during this owner's lease without touching it.
    source=source.replace("OPENBLAS_NUM_THREADS='2'","OPENBLAS_NUM_THREADS='2',OPENBLAS_CORETYPE='Prescott'")
    source=source.replace('used=0.;leases=[]','used=p.PRIOR_GPU_SECONDS;leases=list(lread_previous_leases())')
    source=source.replace("inspected=False","inspected=False;last_resource_check=begin")
    needle="                time.sleep(2)"
    insert="""                if gpu and inspected and now-last_resource_check>=30:
                    verify_uuid()
                    own=set(subprocess.check_output(['docker','top',name,'-eo','pid'],text=True,timeout=8).splitlines()[1:]);own={x.strip() for x in own}
                    actual=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=8).strip().splitlines()
                    assert set(actual)<=own,('external GPU compute conflict',actual)
                    assert not observed_private_holders(),'external private benchmark lock held'
                    last_resource_check=now
                time.sleep(2)"""
    assert needle in source;source=source.replace(needle,insert)
    return source

def start():
    assert os.uname().nodename==p.HOSTNAME and time.time()<p.CPU_END and not OUT.exists(),'duplicate/wrong host/expired'
    previous=ROOT/p.PREVIOUS;assert l.read(previous/'status.json')['status']=='failed' and l.read(previous/'gpu_released.json')['original_lock_released']
    assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+p.PREVIOUS_OWNER,'--format','{{.ID}}'],text=True).strip()
    assert l.read(PRIOR/'status.json')['status']=='complete' and l.read(ROOT/p.LR/'status.json')['status']=='complete'
    assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+p.OWNER,'--format','{{.ID}}'],text=True).strip()
    assert subprocess.check_output(['docker','image','inspect',p.IMAGE,'--format','{{.Id}}'],text=True).strip()==p.IMAGE
    backend=l.read(ROOT/'outputs/lr_system_migration_20261027/cpu_backend_replay.json');assert backend['status']=='passed' and len(backend['images'])==8 and backend['core']=='Prescott'
    migration=l.read(ROOT/'outputs/lr_system_migration_20261027/image_migration_acceptance.json');assert migration['status']=='passed' and migration['original_OCI_manifest']==p.ORIGINAL_IMAGE_MANIFEST and migration['same_OCI_config']==p.IMAGE
    prior=l.read(PRIOR/'plan.json');assert len(prior['source_sha256'])==78
    for n,h in prior['source_sha256'].items():assert l.sha(PRIOR/'source'/n)==h,n
    external=dict(prior['external_source_sha256'])
    lrplan=l.read(ROOT/p.LR/'plan.json')
    for n,h in lrplan['source_sha256'].items():external[p.LR+'/source/'+n]=h
    for n,h in external.items():assert l.sha(ROOT/n)==h,n
    assert shutil.disk_usage(ROOT).free>30*2**30;compile(transformed(),'<supervisor audit>','exec')
    (OUT/'logs').mkdir(parents=True);shutil.copytree(PRIOR/'source',OUT/'source')
    for n in p.NEW:assert n not in prior['source_sha256'];shutil.copy2(Path(__file__).parent/n,OUT/'source'/n)
    files=dict(prior['source_sha256'],**{n:l.sha(OUT/'source'/n) for n in p.NEW});assert len(files)==82
    plan=dict(protocol='LR3e5_frozen_nineview_gpu7',source_sha256=files,inherited_sha256=prior['source_sha256'],external_source_sha256=external,source_commit=os.environ.get('SOURCE_COMMIT',''),source_bundle_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),owner=p.OWNER,host=p.HOST,hostname=p.HOSTNAME,gpu_uuid=p.UUID,selected_gpu=0,start_unix=p.START,cpu_end=p.CPU_END,gpu_end=p.GPU_END,deadline=p.END,arm_gpu_max=p.GPU_MAX,round_deadline=p.END,round_gpu_max=p.GPU_MAX,primary=p.LR,primary_report_sha256=p.REF_REPORT_SHA,primary_cache_sha256=p.REF_CACHE_SHA,baseline=p.PRIOR,baseline_report_sha256=p.PRIOR_REPORT_SHA,new_weight=p.WEIGHT,new_weight_sha256=p.WEIGHT_SHA,previous_attempt=p.PREVIOUS,prior_GPU_seconds=p.PRIOR_GPU_SECONDS,previous_first_error_sha256=l.sha(previous/'first_error.json'),training_updates=0,new_full_AP_runs=1,old_full_AP_runs=0,image=p.IMAGE,original_image_manifest=p.ORIGINAL_IMAGE_MANIFEST,image_migration=migration,cpu_backend_replay=backend,OPENBLAS_CORETYPE=p.OPENBLAS_CORETYPE,resource_guards=[p.PROJECT_LOCK,p.UUID_LOCK],private_lock_read_only_kernel_observation=p.PRIVATE_OBSERVED_LOCK)
    l.save(OUT/'plan.json',plan);batch=l.read(BATCH/'status.json');batch.update(status='running',current=p.ARM);batch['arms'][p.ARM]='preparing';l.save(BATCH/'status.json',batch);l.save(BATCH/'gpu_ledger.json',dict(selected_gpu=0,total_seconds=p.PRIOR_GPU_SECONDS,arms={'LR':p.PRIOR_GPU_SECONDS},max_seconds=p.GPU_MAX))
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        c=subprocess.Popen([sys.executable,'-u',str(OUT/'source/lr_system_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    l.save(OUT/'launch.json',dict(pid=c.pid,unix=time.time(),plan_sha256=l.sha(OUT/'plan.json')));print(json.dumps(l.read(OUT/'launch.json')))

def worker():
    def previous_leases():
        result=l.read(ROOT/p.PREVIOUS/'gpu_released.json')['leases'];d=ROOT/'outputs/lr_system_migration_20261027/old_parity_diagnostic';gate=l.read(d/'lease.json');done=l.read(d/'gpu_released.json');return result+[dict(start=gate['start'],finish=done['finish'],seconds=done['seconds'],gpu=0,original_lock_released=True,diagnostic_only=True)]
    ns=dict(l.__dict__,lread_previous_leases=previous_leases,OUT=OUT,BATCH=BATCH,p=p,LockBundle=LockBundle,verify_uuid=verify_uuid,observed_private_holders=observed_private_holders)
    exec(compile(transformed(),'<finite LR system>','exec'),ns);ns['worker']()
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()

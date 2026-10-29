"""Single inherited nine-view run after terminal G single-scale acceptance."""
import fcntl,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import g_system_spec as p
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');PRIOR=ROOT/p.PRIOR;TRAIN=ROOT/p.TRAIN
sys.path.append(str(PRIOR/'source'))
import abc_spec as s
for k,v in dict(START=p.START,DEADLINE=p.END,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.GPU_END,A_END=p.END,A_GPU_MAX=p.GPU_MAX,GPU_MAX=p.ROUND_GPU_MAX,IMAGE=p.IMAGE,LOCKS={0:p.PROJECT_LOCK}).items():setattr(s,k,v)
import abc_a_launcher as l
BATCH=ROOT/'outputs'/p.NAME;OUT=BATCH/'arms/G'
read,save,sha=l.read,l.save,l.sha

def private_holders():
 path=Path(p.PRIVATE_OBSERVED_LOCK)
 if not path.exists():return []
 st=path.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
 return [line for line in Path('/proc/locks').read_text().splitlines() if key in line]
def verify_uuid():assert subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=8).strip()==p.UUID
class Locks:
 def __init__(self):
  self.files=[]
  try:
   assert os.uname().nodename==p.HOSTNAME and not private_holders()
   for name,mode in ((p.PROJECT_LOCK,'r+'),(p.UUID_LOCK,'a+')):
    f=(ROOT/'outputs'/name).open(mode);self.files.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   assert not private_holders();verify_uuid()
  except BaseException:self.close();raise
 def fileno(self):return self.files[0].fileno()
 def close(self):
  for f in reversed(self.files):f.close()
  self.files=[]
def lease(now,arm_used,total_used):
 prior=read(TRAIN/'gpu_released.json')['total_seconds'];hard=min(p.GPU_END,now+p.GPU_MAX-p.PREVIOUS_GPU_SECONDS-arm_used,now+p.ROUND_GPU_MAX-prior-p.PREVIOUS_GPU_SECONDS-arm_used)
 if hard-now<60:raise TimeoutError('Insufficient remaining G system lease')
 return hard,hard-20

def transformed():
 source=inspect.getsource(l.worker)
 changes={"jobs={'A':'pending'}":"jobs={'G':'pending'}","for card in (0,1):":"for card in (0,):","f=(ROOT/'outputs'/s.LOCKS[card]).open('r+')":"f=Locks()","'ai-m2f-abc-'":"'ai-m2f-Gsys-'","+'/arms/A'":"+'/arms/G'","'abc_a_worker.py',mode":"'g_system_worker.py',mode","state['jobs']['A']":"state['jobs']['G']","batch['arms']['A']":"batch['arms']['G']","'A_delivery_pending'":"'G_system_delivery_pending'","arms={'A':used}":"arms={'G_system':used+p.PREVIOUS_GPU_SECONDS}","OPENBLAS_NUM_THREADS='2'":"OPENBLAS_NUM_THREADS='2',OPENBLAS_CORETYPE='Prescott'","inspected=False":"inspected=False;last_resource_check=begin","rel+'/source/runtime:/workspace/mmdetection'":"rel+'/source/runtime:/workspace/outputs/course_m2f_tiles_20261019_v1/source:/workspace/outputs/course_m2f_geometry_20261020_v1/source:/workspace/'+p.TRAIN+'/source:/workspace/mmdetection'","original_lock_acquired=True,lock=":"gpu_uuid=p.UUID,uuid_guard=p.UUID_LOCK,original_lock_acquired=True,lock="}
 for a,b in changes.items():assert a in source,a;source=source.replace(a,b)
 source=source.replace("total_seconds=used,arms={'G_system':used+p.PREVIOUS_GPU_SECONDS}","total_seconds=used+p.PREVIOUS_GPU_SECONDS+read(ROOT/p.TRAIN/'gpu_released.json')['total_seconds'],arms={'G_single':read(ROOT/p.TRAIN/'gpu_released.json')['total_seconds'],'G_system':used+p.PREVIOUS_GPU_SECONDS}")
 source=source.replace("total_seconds=used,active_owner", "total_seconds=used+p.PREVIOUS_GPU_SECONDS+read(ROOT/p.TRAIN/'gpu_released.json')['total_seconds'],active_owner")
 source=source.replace("selected=card;lock=f","verify_uuid();selected=card;lock=f")
 old='                time.sleep(2)';new="""                if gpu and inspected and now-last_resource_check>=30:
                    verify_uuid();assert not private_holders(),'external benchmark lock held'
                    own={x.strip() for x in subprocess.check_output(['docker','top',name,'-eo','pid'],text=True,timeout=8).splitlines()[1:]}
                    actual=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=8).strip().splitlines()
                    assert set(actual)<=own,('external compute conflict',actual);last_resource_check=now
                time.sleep(2)"""
 assert source.count(old)==1;return source.replace(old,new).replace("p.PREVIOUS_GPU_SECONDS", "gproto.PREVIOUS_GPU_SECONDS").replace("p.TRAIN", "gproto.TRAIN").replace("p.UUID_LOCK", "gproto.UUID_LOCK").replace("p.UUID", "gproto.UUID")

def start():
 assert os.uname().nodename==p.HOSTNAME and time.time()<p.CPU_END and not BATCH.exists()
 tp=read(TRAIN/'plan.json');assert read(TRAIN/'status.json')['status']=='complete' and read(TRAIN/'terminal_acceptance.json')['status']=='passed' and read(TRAIN/'single_readonly_acceptance.json')['status']=='passed'
 assert read(TRAIN/'gpu_released.json')['original_lock_released']
 for owner in (tp['owner_token'],p.OWNER):assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+owner,'--format','{{.ID}}'],text=True).strip()
 prior=read(PRIOR/'plan.json');assert len(prior['source_sha256'])==82
 for n,h in prior['source_sha256'].items():assert sha(PRIOR/'source'/n)==h,n
 external=dict(prior['external_source_sha256']);external.update({p.TRAIN+'/source/'+n:h for n,h in tp['source_sha256'].items()})
 for n,h in external.items():assert sha(ROOT/n)==h,n
 assert sha(PRIOR/'report.json')==p.REF_REPORT_SHA and sha(PRIOR/'evaluate_1536/predictions.segm.json')==p.REF_CACHE_SHA and sha(ROOT/p.OLD_WEIGHT)==p.OLD_WEIGHT_SHA
 assert read(TRAIN/'formal/runs/geometry_long/report.json')['checkpoint_sha256']==sha(ROOT/p.WEIGHT)
 assert shutil.disk_usage(ROOT).free>30*2**30;compile(transformed(),'<G system supervisor>','exec')
 (OUT/'logs').mkdir(parents=True);shutil.copytree(PRIOR/'source',OUT/'source')
 for n in p.NEW:assert n not in prior['source_sha256'];shutil.copy2(Path(__file__).parent/n,OUT/'source'/n)
 files={n:sha(OUT/'source'/n) for n in (*prior['source_sha256'],*p.NEW)}
 prior_attempt=ROOT/'outputs/course_m2f_G_system_20261027_v1/arms/G'
 assert read(prior_attempt/'status.json')['status']=='failed' and not (prior_attempt/'gpu_gate.json').exists() and not (prior_attempt/'gpu_released.json').exists()
 assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner=dcf7b3859c4a4212a134e6c5f0d124c1','--format','{{.ID}}'],text=True).strip()
 failure_binding={str(q.relative_to(ROOT)):sha(q) for q in (prior_attempt/'plan.json',prior_attempt/'status.json',prior_attempt/'first_error.json',prior_attempt/'supervisor.log')}
 prior_gpu_attempt=ROOT/'outputs/course_m2f_G_system_20261027_recovery1/arms/G'
 assert read(prior_gpu_attempt/'status.json')['status']=='failed' and read(prior_gpu_attempt/'gpu_released.json')['original_lock_released']
 assert read(prior_gpu_attempt/'gpu_released.json')['total_seconds']==p.PREVIOUS_GPU_SECONDS and not (prior_gpu_attempt/'captures').exists()
 assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner=b7424773436f4c67b5a2f7db402f9a85','--format','{{.ID}}'],text=True).strip()
 failure_binding.update({str(q.relative_to(ROOT)):sha(q) for q in (prior_gpu_attempt/'plan.json',prior_gpu_attempt/'status.json',prior_gpu_attempt/'first_error.json',prior_gpu_attempt/'supervisor.log',prior_gpu_attempt/'logs/gpu.log',prior_gpu_attempt/'gpu_released.json')})
 plan=dict(prior_failed_attempts=failure_binding,prior_failed_attempt_GPU_seconds=p.PREVIOUS_GPU_SECONDS,protocol='G_single_checkpoint_frozen_nineview',source_sha256=files,inherited_sha256=prior['source_sha256'],external_source_sha256=external,source_bundle_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),owner=p.OWNER,selected_gpu=0,host='gpu7',gpu_uuid=p.UUID,image_id=p.IMAGE,start_unix=p.START,cpu_end=p.CPU_END,gpu_end=p.GPU_END,deadline=p.END,arm_gpu_max=p.GPU_MAX,round_gpu_max=p.ROUND_GPU_MAX,prior_GPU_seconds=read(TRAIN/'gpu_released.json')['total_seconds'],new_weight=p.WEIGHT,new_weight_sha256=sha(ROOT/p.WEIGHT),primary=p.PRIOR,primary_report_sha256=p.REF_REPORT_SHA,primary_cache_sha256=p.REF_CACHE_SHA,training_updates=0,new_full_AP_runs=1,old_AP_runs=0)
 save(OUT/'plan.json',plan);save(BATCH/'status.json',dict(status='running',current='G',arms={'G':'preparing'}))
 with (OUT/'supervisor.log').open('xb',buffering=0) as log:
  child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/g_system_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')));print(json.dumps(read(OUT/'launch.json')))
def worker():
 ns=dict(l.__dict__,OUT=OUT,BATCH=BATCH,p=p,gproto=p,s=s,Locks=Locks,lease=lease,private_holders=private_holders,verify_uuid=verify_uuid)
 exec(compile(transformed(),'<G system bounded supervisor>','exec'),ns);ns['worker']()
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()

"""Bounded GPU8/GPU1 inference with original coordination + UUID locks."""
import fcntl,hashlib,inspect,json,os,shutil,subprocess,sys,time
from pathlib import Path
import g_submit_spec as cfg
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');PRIOR=ROOT/cfg.SYSTEM
sys.path.append(str(PRIOR/'source'))
import abc_spec as s
for k,v in dict(START=cfg.START,DEADLINE=cfg.DEADLINE,NAME=cfg.NAME,A_OWNER=cfg.OWNER,A_CPU_END=cfg.CPU_END,A_GPU_END=cfg.GPU_END,A_END=cfg.DEADLINE,A_GPU_MAX=cfg.GPU_MAX,GPU_MAX=cfg.GPU_MAX,IMAGE=cfg.IMAGE,LOCKS={1:cfg.PROJECT_LOCK}).items():setattr(s,k,v)
import abc_a_launcher as l
BATCH=ROOT/'outputs'/cfg.NAME;OUT=BATCH/'arms/test'
read,save,sha=l.read,l.save,l.sha

def verify_uuid():assert subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=8).strip()==cfg.UUID
class Locks:
 def __init__(self):
  self.files=[]
  try:
   assert os.uname().nodename==cfg.HOSTNAME
   for name,mode in ((cfg.PROJECT_LOCK,'r+'),(cfg.UUID_LOCK,'a+')):
    f=(ROOT/'outputs'/name).open(mode);self.files.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   verify_uuid()
  except BaseException:self.close();raise
 def fileno(self):return self.files[0].fileno()
 def close(self):
  for f in reversed(self.files):f.close()
  self.files=[]

def transformed():
 src=inspect.getsource(l.worker)
 changes={"jobs={'A':'pending'}":"jobs={'test':'pending'}",'for card in (0,1):':'for card in (1,)',"f=(ROOT/'outputs'/s.LOCKS[card]).open('r+')":"f=Locks()","rel='/workspace/outputs/'+s.NAME+'/arms/A'":"rel='/workspace/outputs/'+s.NAME+'/arms/test'","rel+'/source/runtime:/workspace/mmdetection'":"rel+'/source/runtime:/workspace/outputs/course_m2f_tiles_20261019_v1/source:/workspace/outputs/course_m2f_geometry_20261020_v1/source:/workspace/'+cfg.TRAIN+'/source:/workspace/mmdetection'","'abc_a_worker.py',mode":"'g_submit_worker.py',mode","state['jobs']['A']":"state['jobs']['test']","batch['arms']['A']":"batch['arms']['test']","'A_delivery_pending'":"'user_file_delivery_pending'","arms={'A':used}":"arms={'test':used}","'ai-m2f-abc-'":"'ai-m2f-Gcsv-'","OPENBLAS_NUM_THREADS='2'":"OPENBLAS_NUM_THREADS='2',OPENBLAS_CORETYPE='Prescott'","inspected=False":"inspected=False;last_resource_check=begin","original_lock_acquired=True,lock=":"gpu_uuid=cfg.UUID,uuid_guard=cfg.UUID_LOCK,original_lock_acquired=True,lock="}
 # Preserve Python syntax for the fixed GPU iteration.
 changes['for card in (0,1):']='for card in (1,):'
 for old,new in changes.items():assert old in src,old;src=src.replace(old,new)
 src=src.replace('selected=card;lock=f','verify_uuid();selected=card;lock=f')
 old='                time.sleep(2)'
 new="""                if gpu and inspected and now-last_resource_check>=30:
                    verify_uuid()
                    own={x.strip() for x in subprocess.check_output(['docker','top',name,'-eo','pid'],text=True,timeout=8).splitlines()[1:]}
                    actual=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=8).strip().splitlines()
                    assert set(actual)<=own,('external compute conflict',actual);last_resource_check=now
                time.sleep(2)"""
 assert src.count(old)==1
 return src.replace(old,new)

def start():
 assert os.uname().nodename==cfg.HOSTNAME and time.time()<cfg.CPU_END and not BATCH.exists(),'duplicate/expired preparation'
 assert read(PRIOR/'status.json')['status']=='complete' and read(PRIOR/'gpu_released.json')['original_lock_released']
 assert read(PRIOR/'system_readonly_acceptance.json')['status']=='passed'
 prior=read(PRIOR/'plan.json');assert len(prior['source_sha256'])==86 and len(prior['external_source_sha256'])==361
 assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+cfg.OWNER,'--format','{{.ID}}'],text=True).strip()
 for n,h in prior['source_sha256'].items():assert sha(PRIOR/'source'/n)==h,n
 for n,h in prior['external_source_sha256'].items():assert sha(ROOT/n)==h,n
 assert sha(PRIOR/'report.json')==cfg.SYSTEM_REPORT_SHA and sha(ROOT/cfg.WEIGHT)==cfg.WEIGHT_SHA
 assert sha(ROOT/cfg.K)==cfg.K_SHA and sha(ROOT/cfg.INPUT)==cfg.INPUT_SHA
 assert shutil.disk_usage(ROOT).free>30*2**30;compile(transformed(),'<bounded G test supervisor>','exec')
 assert subprocess.check_output(['docker','image','inspect',cfg.IMAGE,'--format','{{.Id}}'],text=True).strip()==cfg.IMAGE
 (OUT/'logs').mkdir(parents=True);shutil.copytree(PRIOR/'source',OUT/'source')
 for n in cfg.NEW:assert n not in prior['source_sha256'];shutil.copy2(Path(__file__).parent/n,OUT/'source'/n)
 files=dict(prior['source_sha256'],**{n:sha(OUT/'source'/n) for n in cfg.NEW});assert len(files)==91
 plan=dict(protocol='authorized_best_G_test_once_gpu8_GPU1',source_sha256=files,inherited_sha256=prior['source_sha256'],external_source_sha256=prior['external_source_sha256'],owner=cfg.OWNER,selected_gpu=1,host='gpu8',gpu_uuid=cfg.UUID,start_unix=cfg.START,cpu_end=cfg.CPU_END,gpu_end=cfg.GPU_END,deadline=cfg.DEADLINE,arm_gpu_max=cfg.GPU_MAX,round_deadline=cfg.DEADLINE,round_gpu_max=cfg.GPU_MAX,source_bundle_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),accepted_system=cfg.SYSTEM,system_report_sha256=cfg.SYSTEM_REPORT_SHA,weight=cfg.WEIGHT,weight_sha256=cfg.WEIGHT_SHA,derived_K=cfg.K,derived_K_sha256=cfg.K_SHA,training_updates=0,AP_runs=0,test_images=2112,test_labels=False,upload_authorized=False,image=cfg.IMAGE,original_gpu1_lock=cfg.PROJECT_LOCK,uuid_guard=cfg.UUID_LOCK,parallel_gpu7_GQ_unchanged=True)
 save(OUT/'plan.json',plan);save(BATCH/'plan.json',plan);save(BATCH/'status.json',dict(status='running',current='test',arms={'test':'preparing'},upload_attempts=0));save(BATCH/'gpu_ledger.json',dict(selected_gpu=1,total_seconds=0,arms={},max_seconds=cfg.GPU_MAX))
 with (OUT/'supervisor.log').open('xb',buffering=0) as log:
  c=subprocess.Popen([sys.executable,'-u',str(OUT/'source/g_submit_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 save(OUT/'launch.json',dict(pid=c.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')));print(json.dumps(read(OUT/'launch.json')))
def worker():
 ns=dict(l.__dict__,OUT=OUT,BATCH=BATCH,cfg=cfg,s=s,Locks=Locks,verify_uuid=verify_uuid);exec(compile(transformed(),'<G test single GPU1>','exec'),ns);ns['worker']()
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()

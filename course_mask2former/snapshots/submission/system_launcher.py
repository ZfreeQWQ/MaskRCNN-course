"""Use existing owner/lock supervisor with one fixed-card finite B-system lease."""
import inspect,os,shutil,subprocess,sys,time,json
from pathlib import Path
import system_spec as p
import abc_spec as s
for key,value in dict(START=p.START,DEADLINE=p.DEADLINE,NAME=p.NAME,A_OWNER=p.OWNER,A_CPU_END=p.CPU_END,A_GPU_END=p.B_GPU_END,A_END=p.B_END,A_GPU_MAX=p.B_GPU_MAX,GPU_MAX=p.GPU_MAX).items():setattr(s,key,value)
import abc_a_launcher as l
ROOT=l.ROOT;BATCH=ROOT/'outputs'/p.NAME;OUT=BATCH/'arms/B_into_A';PRIOR=ROOT/p.PRIOR

def transformed():
    src=inspect.getsource(l.worker)
    changes={"jobs={'A':'pending'}":"jobs={'B_into_A':'pending'}",'for card in (0,1):':'for card in (0,):',"rel='/workspace/outputs/'+s.NAME+'/arms/A'":"rel='/workspace/outputs/'+s.NAME+'/arms/B_into_A'","rel+'/source/runtime:/workspace/mmdetection'":"rel+'/source/runtime:/workspace/outputs/course_m2f_tiles_20261019_v1/source:/workspace/outputs/course_m2f_geometry_20261020_v1/source:/workspace/mmdetection'","'abc_a_worker.py',mode":"'system_worker.py',mode","state['jobs']['A']":"state['jobs']['B_into_A']","batch['arms']['A']":"batch['arms']['B_into_A']","'A_delivery_pending'":"'B_system_delivery_pending'","arms={'A':used}":"arms={'B_into_A':used}","'ai-m2f-abc-'":"'ai-m2f-system-'"}
    for old,new in changes.items():assert old in src,old;src=src.replace(old,new)
    return src

def start():
    assert time.time()<p.CPU_END and not BATCH.exists(),'duplicate/expired system preparation'
    assert l.read(ROOT/p.B/'status.json')['status']=='complete' and l.read(ROOT/p.C/'status.json')['status']=='complete'
    for old in (p.PRIOR,p.B,p.C):assert l.read(ROOT/old/'gpu_released.json')['original_lock_released']
    for owner in ('7f728d93869445c6bd924367569fe2431','bdfc28c7bc634e6c929555b2883eb728','9457c0ccbc8e46a3bee19d73357f9d06',p.OWNER):assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+owner,'--format','{{.ID}}'],text=True).strip()
    prior=l.read(PRIOR/'plan.json');assert len(prior['source_sha256'])==74
    for n,h in prior['source_sha256'].items():assert l.sha(PRIOR/'source'/n)==h
    for n,h in prior['external_source_sha256'].items():assert l.sha(ROOT/n)==h
    assert shutil.disk_usage(ROOT).free>30*2**30;transformed()
    (OUT/'logs').mkdir(parents=True);shutil.copytree(PRIOR/'source',OUT/'source')
    for n in p.NEW:assert n not in prior['source_sha256'];shutil.copy2(Path(__file__).parent/n,OUT/'source'/n)
    files=dict(prior['source_sha256'],**{n:l.sha(OUT/'source'/n) for n in p.NEW});assert len(files)==78
    plan=dict(protocol='B_into_A_full_system',source_sha256=files,inherited_sha256=prior['source_sha256'],external_source_sha256=prior['external_source_sha256'],owner=p.OWNER,selected_gpu=0,start_unix=p.START,cpu_end=p.CPU_END,gpu_end=p.B_GPU_END,deadline=p.B_END,arm_gpu_max=p.B_GPU_MAX,round_deadline=p.DEADLINE,round_gpu_max=p.GPU_MAX,source_commit=os.environ['SOURCE_COMMIT'],baseline=p.PRIOR,baseline_report_sha256=p.REF_SHA,new_weight=p.WEIGHT,new_weight_sha256=p.WEIGHT_SHA,training_updates=0,new_full_AP_runs=1,old_full_AP_runs=0,image=p.IMAGE)
    l.save(OUT/'plan.json',plan);l.save(BATCH/'plan.json',dict(protocol='conditional_B_then_C_full_system',start_unix=p.START,deadline=p.DEADLINE,GPU_max=p.GPU_MAX,max_new_full_val=2,main_training=0,head_fitting=0,selected_gpu=0));l.save(BATCH/'status.json',dict(status='running',current='B_into_A',arms={'B_into_A':'preparing','C_into_retained':'authorized_pending_compatibility'}));l.save(BATCH/'gpu_ledger.json',dict(selected_gpu=0,total_seconds=0,arms={},max_seconds=p.GPU_MAX))
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        c=subprocess.Popen([sys.executable,'-u',str(OUT/'source/system_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    l.save(OUT/'launch.json',dict(pid=c.pid,unix=time.time(),plan_sha256=l.sha(OUT/'plan.json')));print(json.dumps(l.read(OUT/'launch.json')))

def worker():
    ns=dict(l.__dict__,OUT=OUT,BATCH=BATCH);exec(compile(transformed(),'<finite B system>','exec'),ns);ns['worker']()
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()

"""Original coordination locks, exact owner, fixed budget; launch A once."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import abc_spec as s
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');BATCH=ROOT/'outputs'/s.NAME;OUT=BATCH/'arms/A'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def lease(now,arm_used,total_used):
    hard=min(s.A_GPU_END,now+s.A_GPU_MAX-arm_used,now+s.GPU_MAX-total_used)
    if hard-now<60:raise TimeoutError('No finite A GPU headroom')
    return hard,hard-20

def start():
    assert time.time()<s.A_CPU_END and not BATCH.exists(),'Duplicate or expired preparation'
    prior=ROOT/s.RUNTIME;inherited=read(prior/'plan.json')['source_sha256']
    assert read(prior/'report.json')['status']=='passed'
    assert read(ROOT/s.NOISE/'status.json')['status']=='complete'
    for owner in ('f4d0be561fc14e9789cab609579bebec',s.A_OWNER):
        assert not subprocess.check_output(['docker','ps','-a','--filter','label=ai-course-directions-owner='+owner,'--format','{{.ID}}'],text=True).strip()
    assert shutil.disk_usage(ROOT).free>30*2**30
    for n,h in inherited.items():assert sha(prior/'source'/n)==h,n
    (OUT/'logs').mkdir(parents=True);(OUT/'source/runtime').mkdir(parents=True)
    for n in inherited:
        dest=OUT/'source/runtime'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(prior/'source'/n,dest)
    here=Path(__file__).parent
    for n in s.NEW:shutil.copy2(here/n,OUT/'source'/n)
    files={str(p.relative_to(OUT/'source')):sha(p) for p in (OUT/'source').rglob('*.py')}
    assert all(files['runtime/'+n]==h for n,h in inherited.items())
    image=subprocess.check_output(['docker','image','inspect',s.IMAGE,'--format','{{.Id}}'],text=True).strip();assert image==s.IMAGE
    plan=dict(protocol='ABC_A_frozen_pointnoise_best9views',source_sha256=files,inherited_sha256=inherited,
        image_id=image,owner=s.A_OWNER,start_unix=s.START,cpu_end=s.A_CPU_END,gpu_end=s.A_GPU_END,deadline=s.A_END,
        arm_gpu_max=s.A_GPU_MAX,round_deadline=s.DEADLINE,round_gpu_max=s.GPU_MAX,training_updates=0,full_val_AP_runs=1,
        reference=s.REFERENCE,reference_sha256=s.REF_SHA,new_weight=s.WEIGHT,new_weight_sha256=s.WEIGHT_SHA,selected_gpu=None)
    save(OUT/'plan.json',plan)
    save(BATCH/'plan.json',dict(protocol='A_B_C_serial',start_unix=s.START,deadline=s.DEADLINE,gpu_max=s.GPU_MAX,
        arms=['A','B','C'],limits={'A':{'wall':21600,'gpu':18000},'B':{'wall':64800,'gpu':54000},'C':{'wall':21600,'gpu':10800}},
        extension_applied=False,CNN_paused=True,A_plan_sha256=sha(OUT/'plan.json')))
    save(BATCH/'status.json',dict(status='running',current='A',arms={'A':'preparing','B':'authorized_pending_implementation','C':'authorized_pending_implementation'}))
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/abc_a_launcher.py'),'worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')));print(json.dumps(read(OUT/'launch.json')))

def worker():
    with (OUT/'launch_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),unix=time.time()),f)
    plan=read(OUT/'plan.json');state=dict(status='running',phase='prep',jobs={'A':'pending'},selected_gpu=None,steps=[])
    child=None;name=None;lock=None;acquired=None;used=0.;leases=[];selected=None;stop=False;stage='prep'
    def update(**kw):state.update(kw,unix=time.time());save(OUT/'status.json',state)
    def stop_signal(*args):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_signal)
    def intact():assert all(sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
    def release():
        nonlocal lock,acquired,used
        if lock is not None:lock.close();lock=None
        if acquired is not None:
            now=time.time();used+=now-acquired;leases.append(dict(start=acquired,finish=now,seconds=now-acquired,gpu=selected,original_lock_released=True));acquired=None
            save(OUT/'gpu_released.json',dict(total_seconds=used,leases=leases,original_lock_released=True))
            save(BATCH/'gpu_ledger.json',dict(selected_gpu=selected,total_seconds=used,arms={'A':used},max_seconds=s.GPU_MAX))
    def acquire():
        nonlocal lock,acquired,selected,used
        errors=[]
        for card in (0,1):
            f=None;start=None
            try:
                f=(ROOT/'outputs'/s.LOCKS[card]).open('r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
                raw=subprocess.check_output(['nvidia-smi','-i',str(card),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
                index,mem,util=map(int,raw.strip().split(','));assert index==card and mem<=512 and util<=10,raw
                procs=subprocess.check_output(['nvidia-smi','-i',str(card),'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8).strip();assert not procs,procs
                selected=card;lock=f;f=None;acquired=start;hard,work=lease(start,used,used)
                save(OUT/'gpu_gate.json',dict(original_lock_acquired=True,lock=str(ROOT/'outputs'/s.LOCKS[card]),gpu=card,memory_mib=mem,utilization=util,compute=procs,unix=start,hard_end=hard,worker_end=work,wait_seconds=0))
                save(BATCH/'gpu_ledger.json',dict(selected_gpu=card,total_seconds=used,active_owner=s.A_OWNER,max_seconds=s.GPU_MAX));update(selected_gpu=card);return work
            except (OSError,AssertionError,subprocess.SubprocessError) as e:errors.append(dict(gpu=card,error=repr(e)))
            finally:
                if f is not None:
                    f.close()
                    if start is not None:
                        now=time.time();used+=now-start;leases.append(dict(start=start,finish=now,seconds=now-start,gpu=card,idle_check_failed=True,original_lock_released=True))
        save(OUT/'resource_gate_failure.json',errors);raise RuntimeError('No original lock + idle GPU; zero wait')
    def cleanup():
        if name:
            p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=8)
            if p.returncode==0:
                d=json.loads(p.stdout)[0];assert d['Config']['Labels']['ai-course-directions-owner']==s.A_OWNER
                subprocess.run(['docker','stop','-t','5',name],check=True,timeout=15)
        if child is not None:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
    def execute(mode,end,gpu=False):
        nonlocal child,name,stage
        stage=mode;assert time.time()<min(end,s.A_END);intact();name='ai-m2f-abc-'+s.A_OWNER[:10]+'-'+mode
        rel='/workspace/outputs/'+s.NAME+'/arms/A'
        cmd=['docker','run','--rm','--pull','never','--name',name,'--label','ai-course-directions-owner='+s.A_OWNER,'--network','none','--read-only','--cpus','8','--memory','32g','--shm-size','4g','--user',f'{os.getuid()}:{os.getgid()}',
            '-v',str(ROOT)+':/workspace:ro','-v',str(ROOT)+':'+str(ROOT)+':ro','-v',str(OUT)+':'+rel+':rw','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','-w',rel+'/source']
        env=dict(PYTHONPATH=rel+'/source:'+rel+'/source/runtime:/workspace/mmdetection',PATH='/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',XDG_CACHE_HOME='/tmp/cache',MPLCONFIGDIR='/tmp/matplotlib',AI_CLASS_TEST_DEVICE='cpu')
        if gpu:cmd+=['--gpus','device='+str(selected)]
        else:cmd+=['--runtime','runc'];env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
        for k,v in env.items():cmd+=['-e',k+'='+v]
        cmd += [s.IMAGE,'python','abc_a_worker.py',mode]
        save(OUT/(mode+'_command.json'),cmd);save(OUT/'phase_deadlines.json',dict(worker_end=end));begin=time.time();update(phase=mode,container=name)
        with (OUT/'logs'/(mode+'.log')).open('xb',buffering=0) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True);update(client_pid=child.pid);inspected=False
            while child.poll() is None:
                now=time.time();stamp=max(begin,(OUT/'progress.json').stat().st_mtime if (OUT/'progress.json').exists() else begin,(OUT/'logs'/(mode+'.log')).stat().st_mtime)
                if stop or (OUT/'STOP').exists():raise RuntimeError('Stop requested')
                if now>=min(end,s.A_END):raise TimeoutError('A phase deadline')
                if now-stamp>300:raise TimeoutError('5min without durable progress')
                if not inspected and now-begin>1:
                    p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=8)
                    if p.returncode==0:
                        d=json.loads(p.stdout)[0];h=d['HostConfig'];assert d['Config']['Labels']['ai-course-directions-owner']==s.A_OWNER and h['ReadonlyRootfs'] and h['NetworkMode']=='none' and h['NanoCpus']==8000000000
                        if gpu:assert h['DeviceRequests'][0]['DeviceIDs']==[str(selected)]
                        else:assert not h['Devices'] and not h['DeviceRequests'] and h['Runtime']=='runc'
                        save(OUT/(mode+'_container.json'),dict(owner=s.A_OWNER,devices=h['DeviceRequests'],memory=h['Memory'],runtime=h['Runtime'],network=h['NetworkMode'],readonly=h['ReadonlyRootfs'],unix=now));inspected=True
                time.sleep(2)
        assert child.returncode==0,(mode,child.returncode)
        assert inspected
        state['steps'].append(dict(phase=mode,start=begin,finish=time.time(),exit_code=0));child=None;name=None;intact();update(container=None,client_pid=None)
    try:
        update();execute('prep',s.A_CPU_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
        end=acquire();execute('gpu',end,True);release();assert used<=s.A_GPU_MAX
        execute('finish',s.A_END-60);assert read(OUT/'report.json')['status']=='passed'
        save(OUT/'terminal_acceptance.json',dict(status='passed',report_sha256=sha(OUT/'report.json'),comparison_sha256=sha(OUT/'comparison.json'),source_bound=True,GPU_seconds=used,unix=time.time()))
        state['jobs']['A']='passed';update(status='complete',phase='complete')
        batch=read(BATCH/'status.json');batch['arms']['A']='complete';batch['current']='A_delivery_pending';save(BATCH/'status.json',batch)
    except BaseException as e:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(phase=stage,error=repr(e),unix=time.time()))
        try:cleanup()
        finally:
            release();update(status='failed',phase=stage);batch=read(BATCH/'status.json');batch['arms']['A']='failed';batch['current']='A_delivery_pending';save(BATCH/'status.json',batch)
        raise
if __name__=='__main__':start() if sys.argv[1]=='start' else worker()

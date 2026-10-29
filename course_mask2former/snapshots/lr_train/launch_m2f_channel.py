"""One card, original locks, bounded CP+channel30k; no retries or mutable-ledger binding."""
import fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import m2f_channel_spec as s
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/s.NAME
FROZEN=ROOT/'outputs/course_directions_20261015_v1/source'
if FROZEN.is_dir():sys.path.append(str(FROZEN))
import run_course_directions as q
from launch_nyu_sweep import atomic_json as save,sha256 as sha
def read(p):return json.loads(Path(p).read_text())

def freeze(commit):
    assert time.time()<s.CPU_END and not OUT.exists()
    assert len(commit)==40 and all(c in '0123456789abcdef' for c in commit)
    prior=read(FROZEN.parent/'plan.json');assert all(sha(FROZEN/k)==v for k,v in prior['source_sha256'].items())
    provenance,reference=q.verify_inputs(ROOT)
    assert sha(ROOT/s.REFERENCE/'report.json')==s.REFERENCE_SHA
    assert (ROOT/s.REFERENCE/'predictions.segm.json').is_file()
    assert shutil.disk_usage(ROOT).free>=12*2**30
    OUT.mkdir();(OUT/'logs').mkdir();shutil.copytree(FROZEN,OUT/'source')
    origin=Path(__file__).resolve().parent
    for name in s.NEW:
        target=OUT/'source'/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(origin/name,target)
    # The accepted source manifest also contains the category JSON. Preserve it.
    files={name:sha(OUT/'source'/name) for name in (*prior['source_sha256'],*s.NEW)}
    assert all(files[k]==v for k,v in prior['source_sha256'].items())
    image=subprocess.check_output(['docker','image','inspect',q.IMAGE,'--format','{{.Id}}'],text=True).strip()
    plan=dict(protocol='single_CP_channel30k_v1',source_commit=commit,source_sha256=files,inherited_sha256=prior['source_sha256'],
        workspace=str(ROOT),output=str(OUT),image_id=image,owner_token=s.OWNER,expected_provenance=provenance,
        reference_baseline=reference,primary_reference=s.REFERENCE,primary_reference_sha256=s.REFERENCE_SHA,
        start_unix=s.START,cpu_end=s.CPU_END,smoke_end=s.SMOKE_END,formal_end=s.FORMAL_END,eval_end=s.EVAL_END,
        deadline_unix=s.DEADLINE,gpu_max_seconds=s.GPU_MAX,lock_wait_seconds=0,gpu_candidates=[0,1],CNN_paused=True,
        new_AP_runs=1,baseline_cached_AP_runs=1,steps=30000)
    save(OUT/'plan.json',plan)
    with (OUT/'supervisor.log').open('xb',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(OUT/'source/launch_m2f_channel.py'),'worker'],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(OUT/'launch.json',dict(pid=child.pid,unix=time.time(),plan_sha256=sha(OUT/'plan.json')))
    print(json.dumps(read(OUT/'launch.json')))

def worker():
    plan=read(OUT/'plan.json');assert not (OUT/'status.json').exists()
    with (OUT/'launch_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),unix=time.time()),f)
    state=dict(status='running',phase='cpu',jobs={s.JOB:'pending'},selected_gpu=None,steps=[])
    lock=None;acquired=None;used=0.;leases=[];selected=None;child=None;name=None;stage='cpu';stop=False
    def update(**kw):state.update(kw,unix=time.time());save(OUT/'status.json',state)
    def signal_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal_stop)
    def intact():
        assert all(sha(OUT/'source'/k)==v for k,v in plan['source_sha256'].items())
    def release():
        nonlocal lock,acquired,used
        if lock is not None:lock.close();lock=None
        if acquired is not None:
            now=time.time();seconds=now-acquired;used+=seconds
            leases.append(dict(start=acquired,finish=now,seconds=seconds,gpu=selected,original_lock_released=True))
            acquired=None;save(OUT/'gpu_released.json',dict(total_seconds=used,leases=leases,original_lock_released=True))
    def acquire(limit,absolute):
        nonlocal selected,lock,acquired,used
        errors=[]
        for card in ([selected] if selected is not None else [0,1]):
            f=None;started=None
            try:
                f=(ROOT/'outputs'/s.LOCKS[card]).open('r+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                started=time.time()
                vals=subprocess.check_output(['nvidia-smi','-i',str(card),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
                index,mem,util=map(int,vals.strip().split(','));assert index==card and mem<=512 and util<=10,vals
                procs=subprocess.check_output(['nvidia-smi','-i',str(card),'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8).strip();assert not procs,procs
                selected=card;lock=f;f=None;acquired=started
                hard,work=s.lease_end(started,used,absolute,limit)
                save(OUT/(stage+'_gate.json'),dict(original_lock_acquired=True,lock=str(ROOT/'outputs'/s.LOCKS[card]),gpu=card,
                    memory_mib=mem,utilization=util,compute=procs,additional_wait_seconds=0,unix=started,hard_end=hard,worker_end=work))
                update(selected_gpu=selected);return hard,work
            except (OSError,AssertionError,subprocess.SubprocessError) as e:
                errors.append(dict(gpu=card,error=repr(e)))
            finally:
                if f is not None:
                    f.close()
                    if started is not None:
                        finished=time.time();seconds=finished-started;used+=seconds
                        leases.append(dict(start=started,finish=finished,seconds=seconds,gpu=card,original_lock_released=True,idle_check_failed=True))
                        save(OUT/'gpu_released.json',dict(total_seconds=used,leases=leases,original_lock_released=True))
        save(OUT/'resource_gate_failure.json',errors);raise RuntimeError('No single GPU passes original lock and actual idle; zero wait')
    def cleanup():
        if name:
            r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=8)
            if r.returncode==0:
                d=json.loads(r.stdout)[0]
                assert d['Config']['Labels']['ai-course-directions-owner']==s.OWNER
                subprocess.run(['docker','stop','-t','5',name],check=True,timeout=15)
        if child is not None:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
    def execute(mode,end,gpu=False):
        nonlocal child,name,stage
        stage=mode;assert time.time()<end;intact();name='ai-m2f-channel-'+s.OWNER[:10]+'-'+mode
        rel='/workspace/outputs/'+s.NAME
        cmd=['docker','run','--rm','--pull','never','--name',name,'--label','ai-course-directions-owner='+s.OWNER,
            '--network','none','--read-only','--cpus','4','--memory','32g' if gpu else '12g','--shm-size','4g',
            '--user',f'{os.getuid()}:{os.getgid()}','-v',str(ROOT)+':/workspace:ro','-v',str(ROOT)+':'+str(ROOT)+':ro',
            '-v',str(OUT)+':'+rel+':rw','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','-w',rel+'/source']
        env=dict(PYTHONPATH=rel+'/source:'+rel+'/source/tests:/workspace/mmdetection',
            PATH='/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            XDG_CACHE_HOME='/tmp/cache',MPLCONFIGDIR='/tmp/matplotlib',AI_CLASS_TEST_DEVICE='cpu')
        if gpu:cmd+=['--gpus','device='+str(selected)]
        else:cmd+=['--runtime','runc'];env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
        for k,v in env.items():cmd+=['-e',k+'='+v]
        cmd += [plan['image_id'],'python','m2f_channel_worker.py',mode]
        save(OUT/(mode+'_command.json'),cmd);save(OUT/'phase_deadlines.json',dict(worker_end=end))
        begin=time.time();update(phase=mode,container=name)
        with (OUT/'logs'/(mode+'.log')).open('xb',buffering=0) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            update(client_pid=child.pid);inspected=False
            while child.poll() is None:
                now=time.time()
                if stop or (OUT/'STOP').exists():raise RuntimeError('Stop requested')
                if now>=end or now>=s.DEADLINE:raise TimeoutError('Immutable '+mode+' deadline')
                stamp=max(begin,(OUT/'progress.json').stat().st_mtime if (OUT/'progress.json').exists() else begin,(OUT/'logs'/(mode+'.log')).stat().st_mtime)
                if now-stamp>300:raise TimeoutError('5min without durable progress')
                if not inspected and now-begin>1:
                    r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=8)
                    if r.returncode==0:
                        d=json.loads(r.stdout)[0];h=d['HostConfig'];assert d['Config']['Labels']['ai-course-directions-owner']==s.OWNER
                        assert h['NetworkMode']=='none' and h['ReadonlyRootfs'] and h['NanoCpus']==4*10**9
                        if gpu:assert h['DeviceRequests'][0]['DeviceIDs']==[str(selected)]
                        else:assert not h['DeviceRequests'] and not h['Devices'] and h['Runtime']=='runc'
                        save(OUT/(mode+'_container.json'),dict(owner=s.OWNER,devices=h['DeviceRequests'],memory=h['Memory'],network=h['NetworkMode'],readonly=h['ReadonlyRootfs'],runtime=h['Runtime'],unix=now));inspected=True
                time.sleep(2)
        assert child.returncode==0,(mode,child.returncode)
        assert inspected
        state['steps'].append(dict(phase=mode,start=begin,finish=time.time(),exit_code=0));update(container=None,client_pid=None)
        child=None;name=None;intact()
    def accept(smoke):
        phase='smoke' if smoke else 'formal';j=s.job(smoke)
        accepted=q.validate_report(OUT,plan,phase,j);run=OUT/phase/'runs'/s.JOB
        r=read(run/'report.json');a=read(run/'channel_audit.json');n=20 if smoke else 30000
        assert r['config_checks']['channel_only'] and r['active_parameters']==s.ACTIVE and r['state_tensor_bytes']==s.STATE
        assert a['status']=='passed' and a['iterations']==n and all(a['updated'].values()) and a['calls']['train']==n
        assert r['direction_activity']['copy_paste']['calls']==n*2 and r['direction_activity']['copy_paste']['pasted_instances']>0
        for i in range(n if smoke else 5000,n+1,n if smoke else 5000):
            ids=read(run/f'validation_ids_{i}.json');expected=read(ROOT/q.DATA/'annotations'/('smoke_val.json' if smoke else 'instances_val.json'))
            assert ids==[x['id'] for x in expected['images']]
        save(OUT/(phase+'_acceptance.json'),dict(**accepted,channel_audit_sha256=sha(run/'channel_audit.json'),actual_binding=True))
    try:
        update();execute('cpu',s.CPU_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
        stage='smoke';hard,work=acquire(900,s.SMOKE_END)
        for phase in ('smoke','formal'):
            folder=OUT/phase;folder.mkdir();(folder/'runs').mkdir();(folder/'source').symlink_to(OUT/'source',target_is_directory=True)
            execute(phase,work,True);accept(phase=='smoke');release();assert used<=s.GPU_MAX
            if phase=='smoke':
                stage='formal';hard,work=acquire(34200,s.FORMAL_END)
        state['jobs'][s.JOB]='trained_accepted';update()
        stage='evaluate';hard,work=acquire(2700,s.EVAL_END);execute('evaluate',work,True);release();assert used<=s.GPU_MAX
        execute('finish',s.DEADLINE-60)
        r=read(OUT/'comparison.json');assert r['status']=='passed'
        intact();state['jobs'][s.JOB]='passed'
        save(OUT/'terminal_acceptance.json',dict(status='passed',comparison_sha256=sha(OUT/'comparison.json'),
            new_report_sha256=sha(OUT/'evaluate_1536/report.json'),GPU_seconds=used,unix=time.time(),
            checkpoints={p.name:sha(p) for p in (OUT/'formal/runs'/s.JOB).glob('iter_*.pth')}))
        update(status='complete',phase='complete',container=None)
    except BaseException as e:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(error=repr(e),phase=stage,unix=time.time()))
        try:cleanup()
        finally:release();update(status='failed',phase=stage)
        raise

if __name__=='__main__':
    freeze(sys.argv[2]) if sys.argv[1]=='start' else worker()

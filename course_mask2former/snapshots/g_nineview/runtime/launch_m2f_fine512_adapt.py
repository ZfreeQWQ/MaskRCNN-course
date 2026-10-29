"""Two nonblocking GPU0 leases, cumulative1800s, original absolute deadlines."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import launch_m2f_rle_passthrough as reused
import run_m2f_fine512_adapt as worker
from run_m2f_fine512_adapt import START,PREP_END,CAPTURE_END,LABEL_END,TRAIN_END,ANALYSIS_END,DEADLINE,NAME,OWNER,read,save,sha
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/NAME
reused.ROOT,reused.OUT,reused.OWNER=ROOT,OUT,OWNER

def command(stage):
    gpu=stage in ('capture','traincheck');_,cmd=reused.command('gpu' if gpu else 'cpu')
    name='ai-m2f-512adapt-'+OWNER[:10]+'-'+stage
    cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--memory')+1]='32g' if gpu else '8g'
    cmd[cmd.index('m2f_rle_passthrough.py'):]=['run_m2f_fine512_adapt.py',stage]
    return name,cmd

def stop_owned(name):
    r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=6)
    if r.returncode:return
    item=json.loads(r.stdout)[0];assert item['Config']['Labels'].get('ai-course-directions-owner')==OWNER
    subprocess.run(['docker','stop','-t','5',name],timeout=12,check=True)

def main():
    assert time.time()<PREP_END and not (OUT/'plan.json').exists(),'Duplicate or late'
    plan=dict(start_unix=START,prep_end=PREP_END,capture_end=CAPTURE_END,label_end=LABEL_END,train_end=TRAIN_END,analysis_end=ANALYSIS_END,deadline_unix=DEADLINE,
        source_commit=sys.argv[1],owner=OWNER,source_sha256={p.name:sha(p) for p in (OUT/'source').glob('*.py')},
        original_lock=str(ROOT/'outputs/.course_directions_gpu0.lock'),lock_wait_seconds=0,gpu=0,gpu_max_seconds=1800,capture_gpu_max_seconds=1480,
        only_change='new512source uses a2000step adapted verifier copy; old-source probabilities/models/rules frozen',
        fit_images=384,check_images=128,fresh_seed=20261022,m2f_eval_calls=2688,steps=2000,
        minimum_fit=dict(positive_gt=32,positive_scenes=16,positive_area64_gt=8,negative_candidates=128,negative_scenes=32),
        gate='new128 APs strict positive vs best5 and tiny net nonnegative; no fullval',official_best_aps=worker.old.BEST_APS,full_val=False,maskrcnn_paused=True)
    save(OUT/'plan.json',plan);save(OUT/'launch.json',dict(supervisor_pid=os.getpid(),unix=time.time()))
    name=None;process=None;phase='prep';lock=None;acquired=None;leases=[];phases=[]
    def release(failure=False):
        nonlocal lock,acquired
        if lock is not None:lock.close();lock=None
        if acquired is not None:
            ended=time.time();leases.append(dict(start=acquired,finish=ended,seconds=ended-acquired,failure_cleanup=failure));acquired=None
            save(OUT/'gpu_leases.json',dict(leases=leases,cumulative_seconds=sum(x['seconds'] for x in leases),original_lock_released=True))
    def execute(stage,end):
        nonlocal name,process,phase
        phase=stage;name,cmd=command(stage);begin=time.time();assert begin<end
        save(OUT/(stage+'_command.json'),cmd)
        with (OUT/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/'status.json',dict(status='running',phase=stage,container=name,client_pid=process.pid,stage_end=end,unix=time.time()))
            inspected=False
            while process.poll() is None:
                now=time.time()
                if now>=end:raise TimeoutError('Original '+stage+' deadline')
                if not inspected and now-begin>2:
                    raw=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=6)
                    if raw.returncode==0:
                        item=json.loads(raw.stdout)[0];h=item['HostConfig']
                        assert item['Config']['Labels'].get('ai-course-directions-owner')==OWNER
                        assert h['NetworkMode']=='none' and h['NanoCpus']==4*10**9 and h['ReadonlyRootfs']
                        if stage in ('capture','traincheck'):assert h['DeviceRequests'][0]['DeviceIDs']==['0'] and h['Memory']==32*2**30
                        else:assert not h['DeviceRequests'] and not h['Devices'] and h['Runtime']=='runc' and h['Memory']==8*2**30
                        save(OUT/(stage+'_container.json'),dict(owner=OWNER,devices=h['DeviceRequests'],runtime=h['Runtime'],cpus=h['NanoCpus'],memory=h['Memory'],network=h['NetworkMode'],readonly=True,unix=now));inspected=True
                if stage in ('capture','traincheck'):
                    stamp=(OUT/'progress.json').stat().st_mtime if (OUT/'progress.json').exists() else begin
                    if now-max(stamp,begin)>300:raise TimeoutError('5min without persistent progress')
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,start=begin,finished=time.time(),exit_code=0));save(OUT/'phases.json',phases)
    def gpu_gate(stage,absolute,maxseconds):
        nonlocal phase,lock,acquired
        phase=stage+'_gate';assert time.time()<absolute
        lock=(ROOT/'outputs/.course_directions_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
        index,memory,util=map(int,raw.strip().split(','));assert index==0 and memory<=512 and util<=10,raw
        procs=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8).strip();assert not procs,'GPU0 occupied: '+procs
        acquired=time.time();remaining=1800-sum(x['seconds'] for x in leases)-20
        if stage=='traincheck':assert remaining>=180,'Insufficient conservative head budget'
        end=min(absolute,acquired+min(maxseconds,remaining))
        save(OUT/(stage+'_gpu_gate.json'),dict(original_lock_acquired=True,additional_wait_seconds=0,gpu=0,memory_mib=memory,utilization=util,compute_processes=procs,unix=acquired,gpu_stop=end,remaining_cumulative_seconds=remaining))
        return end
    try:
        execute('prep',PREP_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
        end=gpu_gate('capture',CAPTURE_END,1480);save(OUT/'phase_deadlines.json',dict(capture_end=end))
        execute('capture',end);release();execute('labels',LABEL_END)
        if read(OUT/'space_gate.json')['passed']:
            end=gpu_gate('traincheck',TRAIN_END,1800);d=read(OUT/'phase_deadlines.json');d['train_end']=end;save(OUT/'phase_deadlines.json',d)
            execute('traincheck',end);release()
        execute('finish',ANALYSIS_END);report=read(OUT/'report.json');assert report['status']=='passed'
        assert all(sha(OUT/n)==h for n,h in report['artifacts_sha256'].items())
        assert all(sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
        assert read(OUT/'gpu_leases.json')['cumulative_seconds']<=1800
        save(OUT/'terminal_acceptance.json',dict(status='passed',report_sha256=sha(OUT/'report.json'),source_unchanged=True,unix=time.time()))
        save(OUT/'status.json',dict(status='complete',phase=report['outcome'],unix=time.time()))
    except BaseException as exc:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(error=repr(exc),phase=phase,unix=time.time()))
        if name:
            try:stop_owned(name)
            except Exception as err:save(OUT/'cleanup_error.json',dict(error=repr(err),unix=time.time()))
        if process is not None and process.poll() is None:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate()
        release(True);save(OUT/'status.json',dict(status='failed',phase=phase,unix=time.time()));raise
    finally:release()
if __name__=='__main__':main()

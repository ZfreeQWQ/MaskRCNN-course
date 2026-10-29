"""Original budget recovery, preserve complete128, no repeated inference."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import launch_m2f_fine512_adapt as common
import recover_m2f_fine512_adapt as worker
from recover_m2f_fine512_adapt import NAME,OWNER,START,PREP_END,CAPTURE_END,LABEL_END,TRAIN_END,ANALYSIS_END,DEADLINE,read,save,sha
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/NAME;PRIOR=ROOT/'outputs/course_m2f_fine512_adapt_20261021_v1'
common.OUT=common.reused.OUT=OUT;common.OWNER=common.reused.OWNER=OWNER
def command(stage):
    name,cmd=common.command(stage);cmd[cmd.index('run_m2f_fine512_adapt.py')]='recover_m2f_fine512_adapt.py';return name,cmd

def main():
    assert time.time()<CAPTURE_END and not (OUT/'plan.json').exists()
    original=read(PRIOR/'plan.json');assert original['deadline_unix']==DEADLINE and read(PRIOR/'status.json')['status']=='failed'
    leases=list(read(PRIOR/'gpu_leases.json')['leases']);assert sum(x['seconds'] for x in leases)<1480
    plan=dict(original,source_commit=sys.argv[1],owner=OWNER,source_sha256={p.name:sha(p) for p in (OUT/'source').glob('*.py')},
        recovery_of=str(PRIOR),original_plan_sha256=sha(PRIOR/'plan.json'),prior_gpu_seconds=sum(x['seconds'] for x in leases),
        repair='only explicit fit M2F backend flags; reuse complete128; original22CPU passed before prep deadline;2CPU targeted recovery under original capture deadline',additional_lock_wait=0)
    save(OUT/'plan.json',plan);save(OUT/'launch.json',dict(supervisor_pid=os.getpid(),unix=time.time()))
    lock=None;acquired=None;name=None;process=None;phase='recovery_prep';phases=[]
    def release(failed=False):
        nonlocal lock,acquired
        if lock is not None:lock.close();lock=None
        if acquired is not None:
            now=time.time();leases.append(dict(start=acquired,finish=now,seconds=now-acquired,failure_cleanup=failed));acquired=None
        save(OUT/'gpu_leases.json',dict(leases=leases,cumulative_seconds=sum(x['seconds'] for x in leases),original_lock_released=True,includes_prior_failed_allocation=True))
    def execute(stage,end):
        nonlocal process,name,phase
        phase=stage;name,cmd=command(stage);begin=time.time();assert begin<end;save(OUT/(stage+'_command.json'),cmd)
        with (OUT/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/'status.json',dict(status='running',phase=stage,container=name,client_pid=process.pid,stage_end=end,unix=time.time()))
            inspected=False
            while process.poll() is None:
                now=time.time()
                if now>=end:raise TimeoutError('Original '+stage+' deadline')
                if not inspected and now-begin>2:
                    q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=6)
                    if q.returncode==0:
                        x=json.loads(q.stdout)[0];h=x['HostConfig'];assert x['Config']['Labels']['ai-course-directions-owner']==OWNER
                        assert h['NetworkMode']=='none' and h['NanoCpus']==4*10**9 and h['ReadonlyRootfs']
                        if stage in ('capture','traincheck'):assert h['DeviceRequests'][0]['DeviceIDs']==['0'] and h['Memory']==32*2**30
                        else:assert not h['DeviceRequests'] and not h['Devices'] and h['Runtime']=='runc' and h['Memory']==8*2**30
                        save(OUT/(stage+'_container.json'),dict(owner=OWNER,devices=h['DeviceRequests'],runtime=h['Runtime'],cpus=h['NanoCpus'],memory=h['Memory'],network=h['NetworkMode'],readonly=True,unix=now));inspected=True
                if stage in ('capture','traincheck'):
                    stamp=(OUT/'progress.json').stat().st_mtime if (OUT/'progress.json').exists() else begin
                    if now-max(begin,stamp)>300:raise TimeoutError('5min no persistent progress')
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,start=begin,finished=time.time(),exit_code=0));save(OUT/'phases.json',phases)
    def gate(stage,absolute):
        nonlocal lock,acquired,phase
        phase=stage+'_gate';lock=(ROOT/'outputs/.course_directions_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
        index,memory,util=map(int,raw.strip().split(','));assert index==0 and memory<=512 and util<=10,raw
        procs=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8).strip();assert not procs
        acquired=time.time();used=sum(x['seconds'] for x in leases);remaining=(1480 if stage=='capture' else 1780)-used
        assert remaining>180;end=min(absolute,acquired+remaining)
        save(OUT/(stage+'_gpu_gate.json'),dict(original_lock_acquired=True,additional_wait_seconds=0,gpu=0,memory_mib=memory,utilization=util,compute_processes=procs,unix=acquired,gpu_stop=end,prior_cumulative_seconds=used,remaining_cumulative_seconds=1800-used))
        return end
    try:
        execute('prep',min(CAPTURE_END,time.time()+120));assert read(OUT/'recovery_acceptance.json')['status']=='passed'
        end=gate('capture',CAPTURE_END);save(OUT/'phase_deadlines.json',dict(capture_end=end));execute('capture',end);release()
        execute('labels',LABEL_END)
        if read(OUT/'space_gate.json')['passed']:
            end=gate('traincheck',TRAIN_END);d=read(OUT/'phase_deadlines.json');d['train_end']=end;save(OUT/'phase_deadlines.json',d);execute('traincheck',end);release()
        execute('finish',ANALYSIS_END);r=read(OUT/'report.json');assert r['status']=='passed'
        assert all(sha(OUT/n)==h for n,h in r['artifacts_sha256'].items()) and all(sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
        assert sum(x['seconds'] for x in leases)<=1800
        save(OUT/'terminal_acceptance.json',dict(status='passed',report_sha256=sha(OUT/'report.json'),source_unchanged=True,prior_failure_preserved=True,unix=time.time()))
        save(OUT/'status.json',dict(status='complete',phase=r['outcome'],unix=time.time()))
    except BaseException as e:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(error=repr(e),phase=phase,unix=time.time()))
        if name:
            try:common.stop_owned(name)
            except Exception as ee:save(OUT/'cleanup_error.json',dict(error=repr(ee),unix=time.time()))
        if process is not None and process.poll() is None:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate()
        release(True);save(OUT/'status.json',dict(status='failed',phase=phase,unix=time.time()));raise
    finally:
        if lock is not None or acquired is not None:release(True)
if __name__=='__main__':main()

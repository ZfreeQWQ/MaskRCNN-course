"""One GPU0 lease, immutable95min GPU and150min total budget."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import launch_m2f_rle_passthrough as reused
from run_m2f_fine512_fullval import START,PREP_END,GPU_END,ANALYSIS_END,DEADLINE,NAME,OWNER,read,save,sha
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/NAME
reused.ROOT,reused.OUT,reused.OWNER=ROOT,OUT,OWNER
def command(stage):
    _,cmd=reused.command('gpu' if stage=='gpu' else 'cpu');name='ai-m2f-512val-'+OWNER[:10]+'-'+stage
    cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--memory')+1]='32g' if stage=='gpu' else '8g'
    cmd[cmd.index('m2f_rle_passthrough.py'):]=['run_m2f_fine512_fullval.py',stage];return name,cmd
def stop_owned(name):
    r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=6)
    if r.returncode:return
    x=json.loads(r.stdout)[0];assert x['Config']['Labels']['ai-course-directions-owner']==OWNER
    subprocess.run(['docker','stop','-t','5',name],check=True,timeout=12)
def main():
    assert time.time()<PREP_END and not (OUT/'plan.json').exists()
    plan=dict(start_unix=START,prep_end=PREP_END,gpu_end=GPU_END,analysis_end=ANALYSIS_END,deadline_unix=DEADLINE,gpu_max_seconds=5700,owner=OWNER,source_commit=sys.argv[1],source_sha256={p.name:sha(p) for p in (OUT/'source').glob('*.py')},gpu=0,lock_wait_seconds=0,train_updates=0,m2f_calls=10633,full_val_evaluations=1,baseline_aps=.41399673960953054)
    save(OUT/'plan.json',plan);save(OUT/'launch.json',dict(supervisor_pid=os.getpid(),unix=time.time()))
    phase='prep';name=None;process=None;lock=None;acquired=None;phases=[]
    def release(failed=False):
        nonlocal lock,acquired
        if lock is not None:lock.close();lock=None
        if acquired is not None:
            end=time.time();save(OUT/'gpu_released.json',dict(start=acquired,finish=end,seconds=end-acquired,original_lock_released=True,failure_cleanup=failed));acquired=None
    def execute(stage,end):
        nonlocal phase,name,process
        phase=stage;name,cmd=command(stage);start=time.time();assert start<end
        save(OUT/(stage+'_command.json'),cmd)
        with (OUT/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/'status.json',dict(status='running',phase=stage,container=name,client_pid=process.pid,stage_end=end,unix=time.time()))
            inspected=False
            while process.poll() is None:
                now=time.time()
                if now>=end:raise TimeoutError('Original '+stage+' deadline')
                if not inspected and now-start>2:
                    r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=6)
                    if r.returncode==0:
                        x=json.loads(r.stdout)[0];h=x['HostConfig'];assert x['Config']['Labels']['ai-course-directions-owner']==OWNER
                        assert h['NetworkMode']=='none' and h['ReadonlyRootfs'] and h['NanoCpus']==4*10**9
                        if stage=='gpu':assert h['DeviceRequests'][0]['DeviceIDs']==['0'] and h['Memory']==32*2**30
                        else:assert not h['DeviceRequests'] and not h['Devices'] and h['Runtime']=='runc' and h['Memory']==8*2**30
                        save(OUT/(stage+'_container.json'),dict(owner=OWNER,devices=h['DeviceRequests'],runtime=h['Runtime'],cpus=h['NanoCpus'],memory=h['Memory'],network=h['NetworkMode'],readonly=True,unix=now));inspected=True
                if stage=='gpu':
                    stamp=(OUT/'progress.json').stat().st_mtime if (OUT/'progress.json').exists() else start
                    if now-max(start,stamp)>300:raise TimeoutError('5min without persistent progress')
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,start=start,finish=time.time(),exit_code=0));save(OUT/'phases.json',phases)
    try:
        execute('prep',PREP_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
        phase='gpu_gate';lock=(ROOT/'outputs/.course_directions_gpu0.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
        index,mem,util=map(int,raw.strip().split(','));assert index==0 and mem<=512 and util<=10,raw
        procs=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8).strip();assert not procs
        acquired=time.time();end=min(GPU_END,acquired+5680);assert end-acquired>=5100,'Insufficient conservative85min headroom'
        save(OUT/'gpu_gate.json',dict(original_lock_acquired=True,additional_wait_seconds=0,gpu=0,memory_mib=mem,utilization=util,compute_processes=procs,unix=acquired))
        save(OUT/'phase_deadlines.json',dict(gpu_end=end));execute('gpu',end);release()
        assert read(OUT/'gpu_released.json')['seconds']<=5700
        execute('finish',ANALYSIS_END);r=read(OUT/'report.json');assert r['status']=='passed'
        assert all(sha(OUT/n)==h for n,h in r['artifacts_sha256'].items()) and all(sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
        save(OUT/'terminal_acceptance.json',dict(status='passed',report_sha256=sha(OUT/'report.json'),source_unchanged=True,unix=time.time()))
        save(OUT/'status.json',dict(status='complete',phase='evaluated',unix=time.time()))
    except BaseException as e:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(error=repr(e),phase=phase,unix=time.time()))
        if name:
            try:stop_owned(name)
            except Exception as ee:save(OUT/'cleanup_error.json',dict(error=repr(ee),unix=time.time()))
        if process is not None and process.poll() is None:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate()
        release(True);save(OUT/'status.json',dict(status='failed',phase=phase,unix=time.time()));raise
    finally:
        if lock is not None or acquired is not None:release(True)
if __name__=='__main__':main()

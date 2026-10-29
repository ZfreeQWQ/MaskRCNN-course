"""One immutable90min pilot, zero lock wait, <=20min physicalGPU0."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import launch_m2f_rle_passthrough as reused
import run_m2f_fine512 as worker
from run_m2f_fine512 import START,PREP_END,CAPTURE_END,ANALYSIS_END,DEADLINE,NAME,OWNER,read,save,sha

ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/NAME
reused.ROOT,reused.OUT,reused.OWNER=ROOT,OUT,OWNER
def command(stage):
    gpu=stage=='capture';_,cmd=reused.command('gpu' if gpu else 'cpu')
    name='ai-m2f-fine512-'+OWNER[:10]+'-'+stage
    cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--memory')+1]='32g' if gpu else '8g'
    cmd[cmd.index('m2f_rle_passthrough.py'):]=['run_m2f_fine512.py',stage]
    return name,cmd

def stop_owned(name):
    r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=6)
    if r.returncode:return
    item=json.loads(r.stdout)[0];assert item['Config']['Labels'].get('ai-course-directions-owner')==OWNER
    subprocess.run(['docker','stop','-t','5',name],timeout=12,check=True)

def main():
    assert time.time()<PREP_END and not (OUT/'plan.json').exists(),'Duplicate or late'
    plan=dict(start_unix=START,prep_end=PREP_END,capture_end=CAPTURE_END,analysis_end=ANALYSIS_END,
        deadline_unix=DEADLINE,source_commit=sys.argv[1],owner=OWNER,
        source_sha256={p.name:sha(p) for p in (OUT/'source').glob('*.py')},
        original_lock=str(ROOT/'outputs/.course_directions_gpu0.lock'),lock_wait_seconds=0,
        gpu=0,gpu_max_seconds=1200,official_baseline_aps=worker.BEST_APS,maskrcnn_paused=True,
        only_change='four512fixedwindows at1536 augment tiny candidate sources; all models/rules frozen',
        windows=worker.WINDOWS,old_views=5,new_views=4,images=128,seed=20261021,
        weights_updated=False,full_val=False,minimum_opportunity_gt=8,minimum_opportunity_scenes=4,
        gate='paired128 train APs strict positive AND tiny net>=0 AND opportunity gate',
        one_original_budget=True)
    save(OUT/'plan.json',plan);save(OUT/'launch.json',dict(supervisor_pid=os.getpid(),unix=time.time()))
    process=None;name=None;phase='prep';lock=None;acquired=None;phases=[]
    def execute(stage,end):
        nonlocal name,process,phase
        phase=stage;name,cmd=command(stage);begin=time.time();assert begin<end
        save(OUT/(stage+'_command.json'),cmd)
        with (OUT/(stage+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/'status.json',dict(status='running',phase=stage,container=name,
                client_pid=process.pid,stage_end=end,unix=time.time()))
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
                        if stage=='capture':assert h['DeviceRequests'][0]['DeviceIDs']==['0'] and h['Memory']==32*2**30
                        else:assert not h['DeviceRequests'] and not h['Devices'] and h['Runtime']=='runc' and h['Memory']==8*2**30
                        save(OUT/(stage+'_container.json'),dict(owner=OWNER,devices=h['DeviceRequests'],
                            runtime=h['Runtime'],cpus=h['NanoCpus'],memory=h['Memory'],network=h['NetworkMode'],readonly=True,unix=now))
                        inspected=True
                if stage=='capture':
                    stamp=(OUT/'progress.json').stat().st_mtime if (OUT/'progress.json').exists() else begin
                    if now-max(stamp,begin)>300:raise TimeoutError('5min without persistent progress')
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,start=begin,finished=time.time(),exit_code=0));save(OUT/'phases.json',phases)
    try:
        execute('prep',PREP_END);assert read(OUT/'cpu_acceptance.json')['status']=='passed'
        phase='gpu_gate';assert time.time()<PREP_END
        lock=(ROOT/'outputs/.course_directions_gpu0.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8)
        index,memory,util=map(int,raw.strip().split(','));assert index==0 and memory<=512 and util<=10,raw
        procs=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=8).strip()
        assert not procs,'ActualGPU0 occupied: '+procs
        acquired=time.time();end=min(CAPTURE_END,acquired+1180)
        save(OUT/'gpu_gate.json',dict(original_lock_acquired=True,gpu=0,memory_mib=memory,utilization=util,
            compute_processes=procs,unix=acquired,gpu_stop=end,visibility_not_occupancy=True))
        save(OUT/'phase_deadlines.json',dict(capture_end=end))
        execute('capture',end)
        lock.close();lock=None
        save(OUT/'gpu_released.json',dict(unix=time.time(),original_lock_released=True,gpu_allocation_seconds=time.time()-acquired))
        execute('finish',ANALYSIS_END)
        report=read(OUT/'report.json');assert report['status']=='passed' and report['finished_unix']<ANALYSIS_END
        assert all(sha(OUT/n)==h for n,h in report['artifacts_sha256'].items())
        assert all(sha(OUT/'source'/n)==h for n,h in plan['source_sha256'].items())
        assert read(OUT/'gpu_released.json')['gpu_allocation_seconds']<=1200
        save(OUT/'terminal_acceptance.json',dict(status='passed',report_sha256=sha(OUT/'report.json'),
            plan_sha256=sha(OUT/'plan.json'),source_unchanged=True,unix=time.time()))
        save(OUT/'status.json',dict(status='complete',phase=report['outcome'],unix=time.time()))
    except BaseException as exc:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(error=repr(exc),phase=phase,unix=time.time()))
        if name:
            try:stop_owned(name)
            except Exception as err:save(OUT/'cleanup_error.json',dict(error=repr(err),unix=time.time()))
        if process is not None and process.poll() is None:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate()
        save(OUT/'status.json',dict(status='failed',phase=phase,unix=time.time()));raise
    finally:
        if lock is not None:
            lock.close()
            if acquired:save(OUT/'gpu_released.json',dict(unix=time.time(),original_lock_released=True,
                gpu_allocation_seconds=time.time()-acquired,failure_cleanup=True))
if __name__=='__main__':main()

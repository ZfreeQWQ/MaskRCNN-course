"""One CPU-only frozen full validation; exact owner and fixed30min budget."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

START=1792573390;COMPUTE_END=START+1500;DEADLINE=START+1800
OWNER='3531b2dbd6b447f4b3ea5e1f08b6a7ab'
NAME='course_m2f_fine512_quality_fullval_20261021_v1'
ROOT=Path('/mnt/nfs/fenghaibin/AI_Class');OUT=ROOT/'outputs'/NAME
PRIOR=ROOT/'outputs/course_m2f_fine512_quality_20261021_v1'
FULL=ROOT/'outputs/course_m2f_fine512_fullval_20261021_v2'
FILES=('run_m2f_fine512_quality_fullval.py','test_m2f_fine512_quality_fullval.py','launch_m2f_fine512_quality_fullval.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    assert time.time()<START+300 and not (OUT/'plan.json').exists()
    assert sha(PRIOR/'report.json')=='791e65bc9f5704e20ff306d15d661cdb6184a6f0b230ae5a7e36c870c0d8b419'
    assert sha(PRIOR/'selector.json')=='e5f3ef4e39d7f74a965a6d93301e279bbfda4d00f8964a89c5055f9b1ee41af8'
    inherited=json.loads((PRIOR/'plan.json').read_text())['source_sha256']
    for n,h in inherited.items():
        assert sha(PRIOR/'source'/n)==h and n not in FILES and not (OUT/'source'/n).exists(),n
        shutil.copy2(PRIOR/'source'/n,OUT/'source'/n)
    source={p.name:sha(p) for p in (OUT/'source').glob('*.py')};assert len(source)==len(inherited)+3==62
    save(OUT/'plan.json',dict(start_unix=START,binding_end=START+300,AP_end=START+1200,analysis_end=COMPUTE_END,
        deadline_unix=DEADLINE,buffer_used=False,owner=OWNER,source_commit=sys.argv[1],source_sha256=source,
        inherited_sha256=inherited,prior_report_sha256=sha(PRIOR/'report.json'),selector_sha256=sha(PRIOR/'selector.json'),
        CPU=4,memory_gib=8,GPU=False,baseline_APs=.41399673960953054,max_fits=0,max_fullval_AP=1,
        user_authorized_new_validation=True,prior_gate_not_rewritten=True,tiny_per_group_loss_is_reported_risk=True,
        prediction_functions_unchanged=True,no_GPU_lock=True,CNN_paused=True))
    rel='/workspace/outputs/'+NAME;image=next(v for v in json.loads((FULL/'finish_command.json').read_text()) if v.startswith('sha256:'))
    container='ai-m2f-512qualityval-'+OWNER[:10]+'-cpu'
    cmd=['docker','run','--rm','--pull','never','--runtime','runc','--name',container,
        '--label','ai-course-directions-owner='+OWNER,'--network','none','--read-only','--cpus','4','--memory','8g',
        '--shm-size','1g','--user',f'{os.getuid()}:{os.getgid()}',
        '-v',str(ROOT)+':/workspace:ro','-v',str(ROOT)+':'+str(ROOT)+':ro',
        '-v',str(OUT)+':'+rel+':rw','--tmpfs','/tmp:rw,nosuid,nodev,size=1g','-w',rel+'/source']
    env=dict(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONPATH=rel+'/source:/workspace/outputs/course_m2f_tiles_20261019_v1/source:/workspace/outputs/course_m2f_geometry_20261020_v1/source:/workspace/mmdetection',
        PATH='/workspace/envs/mmdet/bin:/usr/local/bin:/usr/bin:/bin')
    for k,v in env.items():cmd+=['-e',k+'='+v]
    cmd += [image,'python','run_m2f_fine512_quality_fullval.py'];save(OUT/'command.json',cmd);save(OUT/'launch.json',dict(pid=os.getpid(),unix=time.time()))
    try:
        with (OUT/'cpu.log').open('x') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/'status.json',dict(status='running',phase='cpu',client_pid=p.pid,container=container,unix=time.time()))
            inspected=False
            while p.poll() is None:
                assert time.time()<COMPUTE_END,'Immutable analysis deadline'
                if not inspected:
                    r=subprocess.run(['docker','inspect',container],capture_output=True,text=True,timeout=5)
                    if r.returncode==0:
                        d=json.loads(r.stdout)[0];h=d['HostConfig'];assert d['Config']['Labels']['ai-course-directions-owner']==OWNER
                        assert not h['Devices'] and not h['DeviceRequests'] and h['Runtime']=='runc'
                        assert h['NanoCpus']==4000000000 and h['Memory']==8*2**30 and h['NetworkMode']=='none' and h['ReadonlyRootfs']
                        save(OUT/'container.json',dict(runtime=h['Runtime'],devices=h['Devices'],device_requests=h['DeviceRequests'],
                            cpus=4,memory_gib=8,network='none',readonly_root=True,owner=OWNER));inspected=True
                time.sleep(2)
            assert p.returncode==0,('CPU worker exit',p.returncode)
        assert inspected and time.time()<COMPUTE_END
        report=json.loads((OUT/'report.json').read_text());assert report['status']=='passed' and report['full_val_evaluations']==1 and report['selector_fits']==0
        assert all(sha(OUT/'source'/n)==h for n,h in source.items())
        assert all(sha(OUT/n)==h for n,h in report['artifacts_sha256'].items())
        save(OUT/'terminal_acceptance.json',dict(status='passed',report_sha256=sha(OUT/'report.json'),source_files=len(source),
                                                artifacts=len(report['artifacts_sha256']),unix=time.time()))
        save(OUT/'status.json',dict(status='complete',phase='evaluated',unix=time.time()))
    except BaseException as exc:
        if not (OUT/'first_error.json').exists():save(OUT/'first_error.json',dict(error=repr(exc),unix=time.time()))
        r=subprocess.run(['docker','inspect',container],capture_output=True,text=True,timeout=5)
        if r.returncode==0:
            d=json.loads(r.stdout)[0];assert d['Config']['Labels']['ai-course-directions-owner']==OWNER
            subprocess.run(['docker','stop','-t','5',container],timeout=12,check=False)
        save(OUT/'status.json',dict(status='failed',phase='cpu',unix=time.time()));raise

if __name__=='__main__':main()

"""One bounded CPU/smoke/two-scale workflow. Stop first error, never retrain."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/mnt/nfs/fenghaibin/AI_Class')
IMAGE='nvidia_cuda-export:11.8.0-cudnn8-devel-ubuntu22.04'
FILES=['course_resolution_probe.py','course_demo.py','course_batch_predict.py',
       'run_course_resolution.py','tests/test_course_resolution.py','tests/test_course_batch_predict.py']


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def idle():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    rows=[list(map(int,line.split(','))) for line in text.strip().splitlines()]
    if not any(i==0 and mem<=512 and util<=10 for i,mem,util in rows):
        raise RuntimeError('GPU0 busy; leave other users untouched')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('Require a NEW output under personal outputs')
    out.mkdir();source=out/'source';source.mkdir()
    base=Path(__file__).resolve().parent
    for rel in FILES:
        target=source/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,target)
    hashes={rel:sha(source/rel) for rel in FILES}
    image_id=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    plan=dict(protocol='fixed-control30k-val-1024-1536-v1',source_sha256=hashes,image=image_id,
              max_seconds=7200,scales=[1024,1536],gpu=0,training=False,test_inputs=False,
              declared_gate=dict(APs_gain_points=1.,AP_drop_max_points=.2,latency_ratio_max=2.5))
    save(out/'plan.json',plan)
    state=dict(status='running',started_at_utc=now(),pid=os.getpid(),steps=[],plan=plan)
    save(out/'status.json',state)
    start=time.monotonic();name=None
    prefix='ai-course-resolution-'+hashlib.sha256(str(out).encode()).hexdigest()[:8]
    container_source='/workspace/'+str(source.relative_to(ROOT))
    frozen='/workspace/outputs/course_budget_20261014_v1/source'
    worker=['python',container_source+'/course_resolution_probe.py']
    steps=[('cpu',['python','-m','unittest','-v','test_course_resolution','test_course_batch_predict'],False)]
    for mode in ('smoke','evaluate'):
        for scale in (1024,1536):
            phase=f'{mode}_{scale}'
            steps.append((phase,worker+['--mode',mode,'--course-eval-scale',str(scale),
                '--output','/workspace/'+str(out.relative_to(ROOT))+'/'+phase],True))
    try:
        for phase,command,gpu in steps:
            if {rel:sha(source/rel) for rel in FILES}!=hashes:
                raise RuntimeError('Frozen workflow source changed')
            if gpu:state['gpu_before_'+phase]=idle()
            state.update(phase=phase,updated_at_utc=now());save(out/'status.json',state)
            name=prefix+'-'+phase.replace('_','-')
            cmd=['docker','run','--rm','--pull','never','--name',name,'--shm-size=8g',
                 '--user',str(os.getuid())+':'+str(os.getgid()),
                 '--volume',str(ROOT)+':/workspace','--volume',str(ROOT)+':'+str(ROOT),
                 '--tmpfs','/tmp:rw,nosuid,nodev,size=1g','--workdir','/tmp']
            if gpu:cmd+=['--gpus','device=0']
            env={'TMPDIR':'/tmp','XDG_CACHE_HOME':'/workspace/.cache',
                'MPLCONFIGDIR':'/workspace/.cache/matplotlib','PYTHONDONTWRITEBYTECODE':'1',
                'PATH':'/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                'PYTHONPATH':container_source+':'+container_source+'/tests:'+frozen+':/workspace/mmdetection',
                'AI_CLASS_TEST_DEVICE':'cpu','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
            if not gpu:env['CUDA_VISIBLE_DEVICES']=''
            for key,value in env.items():cmd+=['--env',key+'='+value]
            cmd+=[image_id]+command
            remaining=7200-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Bounded workflow deadline reached')
            with (out/(phase+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state.update(docker_client_pid=child.pid,container=name);save(out/'status.json',state)
                try:code=child.wait(timeout=min(remaining,300 if phase=='cpu' else 900 if phase.startswith('smoke') else remaining))
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker','stop','--time','10',name],check=False,timeout=30)
                    child.wait(timeout=30);raise
            state['steps'].append(dict(phase=phase,exit_code=code,finished_at_utc=now()))
            save(out/'status.json',state)
            if code:raise RuntimeError('Step failed: '+phase+'; retain first original log')
            if gpu:
                result=json.loads((out/phase/'report.json').read_text())
                expected=8 if phase.startswith('smoke') else 2584
                if result['status']!='passed' or result['completed_images']!=expected:
                    raise RuntimeError('Incomplete or invalid inference stage')
        if {rel:sha(source/rel) for rel in FILES}!=hashes:
            raise RuntimeError('Frozen workflow source changed before final comparison')
        a=json.loads((out/'evaluate_1024/report.json').read_text())
        b=json.loads((out/'evaluate_1536/report.json').read_text())
        if a['input_hashes_sha256']!=b['input_hashes_sha256']:
            raise RuntimeError('Input files differ between scales')
        delta={k:round((b['metrics'][k]-v)*100,4) for k,v in a['metrics'].items()}
        ratio=b['timing']['mean_ms']/a['timing']['mean_ms']
        comparison=dict(status='complete',delta_metric_points=delta,latency_ratio=ratio,
            engineering_gate_passed=delta['course/segm_mAP_s']>=1.-1e-6 and delta['course/segm_mAP']>=-.2-1e-6 and ratio<=2.5,
            note='Same weights/training, unequal inference compute. Not a new trained architecture or independent replication.')
        save(out/'comparison.json',comparison)
        state.update(status='complete',finished_at_utc=now(),training_started=False,
                     test_inputs_used=False,test_annotations_used=False)
        save(out/'status.json',state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc),finished_at_utc=now());save(out/'status.json',state);raise


if __name__=='__main__':main()

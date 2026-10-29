"""Single CP+detail30k plus final1536 evaluation; absolute12h, no replay."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path('/mnt/nfs/fenghaibin/AI_Class')
FROZEN=ROOT/'outputs/course_directions_20261015_v1/source'
if FROZEN.is_dir():
    sys.path.append(str(FROZEN))
import run_course_directions as queue
from m2f_cpdetail import job, JOB_ID, REFERENCE, REFERENCE_SHA
from launch_nyu_sweep import sha256, atomic_json, container_path

FILES=('m2f_cpdetail.py','run_m2f_cpdetail.py','tests/test_m2f_cpdetail.py')
CPU_TESTS=('test_m2f_cpdetail','test_course_directions_training','test_course_directions_workflow',
           'test_course_feature_models','test_course_copy_paste','test_course_resolution',
           'test_course_directions_diagnostics')


def docker(root, plan, phase, command, gpu):
    source=container_path(root/'source',ROOT);output=container_path(root,ROOT)
    name='ai-m2f-cpdetail-'+plan['owner_token'][:10]+'-'+phase
    cmd=['docker','run','--rm','--pull','never','--name',name,
         '--label','ai-course-directions-owner='+plan['owner_token'],
         '--network','none','--read-only','--cpus','4','--memory','32g','--shm-size','4g',
         '--user',f'{os.getuid()}:{os.getgid()}',
         '--volume',str(ROOT)+':/workspace:ro','--volume',str(ROOT)+':'+str(ROOT)+':ro',
         '--volume',str(root)+':'+output+':rw','--tmpfs','/tmp:rw,nosuid,nodev,size=2g',
         '--workdir',source]
    env=dict(PYTHONPATH=source+':'+source+'/tests:/workspace/mmdetection',
        PATH='/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        HOME='/tmp',XDG_CACHE_HOME='/tmp/cache',MPLCONFIGDIR='/tmp/matplotlib',TMPDIR='/tmp',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',AI_CLASS_TEST_DEVICE='cpu',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if gpu: cmd+=['--gpus','device=0']
    else: env['CUDA_VISIBLE_DEVICES']=''
    for k,v in env.items():cmd+=['--env',k+'='+v]
    return name,cmd+[plan['image_id']]+command


def train_args(root,smoke):
    stage='smoke' if smoke else 'formal';j=job(smoke)
    args=['python','m2f_cpdetail.py','train','--workspace','/workspace',
          '--data-root','/workspace/'+queue.DATA,'--checkpoint','/workspace/'+queue.CHECKPOINT,
          '--output',container_path(root/stage/'runs'/JOB_ID,ROOT),'--seed','20261013']
    for k,v in queue.stage_recipe(smoke).items(): args+=['--'+k.replace('_','-'),str(v)]
    args+=['--'+k.replace('_','-') for k,v in j['flags'].items() if v]
    return args


def accept_final(root):
    path=root/'evaluate_1536';r=json.loads((path/'report.json').read_text())
    ref=ROOT/REFERENCE;old=json.loads((ref/'report.json').read_text())
    trained=json.loads((root/'formal/runs'/JOB_ID/'report.json').read_text())
    if sha256(ref/'report.json')!=REFERENCE_SHA:raise ValueError('Primary reference changed')
    if (r['status']!='passed' or r['input_scale']!=1536 or r['completed_images']!=2584
        or r['fixed_case_ids']!=list(queue.FIXED_IDS) or r['checkpoint_sha256']!=trained['checkpoint_sha256']
        or r['training_report_sha256']!=sha256(root/'formal/runs'/JOB_ID/'report.json')
        or r['training_started'] or r['gt_sent_to_model'] or r['test_inputs_used'] or r['test_annotations_used']):
        raise ValueError('Final evaluation acceptance failed')
    hashes=json.loads((path/'input_hashes.json').read_text())
    if hashes!=json.loads((ref/'input_hashes.json').read_text()) or len({x['image_id'] for x in hashes})!=2584:
        raise ValueError('Full official IDs/order/input hashes differ')
    if sha256(path/'input_hashes.json')!=r['input_hashes_sha256']:raise ValueError('Input binding changed')
    if r['active_parameters']!=44402718 or r['state_tensor_bytes']!=177823800:raise ValueError('Detail size mismatch')
    if r['timing']['warmup']!=5 or r['timing']['repeats']!=20:raise ValueError('Timing protocol')
    for i in queue.FIXED_IDS:
        if not (path/f'case_{i}_predictions.json').is_file():raise ValueError('Missing fixed case')
    metrics=queue.metric_values(r['metrics'])
    result=dict(status='passed',primary_reference_report_sha256=REFERENCE_SHA,
        report_sha256=sha256(path/'report.json'),metrics=metrics,reference_metrics=old['metrics'],
        delta_points={k:round(100*(metrics[k]-old['metrics'][k]),4) for k in queue.METRICS},
        latency_ratio=r['timing']['mean_ms']/old['timing']['mean_ms'],
        peak_memory_ratio=r['timing']['peak_allocated_mib']/old['timing']['peak_allocated_mib'],
        six_checkpoint_sha256={p.name:sha256(p) for p in sorted((root/'formal/runs'/JOB_ID).glob('iter_*.pth'))},
        full_val_order_and_hashes_match=True,single_seed=True,test_inputs_used=False)
    atomic_json(root/'comparison.json',result);return result


class Single(queue.Supervisor):
    def __init__(self,root,plan):
        super().__init__(root,plan);self.state['jobs']={JOB_ID:'pending'}
    def check_stop(self):
        if self.stop_requested or (self.root/'STOP').exists():raise queue.StopRequested('Stop requested')
        if self.remaining()<=0:raise TimeoutError('12h absolute budget includes all preparation/evaluation')
    def execute(self):
        self.save()
        try:
            code="import unittest; s=unittest.defaultTestLoader.loadTestsFromNames(%r); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(0 if r.wasSuccessful() and not r.skipped and r.testsRun>=100 else 1)"%list(CPU_TESTS)
            n,c=docker(self.root,self.plan,'cpu',['python','-c',code],False)
            self.run_command('cpu',n,c,900)
            for stage,smoke in [('smoke',True),('formal',False)]:
                out=self.root/stage;out.mkdir();(out/'runs').mkdir();(out/'source').symlink_to(self.root/'source',target_is_directory=True)
                self.wait_idle()
                n,c=docker(self.root,self.plan,stage,train_args(self.root,smoke),True)
                self.run_command(stage,n,c,min(self.remaining(),1800 if smoke else 43200))
                accepted=queue.validate_report(self.root,self.plan,stage,job(smoke))
                report=json.loads((out/'runs'/JOB_ID/'report.json').read_text())
                activity=report['direction_activity']
                steps=20 if smoke else 30000
                if (activity['copy_paste'].get('calls')!=steps*2 or activity['copy_paste'].get('pasted_instances',0)<=0
                    or activity['feature'].get('detail_train_samples')!=steps*2
                    or report['config_checks'].get('cp_plus_detail_only') is not True):raise ValueError('Combination not actually exercised')
                self.state['steps'][-1]['acceptance']=accepted
                self.save()
            self.state['jobs'][JOB_ID]='trained_accepted';self.save()
            self.wait_idle()
            n,c=docker(self.root,self.plan,'evaluate',['python','m2f_cpdetail.py','evaluate',
                '--run',container_path(self.root/'formal/runs'/JOB_ID,ROOT),
                '--output',container_path(self.root/'evaluate_1536',ROOT)],True)
            self.run_command('evaluate_1536',n,c,self.remaining())
            self.check_stop();result=accept_final(self.root)
            self.state['steps'][-1]['acceptance']=result
            queue.assert_intact(self.root,self.plan)
            self.state['jobs'][JOB_ID]='passed'
            self.save(status='complete',phase='complete',finished_at_utc=queue.now(),container=None,docker_client_pid=None)
        except BaseException as exc:
            self.stop_owned()
            error=dict(type=type(exc).__name__,message=str(exc),phase=self.state.get('phase'),original_log=self.state.get('current_log'),recorded_at_utc=queue.now())
            if not (self.root/'first_error.json').exists():atomic_json(self.root/'first_error.json',error)
            self.save(status='failed',first_error=error,finished_at_utc=queue.now());raise


def worker(root):
    plan=json.loads((root/'plan.json').read_text())
    if (root/'status.json').exists() or plan['output']!=str(root):raise ValueError('No worker replay')
    s=Single(root,plan)
    try:
        with (ROOT/'outputs/.course_directions_gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            signal.signal(signal.SIGHUP,signal.SIG_IGN)
            for event in (signal.SIGINT,signal.SIGTERM):signal.signal(event,lambda *_:setattr(s,'stop_requested',True))
            s.execute()
    except BaseException as exc:
        if not (root/'first_error.json').exists():
            error=dict(type=type(exc).__name__,message=str(exc),phase='locking',recorded_at_utc=queue.now())
            atomic_json(root/'first_error.json',error);s.save(status='failed',first_error=error)
        raise


def start(root,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('Recorded source commit required')
    if root.exists() or root==ROOT/'outputs' or not root.is_relative_to(ROOT/'outputs'):raise ValueError('New personal output only')
    created=time.time()
    old=json.loads((FROZEN.parent/'plan.json').read_text())
    for rel,digest in old['source_sha256'].items():
        if sha256(FROZEN/rel)!=digest:raise ValueError('Frozen dependency changed '+rel)
    if shutil.disk_usage(ROOT).free<12*2**30:raise ValueError('Insufficient checkpoint reserve')
    provenance,reference=queue.verify_inputs(ROOT)
    if sha256(ROOT/REFERENCE/'report.json')!=REFERENCE_SHA:raise ValueError('CP1536 reference changed')
    root.mkdir();(root/'logs').mkdir();shutil.copytree(FROZEN,root/'source')
    origin=Path(__file__).resolve().parent
    for rel in FILES:
        dst=root/'source'/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(origin/rel,dst)
    mapping={rel:sha256(root/'source'/rel) for rel in (*old['source_sha256'],*FILES)}
    image=subprocess.check_output(['docker','image','inspect',queue.IMAGE,'--format','{{.Id}}'],text=True).strip()
    plan=dict(protocol='single-cpdetail30k-final1536-v1',workspace=str(ROOT),output=str(root),
        source_commit=commit,source_sha256=mapping,created_at_utc=queue.now(),deadline_unix=created+43200,
        max_seconds=43200,max_gpu_wait_seconds=1800,gpu=0,image_id=image,owner_token=uuid.uuid4().hex,
        expected_provenance=provenance,reference_baseline=reference,primary_reference=REFERENCE,
        primary_reference_sha256=REFERENCE_SHA,jobs=[job()],recipe=queue.BASE_RECIPE,cpu_tests=CPU_TESTS,
        test_inputs_used=False,test_annotations_used=False,no_parameter_search=True)
    atomic_json(root/'plan.json',plan)
    cmd=[sys.executable,'-u',str(root/'source/run_m2f_cpdetail.py'),'worker','--output',str(root)]
    with (root/'workflow.log').open('xb',buffering=0) as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    launch=dict(pid=child.pid,command=cmd,started_at_utc=queue.now(),deadline_unix=plan['deadline_unix'],plan_sha256=sha256(root/'plan.json'))
    atomic_json(root/'launch.json',launch);print(json.dumps(launch,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','worker']);p.add_argument('--output',required=True);p.add_argument('--source-commit')
    a=p.parse_args();r=Path(a.output).resolve();start(r,a.source_commit) if a.mode=='start' else worker(r)

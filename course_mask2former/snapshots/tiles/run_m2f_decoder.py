"""Bounded fixed-weight decoder diagnosis; no training or GPU1 mutation."""
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
FROZEN=ROOT/'outputs/course_m2f_cpdetail_20261018_v1/source'
if FROZEN.is_dir():sys.path.append(str(FROZEN))
import run_course_directions as queue
from launch_nyu_sweep import sha256, atomic_json, container_path
from m2f_decoder_probe import ARMS, MANIFEST, MANIFEST_SHA, CAP_BYTES, budget, read

PREPARATION_UNIX=1792328160.0  # 2026-10-18 20:56 Beijing, first preparation
DEADLINE=1792335360.0  # original2h, never reset on launch/recovery
FILES=('m2f_decoder_probe.py','run_m2f_decoder.py','tests/test_m2f_decoder.py')
UPSTREAM=('mmdet/models/dense_heads/mask2former_head.py',
          'mmdet/models/dense_heads/maskformer_head.py',
          'mmdet/models/seg_heads/panoptic_fusion_heads/maskformer_fusion_head.py',
          'mmdet/models/detectors/maskformer.py')


def docker(root,plan,phase,command,gpu=False):
    src=container_path(root/'source',ROOT);out=container_path(root,ROOT)
    name='ai-m2f-decoder-'+plan['owner_token'][:10]+'-'+phase
    cmd=['docker','run','--rm','--pull','never','--name',name,'--label',
        'ai-course-directions-owner='+plan['owner_token'],'--network','none','--read-only',
        '--cpus','4','--memory','32g','--shm-size','4g','--user',f'{os.getuid()}:{os.getgid()}',
        '--volume',str(ROOT)+':/workspace:ro','--volume',str(ROOT)+':'+str(ROOT)+':ro',
        '--volume',str(root)+':'+out+':rw','--tmpfs','/tmp:rw,nosuid,nodev,size=2g','--workdir',src]
    env=dict(PYTHONPATH=src+':'+src+'/tests:/workspace/mmdetection',
        PATH='/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        HOME='/tmp',XDG_CACHE_HOME='/tmp/cache',MPLCONFIGDIR='/tmp/matplotlib',TMPDIR='/tmp',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if gpu:cmd+=['--gpus','device=0']
    else:env['CUDA_VISIBLE_DEVICES']=''
    for k,v in env.items():cmd+=['--env',k+'='+v]
    return name,cmd+[plan['image_id']]+command


def accept(root,plan):
    m=read(ROOT/MANIFEST)
    for arm,spec in ARMS.items():
        p=root/arm;r=read(p/'report.json')
        if (r['status']!='passed' or r['image_ids']!=m['image_ids'] or r['completed_images']!=256
            or r['layer_count']!=10 or r['query_count']!=100 or r['recorder_calls']!=256
            or r['model_eval_calls']!=257 or r['train_calls']!=0
            or not all(r[k] for k in ('first_image_off_on_equivalent','all_final_layers_equivalent','rle_roundtrip_verified'))
            or any(r[k] for k in ('training_started','gt_sent_to_model','test_inputs_used','test_annotations_used','ap_reevaluated'))
            or r['checkpoint_sha256']!=spec['weight_sha'] or r['training_report_sha256']!=spec['report_sha']
            or r['reference_report_sha256']!=spec['reference_sha'] or r['manifest_sha256']!=MANIFEST_SHA
            or r['active_parameters']!=spec['parameters'] or r['state_tensor_bytes']!=spec['state_bytes']):
            raise ValueError('Decoder arm acceptance failed: '+arm)
        if len(r['prediction_sha256'])!=256 or any(sha256(p/f)!=h for f,h in r['prediction_sha256'].items()):
            raise ValueError('Full captured-file SHA binding failed')
        if sha256(p/'input_hashes.json')!=r['input_hashes_sha256']:
            raise ValueError('Input hash binding failed')
        for file,digest in (('report.json',spec['report_sha']),('iter_30000.pth',spec['weight_sha'])):
            if sha256(ROOT/spec['run']/file)!=digest:raise ValueError('Original accepted arm changed')
    if read(root/'copy_paste/input_hashes.json')!=read(root/'cp_detail/input_hashes.json'):
        raise ValueError('Arms not paired')
    r=read(root/'analysis/report.json')
    if (r['status']!='passed' or set(r['arms'])!=set(ARMS) or r['original_small_gt']<100
        or not r['gt_used_only_after_inference'] or r['report_sha256']!={a:sha256(root/a/'report.json') for a in ARMS}):
        raise ValueError('CPU analysis failed')
    if any(sha256(ROOT/'mmdetection'/f)!=h for f,h in plan['upstream_sha256'].items()):
        raise ValueError('Upstream implementation changed')
    queue.assert_intact(root,plan)
    return dict(status='passed',arms={a:sha256(root/a/'report.json') for a in ARMS},
        analysis_report_sha256=sha256(root/'analysis/report.json'),output_bytes=budget(root,DEADLINE),
        limits='No training, no full-val AP, fixed256 per arm, diagnostic upper bounds not performance gains')


class Supervisor(queue.Supervisor):
    def __init__(self,root,plan):
        super().__init__(root,plan);self.state['jobs']={'decoder_diagnostic':'pending'}
        self.lock=None

    def check_stop(self):
        if self.stop_requested or (self.root/'STOP').exists():raise queue.StopRequested('Stop requested')
        if self.remaining()<=0:raise TimeoutError('Original2h preparation-inclusive budget expired')
        budget(self.root,self.plan['deadline_unix'])

    def lock_idle(self):
        began=time.monotonic();self.lock=(ROOT/'outputs/.course_directions_gpu0.lock').open('a')
        acquired=False
        while True:
            self.check_stop()
            if not acquired:
                try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                except BlockingIOError:pass
            usage=queue.gpu_snapshot().get(0)
            if acquired and usage and usage['memory_used_mib']<=512 and usage['utilization']<=10:
                self.save(lock_acquired=True,gpu0_before=usage);return
            self.save(phase='waiting_original_gpu0_lock_and_idle',lock_acquired=acquired)
            if time.monotonic()-began>=300:raise TimeoutError('Original GPU0 lock/idle5min budget exhausted; no bypass')
            time.sleep(min(10,max(0.,self.remaining())))

    def execute(self):
        self.save()
        try:
            code=("import unittest,json; from pathlib import Path; "
                "s=unittest.defaultTestLoader.loadTestsFromName('test_m2f_decoder'); "
                "r=unittest.TextTestRunner(verbosity=2).run(s); "
                "ok=r.wasSuccessful() and not r.skipped and r.testsRun>=20; "
                "Path(%r).write_text(json.dumps(dict(status='passed' if ok else 'failed',tests=r.testsRun,skipped=len(r.skipped),errors=len(r.errors),failures=len(r.failures)))); "
                "raise SystemExit(0 if ok else 1)")%container_path(self.root/'cpu_report.json',ROOT)
            n,c=docker(self.root,self.plan,'cpu',['python','-c',code])
            self.run_command('cpu',n,c,min(self.remaining(),900))
            self.lock_idle()
            for arm in ARMS:
                n,c=docker(self.root,self.plan,arm,['python','m2f_decoder_probe.py','capture',
                    '--root',container_path(self.root,ROOT),'--arm',arm,'--deadline',str(DEADLINE)],True)
                self.run_command(arm,n,c,self.remaining())
            # This phase reads saved predictions/GT, no GPU or model invocation.
            n,c=docker(self.root,self.plan,'analyze',['python','m2f_decoder_probe.py','analyze',
                '--root',container_path(self.root,ROOT),'--deadline',str(DEADLINE)])
            self.run_command('analyze',n,c,self.remaining())
            self.check_stop();result=accept(self.root,self.plan)
            atomic_json(self.root/'terminal_acceptance.json',result)
            self.state['jobs']['decoder_diagnostic']='passed'
            self.save(status='complete',phase='complete',acceptance=result,finished_at_utc=queue.now(),container=None,docker_client_pid=None)
        except BaseException as exc:
            self.stop_owned()
            error=dict(type=type(exc).__name__,message=str(exc),phase=self.state.get('phase'),
                original_log=self.state.get('current_log'),recorded_at_utc=queue.now())
            if not (self.root/'first_error.json').exists():atomic_json(self.root/'first_error.json',error)
            self.save(status='failed',first_error=error,finished_at_utc=queue.now());raise
        finally:
            if self.lock is not None:self.lock.close()


def start(root,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('Source commit required')
    if root!=ROOT/'outputs/course_m2f_decoder_20261018_v1' or root.exists():raise ValueError('Exact new output only; no replay')
    if time.time()>=DEADLINE-1800:raise TimeoutError('Insufficient original budget for two full captures')
    old=read(FROZEN.parent/'plan.json')
    if any(sha256(FROZEN/f)!=h for f,h in old['source_sha256'].items()):raise ValueError('Frozen dependency drift')
    if sha256(ROOT/MANIFEST)!=MANIFEST_SHA:raise ValueError('Manifest identity')
    for spec in ARMS.values():
        if (sha256(ROOT/spec['run']/'report.json')!=spec['report_sha'] or
            sha256(ROOT/spec['run']/'iter_30000.pth')!=spec['weight_sha'] or
            sha256(ROOT/spec['reference']/'report.json')!=spec['reference_sha']):raise ValueError('Accepted input identity')
    if shutil.disk_usage(ROOT).free<8*2**30:raise ValueError('Need output reserve')
    root.mkdir();(root/'logs').mkdir();shutil.copytree(FROZEN,root/'source')
    for rel in FILES:
        dst=root/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__).resolve().parent/rel,dst)
    image=subprocess.check_output(['docker','image','inspect',queue.IMAGE,'--format','{{.Id}}'],text=True).strip()
    plan=dict(protocol='fixed-two-arm-decoder256-v1',workspace=str(ROOT),output=str(root),
        source_commit=commit,source_sha256={f:sha256(root/'source'/f) for f in (*old['source_sha256'],*FILES)},
        upstream_sha256={f:sha256(ROOT/'mmdetection'/f) for f in UPSTREAM},owner_token=uuid.uuid4().hex,
        preparation_unix=PREPARATION_UNIX,deadline_unix=DEADLINE,max_seconds=7200,max_gpu_wait_seconds=300,
        max_output_bytes=CAP_BYTES,image_id=image,gpu=0,arms=ARMS,manifest=MANIFEST,manifest_sha256=MANIFEST_SHA,
        val_annotations=queue.DATA+'/annotations/instances_val.json',test_inputs_used=False,test_annotations_used=False,
        training_started=False,ap_reevaluated=False,created_at_utc=queue.now())
    atomic_json(root/'plan.json',plan)
    cmd=[sys.executable,'-u',str(root/'source/run_m2f_decoder.py'),'worker','--output',str(root)]
    with (root/'workflow.log').open('xb',buffering=0) as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    launch=dict(pid=p.pid,command=cmd,started_at_utc=queue.now(),deadline_unix=DEADLINE,plan_sha256=sha256(root/'plan.json'))
    atomic_json(root/'launch.json',launch);print(json.dumps(launch,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','worker']);p.add_argument('--output',required=True);p.add_argument('--source-commit')
    a=p.parse_args();r=Path(a.output).resolve()
    if a.mode=='start':start(r,a.source_commit)
    else:
        if (r/'status.json').exists():raise ValueError('No worker replay')
        s=Supervisor(r,read(r/'plan.json'))
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        for event in (signal.SIGTERM,signal.SIGINT):signal.signal(event,lambda *_:setattr(s,'stop_requested',True))
        s.execute()

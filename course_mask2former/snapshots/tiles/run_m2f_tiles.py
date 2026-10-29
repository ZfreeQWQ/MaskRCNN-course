"""Authorized single global+overlapping-tile experiment, absolute4h, GPU0 only."""
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
FROZEN=ROOT/'outputs/course_m2f_decoder_20261018_v1/source'
if FROZEN.is_dir():sys.path.append(str(FROZEN))
import run_course_directions as queue
from launch_nyu_sweep import sha256,atomic_json,container_path
from run_m2f_decoder import docker as inherited_docker
from m2f_tiles import PREPARATION_UNIX,DEADLINE,SPEC,METRICS,MANIFEST,MANIFEST_SHA,read,DECODER,DECODER_REPORT_SHA,DECODER_GT_SHA

FILES=('m2f_tiles.py','run_m2f_tiles.py','tests/test_m2f_tiles.py')
OUTPUT=ROOT/'outputs/course_m2f_tiles_20261018_v1'


def docker(root,plan,phase,command,gpu=False):
    old,cmd=inherited_docker(root,plan,phase,command,gpu)
    name='ai-m2f-tiles-'+plan['owner_token'][:10]+'-'+phase
    cmd[cmd.index('--name')+1]=name
    return name,cmd


def accept(root,plan):
    r=read(root/'evaluate/report.json');ref=ROOT/SPEC['reference'];old=read(ref/'report.json')
    analysis=read(root/'analysis/report.json');m=read(ROOT/MANIFEST)
    ids=[x['image_id'] for x in read(ref/'input_hashes.json')]
    if (r['status']!='passed' or r['completed_images']!=2584 or r['image_ids']!=ids or len(set(ids))!=2584
        or r['subset_ids']!=m['image_ids'] or r['fixed_case_ids']!=list(queue.FIXED_IDS)
        or r['eval_calls']!=13045 or r['train_calls']!=0
        or any(r[k] for k in ('training_started','test_inputs_used','test_annotations_used','gt_sent_to_model'))
        or not r['global_input_equivalent'] or not r['rle_roundtrip_verified']
        or r['input_scale']!=1536 or r['original_shape']!=[1024,1024]
        or r['tile_origins']!=[[0,0],[384,0],[0,384],[384,384]] or r['tile_size']!=640
        or r['tile_max_area']!=1024 or r['mask_nms_iou']!=.5 or r['max_instances']!=100):
        raise ValueError('Full original-grid tiled evaluation protocol failed')
    if (read(root/'evaluate/input_hashes.json')!=read(ref/'input_hashes.json')
        or sha256(root/'evaluate/input_hashes.json')!=r['input_hashes_sha256']):raise ValueError('Input identity/order drift')
    if (r['checkpoint_sha256']!=SPEC['weight_sha'] or r['training_report_sha256']!=SPEC['report_sha']
        or r['reference_report_sha256']!=SPEC['reference_sha'] or sha256(ref/'report.json')!=SPEC['reference_sha']
        or sha256(ROOT/SPEC['run']/'iter_30000.pth')!=SPEC['weight_sha']):raise ValueError('Accepted source weights changed')
    expected=set(m['image_ids'])|set(queue.FIXED_IDS)
    if set(r['captured_ids'])!=expected or len(r['capture_sha256'])!=len(expected):raise ValueError('Subset/eight incomplete')
    for file,digest in r['capture_sha256'].items():
        if sha256(root/'evaluate'/file)!=digest:raise ValueError('Subset capture SHA')
    for identity in queue.FIXED_IDS:
        if not (root/'evaluate'/f'case_{identity}_predictions.json').is_file():raise ValueError('Missing eight cases')
    t=r['timing']
    if t['warmup']!=5 or t['repeats']!=20 or t['views_per_call']!=5 or len(t['all_seconds'])!=20:
        raise ValueError('Composite timing protocol')
    if r['active_parameters']!=44400414 or r['state_tensor_bytes']!=177814584:raise ValueError('Model architecture changed')
    queue.metric_values(r['metrics'])
    if (analysis['status']!='passed' or analysis['report_sha256']!=sha256(root/'evaluate/report.json')
        or analysis['reference_report_sha256']!=SPEC['reference_sha'] or analysis['small']['total']!=1187
        or analysis['original_never230']['total']!=230 or analysis['original_early_lost51']['total']!=51):
        raise ValueError('Original diagnostic cohort analysis failed')
    queue.assert_intact(root,plan)
    if any(sha256(ROOT/'mmdetection'/f)!=h for f,h in plan['upstream_sha256'].items()):raise ValueError('Upstream drift')
    result=dict(status='passed',report_sha256=sha256(root/'evaluate/report.json'),analysis_report_sha256=sha256(root/'analysis/report.json'),
        checkpoint_sha256=SPEC['weight_sha'],reference_report_sha256=SPEC['reference_sha'],
        metrics=r['metrics'],delta_points=analysis['delta_points'],latency_ratio=analysis['latency_ratio'],peak_memory_ratio=analysis['peak_memory_ratio'],
        engineering_targets_passed=analysis['engineering_targets_passed'],full_val_order_and_hashes_match=True,
        training_started=False,note='Passed execution/acceptance does not imply performance targets passed; fixed-weight5-view extra compute.')
    atomic_json(root/'comparison.json',analysis);return result


class Supervisor(queue.Supervisor):
    def __init__(self,root,plan):
        super().__init__(root,plan);self.state['jobs']={'global_tiles':'pending'};self.lock=None
    def check_stop(self):
        if self.stop_requested or (self.root/'STOP').exists():raise queue.StopRequested('Stop requested')
        if self.remaining()<=0:raise TimeoutError('Absolute4h including preparation; no extension')
    def lock_idle(self):
        began=time.monotonic();self.lock=(ROOT/'outputs/.course_directions_gpu0.lock').open('a');acquired=False
        while True:
            self.check_stop()
            if not acquired:
                try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
                except BlockingIOError:pass
            usage=queue.gpu_snapshot().get(0)
            if acquired and usage and usage['memory_used_mib']<=512 and usage['utilization']<=10:
                self.save(lock_acquired=True,gpu0_before=usage);return
            self.save(phase='waiting_original_gpu0_lock_and_idle',lock_acquired=acquired)
            if time.monotonic()-began>=300:raise TimeoutError('GPU0 lock/idle5min exhausted; no bypass')
            time.sleep(min(10,max(0.,self.remaining())))
    def execute(self):
        self.save()
        try:
            code=("import unittest,json; from pathlib import Path; "
                "s=unittest.defaultTestLoader.loadTestsFromName('test_m2f_tiles'); "
                "r=unittest.TextTestRunner(verbosity=2).run(s); ok=r.wasSuccessful() and not r.skipped and r.testsRun>=20; "
                "Path(%r).write_text(json.dumps(dict(status='passed' if ok else 'failed',tests=r.testsRun,skipped=len(r.skipped),errors=len(r.errors),failures=len(r.failures)))); "
                "raise SystemExit(0 if ok else 1)")%container_path(self.root/'cpu_report.json',ROOT)
            n,c=docker(self.root,self.plan,'cpu',['python','-c',code]);self.run_command('cpu',n,c,min(900,self.remaining()))
            n,c=docker(self.root,self.plan,'binding',['python','m2f_tiles.py','binding','--root',container_path(self.root,ROOT)])
            self.run_command('binding',n,c,min(600,self.remaining()))
            self.lock_idle()
            n,c=docker(self.root,self.plan,'evaluate',['python','m2f_tiles.py','evaluate','--root',container_path(self.root,ROOT)],True)
            self.run_command('evaluate',n,c,self.remaining())
            # Release GPU lock before CPU-only pairing; no second GPU task.
            self.lock.close();self.lock=None
            n,c=docker(self.root,self.plan,'analyze',['python','m2f_tiles.py','analyze','--root',container_path(self.root,ROOT)])
            self.run_command('analyze',n,c,self.remaining())
            self.check_stop();result=accept(self.root,self.plan);self.check_stop()
            atomic_json(self.root/'terminal_acceptance.json',result);self.state['jobs']['global_tiles']='passed'
            self.save(status='complete',phase='complete',acceptance=result,finished_at_utc=queue.now(),container=None,docker_client_pid=None)
        except BaseException as exc:
            self.stop_owned();error=dict(type=type(exc).__name__,message=str(exc),phase=self.state.get('phase'),
                original_log=self.state.get('current_log'),recorded_at_utc=queue.now())
            if not (self.root/'first_error.json').exists():atomic_json(self.root/'first_error.json',error)
            self.save(status='failed',first_error=error,finished_at_utc=queue.now());raise
        finally:
            if self.lock is not None:self.lock.close()


def start(root,commit):
    if root!=OUTPUT or root.exists():raise ValueError('Exact new output only; no replay')
    if not re.fullmatch('[0-9a-f]{40}',commit or ''):raise ValueError('Source commit required')
    if time.time()>=DEADLINE-3600:raise TimeoutError('Insufficient original budget; no reset')
    old=read(FROZEN.parent/'plan.json')
    if any(sha256(FROZEN/f)!=h for f,h in old['source_sha256'].items()):raise ValueError('Frozen dependency drift')
    if any(sha256(ROOT/'mmdetection'/f)!=h for f,h in old['upstream_sha256'].items()):raise ValueError('Upstream implementation drift')
    for path,digest in [(ROOT/SPEC['run']/'report.json',SPEC['report_sha']),
        (ROOT/SPEC['run']/'iter_30000.pth',SPEC['weight_sha']),
        (ROOT/SPEC['reference']/'report.json',SPEC['reference_sha']),
        (ROOT/DECODER/'analysis/report.json',DECODER_REPORT_SHA),
        (ROOT/DECODER/'analysis/copy_paste_per_gt.json',DECODER_GT_SHA),(ROOT/MANIFEST,MANIFEST_SHA)]:
        if sha256(path)!=digest:raise ValueError('Accepted reference changed: '+str(path))
    if shutil.disk_usage(ROOT).free<8*2**30:raise ValueError('Insufficient output reserve')
    root.mkdir();(root/'logs').mkdir();shutil.copytree(FROZEN,root/'source')
    origin=Path(__file__).resolve().parent
    for rel in FILES:
        dst=root/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin/rel,dst)
    plan=dict(protocol='fixed-cp-global4tiles-full-val-v1',workspace=str(ROOT),output=str(root),source_commit=commit,
        source_sha256={f:sha256(root/'source'/f) for f in (*old['source_sha256'],*FILES)},upstream_sha256=old['upstream_sha256'],
        owner_token=uuid.uuid4().hex,preparation_unix=PREPARATION_UNIX,deadline_unix=DEADLINE,max_seconds=14400,
        max_gpu_wait_seconds=300,image_id=old['image_id'],gpu=0,reference=SPEC,manifest=MANIFEST,manifest_sha256=MANIFEST_SHA,
        no_parameter_search=True,training_started=False,test_inputs_used=False,test_annotations_used=False,created_at_utc=queue.now())
    atomic_json(root/'plan.json',plan)
    cmd=[sys.executable,'-u',str(root/'source/run_m2f_tiles.py'),'worker','--output',str(root)]
    with (root/'workflow.log').open('xb',buffering=0) as log:
        p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    launch=dict(pid=p.pid,command=cmd,started_at_utc=queue.now(),deadline_unix=DEADLINE,plan_sha256=sha256(root/'plan.json'))
    atomic_json(root/'launch.json',launch);print(json.dumps(launch,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','worker']);p.add_argument('--output',required=True);p.add_argument('--source-commit');a=p.parse_args();r=Path(a.output).resolve()
    if a.mode=='start':start(r,a.source_commit)
    else:
        if (r/'status.json').exists():raise ValueError('No worker replay')
        s=Supervisor(r,read(r/'plan.json'));signal.signal(signal.SIGHUP,signal.SIG_IGN)
        for event in (signal.SIGTERM,signal.SIGINT):signal.signal(event,lambda *_:setattr(s,'stop_requested',True))
        s.execute()

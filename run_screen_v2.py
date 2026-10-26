"""Two explicitly declared candidates against an audited same-budget control."""
import fcntl
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import yaml
from course.common import atomic_json,configuration
from course.screen_protocol import compare_protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',default='screen_v2')
    parser.add_argument('--candidates',nargs=2,default=['j01_p2_queries','j02_geometry_queries'])
    parser.add_argument('--gpu-audit',default='runs/screen_v2_gpu_audit.json')
    parser.add_argument('--smoke-prefix',default='screen_v2_smoke_r2_')
    parser.add_argument('--control-dir',default='runs/screen_v1/i00_control')
    parser.add_argument('--allowed-change',nargs='*',default=[])
    args=parser.parse_args()
    assert len(set(args.candidates))==2, 'Two distinct candidates required'
    assert '/' not in args.suite and '\\' not in args.suite and args.suite.startswith('screen_')
    root=Path(__file__).resolve().parent
    suite=root/'runs'/args.suite
    suite.mkdir(parents=True,exist_ok=True)
    lock=(suite/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (suite/'suite.json').exists(): raise RuntimeError('Existing suite: inspect, never overwrite')
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    gpu_memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(gpu_memory.splitlines())>=2 and all(int(v.strip())<256 for v in gpu_memory.splitlines()[:2]), 'GPUs occupied; never stop unrelated jobs'
    reference_dir=(root/args.control_dir).resolve()
    assert reference_dir.is_relative_to(root/'runs'), 'Control must be a local audited run'
    reference=json.loads((reference_dir/'status.json').read_text())
    assert reference['status']=='completed' and reference['step']==3000
    reference_protocol=json.loads((reference_dir/'protocol.json').read_text())
    reference_config=yaml.safe_load(reference_protocol['config'])
    for name,sha in reference_protocol['geometry_assets'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    panel=json.loads((root/'runs/screen_v1/panel_screen.json').read_text())
    atomic_json(suite/'panel_screen.json',panel)
    panel_sha=hashlib.sha256((suite/'panel_screen.json').read_bytes()).hexdigest()
    assert panel_sha==reference_protocol['panel_sha256']
    weights=root/'runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth'
    weights_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert weights_sha==reference_protocol['weights_sha256']
    jobs={n:dict(gpu=str(i),config=f'configs/screen/{n}.yaml',status='pending')
          for i,n in enumerate(args.candidates)}
    for job in jobs.values():
        cfg=yaml.safe_load(configuration(job['config']).dump())
        job['declared_training_differences']=compare_protocol(cfg,reference_config,args.allowed_change)
    audit=json.loads((root/args.gpu_audit).read_text())
    assert audit['status']=='passed' and len(audit['comparisons'])==4
    assert {row['config'] for row in audit['comparisons']}==set(args.candidates), 'Wrong candidate GPU audit'
    for name in jobs:
        smoke=json.loads((root/f'runs/{args.smoke_prefix}{name}/status.json').read_text())
        assert smoke['status']=='smoke_test_passed' and smoke['step']==10
        smoke_dir=root/f'runs/{args.smoke_prefix}{name}'
        assert any((smoke_dir/n).is_file() for n in ('query_gradient_step10.json','local_gradient_step10.json'))
        coverage=configuration(jobs[name]['config']).COURSE.COVERAGE
        if coverage.MATCH_POINTS or coverage.LOSS_POINTS:
            evidence=json.loads((smoke_dir/'coverage_step10.json').read_text())
            assert evidence['positive_coverage_seen'], 'Small-instance sampling was never exercised in smoke'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    archive=subprocess.check_output(['git','archive','--format=zip',revision],cwd=root)
    source=suite/'source';source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source/n).resolve().is_relative_to(source) for n in z.namelist())
        z.extractall(source)
    for name in ('weights','data_cache'): (source/name).symlink_to(root/name,target_is_directory=True)
    atomic_json(source/'.source_revision.json',dict(commit=revision,dirty=False,immutable_snapshot=True,
                 archive_sha256=hashlib.sha256(archive).hexdigest()))
    reference_scores=[{k:v.get(k) for k in ('additional_training_step','mask_aps','mask_ap','mask_aps50','mask_aps75','mask_ars')}
                      for v in reference['validation_results']]
    state=dict(status='preflight',commit=revision,source=str(source),controller_pid=os.getpid(),jobs=jobs,
               started_unix=time.time(),panel_sha256=panel_sha,weights_sha256=weights_sha,
               matched_control=str(reference_dir),control_metrics=reference_scores,
               control_status_sha256=hashlib.sha256((reference_dir/'status.json').read_bytes()).hexdigest(),
               control_reuse='Same trained633 tensors, shared model initialization, optimizer/update budget, data/geometry hashes and panel; explicit training-factor exceptions recorded per job; CPU/GPU initial parity',
               allowed_training_changes=args.allowed_change,
               rng_caveat='Same seed/sampler; scale or point-budget changes can alter RNG consumption. Coverage floors use deterministic anchors and keep full original point draws; unit tests check RNG parity. Identical later optimization trajectories or wall time are NOT claimed',
               policy='Two +3000-step opt-in candidates; fixed200 at0/1000/2000/3000; no full val/test; heartbeat reviews and continues',
               gpu_parity=audit,test_used=False,confirmation_used=False)
    processes,logs={},[]

    def update(**values):
        state.update(values);state['updated_unix']=time.time()
        atomic_json(suite/'suite.json',state)
    def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        update()
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        for name,job in jobs.items():
            log=(suite/(name+'.log')).open('a');logs.append(log)
            p=subprocess.Popen([sys.executable,'-u','train_screen.py','--config',job['config'],'--weights',str(weights),
                '--panel',str(suite/'panel_screen.json'),'--output',str(suite/name)],cwd=source,
                env=dict(env,CUDA_VISIBLE_DEVICES=job['gpu']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=p;job.update(status='running',pid=p.pid)
        update(status='running')
        while True:
            for name,p in processes.items():
                path=suite/name/'status.json'
                if path.exists():
                    s=json.loads(path.read_text())
                    jobs[name].update(status=s['status'],step=s['step'],screen_metrics=[{k:v.get(k) for k in
                        ('additional_training_step','mask_aps','mask_ap','mask_aps50','mask_aps75','mask_ars')} for v in s['validation_results']])
                if p.poll() is not None and p.returncode: raise RuntimeError(f'{name} failed: {p.returncode}')
            update()
            if all(p.poll() is not None for p in processes.values()):
                assert all(j['status']=='completed' for j in jobs.values())
                deltas={n:j['screen_metrics'][-1]['mask_aps']-reference_scores[-1]['mask_aps'] for n,j in jobs.items()}
                update(status='completed',finished_unix=time.time(),paired_final_aps_deltas=deltas,
                       next_action='Autonomous heartbeat: analyze trends and error groups; record results; implement next evidence-driven quick experiment, no automatic full val')
                break
            time.sleep(15)
    except BaseException as error:
        for p in processes.values():
            if p.poll() is None: os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
        update(status='stopped' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        for log in logs: log.close()


if __name__=='__main__': main()

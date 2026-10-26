"""Train1280 vs existing trained1024 P2 control, BOTH evaluated at1280/fixed200."""
import fcntl
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
import torch
import yaml
from course.common import atomic_json,configuration
from course.resolution_screen import validate_resolution_pair,assert_same_initial_state,paired_scores


def main():
    root=Path(__file__).resolve().parent
    lock=(root/'runs/.screen_v8.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(v)<256 for v in memory.splitlines()[:2]),'GPUs occupied; never stop unrelated work'
    suite=root/'runs/screen_v8';assert not suite.exists(),'Existing output: inspect, never overwrite'
    control=root/'runs/screen_v2/j01_p2_queries'
    previous=json.loads((control/'status.json').read_text());assert previous['status']=='completed' and previous['step']==3000
    protocol=json.loads((control/'protocol.json').read_text())
    configs=['q00_p2_eval1280','q01_p2_train1280']
    cfgs=[yaml.safe_load(configuration(f'configs/screen/{n}.yaml').dump()) for n in configs]
    differences=validate_resolution_pair(cfgs[1],cfgs[0],yaml.safe_load(protocol['config']))
    for name,sha in protocol['geometry_assets'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    panel=root/'runs/screen_v1/panel_screen.json'
    panel_sha=hashlib.sha256(panel.read_bytes()).hexdigest();assert panel_sha==protocol['panel_sha256']
    weights=root/'runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth'
    weights_sha=hashlib.sha256(weights.read_bytes()).hexdigest();assert weights_sha==protocol['weights_sha256']
    checkpoints={0:control/'model_initialized_inference.pth',1000:control/'model_0000999.pth',
                 2000:control/'model_0001999.pth',3000:control/'model_final.pth'}
    initial=torch.load(checkpoints[0],map_location='cpu',weights_only=False)['model']
    smoke=root/'runs/screen_v8_smoke_q01_p2_train1280'
    assert json.loads((smoke/'status.json').read_text())['status']=='smoke_test_passed'
    initialized=torch.load(smoke/'model_initialized_inference.pth',map_location='cpu',weights_only=False)['model']
    shared_count=assert_same_initial_state(initial,initialized)
    del initial,initialized
    gpu=json.loads((root/'runs/screen_v8_gpu_audit.json').read_text())
    assert gpu['status']=='passed' and len(gpu['comparisons'])==4
    assert gpu['input_size']==1280 and gpu['reference']==configs[0]
    assert {r['config'] for r in gpu['comparisons']}==set(configs)
    data_audit=root/'runs/screen_v8_data_audit.json';data=json.loads(data_audit.read_text())
    assert data['status']=='passed' and data['train_only'] and len(data['rows'])==4
    for row in data['rows']:
        assert row['config'] in configs
        expected=1024 if row['config']==configs[0] else 1280
        assert row['shape']==[11,expected,expected]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    archive=subprocess.check_output(['git','archive','--format=zip',revision],cwd=root)
    source=suite/'source';source.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source/n).resolve().is_relative_to(source) for n in z.namelist());z.extractall(source)
    for n in ('weights','data_cache'):(source/n).symlink_to(root/n,target_is_directory=True)
    atomic_json(source/'.source_revision.json',dict(commit=revision,dirty=False,immutable_snapshot=True,
                archive_sha256=hashlib.sha256(archive).hexdigest()))
    state=dict(status='running',commit=revision,source=str(source),controller_pid=os.getpid(),started_unix=time.time(),
               differences=differences,matched_control=str(control),panel_sha256=panel_sha,weights_sha256=weights_sha,
               identical_initial_tensors=shared_count,test_used=False,full_val=False,confirmation_used=False,
               data_audit_sha256=hashlib.sha256(data_audit.read_bytes()).hexdigest(),
               policy='Same +3000 update/data-exposure budget; train1024 vs1280; both eval1280 on frozen200. Higher resolution costs more compute; not FLOP/wall-time matched. No test without meaningful reproducible gains.',
               control_checkpoint_sha256={str(k):hashlib.sha256(v.read_bytes()).hexdigest() for k,v in checkpoints.items()},
               jobs={configs[0]:dict(gpu='0',role='reuse_control_re_evaluation',status='pending',step=0,screen_metrics=[]),
                     configs[1]:dict(gpu='1',role='train1280',status='pending',step=0,screen_metrics=[])})
    processes,logs={},[]
    def update(**values):
        path=suite/configs[1]/'status.json'
        if path.exists():
            d=json.loads(path.read_text());state['jobs'][configs[1]].update(status=d['status'],step=d['step'],
                screen_metrics=[{k:v[k] for k in ('additional_training_step','mask_aps','mask_ap','mask_aps75','mask_ars','input_size')} for v in d['validation_results']])
        state.update(values);state['updated_unix']=time.time();atomic_json(suite/'suite.json',state)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def start(name,command,log_path,gpu_id):
        log=log_path.open('w');logs.append(log)
        p=subprocess.Popen(command,cwd=source,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_id,
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=p;state['jobs'][name]['pid']=p.pid;return p
    try:
        atomic_json(suite/'panel_screen.json',json.loads(panel.read_text()));update()
        name=configs[1]
        candidate_dir=suite/name
        start(name,[sys.executable,'-u','train_screen.py','--config',f'configs/screen/{name}.yaml','--weights',str(weights),
              '--panel',str(suite/'panel_screen.json'),'--output',str(candidate_dir),'--training-size','1280'],suite/(name+'.log'),'1')
        reference_results=[]
        reference_dir=suite/configs[0];reference_dir.mkdir()
        for step,path in checkpoints.items():
            name=configs[0];out=reference_dir/f'screen_step{step}.json'
            state['jobs'][name].update(status='screen_evaluation',step=step)
            p=start(name,[sys.executable,'-u','evaluate_rgbd.py','--config',f'configs/screen/{name}.yaml',
                '--weights',str(path),'--strict-load','--image-ids',str(suite/'panel_screen.json'),'--diagnostics',
                '--output',str(out)],reference_dir/f'screen_step{step}.log','0')
            update()
            while p.poll() is None:
                if processes[configs[1]].poll() not in (None,0):raise RuntimeError('Candidate training failed')
                update()
                time.sleep(5)
            assert p.returncode==0,f'Control evaluation failed at{step}'
            d=json.loads(out.read_text());assert d['panel_sha256']==panel_sha and d['input_size']==1280 and d['validation_images']==200
            d['additional_training_step']=step;reference_results.append(d)
            state['jobs'][name]['screen_metrics']=[{k:v[k] for k in ('additional_training_step','mask_aps','mask_ap','mask_aps75','mask_ars','input_size')} for v in reference_results]
            update()
        state['jobs'][configs[0]].update(status='completed',step=3000)
        while True:
            p=processes[configs[1]];file=candidate_dir/'status.json'
            if file.exists():
                d=json.loads(file.read_text());state['jobs'][configs[1]].update(status=d['status'],step=d['step'],
                    screen_metrics=[{k:v[k] for k in ('additional_training_step','mask_aps','mask_ap','mask_aps75','mask_ars','input_size')} for v in d['validation_results']])
            if p.poll() not in (None,0):raise RuntimeError('Candidate training failed')
            update()
            if p.poll()==0:
                assert d['status']=='completed' and d['step']==3000
                deltas=paired_scores(d['validation_results'],reference_results)
                update(status='completed',paired_aps_deltas=deltas,finished_unix=time.time(),
                    paired_final_aps_deltas={configs[1]:deltas['3000']},
                    next_action='Record same-inference-resolution results; only >=.01 repeated gains may enter independent confirmation/seed checks. Otherwise continue optimization; NO test.')
                break
            time.sleep(15)
    except BaseException as exc:
        for p in processes.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':main()

"""Start D fine-tuning on GPUs 2/3, then resume on four GPUs at a saved checkpoint.

The expansion waits for round-one test/export completion and idle GPUs 0/1.
The global batch stays 8. Optimizer/scheduler are restored during expansion;
dataloader order after resume is not bit-exact. Every process/log is owned here.
"""
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path('seg_project/experiments/mask2former/d_continue6_20261026')
ROUND=Path('seg_project/experiments/mask2former/optimization_round1_20261026')
CONFIG='seg_project/configs/mask2former/d_continue6_20261026.yaml'


def write_state(state):
    state=dict(state,updated=time.time())
    temp=ROOT/'manager_state.json.partial'
    temp.write_text(json.dumps(state,indent=2))
    temp.replace(ROOT/'manager_state.json')


def parse_gpu_memory(output):
    memory={}; current=None
    for line in output.splitlines():
        match=re.search(r'\|\s*(\d+)\s+Iluvatar\b',line)
        if match: current=int(match.group(1))
        match=re.search(r'(\d+)MiB\s*/\s*\d+MiB',line)
        if match and current is not None:
            memory[current]=int(match.group(1)); current=None
    return memory


def spare_gpus_idle():
    result=subprocess.run(['ixsmi'],capture_output=True,text=True,timeout=15,check=True)
    memory=parse_gpu_memory(result.stdout)
    return all(index in memory and memory[index]<512 for index in (0,1))


def last_checkpoint():
    tag=ROOT/'last_checkpoint'
    if not tag.exists(): return None
    name=tag.read_text().strip()
    if not name or Path(name).name!=name: raise ValueError('Unexpected checkpoint tag')
    path=ROOT/name
    return name if path.is_file() and path.stat().st_size>0 else None


def launch(gpus,resume,attempt):
    command=['./seg_project/run.sh','python','-m','torch.distributed.run','--standalone',
             '--nnodes=1',f'--nproc-per-node={len(gpus)}','seg_project/tools/train_m2f.py',
             '--config-file',CONFIG]
    if resume: command.append('--resume')
    log=(ROOT/f'attempt{attempt}.log').open('w')
    process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus))),
                             stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state=dict(status='running',manager_pid=os.getpid(),training_pid=process.pid,
               attempt=attempt,gpus=gpus,global_batch=8,resume=resume,command=command,started=time.time())
    write_state(state)
    print(json.dumps(state),flush=True)
    return process,log,state


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    smoke=Path('seg_project/experiments/mask2former/d_continue6_smoke_20261026/job.exit_code')
    if smoke.read_text().strip()!='0': raise ValueError('Training smoke must pass')
    if (ROOT/'manager_state.json').exists() or (ROOT/'last_checkpoint').exists():
        raise FileExistsError('Training output already exists; inspect before resuming')
    process,log,state=launch([2,3],False,1)
    checkpoint_when_ready=None
    waiting_for_new_checkpoint=False
    try:
        while process.poll() is None:
            if (ROUND/'test_complete.json').exists() and spare_gpus_idle():
                checkpoint=last_checkpoint()
                if not waiting_for_new_checkpoint:
                    checkpoint_when_ready=checkpoint
                    waiting_for_new_checkpoint=True
                    state['expansion_status']='waiting for next completed checkpoint'
                    write_state(state)
                elif checkpoint is not None and checkpoint!=checkpoint_when_ready:
                    # The tag is published after torch.save completes. Stop the owned
                    # torchrun launcher, which shuts down its worker processes.
                    state['expansion_checkpoint']=checkpoint
                    write_state(state)
                    process.send_signal(signal.SIGTERM)
                    code=process.wait(timeout=120)
                    log.close()
                    (ROOT/'attempt1.exit_code').write_text(str(code))
                    (ROOT/'gpu_expansion.json').write_text(json.dumps(dict(
                        checkpoint=checkpoint,old_pid=process.pid,old_exit=code,
                        old_gpus=[2,3],new_gpus=[0,1,2,3],global_batch=8,
                        reason='Round-one submission complete; expand at newly saved checkpoint.',
                        resume_note='At most the few steps since checkpoint publication are replayed; loader order is not bit-exact.',
                        timestamp=time.time()),indent=2))
                    process,log,state=launch([0,1,2,3],True,2)
                    break
            time.sleep(5)
        code=process.wait()
        log.close()
        (ROOT/f'attempt{state["attempt"]}.exit_code').write_text(str(code))
        state.update(status='completed' if code==0 else 'failed',exit_code=code,finished=time.time(),
                     last_checkpoint=last_checkpoint())
        write_state(state)
        if code: raise RuntimeError(f'Training failed with exit code {code}; inspect attempt log')
    except BaseException as error:
        state.update(status='manager_error',error=repr(error),training_alive=process.poll() is None)
        write_state(state)
        raise


if __name__=='__main__': main()

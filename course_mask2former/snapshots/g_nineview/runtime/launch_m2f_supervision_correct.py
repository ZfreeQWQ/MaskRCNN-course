"""Immutable budget; one original GPU0 lock acquisition, no extra training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import launch_m2f_rle_passthrough as reused
import run_m2f_supervision_correct as worker
from run_m2f_supervision_correct import START,PREP_END,TRAIN_END,GPU_END,EVAL_END,DEADLINE,NAME,OWNER,read,save,sha

ROOT=Path("/mnt/nfs/fenghaibin/AI_Class")
OUT=ROOT/"outputs"/NAME
reused.ROOT,reused.OUT,reused.OWNER=ROOT,OUT,OWNER


def command(phase):
    gpu=phase in ("traincheck","predict")
    _,cmd=reused.command("gpu" if gpu else "cpu")
    name="ai-m2f-correct-"+OWNER[:10]+"-"+phase
    cmd[cmd.index("--name")+1]=name
    cmd[cmd.index("--memory")+1]="16g" if phase=="traincheck" else "32g" if gpu else "8g"
    cmd[cmd.index("m2f_rle_passthrough.py"):]=["run_m2f_supervision_correct.py",phase]
    assert gpu==("--gpus" in cmd)
    return name,cmd


def stop_owned(name):
    try:item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
    except subprocess.CalledProcessError:return
    assert item["Config"]["Labels"].get("ai-course-directions-owner")==OWNER
    subprocess.run(["docker","stop","-t","5",name],timeout=12,check=True)


def main():
    assert not (OUT/"plan.json").exists(),"No duplicate dispatch"
    assert time.time()<PREP_END
    plan=dict(start_unix=START,prep_end=PREP_END,train_check_end=TRAIN_END,gpu_end=GPU_END,
        evaluation_end=EVAL_END,deadline_unix=DEADLINE,source_commit=sys.argv[1],owner=OWNER,
        source_sha256={p.name:sha(p) for p in (OUT/"source").glob("*.py")},
        original_lock=str(ROOT/"outputs/.course_directions_gpu0.lock"),lock_wait_seconds=0,
        gpu=0,gpu_max_seconds=1800,baseline_aps=.41095135423262297,maskrcnn_paused=True,
        m2f_weights_frozen=True,only_change="train gt_area>1024 targets become coarse identity",
        fit=384,check=128,old_fit_probe="first64 sorted fit IDs",steps=3000,batch=32,
        lr=.001,weight_decay=.0001,seed=20261013,
        gate="check128 small ten-threshold net>0; tiny net>=0 vs unmodified best masks",
        only_one_full_val_after_pass=True,no_parameter_search=True)
    save(OUT/"plan.json",plan);save(OUT/"launch.json",dict(supervisor_pid=os.getpid(),unix=time.time()))
    phases=[];name=None;process=None;phase="prep"
    def execute(stage,end):
        nonlocal name,process,phase
        assert time.time()<end
        name,cmd=command(stage);phase=stage;begin=time.time()
        save(OUT/(stage+"_command.json"),cmd)
        with (OUT/(stage+".log")).open("x") as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/"status.json",dict(status="running",phase=stage,container=name,client_pid=process.pid,stage_end=end,unix=time.time()))
            inspected=False
            while process.poll() is None:
                now=time.time()
                if now>=end:raise TimeoutError("Immutable "+stage+" cutoff")
                if not inspected and now-begin>3:
                    try:
                        item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
                        assert item["Config"]["Labels"].get("ai-course-directions-owner")==OWNER
                        if stage in ("traincheck","predict"):
                            assert item["HostConfig"]["DeviceRequests"][0]["DeviceIDs"]==["0"]
                        else:assert not item["HostConfig"].get("DeviceRequests")
                        save(OUT/(stage+"_container.json"),dict(owner=OWNER,
                            devices=item["HostConfig"].get("DeviceRequests"),cpus=item["HostConfig"]["NanoCpus"],
                            memory=item["HostConfig"]["Memory"],network=item["HostConfig"]["NetworkMode"],unix=now))
                        inspected=True
                    except subprocess.CalledProcessError:pass
                if stage in ("traincheck","predict"):
                    stamp=(OUT/"progress.json").stat().st_mtime if (OUT/"progress.json").exists() else begin
                    if now-max(stamp,begin)>300:raise TimeoutError("5min no persistent progress")
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,start=begin,finished=time.time(),exit_code=0));save(OUT/"phases.json",phases)
    try:
        execute("prep",PREP_END)
        assert read(OUT/"cpu_acceptance.json")["status"]=="passed"
        phase="gpu_gate"
        assert time.time()<PREP_END
        with (ROOT/"outputs/.course_directions_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            raw=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"],text=True,timeout=8)
            index,memory,util=map(int,raw.strip().split(","))
            assert index==0 and memory<=512 and util<=10,raw
            procs=subprocess.check_output(["nvidia-smi","-i","0","--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits"],text=True,timeout=8).strip()
            assert not procs,"Actual occupied GPU0: "+procs
            gpu_end=min(GPU_END,time.time()+1800)
            save(OUT/"gpu_gate.json",dict(lock_acquired=True,memory_mib=memory,utilization=util,
                compute_processes=procs,gpu=0,unix=time.time(),gpu_end=gpu_end))
            save(OUT/"phase_deadlines.json",dict(train_end=min(TRAIN_END,gpu_end),predict_end=gpu_end,deadline=DEADLINE))
            execute("traincheck",min(TRAIN_END,gpu_end))
            gate=read(OUT/"learned_gate.json")
            if gate["gate_passed"]:execute("predict",gpu_end)
        save(OUT/"gpu_released.json",dict(unix=time.time(),original_lock_released=True))
        if gate["gate_passed"]:
            execute("evaluate",EVAL_END)
            execute("analyze",DEADLINE)
        else:
            worker.ROOT,worker.OUT=ROOT,OUT
            worker.stopped_report()
        report=read(OUT/"report.json")
        assert report["status"]=="passed" and report["finished_unix"]<DEADLINE
        assert all(sha(OUT/n)==h for n,h in report["artifacts_sha256"].items())
        assert all(sha(OUT/"source"/n)==h for n,h in plan["source_sha256"].items())
        assert report["full_val_evaluations"]==int(gate["gate_passed"])
        save(OUT/"terminal_acceptance.json",dict(status="passed",report_sha256=sha(OUT/"report.json"),
            plan_sha256=sha(OUT/"plan.json"),source_unchanged=True,gate_enforced=True,unix=time.time()))
        save(OUT/"status.json",dict(status="complete",phase=report["outcome"],unix=time.time()))
    except BaseException as exc:
        save(OUT/"first_error.json",dict(error=repr(exc),phase=phase,unix=time.time()))
        if name:
            try:stop_owned(name)
            except Exception as e:save(OUT/"cleanup_error.json",dict(error=repr(e),unix=time.time()))
        if process is not None and process.poll() is None:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate()
        save(OUT/"status.json",dict(status="failed",phase=phase,unix=time.time()))
        raise


if __name__=="__main__":
    main()

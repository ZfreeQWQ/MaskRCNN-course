"""Immutable 60min/20min, original GPU0 lock and actual occupancy gate."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import launch_m2f_rle_passthrough as reused
import run_m2f_tiny_hflip as worker
from run_m2f_tiny_hflip import START,PREP_END,TRAIN_END,GPU_END,EVAL_END,DEADLINE,NAME,OWNER,read,save,sha

ROOT=Path("/mnt/nfs/fenghaibin/AI_Class")
OUT=ROOT/"outputs"/NAME
reused.ROOT,reused.OUT,reused.OWNER=ROOT,OUT,OWNER


def command(stage):
    gpu=stage in ("traincheck","predict")
    _,cmd=reused.command("gpu" if gpu else "cpu")
    name="ai-m2f-tinyflip-"+OWNER[:10]+"-"+stage
    cmd[cmd.index("--name")+1]=name
    cmd[cmd.index("--memory")+1]="32g" if gpu else "8g"
    cmd[cmd.index("m2f_rle_passthrough.py"):]=["run_m2f_tiny_hflip.py",stage]
    assert gpu==("--gpus" in cmd)
    return name,cmd


def stop_owned(name):
    try:item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
    except subprocess.CalledProcessError:return
    assert item["Config"]["Labels"].get("ai-course-directions-owner")==OWNER
    subprocess.run(["docker","stop","-t","5",name],timeout=12,check=True)


def main():
    assert time.time()<PREP_END and not (OUT/"plan.json").exists(),"Duplicate or late"
    plan=dict(start_unix=START,prep_end=PREP_END,train_end=TRAIN_END,prediction_end=GPU_END,
        evaluation_end=EVAL_END,deadline_unix=DEADLINE,source_commit=sys.argv[1],owner=OWNER,
        source_sha256={p.name:sha(p) for p in (OUT/"source").glob("*.py")},
        original_lock=str(ROOT/"outputs/.course_directions_gpu0.lock"),lock_wait_seconds=0,
        gpu=0,gpu_max_seconds=1200,baseline_aps=worker.BEST_APS,maskrcnn_paused=True,
        unique_change="training-only six-channel horizontal flip p=.5",flip_seed=20261021,
        feature_grid=[16,16],steps=2000,batch=32,positive_negative_ratio="1:1 uniform",
        fit_images=640,check_images=128,lr=.001,weight_decay=.0001,seed=20261013,
        probability_threshold=.5,max_new=5,merge_iou=.5,max_instances=100,
        old_models_frozen=True,independent_augmentation_rng=True,inference_tta=False,
        learned_gate="same128 APs positive AND tiny one-to-one net>=0",one_val_after_gate=True)
    save(OUT/"plan.json",plan);save(OUT/"launch.json",dict(supervisor_pid=os.getpid(),unix=time.time()))
    name=None;process=None;phase="prep";phases=[];acquired=None;lock=None
    def execute(stage,end):
        nonlocal name,process,phase
        assert time.time()<end
        name,cmd=command(stage);phase=stage;begin=time.time()
        save(OUT/(stage+"_command.json"),cmd)
        with (OUT/(stage+".log")).open("x") as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
            save(OUT/"status.json",dict(status="running",phase=stage,container=name,
                client_pid=process.pid,stage_end=end,unix=time.time()))
            inspected=False
            while process.poll() is None:
                now=time.time()
                if now>=end:raise TimeoutError("Immutable "+stage+" cutoff")
                if not inspected and now-begin>3:
                    try:
                        item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
                        hc=item["HostConfig"]
                        assert item["Config"]["Labels"].get("ai-course-directions-owner")==OWNER
                        assert hc["NetworkMode"]=="none" and hc["NanoCpus"]==4000000000
                        if stage in ("traincheck","predict"):
                            assert hc["DeviceRequests"][0]["DeviceIDs"]==["0"] and hc["Memory"]==32*2**30
                        else:
                            assert not hc.get("DeviceRequests") and hc["Runtime"]=="runc" and hc["Memory"]==8*2**30
                        save(OUT/(stage+"_container.json"),dict(owner=OWNER,devices=hc.get("DeviceRequests"),
                            runtime=hc["Runtime"],cpus=hc["NanoCpus"],memory=hc["Memory"],
                            network=hc["NetworkMode"],unix=now))
                        inspected=True
                    except subprocess.CalledProcessError:pass
                if stage in ("traincheck","predict"):
                    stamp=(OUT/"progress.json").stat().st_mtime if (OUT/"progress.json").exists() else begin
                    if now-max(stamp,begin)>300:raise TimeoutError("5min no persistent progress")
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,start=begin,finished=time.time(),exit_code=0))
        save(OUT/"phases.json",phases)
    try:
        execute("prep",PREP_END)
        assert read(OUT/"cpu_acceptance.json")["status"]=="passed"
        phase="gpu_gate";assert time.time()<PREP_END
        lock=(ROOT/"outputs/.course_directions_gpu0.lock").open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"],text=True,timeout=8)
        index,memory,util=map(int,raw.strip().split(","))
        assert index==0 and memory<=512 and util<=10,raw
        procs=subprocess.check_output(["nvidia-smi","-i","0","--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"],text=True,timeout=8).strip()
        assert not procs,"Actual GPU0 occupied: "+procs
        acquired=time.time();gpu_stop=min(GPU_END,acquired+1180)
        save(OUT/"gpu_gate.json",dict(lock_acquired=True,gpu=0,memory_mib=memory,utilization=util,
            compute_processes=procs,unix=acquired,gpu_stop=gpu_stop,visibility_not_occupancy=True))
        limits=dict(train_end=min(TRAIN_END,gpu_stop),predict_end=gpu_stop)
        save(OUT/"phase_deadlines.json",limits)
        execute("traincheck",limits["train_end"])
        if read(OUT/"learned_gate.json")["gate_passed"]:execute("predict",gpu_stop)
        lock.close();lock=None
        save(OUT/"gpu_released.json",dict(unix=time.time(),original_lock_released=True,
            gpu_allocation_seconds=time.time()-acquired))
        execute("finish",EVAL_END)
        report=read(OUT/"report.json")
        assert report["status"]=="passed" and report["finished_unix"]<EVAL_END
        assert all(sha(OUT/n)==h for n,h in report["artifacts_sha256"].items())
        assert all(sha(OUT/"source"/n)==h for n,h in plan["source_sha256"].items())
        assert read(OUT/"gpu_released.json")["gpu_allocation_seconds"]<=1200
        save(OUT/"terminal_acceptance.json",dict(status="passed",report_sha256=sha(OUT/"report.json"),
            plan_sha256=sha(OUT/"plan.json"),source_unchanged=True,gates_enforced=True,unix=time.time()))
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
    finally:
        if lock is not None:
            lock.close()
            if acquired:save(OUT/"gpu_released.json",dict(unix=time.time(),original_lock_released=True,
                gpu_allocation_seconds=time.time()-acquired,failure_cleanup=True))


if __name__=="__main__":main()

"""Fail closed on original GPU0 flock/actual occupancy; no budget reset."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import launch_m2f_rle_passthrough as reused
from run_m2f_boundary_refine import START,PREP_END,TRAIN_END,EVAL_END,DEADLINE,NAME,OWNER,read,save,sha

ROOT=Path("/mnt/nfs/fenghaibin/AI_Class")
OUT=ROOT/"outputs"/NAME
reused.ROOT,reused.OUT,reused.OWNER=ROOT,OUT,OWNER


def command(phase):
    gpu=phase in ("train","predict")
    _,cmd=reused.command("gpu" if gpu else "cpu")
    name="ai-m2f-boundary-"+OWNER[:10]+"-"+phase
    cmd[cmd.index("--name")+1]=name
    cmd[cmd.index("--memory")+1]="16g" if phase!="predict" else "32g"
    cmd[cmd.index("m2f_rle_passthrough.py"):]=["run_m2f_boundary_refine.py",phase]
    assert gpu==("--gpus" in cmd)
    if gpu:assert cmd[cmd.index("--gpus")+1]=="device=0"
    else:assert cmd[cmd.index("--runtime")+1]=="runc"
    return name,cmd


def owned_stop(name):
    try:item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
    except subprocess.CalledProcessError:return
    assert item["Config"]["Labels"].get("ai-course-directions-owner")==OWNER
    subprocess.run(["docker","stop","-t","5",name],timeout=12,check=True)


def main():
    assert not (OUT/"plan.json").exists(),"Refuse duplicate launch"
    assert time.time()<PREP_END
    plan=dict(start_unix=START,prep_end=PREP_END,train_end=TRAIN_END,evaluation_end=EVAL_END,
        deadline_unix=DEADLINE,source_commit=sys.argv[1],owner=OWNER,gpu=0,cpu=4,
        source_sha256={p.name:sha(p) for p in (OUT/"source").glob("*.py")},
        baseline_aps=.41095135423262297,maskrcnn_paused=True,m2f_weights_frozen=True,
        original_lock=str(ROOT/"outputs/.course_directions_gpu0.lock"),lock_wait_seconds=0,
        refiner_steps=3000,batch=32,adamw_lr=.001,weight_decay=.0001,grad_clip=1.,
        seed=20261013,patch=128,radius=4,depth_feature="log1p(metres), valid separate",
        fit_supervision="best overlapping same-category GT; unmatched identity; uniform per-instance band BCE",
        no_parameter_search=True,stage_max_seconds=dict(prep=3600,train=5400,evaluation=3600,analysis=1800))
    save(OUT/"plan.json",plan);save(OUT/"launch.json",dict(supervisor_pid=os.getpid(),unix=time.time()))
    phases=[];name=None;phase="prep";process=None
    def execute(stage,end):
        nonlocal name,phase,process
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
                if now>=end:raise TimeoutError("Immutable "+stage+" deadline")
                if not inspected and now-begin>3:
                    try:
                        item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
                        assert item["Config"]["Labels"].get("ai-course-directions-owner")==OWNER
                        if stage in ("train","predict"):
                            assert item["HostConfig"]["DeviceRequests"][0]["DeviceIDs"]==["0"]
                        else:assert not item["HostConfig"].get("DeviceRequests")
                        save(OUT/(stage+"_container.json"),dict(name=name,owner=OWNER,
                            device_requests=item["HostConfig"].get("DeviceRequests"),
                            cpus=item["HostConfig"]["NanoCpus"],memory=item["HostConfig"]["Memory"],
                            network=item["HostConfig"]["NetworkMode"],unix=now))
                        inspected=True
                    except subprocess.CalledProcessError:pass
                if stage in ("train","predict"):
                    last=max(begin,(OUT/"progress.json").stat().st_mtime if (OUT/"progress.json").exists() else begin)
                    if now-last>300:raise TimeoutError(stage+" has no progress for5min")
                time.sleep(2)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd)
        phases.append(dict(phase=stage,started=begin,finished=time.time(),exit_code=0))
        save(OUT/"phases.json",phases)
    try:
        execute("prep",PREP_END)
        assert read(OUT/"cpu_acceptance.json")["status"]=="passed"
        if not read(OUT/"headroom_gate.json")["gate_passed"]:
            artifacts=("cpu_acceptance.json","headroom_gate.json","prepared_manifest.json","binding.json")
            save(OUT/"report.json",dict(status="passed",outcome="stopped_no_train_headroom",
                start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),
                full_val_evaluations=0,refiner_trained=False,gpu_used=False,m2f_updated=False,
                artifacts_sha256={n:sha(OUT/n) for n in artifacts}))
        else:
            phase="gpu_gate"
            assert time.time()<PREP_END
            with (ROOT/"outputs/.course_directions_gpu0.lock").open("a") as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                raw=subprocess.check_output(["nvidia-smi","-i","0",
                    "--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],timeout=8,text=True)
                index,memory,util=map(int,raw.strip().split(","))
                assert index==0 and memory<=512 and util<=10,raw
                processes=subprocess.check_output(["nvidia-smi","-i","0",
                    "--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"],
                    timeout=8,text=True).strip()
                assert not processes,"Actual GPU0 compute occupation: "+processes
                save(OUT/"gpu_gate.json",dict(lock_acquired=True,gpu=0,memory_mib=memory,
                    utilization=util,compute_processes=processes,visibility_not_occupation=True,unix=time.time()))
                train_end=min(TRAIN_END,time.time()+5400)
                save(OUT/"phase_deadlines.json",dict(train_end=train_end,deadline=DEADLINE))
                execute("train",train_end)
                assert read(OUT/"fit_report.json")["native_steps"]==[3000]
                end=min(EVAL_END,time.time()+3600)
                save(OUT/"phase_deadlines.json",dict(train_end=train_end,predict_end=end,evaluation_end=end,deadline=DEADLINE))
                execute("predict",end)
            save(OUT/"gpu_released.json",dict(unix=time.time(),original_lock_released=True))
            execute("evaluate",end)
            execute("analyze",min(DEADLINE,time.time()+1800))
        report=read(OUT/"report.json")
        assert report["status"]=="passed" and report["finished_unix"]<DEADLINE
        assert all(sha(OUT/n)==h for n,h in report["artifacts_sha256"].items())
        assert all(sha(OUT/"source"/n)==h for n,h in plan["source_sha256"].items())
        save(OUT/"terminal_acceptance.json",dict(status="passed",report_sha256=sha(OUT/"report.json"),
            plan_sha256=sha(OUT/"plan.json"),artifacts_verified=True,source_unchanged=True,unix=time.time()))
        save(OUT/"status.json",dict(status="complete",phase=report["outcome"],unix=time.time()))
    except BaseException as exc:
        save(OUT/"first_error.json",dict(phase=phase,error=repr(exc),unix=time.time()))
        if name:
            try:owned_stop(name)
            except Exception as cleanup:save(OUT/"cleanup_error.json",dict(error=repr(cleanup),unix=time.time()))
        if process is not None and process.poll() is None:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.terminate()
        save(OUT/"status.json",dict(status="failed",phase=phase,unix=time.time()))
        raise


if __name__=="__main__":
    main()

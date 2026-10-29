"""Original-lock, one GPU, fixed deadlines; CPU evaluation after GPU release."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import launch_m2f_rle_passthrough as reused
from run_m2f_quality_select import START, CPU_END, GPU_END, EVAL_END, DEADLINE, read, save, sha

ROOT = Path("/mnt/nfs/fenghaibin/AI_Class")
OUT = ROOT/"outputs/course_m2f_quality_select_20261020_v1"
OWNER = "6b827c784a39442e93ed922e705b9af5"
reused.ROOT, reused.OUT, reused.OWNER = ROOT, OUT, OWNER


def command(phase):
    name, args = reused.command(phase)
    name = "ai-m2f-quality-" + OWNER[:10] + "-" + phase
    args[args.index("--name")+1] = name
    args[args.index("m2f_rle_passthrough.py")] = "run_m2f_quality_select.py"
    if phase != "gpu":
        args[args.index("--memory")+1] = "8g"
    assert (phase == "gpu") == ("--gpus" in args)
    if phase == "gpu":
        assert args[args.index("--gpus")+1] == "device=0"
    return name, args


def owned_stop(name):
    try:
        item = json.loads(subprocess.check_output(["docker", "inspect", name], timeout=6))[0]
    except subprocess.CalledProcessError:
        return
    assert item["Config"]["Labels"].get("ai-course-directions-owner") == OWNER
    subprocess.run(["docker", "stop", "-t", "5", name], timeout=12, check=True)


def main():
    assert not (OUT/"plan.json").exists(), "No duplicate launch"
    assert time.time() < CPU_END
    plan = dict(start_unix=START, deadline_unix=DEADLINE, cpu_end=CPU_END, gpu_end=GPU_END,
        evaluation_end=EVAL_END, gpu_max_seconds=2700, owner=OWNER, gpu=0,
        source_commit=sys.argv[1], source_sha256={p.name:sha(p) for p in (OUT/"source").glob("*.py")},
        m2f_training=False, selector_training="CPU only, official train384; check128 disjoint scenes",
        fit_images=384, check_images=128, full_val_images=2584, full_val_evaluations=1,
        original_lock=str(ROOT/"outputs/.course_directions_gpu0.lock"), lock_wait_seconds=0,
        maskrcnn_paused=True)
    save(OUT/"plan.json", plan)
    save(OUT/"launch.json", dict(supervisor_pid=os.getpid(), unix=time.time()))
    name, phase, process = None, "cpu", None
    phases = []
    def execute(stage, end):
        nonlocal name, phase, process
        phase = stage
        assert time.time() < end
        name, cmd = command(stage)
        began = time.time()
        with (OUT/(stage+".log")).open("x") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
            save(OUT/"status.json", dict(status="running", phase=phase, container=name,
                                       client_pid=process.pid, stage_end=end, unix=time.time()))
            while process.poll() is None:
                now = time.time()
                if now >= end:
                    raise TimeoutError("Approved "+stage+" absolute deadline")
                if stage == "gpu":
                    last = max(began, (OUT/"progress.json").stat().st_mtime if (OUT/"progress.json").exists() else began)
                    if now-last > 300:
                        raise TimeoutError("GPU stage no persistent progress for5min")
                time.sleep(2)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        phases.append(dict(phase=stage, start=began, end=time.time(), exit_code=0))
        save(OUT/"phases.json", phases)
    try:
        execute("cpu", CPU_END)
        assert read(OUT/"cpu_acceptance.json")["status"] == "passed"
        phase = "gpu_gate"
        assert time.time() < CPU_END
        with (ROOT/"outputs/.course_directions_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raw = subprocess.check_output(["nvidia-smi", "-i", "0",
                "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], timeout=8, text=True)
            index, memory, utilization = map(int, raw.strip().split(","))
            assert index == 0 and memory <= 512 and utilization <= 10, raw
            processes = subprocess.check_output(["nvidia-smi", "-i", "0",
                "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
                timeout=8, text=True).strip()
            assert not processes, "Actual GPU0 compute process: "+processes
            save(OUT/"gpu_gate.json", dict(lock_acquired=True, gpu=0, memory_mib=memory,
                utilization=utilization, compute_processes=processes, visibility_not_occupancy=True, unix=time.time()))
            assert time.time() < CPU_END
            end = min(GPU_END, time.time()+2700)
            save(OUT/"phase_deadlines.json", dict(gpu_end=end, absolute_gpu_end=GPU_END,
                                                  deadline=DEADLINE, unix=time.time()))
            execute("gpu", end)
        save(OUT/"gpu_released.json", dict(unix=time.time(), original_lock_released=True))
        report = read(OUT/"gpu_report.json")
        assert report["status"] == "passed" and report["strict_parity"]
        assert report["calls"] == dict(eval_calls=2725, train_calls=0)
        assert report["captured_train_images"] == 512 and report["parity_images"] == 8
        assert report["timing"]["repeats"] == 20 and report["model_state_before"] == report["model_state_after"]
        execute("evaluate", EVAL_END)
        execute("analyze", DEADLINE)
        report = read(OUT/"report.json")
        assert report["status"] == "passed" and report["finished_unix"] < DEADLINE
        assert report["full_val_evaluations"] == 1 and report["complete_val_images"] == 2584
        assert not report["m2f_weights_updated"] and report["selector_fitted"]
        assert all(sha(OUT/"source"/p) == h for p,h in plan["source_sha256"].items())
        assert all(sha(OUT/p) == h for p,h in report["artifacts_sha256"].items())
        save(OUT/"terminal_acceptance.json", dict(status="passed", report_sha256=sha(OUT/"report.json"),
            selector_sha256=report["selector_sha256"], plan_sha256=sha(OUT/"plan.json"),
            source_unchanged=True, artifacts_verified=True, unix=time.time()))
        save(OUT/"status.json", dict(status="complete", phase="complete", unix=time.time()))
    except BaseException as exc:
        save(OUT/"first_error.json", dict(error=repr(exc), phase=phase, unix=time.time()))
        if name:
            try: owned_stop(name)
            except Exception as cleanup:
                save(OUT/"cleanup_error.json", dict(error=repr(cleanup), name=name, unix=time.time()))
        if process is not None and process.poll() is None:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.terminate()
        save(OUT/"status.json", dict(status="failed", phase=phase, unix=time.time()))
        raise


if __name__ == "__main__":
    main()

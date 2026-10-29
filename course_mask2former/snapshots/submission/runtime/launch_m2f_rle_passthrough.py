"""One original-lock, physical-GPU0 deployment check with immutable deadline."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from m2f_rle_passthrough import START, DEADLINE, CPU_END, GATE_END, GPU_END, read, save, sha
from m2f_geometry_compact_check import OLD, GEOM

ROOT = Path("/mnt/nfs/fenghaibin/AI_Class")
OUT = ROOT/"outputs/course_m2f_rle_passthrough_20261020_v1"
IMAGE = "sha256:eb5bed0d49efd109b91d5365131cc7f25391b8695347086d2e3dd12953d7702c"
OWNER = "a72cd0aa39da42f08a0cb53f8c8d7d52"


def command(phase):
    rel = "/workspace/"+str(OUT.relative_to(ROOT))
    name = "ai-m2f-rle-"+OWNER[:10]+"-"+phase
    cmd = ["docker", "run", "--rm", "--pull", "never", "--name", name,
        "--label", "ai-course-directions-owner="+OWNER, "--network", "none", "--read-only",
        "--cpus", "4", "--memory", "32g", "--shm-size", "4g",
        "--user", f"{os.getuid()}:{os.getgid()}", "-v", str(ROOT)+":/workspace:ro",
        "-v", str(ROOT)+":"+str(ROOT)+":ro", "-v", str(OUT)+":"+rel+":rw",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "-w", rel+"/source"]
    env = dict(PYTHONPATH=rel+"/source:/workspace/"+OLD+"/source:/workspace/"+GEOM+"/source:/workspace/mmdetection",
        PATH="/workspace/envs/mmdet/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        HOME="/tmp", MPLCONFIGDIR="/tmp/matplotlib", PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if phase == "gpu":
        cmd += ["--gpus", "device=0"]
    else:
        cmd += ["--runtime", "runc"]
        env.update(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    for key, value in env.items():
        cmd += ["-e", key+"="+value]
    return name, cmd+[IMAGE, "python", "m2f_rle_passthrough.py", phase, "--output", rel]


def main():
    assert not (OUT/"plan.json").exists(), "No duplicate launch"
    assert time.time() < CPU_END, "Preparation gate expired"
    plan = dict(start_unix=START, deadline_unix=DEADLINE, owner=OWNER, gpu=0,
        source_sha256={p.name: sha(p) for p in (OUT/"source").glob("*.py")},
        source_commit=sys.argv[1], training=False, full_val=False, cpu_deadline=CPU_END,
        gate_deadline=GATE_END, gpu_deadline=GPU_END, gpu_max_seconds=1500,
        original_lock=str(ROOT/"outputs/.course_directions_gpu0.lock"))
    save(OUT/"plan.json", plan)
    phase, name = "cpu", None
    def execute(stage, end):
        nonlocal phase, name
        phase = stage
        name, cmd = command(stage)
        assert time.time() < end
        save(OUT/"status.json", dict(status="running", phase=phase, container=name, unix=time.time()))
        with (OUT/(stage+".log")).open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=end-time.time(), close_fds=True)
    try:
        execute("cpu", CPU_END)
        phase = "lock"
        assert time.time() < GATE_END
        with (ROOT/"outputs/.course_directions_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            raw = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], timeout=8, text=True)
            index, memory, utilization = map(int, raw.strip().split(","))
            assert index == 0 and memory <= 512 and utilization <= 10, raw
            processes = subprocess.check_output(["nvidia-smi", "-i", "0",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits"], timeout=8, text=True).strip()
            assert not processes, "GPU0 compute process already exists: "+processes
            save(OUT/"gpu_gate.json", dict(lock_acquired=True, gpu=0, memory_mib=memory,
                utilization=utilization, compute_processes=processes,
                visibility_is_not_occupancy=True, unix=time.time()))
            assert time.time() < GATE_END
            execute("gpu", min(GPU_END, time.time()+1500))
        report = read(OUT/"report.json")
        assert report["status"] == "passed" and report["strict_parity"]
        assert report["calls"] == dict(eval_calls=1570, train_calls=0)
        assert report["checked_images"] == len(set(report["image_ids"])) == 264
        assert all(report["timing"][k]["repeats"] == 20 for k in ("reference", "passthrough"))
        assert not report["ap_reevaluated"] and not report["training_started"]
        assert all(sha(OUT/"source"/p) == h for p,h in plan["source_sha256"].items())
        save(OUT/"terminal_acceptance.json", dict(status="passed", report_sha256=sha(OUT/"report.json"), unix=time.time()))
        save(OUT/"status.json", dict(status="complete", phase="complete", unix=time.time()))
    except BaseException as exc:
        save(OUT/"first_error.json", dict(error=repr(exc), phase=phase, unix=time.time()))
        if name:
            try:
                item = json.loads(subprocess.check_output(["docker", "inspect", name], timeout=5))[0]
                if item["Config"]["Labels"].get("ai-course-directions-owner") == OWNER:
                    subprocess.run(["docker", "stop", "-t", "5", name], timeout=10, check=False)
            except Exception:
                pass
        save(OUT/"status.json", dict(status="failed", phase=phase, unix=time.time()))
        raise


if __name__ == "__main__":
    main()

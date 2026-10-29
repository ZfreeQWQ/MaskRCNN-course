"""Bounded CPU-only launcher; no GPU reservation or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path("/mnt/nfs/fenghaibin/AI_Class")
OUTPUT = ROOT / "outputs/course_m2f_geometry_20261020_v1"
START, DEADLINE = 1792469220, 1792472820
IMAGE = "sha256:eb5bed0d49efd109b91d5365131cc7f25391b8695347086d2e3dd12953d7702c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(commit):
    if not START <= time.time() < START + 600:
        raise TimeoutError("Preparation window closed; do not reset")
    if (OUTPUT / "plan.json").exists():
        raise ValueError("Existing task; do not duplicate")
    owner = uuid.uuid4().hex
    name = "ai-m2f-geometry-" + owner[:10]
    inside = "/workspace/outputs/" + OUTPUT.name
    names = ("m2f_cached_geometry_refine.py", "m2f_cached_rank_audit.py", "launch_m2f_cached_geometry.py")
    plan = dict(source_commit=commit, preparation_unix=START, deadline_unix=DEADLINE,
        cpu_limit=4, memory_limit_gib=8, gpu_devices=[], network="none", runtime="runc",
        owner=owner, container_name=name, image=IMAGE,
        source_sha256={n: digest(OUTPUT / "source" / n) for n in names},
        rule=dict(global_scores_order_count_fixed=True, global_small_max_area=1024,
            local_max_area=1024, minimum_mask_iou=.5, tie_break=["local_score", "original_index"],
            add_instances=False, remove_instances=False, retrain=False),
        gpu_calls=0, model_loads=0, parameter_search=False, test_inputs_used=False,
        main_baseline_AP_APs=[.715, .388], mrcnn_new_experiments_paused=True)
    command = ["docker", "run", "--rm", "--pull", "never", "--runtime", "runc",
        "--name", name, "--label", "ai-course-directions-owner=" + owner,
        "--network", "none", "--read-only", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--volume", str(ROOT) + ":/workspace:ro", "--volume", str(OUTPUT) + ":" + inside + ":rw",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--workdir", inside + "/source"]
    env = dict(CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void", PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1", PYTHONPATH=inside + "/source", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLCONFIGDIR="/tmp/matplotlib",
        XDG_CACHE_HOME="/tmp/cache")
    for k, v in env.items():
        command += ["--env", k + "=" + v]
    command += [IMAGE, "/workspace/envs/mmdet/bin/python", "m2f_cached_geometry_refine.py", "--output", inside]
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2))
    try:
        with (OUTPUT / "worker.log").open("x") as log:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1, DEADLINE-time.time()))
        if result.returncode:
            raise RuntimeError("CPU worker failed; inspect preserved first_error and log")
        report = json.loads((OUTPUT / "report.json").read_text())
        assert report["status"] == "passed" and report["deadline_unix"] == DEADLINE
        assert report["gpu_calls"] == report["model_loads"] == report["weight_updates"] == 0
        assert time.time() < DEADLINE
        print(json.dumps(dict(status="complete", metrics=report["metrics"], report_sha256=digest(OUTPUT/"report.json"))))
    except BaseException as exc:
        # Inspect first; never stop a different task or an unverified container.
        try:
            found = subprocess.run(["docker", "inspect", name], capture_output=True, text=True, timeout=5)
            if found.returncode == 0:
                c = json.loads(found.stdout)[0]
                if c["Config"]["Labels"].get("ai-course-directions-owner") == owner:
                    subprocess.run(["docker", "stop", "-t", "3", name], timeout=8, check=False)
        except Exception:
            pass
        (OUTPUT / "launcher_error.json").write_text(json.dumps(dict(error=repr(exc), time_unix=time.time()), indent=2))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source-commit", required=True)
    main(p.parse_args().source_commit)

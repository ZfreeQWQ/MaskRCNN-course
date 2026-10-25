"""Frozen validated d01@1280 test inference, two disjoint shards, no training."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from course.common import atomic_json
from course.submission import merge_shards


def main():
    root = Path(__file__).resolve().parent
    destination = root / "runs/submission_d01_1280_v1"
    destination.mkdir(exist_ok=False)
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    assert len(memory.splitlines()) >= 2 and all(int(v) < 256 for v in memory.splitlines()[:2]), "Do not compete for occupied GPUs"
    weights = root / "runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == "f33d2d37c422d6624d0e77bcb7843d4c7ca5af190241ecf1f02a24458ba8a11b"
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=root)
    source = destination / "source"
    source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source/n).resolve().is_relative_to(source) for n in z.namelist())
        z.extractall(source)
    (source / "data_cache").symlink_to(root / "data_cache", target_is_directory=True)
    metadata = Path("/workspace/blender/test_images.json")
    state = dict(status="running", source_commit=revision, controller_pid=os.getpid(),
                 archive_sha256=hashlib.sha256(archive).hexdigest(), training=False,
                 test_used="inference only; no test labels or tuning", selection="existing full-val d01@1280 APS0.5079599315",
                 config="configs/v4/h00_fast1280.yaml", started_unix=time.time(), shards={})
    atomic_json(destination / "status.json", state)
    processes = []
    logs = []
    try:
        for i in range(2):
            output = destination / f"part{i}.csv"
            log = (destination / f"part{i}.log").open("x")
            logs.append(log)
            command = [sys.executable, "-u", "export_submission.py", "--config", state["config"],
                       "--weights", str(weights), "--output", str(output), "--shards", "2", "--shard-index", str(i),
                       "--assume-constant-training-intrinsics"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            processes.append(subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT))
            state["shards"][str(i)] = dict(pid=processes[-1].pid, gpu=i)
        while True:
            for i, p in enumerate(processes):
                progress = destination / f"part{i}.status.json"
                if progress.exists():
                    s = json.loads(progress.read_text())
                    state["shards"][str(i)].update({k:s[k] for k in ("status", "processed", "total")})
                if p.poll() is not None and p.returncode:
                    raise RuntimeError(f"Shard{i} failed: {p.returncode}")
            atomic_json(destination / "status.json", state)
            if all(p.poll() is not None for p in processes):
                break
            time.sleep(10)
        for i in range(2):
            s = json.loads((destination/f"part{i}.status.json").read_text())
            assert s["status"] == "completed" and not s["smoke_only"] and s["processed"] == 1056
        state["status"] = "validating"
        atomic_json(destination / "status.json", state)
        validation = merge_shards([destination/f"part{i}.csv" for i in range(2)], metadata, destination/"submission.csv")
        final = destination/"submission.csv"
        state.update(status="completed", validation=validation, output=str(final), bytes=final.stat().st_size,
                     sha256=hashlib.sha256(final.read_bytes()).hexdigest(), finished_unix=time.time())
        atomic_json(destination / "status.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(destination / "status.json", state)
        # These exact children belong to this exporter, never touch other jobs.
        for p in processes:
            if p.poll() is None:
                p.terminate()
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()

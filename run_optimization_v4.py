"""Exactly two full-val inference comparisons, no training or automatic search."""
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
from course.common import atomic_json


def main():
    root = Path(__file__).resolve().parent
    suite = root / "runs/optimization_v4"
    suite.mkdir(parents=True, exist_ok=True)
    lock = (suite / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / "suite.json").exists():
        raise RuntimeError("V4 output exists: inspect, never overwrite or silently restart")
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=root)
    source = suite / "source"
    source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source / n).resolve().is_relative_to(source) for n in z.namelist())
        z.extractall(source)
    for name in ("weights", "data_cache"):
        (source / name).symlink_to(root/name, target_is_directory=True)
    archive_sha = hashlib.sha256(archive).hexdigest()
    atomic_json(source / ".source_revision.json", dict(commit=revision, archive_sha256=archive_sha,
                                                      immutable_snapshot=True, dirty=False))
    v2 = json.loads((root / "runs/optimization_v2/suite.json").read_text())
    v3 = json.loads((root / "runs/optimization_v3/suite.json").read_text())
    assert v2["status"] == v3["status"] == "completed"
    checkpoints = [v2["jobs"]["d01_geometry_multiscale"]["inference_export"],
                   v3["jobs"]["g02_geometry_detail"]["inference_export"]]
    jobs = {}
    for name, gpu, payload, size in zip(("h01_scale1536", "h02_detail1280"), ("0", "1"), checkpoints, (1536, 1280)):
        path = Path(payload["inference_weights"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["sha256"]
        jobs[name] = dict(config=f"configs/v4/{name}.yaml", gpu=gpu, weights=str(path),
                          weights_sha256=payload["sha256"], input_size=size, status="pending")
    state = dict(status="preflight", commit=revision, archive_sha256=archive_sha, source=str(source),
                 started_unix=time.time(), jobs=jobs, reference=v3["jobs"]["g01_scale1280"]["metrics"],
                 policy="Two fixed full2584-val inference jobs only; frozen weights; no training, test, TTA or extra jobs",
                 test_used=False, training=False)
    processes, logs = {}, []

    def update(**values):
        state.update(values)
        state["updated_unix"] = time.time()
        atomic_json(suite / "suite.json", state)
        ref = state["reference"]
        rows = ["# Optimization V4", "", "Source: `"+revision+"`", "",
                "All result rows are full2584 val. Weights remain unchanged; no new training.", "",
                "| Variant | State | Input | APS | AP | Delta APS vs best1280 | Model ms | Input+model ms |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                f"| V2 geometry1280 reference | completed | 1280 | {ref['mask_aps']:.6f} | {ref['mask_ap']:.6f} | 0 | {ref['inference_seconds_per_image']*1000:.2f} | {ref['model_plus_input_seconds_per_image']*1000:.2f} |"]
        for name, job in jobs.items():
            m = job.get("metrics")
            if m:
                rows.append(f"| {name} | completed | {job['input_size']} | {m['mask_aps']:.6f} | {m['mask_ap']:.6f} | {m['mask_aps']-ref['mask_aps']:+.6f} | {m['inference_seconds_per_image']*1000:.2f} | {m['model_plus_input_seconds_per_image']*1000:.2f} |")
            else:
                rows.append(f"| {name} | {job['status']} | {job['input_size']} | — | — | — | — | — |")
        (suite / "RESULTS.md").write_text("\n".join(rows)+"\n")

    def env(gpu):
        return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def run(command, log, gpu):
        with (suite/log).open("a") as stream:
            subprocess.run([sys.executable]+command, cwd=source, env=env(gpu), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    try:
        update(active="cpu_tests")
        run(["-m", "unittest", "test_rgbd_data", "test_experiments", "test_geometry_v2", "test_roi_v2",
             "test_v3_data", "test_v3_refinement", "test_v4_data", "-v"], "preflight_cpu.log", "")
        update(active="paired_input_benchmark_and_gpu_parity")
        run(["-u", "benchmark_v4_input.py"], "input_benchmark.log", "0")
        audit = json.loads((suite / "input_benchmark.json").read_text())
        assert audit["status"] == "passed" and len(audit["gpu_parity"]) == 4
        state["input_benchmark"] = audit
        for name, job in jobs.items():
            update(active="smoke_"+name)
            run(["-u", "evaluate_rgbd.py", "--config", job["config"], "--weights", job["weights"],
                 "--strict-load", "--samples", "2", "--output", str(suite / ("smoke_"+name+".json"))],
                "smoke_"+name+".log", job["gpu"])
            smoke = json.loads((suite / ("smoke_"+name+".json")).read_text())
            assert smoke["validation_images"] == 2 and smoke["input_size"] == job["input_size"]
        if (suite / "STOP_AFTER_CURRENT").exists():
            update(status="stopped_before_formal", active=None)
            return
        update(status="running", active="two_full_val_inference_jobs")
        for name, job in jobs.items():
            log = (suite / (name+".log")).open("a")
            logs.append(log)
            process = subprocess.Popen([sys.executable, "-u", "evaluate_rgbd.py", "--config", job["config"],
                           "--weights", job["weights"], "--strict-load", "--samples", "0",
                           "--output", str(suite / (name+".json"))], cwd=source, env=env(job["gpu"]),
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = process
            job.update(status="running", pid=process.pid, started_unix=time.time())
            update()
        finished = set()
        while len(finished) != len(jobs):
            for name, process in processes.items():
                if name in finished or process.poll() is None:
                    continue
                if process.returncode:
                    raise RuntimeError(name+" failed with code "+str(process.returncode))
                result = json.loads((suite / (name+".json")).read_text())
                assert result["validation_images"] == 2584 and result["mask_size"] == [1024, 1024]
                assert result["input_size"] == jobs[name]["input_size"]
                jobs[name].update(status="completed", metrics=result, finished_unix=time.time())
                finished.add(name)
                update()
            if len(finished) < len(jobs):
                time.sleep(20)
        update(status="completed", active=None, finished_unix=time.time())
    except BaseException as error:
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
        update(status="failed", active=None, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()

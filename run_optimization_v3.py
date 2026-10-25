"""Bounded V3: full-val1280 on best V2 + one fresh-COCO detail30k run.

All work uses one committed archive. Evaluation is serialized on GPU1;
training runs on GPU0. No search, combinations, test access or automatic retry.
"""
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
    suite = root / "runs/optimization_v3"
    suite.mkdir(parents=True, exist_ok=True)
    lock = (suite / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / "suite.json").exists():
        raise RuntimeError("Existing V3 queue; inspect rather than overwrite/retrain")
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=root)
    source = suite / "source"
    source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source / n).resolve().is_relative_to(source) for n in z.namelist())
        z.extractall(source)
    for name in ("weights", "data_cache"):
        (source / name).symlink_to(root / name, target_is_directory=True)
    atomic_json(source / ".source_revision.json", dict(commit=revision, dirty=False,
                immutable_snapshot=True, archive_sha256=hashlib.sha256(archive).hexdigest()))
    v2 = json.loads((root / "runs/optimization_v2/suite.json").read_text())
    best = v2["jobs"]["d01_geometry_multiscale"]
    weight = Path(best["inference_export"]["inference_weights"])
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == best["inference_export"]["sha256"]
    diagnostic = json.loads((root / "runs/optimization_v3_diagnostics/status.json").read_text())
    assert diagnostic["status"] == "completed" and diagnostic["fixed_samples"] == 400
    state = dict(status="preflight", commit=revision, source=str(source), started_unix=time.time(), jobs={},
                 baseline=best["metrics"], diagnostic_source=diagnostic["commit"],
                 policy="Full-val1280 unchanged V2 weights + one fresh-COCO detail30k/infer1024; stop after both",
                 test_used=False, train_val_never_merged=True)
    processes, handles = {}, []

    def update(**values):
        state.update(values)
        state["updated_unix"] = time.time()
        atomic_json(suite / "suite.json", state)
        base = state["baseline"]
        lines = ["# Optimization V3", "", "Source: `"+revision+"`", "",
                 "Full2584 val only; smoke/val400 are excluded from final result rows.", "",
                 "| Variant | State | Input | APS | AP | Delta APS vs V2 | Model ms | Input+model ms |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- |",
                 f"| V2 geometry best | completed | 1024 | {base['mask_aps']:.6f} | {base['mask_ap']:.6f} | 0 | {base['inference_seconds_per_image']*1000:.2f} | {base['model_plus_input_seconds_per_image']*1000:.2f} |"]
        for name, size in (("g01_scale1280", 1280), ("g02_geometry_detail", 1024)):
            job = state["jobs"].get(name, {})
            m = job.get("metrics")
            if m:
                lines.append(f"| {name} | completed | {size} | {m['mask_aps']:.6f} | {m['mask_ap']:.6f} | {m['mask_aps']-base['mask_aps']:+.6f} | {m['inference_seconds_per_image']*1000:.2f} | {m['model_plus_input_seconds_per_image']*1000:.2f} |")
            else:
                lines.append(f"| {name} | {job.get('status', 'pending')} | {size} | — | — | — | — | — |")
        (suite / "RESULTS.md").write_text("\n".join(lines)+"\n")

    def env(gpu):
        return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def run(command, log_name, gpu="0"):
        with (suite / log_name).open("a") as stream:
            subprocess.run([sys.executable]+command, cwd=source, env=env(gpu), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    def launch(name, command, log_name, gpu):
        log = (suite / log_name).open("a")
        handles.append(log)
        process = subprocess.Popen([sys.executable]+command, cwd=source, env=env(gpu),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = process
        state["jobs"][name] = dict(status="running", pid=process.pid, started_unix=time.time())
        update()

    try:
        update(active="cpu_tests")
        run(["-m", "unittest", "test_rgbd_data", "test_experiments", "test_geometry_v2", "test_roi_v2",
             "test_v3_data", "test_v3_refinement", "-v"], "preflight_cpu.log", "")
        update(active="initial_gpu_parity")
        run(["-u", "audit_v3_initial_gpu.py"], "initial_gpu_parity.log")
        assert json.loads((suite / "initial_gpu_parity.json").read_text())["status"] == "passed"
        smoke = suite / "smoke/g02_geometry_detail"
        update(active="smoke_g02")
        run(["-u", "train_rgbd.py", "--config", "configs/v3/g02_geometry_detail.yaml", "--output", str(smoke),
             "--max-iter", "10", "--smoke-test", "--eval-gpu", "1"], "smoke_g02.log")
        assert json.loads((smoke / "status.json").read_text())["status"] == "smoke_test_passed"
        if (suite / "STOP_AFTER_CURRENT").exists():
            update(status="stopped_before_formal", active=None)
            return
        scale_output = suite / "g01_scale1280_full.json"
        detail = suite / "g02_geometry_detail"
        update(status="running", active="full1280_and_detail_training")
        launch("g01_scale1280", ["-u", "evaluate_rgbd.py", "--config", "configs/v3/g01_scale1280.yaml",
                "--weights", str(weight), "--strict-load", "--samples", "0", "--output", str(scale_output)],
               "g01_scale1280_full.log", "1")
        launch("g02_geometry_detail", ["-u", "train_rgbd.py", "--config", "configs/v3/g02_geometry_detail.yaml",
                "--output", str(detail), "--max-iter", "30000", "--eval-gpu", "1"], "g02_geometry_detail.log", "0")
        completed = set()
        while len(completed) < len(processes):
            for name, process in processes.items():
                if name in completed or process.poll() is None:
                    continue
                if process.returncode:
                    raise RuntimeError(name+" failed with exit "+str(process.returncode))
                if name == "g01_scale1280":
                    metrics = json.loads(scale_output.read_text())
                    assert metrics["validation_images"] == 2584 and metrics["input_size"] == 1280
                    state["jobs"][name].update(status="completed", metrics=metrics, weights_unchanged=True,
                                              checkpoint_sha256=best["inference_export"]["sha256"])
                else:
                    result = json.loads((detail / "status.json").read_text())
                    assert result["status"] == "completed" and result["step"] == 30000
                    metrics = result["final_metrics"]
                    assert metrics["validation_images"] == 2584 and metrics["input_size"] == 1024
                    destination = detail / "model_final_inference.pth"
                    run(["export_inference_weights.py", "--config", "configs/v3/g02_geometry_detail.yaml",
                         "--weights", str(detail / "model_final.pth"), "--output", str(destination)], "g02_export.log", "")
                    state["jobs"][name].update(status="completed", metrics=metrics,
                                              inference_export=json.loads(destination.with_suffix(".json").read_text()))
                state["jobs"][name]["finished_unix"] = time.time()
                completed.add(name)
                update()
            if len(completed) < len(processes):
                time.sleep(20)
        update(status="completed", active=None, finished_unix=time.time())
    except BaseException as error:
        # Only terminate process groups created and owned by THIS controller.
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        update(status="failed", active=None, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for log in handles:
            log.close()


if __name__ == "__main__":
    main()

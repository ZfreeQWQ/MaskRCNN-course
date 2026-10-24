"""Bounded two-run V2 queue; immutable source, no tuning or automatic expansion.

GPU0 trains; GPU1 evaluates. Both runs start from the permitted COCO weights,
not an already fine-tuned checkpoint. V1 stop controls remain untouched.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from course.common import atomic_json


NAMES = ["d01_geometry_multiscale", "d02_roi_supervision"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="runs/optimization_v2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    suite = (root / args.suite).resolve()
    assert suite.is_relative_to(root / "runs")
    suite.mkdir(parents=True, exist_ok=True)
    lock = (suite / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = suite / "suite.json"
    if manifest.exists():
        state = json.loads(manifest.read_text())
        source = Path(state["source"])
        assert state["order"] == NAMES
    else:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        if dirty:
            raise RuntimeError("Commit audited changes before snapshot: " + dirty)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        archive = subprocess.check_output(["git", "archive", "--format=zip", commit], cwd=root)
        source = suite / "source"
        source.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            assert all((source / n).resolve().is_relative_to(source) for n in z.namelist())
            z.extractall(source)
        for name in ("weights", "data_cache"):
            (source / name).symlink_to(root / name, target_is_directory=True)
        atomic_json(source / ".source_revision.json", dict(commit=commit, dirty=False,
                    archive_sha256=hashlib.sha256(archive).hexdigest(), immutable_snapshot=True))
        state = dict(status="preflight", commit=commit, source=str(source), order=NAMES,
                     baseline=json.loads((root / "reports/baseline_frozen.json").read_text()),
                     started_unix=time.time(), jobs={},
                     policy="two independent 30k final/full-val runs; no high-resolution, TTA or combinations")

    def update(**changes):
        state.update(changes)
        state["updated_unix"] = time.time()
        atomic_json(manifest, state)
        baseline = state["baseline"]["metrics"]
        rows = ["# Optimization V2", "", "Source: `" + state["commit"] + "`", "",
                "Final 30k checkpoints / full 2584-image val. Smoke scores are NOT results.", "",
                "| Experiment | State | APS | AP | Delta APS | Model ms | Input+model ms |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                f"| frozen baseline | completed | {baseline['mask_aps']:.6f} | {baseline['mask_ap']:.6f} | 0 | {baseline['inference_seconds_per_image']*1000:.2f} | not recorded |"]
        for name in NAMES:
            job = state["jobs"].get(name, {})
            m = job.get("metrics")
            if m:
                rows.append(f"| {name} | completed | {m['mask_aps']:.6f} | {m['mask_ap']:.6f} | {m['mask_aps']-baseline['mask_aps']:+.6f} | {m['inference_seconds_per_image']*1000:.2f} | {m['model_plus_input_seconds_per_image']*1000:.2f} |")
            else:
                rows.append(f"| {name} | {job.get('status', 'pending')} | — | — | — | — | — |")
        (suite / "RESULTS.md").write_text("\n".join(rows)+"\n")

    def command_run(command, log, gpu="0"):
        with log.open("a") as stream:
            subprocess.run(command, cwd=source, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        update(status="preflight", active="cpu_tests")
        command_run([sys.executable, "-m", "unittest", "test_rgbd_data", "test_experiments",
                     "test_geometry_v2", "test_roi_v2", "-v"], suite / "preflight_cpu.log", "")
        for name in NAMES:
            smoke = suite / "smoke" / name
            status_path = smoke / "status.json"
            result = json.loads(status_path.read_text()) if status_path.exists() else {}
            if result.get("status") == "smoke_test_passed":
                continue
            if result:
                raise RuntimeError("Inspect failed/incomplete smoke; never overwrite: " + str(smoke))
            update(active="smoke_"+name)
            command_run([sys.executable, "-u", "train_rgbd.py", "--config", f"configs/v2/{name}.yaml",
                         "--output", str(smoke), "--max-iter", "10", "--smoke-test", "--eval-gpu", "1"],
                        suite / ("smoke_"+name+".log"))
            assert json.loads(status_path.read_text())["status"] == "smoke_test_passed"
        for name in NAMES:
            # Optional stop file is checked before every new formal job.
            if (suite / "STOP_AFTER_CURRENT").exists():
                update(status="stopped_after_current", active=None)
                return
            output = suite / name
            status_path = output / "status.json"
            result = json.loads(status_path.read_text()) if status_path.exists() else {}
            if result and result.get("status") != "completed":
                raise RuntimeError("Existing incomplete run requires manual inspection: " + str(output))
            if not result:
                state["jobs"][name] = dict(status="training", output=str(output), started_unix=time.time())
                update(status="running", active=name)
                command_run([sys.executable, "-u", "train_rgbd.py", "--config", f"configs/v2/{name}.yaml",
                             "--output", str(output), "--max-iter", "30000", "--eval-gpu", "1"],
                            suite / (name+".log"))
                result = json.loads(status_path.read_text())
            assert result["status"] == "completed" and result["step"] == 30000
            assert result["final_metrics"]["validation_images"] == 2584
            inference = output / "model_final_inference.pth"
            if not inference.exists():
                command_run([sys.executable, "export_inference_weights.py", "--config", f"configs/v2/{name}.yaml",
                             "--weights", str(output/"model_final.pth"), "--output", str(inference)],
                            suite / (name+"_export.log"), "")
            state["jobs"][name] = dict(status="completed", output=str(output), metrics=result["final_metrics"],
                                      inference_export=json.loads(inference.with_suffix(".json").read_text()),
                                      finished_unix=time.time())
            update()
        update(status="completed", active=None, finished_unix=time.time())
    except BaseException as error:
        update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

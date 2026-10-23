"""Sequential fixed-budget ablations from an immutable committed source archive.

One GPU trains, one evaluates. Stop on any failed preflight/train/evaluation.
No automatic tuning, best-checkpoint picking, val training, or test access.
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


NAMES = ["e01_small_resample", "e02_small_crop", "e03_rgbd_copy_paste", "e04_small_loss",
         "e05_foreground_points", "e06_depth_p2_gate", "e07_resolution1280"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="runs/ablations_v1")
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
    else:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        if dirty:
            raise RuntimeError("Commit source changes before running a versioned suite: " + dirty)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        archive = subprocess.check_output(["git", "archive", "--format=zip", commit], cwd=root)
        source = suite / "source"
        source.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for name in z.namelist():
                assert (source / name).resolve().is_relative_to(source)
            z.extractall(source)
        for name in ("weights", "data_cache"):
            (source / name).symlink_to(root / name, target_is_directory=True)
        atomic_json(source / ".source_revision.json", dict(commit=commit, dirty=False,
                    archive_sha256=hashlib.sha256(archive).hexdigest(), immutable_snapshot=True))
        state = dict(status="preflight", commit=commit, source=str(source), order=NAMES,
                     baseline=json.loads((root / "reports/baseline_frozen.json").read_text()),
                     started_unix=time.time(), jobs={}, policy="30k final checkpoints, full val, no combinations")
        atomic_json(manifest, state)

    def update(**values):
        state.update(values)
        state["updated_unix"] = time.time()
        atomic_json(manifest, state)
        # Lightweight report can be read while the queue is running.
        rows = ["# Single-factor ablations", "", "Source commit: `" + state["commit"] + "`", "",
                "Final checkpoints / full val only. Scores are 0–1. Smoke results are excluded.", "",
                "| Experiment | State | APS | AP | Delta APS | ms/image |", "| --- | --- | --- | --- | --- | --- |"]
        base = state["baseline"]["metrics"]
        rows.append(f"| baseline | completed | {base['mask_aps']:.6f} | {base['mask_ap']:.6f} | 0 | {base['inference_seconds_per_image']*1000:.2f} |")
        for name in NAMES:
            job = state["jobs"].get(name, {})
            m = job.get("metrics")
            if m:
                rows.append(f"| {name} | completed | {m['mask_aps']:.6f} | {m['mask_ap']:.6f} | {m['mask_aps']-base['mask_aps']:+.6f} | {m['inference_seconds_per_image']*1000:.2f} |")
            else:
                rows.append(f"| {name} | {job.get('status', 'pending')} | — | — | — | — |")
        (suite / "RESULTS.md").write_text("\n".join(rows) + "\n")

    def command_run(command, log, gpu="0"):
        with log.open("a") as stream:
            subprocess.run(command, cwd=source, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        update(status="preflight", active="cpu_tests")
        command_run([sys.executable, "-m", "unittest", "test_rgbd_data", "test_experiments", "-v"], suite / "preflight_cpu.log", "")
        # Test every implementation before committing many hours of GPU time.
        for name in NAMES:
            smoke = suite / "smoke" / name
            status = json.loads((smoke / "status.json").read_text()) if (smoke / "status.json").exists() else {}
            if status.get("status") == "smoke_test_passed":
                continue
            if status:
                raise RuntimeError("Failed/incomplete smoke needs inspection; will not overwrite: " + str(smoke))
            update(status="preflight", active=name)
            command_run([sys.executable, "-u", "train_rgbd.py", "--config", f"configs/experiments/{name}.yaml",
                         "--output", str(smoke), "--max-iter", "10", "--smoke-test", "--eval-gpu", "1"],
                        suite / ("smoke_" + name + ".log"))
            assert json.loads((smoke / "status.json").read_text())["status"] == "smoke_test_passed"
        for name in NAMES:
            run = suite / name
            control_path = suite / "queue_control.json"
            if control_path.exists():
                control = json.loads(control_path.read_text())
                stop_after = control["stop_after"]
                assert stop_after in NAMES
                if NAMES.index(name) > NAMES.index(stop_after):
                    for deferred in NAMES[NAMES.index(name):]:
                        state["jobs"].setdefault(deferred, dict(status="deferred_by_user"))
                    update(status="stopped_by_user", active=None, stop_after=stop_after,
                           finished_unix=time.time())
                    return
            result = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
            if result and result.get("status") != "completed":
                raise RuntimeError("Existing interrupted experiment needs manual inspection: " + str(run))
            if not result:
                state["jobs"][name] = dict(status="training", started_unix=time.time(), output=str(run))
                update(status="running", active=name)
                command_run([sys.executable, "-u", "train_rgbd.py", "--config", f"configs/experiments/{name}.yaml",
                             "--output", str(run), "--max-iter", "30000", "--eval-gpu", "1"], suite / (name + ".log"))
                result = json.loads((run / "status.json").read_text())
            assert result["status"] == "completed" and result["step"] == 30000
            metrics = result["final_metrics"]
            assert metrics["validation_images"] == 2584
            inference = run / "model_final_inference.pth"
            if not inference.exists():
                command_run([sys.executable, "export_inference_weights.py", "--config", f"configs/experiments/{name}.yaml",
                             "--weights", str(run / "model_final.pth"), "--output", str(inference)], suite / (name + "_export.log"), "")
            state["jobs"][name] = dict(status="completed", metrics=metrics, output=str(run),
                                      inference_export=json.loads(inference.with_suffix(".json").read_text()),
                                      finished_unix=time.time())
            update()
        update(status="completed", active=None, finished_unix=time.time())
    except BaseException as error:
        update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

"""Bounded two-arm screening, 3000 extra updates each; never run full val/test."""
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
from course.screen_data import make_panels


def main():
    root = Path(__file__).resolve().parent
    suite = root / "runs/screen_v1"
    suite.mkdir(parents=True, exist_ok=True)
    lock = (suite / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (suite / "suite.json").exists():
        raise RuntimeError("Existing suite: never overwrite")
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
    atomic_json(source / ".source_revision.json", dict(commit=revision, dirty=False, immutable_snapshot=True,
                archive_sha256=hashlib.sha256(archive).hexdigest()))
    index = json.loads((root / "data_cache/index.json").read_text())
    coco = json.loads((root / "data_cache/coco_val.json").read_text())
    screen, confirmation = make_panels(index, coco)
    atomic_json(suite / "panel_screen.json", screen)
    atomic_json(suite / "panel_confirmation.json", confirmation)
    # Confirmation panel is held out of this queue; use only for shortlisted
    # directions in a later explicit experiment, never optimize on both panels.
    weights = root / "runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth"
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert sha == "f33d2d37c422d6624d0e77bcb7843d4c7ca5af190241ecf1f02a24458ba8a11b"
    jobs = {name: dict(gpu=str(gpu), config=f"configs/screen/{name}.yaml", status="pending")
            for gpu, name in enumerate(("i00_control", "i01_early_dilate"))}
    state = dict(status="preflight", commit=revision, source=str(source), weights_sha256=sha,
                 jobs=jobs, started_unix=time.time(), controller_pid=os.getpid(),
                 policy="Exactly two matched +3000-step task-checkpoint continuations; fixed200 val at 0/1000/2000/3000; NO full val or test",
                 screen=screen, confirmation_used=False)
    processes, logs = {}, []

    def update(**values):
        state.update(values); state["updated_unix"] = time.time()
        atomic_json(suite / "suite.json", state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        update()
        env_base = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        for name, job in jobs.items():
            log = (suite / (name + ".log")).open("a"); logs.append(log)
            p = subprocess.Popen([sys.executable, "-u", "train_screen.py", "--config", job["config"],
                       "--weights", str(weights), "--panel", str(suite / "panel_screen.json"),
                       "--output", str(suite / name)], cwd=source,
                       env=dict(env_base, CUDA_VISIBLE_DEVICES=job["gpu"]),
                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes[name] = p
            job.update(status="running", pid=p.pid)
        update(status="running")
        while True:
            for name, p in processes.items():
                progress = suite / name / "status.json"
                if progress.exists():
                    s = json.loads(progress.read_text())
                    jobs[name].update(status=s["status"], step=s["step"],
                        screen_metrics=[{k:r.get(k) for k in ("additional_training_step", "mask_aps", "mask_ap", "mask_aps50", "mask_aps75", "mask_ars")}
                                        for r in s["validation_results"]])
                if p.poll() is not None and p.returncode:
                    raise RuntimeError(f"{name} failed with code {p.returncode}; inspect its log")
            update()
            if all(p.poll() is not None for p in processes.values()):
                assert all(j["status"] == "completed" for j in jobs.values())
                control = jobs["i00_control"]["screen_metrics"][-1]
                variant = jobs["i01_early_dilate"]["screen_metrics"][-1]
                update(status="completed", finished_unix=time.time(),
                       paired_final_aps_delta=variant["mask_aps"] - control["mask_aps"],
                       next_action="Review screen and error groups; no automatic extension or full val")
                break
            time.sleep(15)
    except BaseException as error:
        # Each trainer owns its eval subprocess and dataloader workers in the
        # same session, so terminate the exact job process groups together.
        for p in processes.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes.values():
            if p.poll() is None:
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
        update(status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
               error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for log in logs: log.close()


if __name__ == "__main__":
    main()

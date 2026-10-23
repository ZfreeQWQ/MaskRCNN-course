"""Adopt a live trainer and stop after its final evaluation; NEVER start training.

Used to replace the old dispatch-only supervisor without interrupting its child.
The committed training snapshot, optimizer, checkpoints and evaluator are untouched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from course.common import atomic_json


def trainer_identity(pid, run):
    proc = Path("/proc") / str(pid)
    try:
        args = (proc / "cmdline").read_bytes().split(b"\0")
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if (b"train_rgbd.py" not in args or b"--output" not in args or
                args[args.index(b"--output")+1] != str(run).encode() or stat[0] == "Z"):
            return None
        return stat[19]  # Linux /proc stat starttime (field 22).
    except (OSError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="runs/ablations_v1")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trainer-pid", required=True, type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    suite = (root / args.suite).resolve()
    assert suite.is_relative_to(root / "runs")
    state = json.loads((suite / "suite.json").read_text())
    assert args.experiment in state["order"]
    assert state["active"] == args.experiment
    run = suite / args.experiment
    birth = trainer_identity(args.trainer_pid, run)
    assert birth is not None, "Specified PID is not this experiment's live trainer"
    deferred = state["order"][state["order"].index(args.experiment)+1:]
    for name in deferred:
        assert not (suite / name / "status.json").exists(), "A later formal run already started"
    if args.check_only:
        print("ADOPTION_CHECK_PASSED", args.trainer_pid, args.experiment, "DEFER", deferred, flush=True)
        return
    lock = (suite / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control = dict(stop_after=args.experiment, reason="Explicit user request: finish sixth experiment, then stop",
                   requested_unix=time.time(), adopted_trainer_pid=args.trainer_pid,
                   trainer_starttime=birth, controller_pid=os.getpid())
    atomic_json(suite / "queue_control.json", control)
    state.update(status="running", stop_after=args.experiment, control=control)
    for name in deferred:
        state["jobs"][name] = dict(status="deferred_by_user")
    atomic_json(suite / "suite.json", state)
    print("ADOPTED", json.dumps(control), flush=True)

    def cpu_command(command):
        subprocess.run(command, cwd=Path(state["source"]), check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1"))

    try:
        while True:
            current = json.loads((run / "status.json").read_text())
            if current["status"] == "completed":
                break
            if current["status"] == "failed":
                raise RuntimeError(current.get("error", "Trainer failed"))
            if trainer_identity(args.trainer_pid, run) != birth:
                # Re-read once in case the trainer published completion just as
                # the process exited between the first read and identity check.
                if json.loads((run / "status.json").read_text())["status"] == "completed":
                    continue
                raise RuntimeError("Adopted trainer exited without completed final evaluation")
            time.sleep(10)
        assert current["step"] == 30000 and current["final_metrics"]["validation_images"] == 2584
        inference = run / "model_final_inference.pth"
        if not inference.exists():
            cpu_command([sys.executable, "export_inference_weights.py", "--config",
                         f"configs/experiments/{args.experiment}.yaml", "--weights", str(run / "model_final.pth"),
                         "--output", str(inference)])
        state["jobs"][args.experiment] = dict(status="completed", metrics=current["final_metrics"],
            output=str(run), inference_export=json.loads(inference.with_suffix(".json").read_text()),
            finished_unix=time.time())
        state.update(status="stopped_by_user", active=None, finished_unix=time.time(), updated_unix=time.time())
        atomic_json(suite / "suite.json", state)
        subprocess.run([sys.executable, "collect_ablation_results.py"], cwd=root, check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1"))
        print("STOPPED_AFTER_REQUESTED_EXPERIMENT", args.experiment, "NO_FURTHER_TRAINING", flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc(), updated_unix=time.time())
        atomic_json(suite / "suite.json", state)
        raise


if __name__ == "__main__":
    main()

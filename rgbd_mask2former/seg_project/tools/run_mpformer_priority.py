#!/usr/bin/env python
"""Validate, smoke-test and launch the algorithm-priority D+MP run persistently."""
import datetime
import fcntl
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/mask2former"
OUTPUT = EXPERIMENTS / "round2_mpformer_priority_20261026"
SMOKE = EXPERIMENTS / "round2_mpformer_smoke_20261026"
TRAIN = EXPERIMENTS / "round2_mpformer_20261026"
SOURCES = ["rgbd_seg/m2f/mpformer.py", "rgbd_seg/m2f/mpformer_trainer.py",
           "rgbd_seg/m2f/round2_inference.py", "rgbd_seg/m2f/round2_trainer.py",
           "rgbd_seg/m2f/trainer.py", "rgbd_seg/m2f/model.py", "rgbd_seg/m2f/mapper.py",
           "rgbd_seg/m2f/config.py", "rgbd_seg/m2f/checkpoint.py",
           "tools/train_m2f.py", "tools/train_mpformer.py", "tools/verify_mpformer_inference.py",
           "configs/mask2former/round2_mpformer_20261026.yaml",
           "configs/mask2former/round2_d_control_20261026.yaml"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(state):
    state["updated_utc"] = now()
    tmp = OUTPUT / "state.tmp"
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False))
    tmp.replace(OUTPUT / "state.json")


def run_stage(state, stage, command):
    if {p: sha(ROOT / p) for p in SOURCES} != state["source_sha256"]:
        raise RuntimeError("Controlled training source changed")
    log = OUTPUT / (stage + ".log")
    job = {"stage": stage, "status": "running", "started_utc": now(), "command": command, "log": str(log)}
    state["jobs"].append(job)
    state["stage"] = stage
    write(state)
    with log.open("x") as stream:
        proc = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT)
        job["pid"] = proc.pid
        write(state)
        print(now(), stage, "pid", proc.pid, flush=True)
        code = proc.wait()
    job["exit_code"], job["ended_utc"] = code, now()
    (OUTPUT / (stage + ".exit_code")).write_text(str(code) + "\n")
    if code:
        job["status"] = "failed"
        write(state)
        raise RuntimeError(stage + " exited with code " + str(code))
    job["status"] = "completed"
    write(state)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "manager.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "state.json").exists() or SMOKE.exists() or TRAIN.exists():
            raise FileExistsError("Existing MP run; inspect rather than overwrite")
        deferred = json.loads((EXPERIMENTS / "round2_dh_control_20261026/state.json").read_text())
        if deferred["status"] != "deferred_by_user" or not Path(deferred["deferred_checkpoint"]).is_file():
            raise RuntimeError("D+H has not stopped with a preserved checkpoint")
        checks = EXPERIMENTS / "round2_mpformer_checks_20261026"
        if int((checks / "ddp.exit_code").read_text()) != 0 or not json.loads((checks / "ddp_empty_targets.json").read_text())["valid"]:
            raise RuntimeError("Four-GPU unequal/empty-target checks have not passed")
        weights = EXPERIMENTS / "d_continue6_results_20261026/best_segm_APs.pth"
        state = {"status": "running", "stage": "initial_inference_audit", "pid": os.getpid(),
                 "started_utc": now(), "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
                 "weights": str(weights), "weights_sha256": sha(weights),
                 "baseline_full_val_APs": 52.91622131864694,
                 "algorithm": "RGB-D Mask2Former + all-layer mask-piloted training",
                 "recipe": {"groups": 1, "mask_noise": 0, "label_noise": 0,
                            "ordinary_query_positions": "preserved from existing D",
                            "pilot_attention": "area-pooled GT support, every cross-attention layer",
                            "ordinary_reads_pilots": False, "inference_pilots": False,
                            "S": False, "H": False},
                 "training": {"gpus": 4, "batch": 8, "iterations": 3608, "epochs": 2,
                              "initial_base_lr": 1e-5, "new_embedding_lr": 1e-4,
                              "train_size": 1024, "val_size": 1536, "eval_period": 1804,
                              "seed": 20261020, "optimizer": "fresh AdamW, weights-only initialization"},
                 "ordinary_D_control_deferred": True, "jobs": []}
        write(state)
        try:
            run_stage(state, "initial_inference_audit", [str(ROOT / "run.sh"), "python",
                                                        str(ROOT / "tools/verify_mpformer_inference.py")])
            if not json.loads((checks / "initial_inference_parity.json").read_text())["valid"]:
                raise RuntimeError("Initial inference parity audit failed")
            command = [str(ROOT / "run.sh"), "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                       str(ROOT / "tools/train_mpformer.py"), "--config-file",
                       str(ROOT / "configs/mask2former/round2_mpformer_20261026.yaml")]
            smoke = command + ["--eval-limit", "16", "SOLVER.MAX_ITER", "60", "SOLVER.STEPS", "(48,56)",
                               "SOLVER.WARMUP_ITERS", "5", "SOLVER.CHECKPOINT_PERIOD", "60",
                               "TEST.EVAL_PERIOD", "60", "OUTPUT_DIR", str(SMOKE)]
            run_stage(state, "four_gpu_smoke", smoke)
            rows = [json.loads(line) for line in (SMOKE / "mp_diagnostics.jsonl").read_text().splitlines()]
            if not rows or rows[-1]["completed_iterations"] != 60:
                raise RuntimeError("Incomplete MP smoke diagnostics")
            records = [rank for row in rows for rank in row["ranks"]]
            if not all(rank["enabled"] and rank["pilot_queries"] > 0 for rank in records):
                raise RuntimeError("Mask-piloted training branch was inactive")
            if not any(rank["label_gradient_norm"] is not None and math.isfinite(rank["label_gradient_norm"])
                       and rank["label_gradient_norm"] > 0 for rank in records):
                raise RuntimeError("New label embedding received no finite gradient")
            validation = json.loads((SMOKE / "inference_subset/round2_diagnostics.json").read_text())
            if validation["images"] != 16 or validation["refined_queries"] != 0:
                raise RuntimeError("Unexpected smoke validation geometry or refinement")
            if (SMOKE / "best_metrics.json").exists():
                raise RuntimeError("Subset validation must not select a best checkpoint")
            state["smoke_passed"] = True
            write(state)
            run_stage(state, "algorithm_training", command)
            state["best"] = json.loads((TRAIN / "best_metrics.json").read_text())
            validation = json.loads((TRAIN / "inference/round2_diagnostics.json").read_text())
            if validation["images"] != 2584:
                raise RuntimeError("Incomplete final full validation")
            state["status"], state["stage"] = "completed", "completed"
            state["completed_utc"] = now()
            write(state)
        except Exception as error:
            state["status"], state["error"] = "failed", repr(error)
            write(state)
            raise


if __name__ == "__main__":
    main()

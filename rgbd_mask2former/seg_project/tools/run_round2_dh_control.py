#!/usr/bin/env python
"""Persistently run two matched ablations, stopping on any failed stage."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/mask2former"
OUTPUT = EXPERIMENTS / "round2_dh_control_20261026"
SMOKE = EXPERIMENTS / "round2_dh_smoke_v3_20261026"
SOURCE_PATHS = [
    "rgbd_seg/m2f/model.py", "rgbd_seg/m2f/criterion.py", "rgbd_seg/m2f/hires_mask.py",
    "rgbd_seg/m2f/trainer.py", "rgbd_seg/m2f/config.py", "rgbd_seg/m2f/mapper.py",
    "rgbd_seg/m2f/experimental_inference.py", "rgbd_seg/m2f/round2_inference.py",
    "rgbd_seg/m2f/round2_trainer.py", "tools/train_m2f.py", "tools/train_round2.py",
    "configs/mask2former/round2_d_control_20261026.yaml",
    "configs/mask2former/round2_dh_20261026.yaml",
]


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


def verify_smoke():
    code = int(SMOKE.with_suffix(".exit_code").read_text().strip())
    diagnostics = json.loads((SMOKE / "inference_subset/round2_diagnostics.json").read_text())
    if code != 0 or diagnostics["images"] != 16 or diagnostics["refined_queries"] <= 0 or diagnostics["changed_pixels"] <= 0:
        raise RuntimeError("Four-GPU paired-validation smoke did not pass")
    if not any((r["head_residual_norm"] or 0) > 0 for r in diagnostics["records"]):
        raise RuntimeError("The refinement head remained zero-initialized")
    if not (SMOKE / "model_final.pth").exists():
        raise RuntimeError("Smoke final checkpoint is missing")
    audit = json.loads((SMOKE / "independent_inference_audit.json").read_text())
    if not audit["valid"]:
        raise RuntimeError("Independent inference parity audit failed")
    control = EXPERIMENTS / "round2_d_control_smoke_20261026"
    if int(control.with_suffix(".exit_code").read_text().strip()) != 0:
        raise RuntimeError("Ordinary D four-GPU smoke failed")
    control_diagnostics = json.loads((control / "inference_subset/round2_diagnostics.json").read_text())
    if control_diagnostics["images"] != 8 or control_diagnostics["refined_queries"] != 0:
        raise RuntimeError("Invalid ordinary D smoke validation")
    return {k: v for k, v in diagnostics.items() if k != "records"}


def check_config_pair():
    import yaml
    from rgbd_seg.m2f.config import setup_cfg

    def flatten(node, prefix=""):
        result = {}
        for key, value in node.items():
            path = prefix + key
            if isinstance(value, dict):
                result.update(flatten(value, path + "."))
            else:
                result[path] = value
        return result

    configs = [setup_cfg(ROOT / ("configs/mask2former/" + name + ".yaml"))
               for name in ("round2_d_control_20261026", "round2_dh_20261026")]
    flattened = [flatten(yaml.safe_load(cfg.dump())) for cfg in configs]
    differences = {key: [flattened[0].get(key), flattened[1].get(key)]
                   for key in flattened[0].keys() | flattened[1].keys()
                   if flattened[0].get(key) != flattened[1].get(key)}
    allowed = {"OUTPUT_DIR", "PROJECT.EXPERIMENT", "PROJECT.HIRES_MASK.ENABLED",
               "PROJECT.HIRES_MASK.MAX_PAIRS_PER_STEP"}
    if set(differences) - allowed:
        raise AssertionError("Unexpected differences between controlled configs: " + repr(differences))
    if any(cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED for cfg in configs):
        raise AssertionError("S must be disabled in both experiments")
    if configs[0].PROJECT.HIRES_MASK.ENABLED or not configs[1].PROJECT.HIRES_MASK.ENABLED:
        raise AssertionError("Expected ordinary D control and D+H candidate")
    return {"valid": True, "differences": differences}


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "manager.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "state.json").exists():
            raise FileExistsError("Existing round2 state: inspect before starting another manager")
        smoke = verify_smoke()
        comparison = check_config_pair()
        source_hashes = {p: sha(ROOT / p) for p in SOURCE_PATHS}
        weights = EXPERIMENTS / "d_continue6_results_20261026/best_segm_APs.pth"
        state = {"status": "starting", "pid": os.getpid(), "started_utc": now(),
                 "smoke": smoke, "config_comparison": comparison, "source_sha256": source_hashes,
                 "initial_weights": str(weights), "weights_sha256": sha(weights),
                 "selection": "full val refined segm/APs for DH; full val mask-first segm/APs for D",
                 "validation_images": 2584, "train_only": True,
                 "training_budget": {"iterations": 3608, "epochs": 2, "global_batch": 8,
                                     "gpus": 4, "base_lr": 1e-5, "new_head_lr": 1e-4,
                                     "optimizer": "fresh AdamW, weights-only initialization",
                                     "seed": 20261020, "train_size": 1024, "val_size": 1536,
                                     "hires_max_pairs_per_step": 16},
                 "planned_jobs": ["round2_dh_20261026", "round2_d_control_20261026"],
                 "jobs": []}
        write(state)
        try:
            for name in state["planned_jobs"]:
                if source_hashes != {p: sha(ROOT / p) for p in SOURCE_PATHS}:
                    raise RuntimeError("Training source changed between controlled experiments")
                if (EXPERIMENTS / name).exists():
                    raise FileExistsError("Refusing to overwrite training output: " + name)
                command = [str(ROOT / "run.sh"), "torchrun", "--standalone", "--nnodes=1",
                           "--nproc_per_node=4", str(ROOT / "tools/train_round2.py"),
                           "--config-file", str(ROOT / ("configs/mask2former/" + name + ".yaml")),
                           "--head-lr-multiplier", "10", "--inference-chunk-size", "4"]
                log = OUTPUT / (name + ".log")
                job = {"name": name, "status": "running", "started_utc": now(),
                       "command": command, "log": str(log)}
                state["jobs"].append(job)
                state["status"], state["active_job"] = "running", name
                with log.open("x") as stream:
                    proc = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                                            stdout=stream, stderr=subprocess.STDOUT)
                    job["pid"] = proc.pid
                    write(state)
                    print(now(), "started", name, "pid", proc.pid, flush=True)
                    code = proc.wait()
                job["exit_code"], job["ended_utc"] = code, now()
                (OUTPUT / (name + ".exit_code")).write_text(str(code) + "\n")
                if code:
                    raise RuntimeError(name + " failed, exit " + str(code))
                metrics = json.loads((EXPERIMENTS / name / "best_metrics.json").read_text())
                diagnostics = json.loads((EXPERIMENTS / name / "inference/round2_diagnostics.json").read_text())
                if diagnostics["images"] != 2584:
                    raise RuntimeError("Incomplete full validation for " + name)
                job["status"], job["best"] = "completed", metrics
                write(state)
            state["status"], state["active_job"] = "completed", None
            state["completed_utc"] = now()
            write(state)
        except Exception as error:
            state["status"], state["error"] = "failed", repr(error)
            if state["jobs"] and state["jobs"][-1]["status"] == "running":
                state["jobs"][-1]["status"] = "failed"
            write(state)
            raise


if __name__ == "__main__":
    run()

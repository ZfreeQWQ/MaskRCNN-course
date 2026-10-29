#!/usr/bin/env python
"""Finite 2x2 experiment queue; durable workers, matched controls and full-val selection."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from algorithm_queue_worker import atomic, process_identity, now
from run_algorithm_queue import trainers, json_rows, sha, read

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/mask2former"
OUT = EXP / "generalization_20261027"
PARENT = EXP / "round3_best_algorithm_extend_20261026"
SMOKE_NAME = "crop_local_smoke_v2"
SOURCE_FILES = ["rgbd_seg/m2f/generalization.py", "tools/train_generalization.py",
                "tools/test_generalization.py", "tools/report_small_generalization.py",
                "tools/run_generalization_queue.py", "tools/export_generalization.py"]


class Queue:
    def __init__(self):
        self.path = OUT / "state.json"
        if self.path.exists():
            self.state = read(self.path)
        else:
            frozen = read(OUT / "preflight_sources.json")
            hashes = dict(frozen["protected_sha256"])
            hashes.update({name: sha(ROOT / name) for name in SOURCE_FILES})
            hashes.update({str(p.relative_to(ROOT)): sha(p) for p in (OUT / "configs").glob("*.yaml")})
            baseline = read(PARENT / "best_metrics.json")["metrics"]["segm"]
            self.state = {"status": "prepared", "stage": "prepared", "jobs": {}, "candidates": {},
                          "created_utc": now(), "protected_sha256": hashes,
                          "baseline_weights_sha256": frozen["baseline_weights_sha256"],
                          "baseline": {"APs": baseline["APs"], "AP": baseline["AP"],
                                       "weights": str(PARENT / "best_segm_APs.pth")},
                          "plan": {"arms": ["control", "crop", "local", "crop_local"],
                                   "epochs_per_arm": 2, "gpus": 4, "global_batch": 8,
                                   "initialization": "same selected parent weights; fresh AdamW",
                                   "seed": 20261027, "test_annotations_used": False,
                                   "train_size": 1024, "full_val_size": 1536, "full_val_images": 2584,
                                   "test_goal_APs": 50, "reported_current_Kaggle_APs": 41}}
        self.state.update(manager_pid=os.getpid(), manager_start_ticks=process_identity(os.getpid()))
        self.save()

    def save(self):
        self.state["updated_utc"] = now()
        atomic(self.path, self.state)

    def guard(self):
        changed = [p for p, expected in self.state["protected_sha256"].items() if sha(ROOT / p) != expected]
        if changed:
            raise RuntimeError("Experiment source/config changed: " + repr(changed))
        if sha(PARENT / "best_segm_APs.pth") != self.state["baseline_weights_sha256"]:
            raise RuntimeError("Frozen parent changed")

    def idle(self):
        clear = 0
        while clear < 2:
            active = trainers()
            clear = 0 if active else clear + 1
            self.state["active_trainers"] = active
            self.save()
            if clear < 2:
                time.sleep(15 if active else 2)

    def job(self, name, command, progress=None, timeout=43200):
        self.guard()
        self.state["stage"] = name
        folder = OUT / "jobs" / name
        jobs = self.state["jobs"]
        if name not in jobs:
            self.idle()
            folder.mkdir(parents=True, exist_ok=False)
            spec = {"command": command, "cwd": str(ROOT.parent), "timeout_seconds": timeout,
                    "stall_seconds": 5400, "poll_seconds": 15}
            if progress:
                spec["progress_path"] = str(progress)
            atomic(folder / "command.json", spec)
            with (folder / "worker.log").open("x") as log:
                worker = subprocess.Popen([sys.executable, str(ROOT / "tools/algorithm_queue_worker.py"), str(folder)],
                                          cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            jobs[name] = {"status": "running", "pid": worker.pid,
                          "start_ticks": process_identity(worker.pid), "folder": str(folder), "command": command}
            self.save()
        job = jobs[name]
        if job["status"] == "failed":
            raise RuntimeError("Inspect failed job before resuming: " + name)
        if job["status"] == "completed":
            return
        while not (folder / "result.json").exists():
            if process_identity(job["pid"]) != job["start_ticks"]:
                raise RuntimeError("Worker vanished without durable result: " + name)
            self.save()
            time.sleep(15)
        job.update(read(folder / "result.json"))
        self.save()
        self.idle()
        if job["status"] != "completed":
            raise RuntimeError("Job failed: " + name)

    def smoke_check(self):
        folder = EXP / ("round4_" + SMOKE_NAME + "_20261027")
        if read(folder / "inference_subset/round2_diagnostics.json")["images"] != 16:
            raise RuntimeError("Missing smoke inference")
        if (folder / "best_metrics.json").exists():
            raise RuntimeError("Subset validation must never select a final checkpoint")
        rows = json_rows(folder / "metrics.json")
        updates = [r for r in rows if "total_loss" in r]
        if not updates or max(r.get("iteration", -1) for r in updates) != 59:
            raise RuntimeError("Incomplete smoke updates")
        if not all(math.isfinite(r["total_loss"]) for r in rows if "total_loss" in r):
            raise RuntimeError("Nonfinite smoke loss")
        for row in rows:
            if "loss_local_bce" in row and not (math.isfinite(row["loss_local_bce"]) and row["loss_local_bce"] > 0):
                raise RuntimeError("Inactive local supervision")
        records = json_rows(folder / "generalization_diagnostics.jsonl")
        if not records or any(len(r["ranks"]) != 4 for r in records):
            raise RuntimeError("Missing four-rank diagnostics")
        if not sum(s["focus_crops"] for r in records for s in r["ranks"]):
            raise RuntimeError("No focus crops observed")
        if not sum(s["local_supervision"]["selected_pairs"] for r in records for s in r["ranks"]):
            raise RuntimeError("No locally supervised instances observed")
        for row in records:
            for rank in row["ranks"]:
                local = rank["local_supervision"]
                if local["selected_pairs"] and (local["foreground_label_mean"] < .99 or
                                                local["sampled_logits_abs_mean"] <= 0):
                    raise RuntimeError("Foreground coordinates or sampled logits are inactive")
        self.state["smoke_check"] = {"valid": True, "iterations": 60, "gpus": 4,
                                     "val_images": 16, "subset_used_for_selection": False,
                                     "max_peak_gpu_bytes": max(s["peak_gpu_bytes"] for r in records for s in r["ranks"])}
        self.save()

    def execute(self):
        if self.state["status"] == "completed":
            return
        if not read(OUT / "cpu_tests.json")["valid"]:
            raise RuntimeError("Numerical/mapper tests not passed")
        self.state.update(status="running", error=None)
        self.save()
        prefix = [str(ROOT / "run.sh"), "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                  str(ROOT / "tools/train_generalization.py"), "--config-file"]
        smoke_cfg = OUT / "configs/crop_local_smoke.yaml"
        self.job(SMOKE_NAME, prefix + [str(smoke_cfg), "--eval-limit", "16"],
                 EXP / ("round4_" + SMOKE_NAME + "_20261027/metrics.json"), timeout=3600)
        self.smoke_check()
        for arm in self.state["plan"]["arms"]:
            folder = EXP / ("round4_" + arm + "_20261027")
            config = OUT / "configs" / (arm + ".yaml")
            self.job(arm + "_train", prefix + [str(config)], folder / "metrics.json", timeout=43200)
            rows = json_rows(folder / "metrics.json")
            diag = read(folder / "inference/round2_diagnostics.json")
            updates = [r for r in rows if "total_loss" in r]
            if not updates or max(r.get("iteration", -1) for r in updates) != 3607 or diag["images"] != 2584:
                raise RuntimeError("Incomplete training/full validation: " + arm)
            if diag["ranking"] != "mask-first" or diag["refined_queries"]:
                raise RuntimeError("Unexpected evaluation pipeline")
            best = read(folder / "best_metrics.json")
            score = best["metrics"]["segm"]
            if not all(math.isfinite(score[k]) and 0 <= score[k] <= 100 for k in ("APs", "AP")):
                raise RuntimeError("Invalid full-val scores")
            slices = OUT / (arm + "_slices.json")
            self.job(arm + "_slices", [str(ROOT / "run.sh"), "python",
                     str(ROOT / "tools/report_small_generalization.py"), "--annotations",
                     str(ROOT / "datasets/blender/annotations/instances_val.json"), "--predictions",
                     str(folder / "best_val_predictions.json"), "--output", str(slices)], timeout=3600)
            independent = read(slices)
            if abs(independent["APs"] - score["APs"]) > 1e-6:
                raise RuntimeError("Independent COCO evaluation disagrees")
            self.state["candidates"][arm] = {
                "name": arm, "APs": score["APs"], "AP": score["AP"], "full_val_images": 2584,
                "config": str(config), "output": str(folder), "weights": str(folder / "best_segm_APs.pth"),
                "weights_sha256": sha(folder / "best_segm_APs.pth"), "iteration": best["iteration"],
                "slices": str(slices), "epochs": 2}
            self.save()
        baseline = self.state["baseline"]
        eligible = {k: v for k, v in self.state["candidates"].items() if v["AP"] >= baseline["AP"] - .3}
        winner = max(eligible, key=lambda k: (eligible[k]["APs"], eligible[k]["AP"])) if eligible else None
        improved = winner is not None and eligible[winner]["APs"] > baseline["APs"] + .2
        self.state["factorial_effects_APs"] = {
            "crop_vs_control": self.state["candidates"]["crop"]["APs"] - self.state["candidates"]["control"]["APs"],
            "local_vs_control": self.state["candidates"]["local"]["APs"] - self.state["candidates"]["control"]["APs"],
            "combination_vs_control": self.state["candidates"]["crop_local"]["APs"] - self.state["candidates"]["control"]["APs"]}
        self.state["recommended_model"] = eligible[winner] if improved else dict(baseline, name="retained_parent")
        self.save()
        if improved:
            self.job("selected_test_export", [str(ROOT / "run.sh"), "python",
                     str(ROOT / "tools/export_generalization.py"), "--selection", str(self.path),
                     "--output-dir", str(OUT / "selected_test_submission")], timeout=7200)
        self.state.update(status="completed", stage="completed", completed_utc=now(),
                          hidden_test_goal_verified=False)
        self.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        state = read(OUT / "state.json")
        print(json.dumps({k: v for k, v in state.items() if k != "protected_sha256"}, indent=2))
    else:
        with (OUT / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue = Queue()
            try:
                queue.execute()
            except BaseException as error:
                queue.state.update(status="failed", error=repr(error))
                queue.save()
                raise

#!/usr/bin/env python
"""Finite, restartable four-GPU algorithm queue. No control/ablation jobs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from algorithm_queue_worker import atomic, process_identity

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/mask2former"
OUT = EXP / "algorithm_queue_20261026"
MP = EXP / "round2_mpformer_20261026"
MP_MANAGER = EXP / "round2_mpformer_priority_20261026/state.json"
BASE_CFG = ROOT / "configs/mask2former/round2_d_control_20261026.yaml"
MP_CFG = ROOT / "configs/mask2former/round2_mpformer_20261026.yaml"
FREQ_CFG = ROOT / "configs/mask2former/round3_frequency_20261026.yaml"
BASE_WEIGHTS = EXP / "d_continue6_results_20261026/best_segm_APs.pth"
BASE_APS, BASE_AP = 52.91622131864694, 81.6031070251422
PLAN = [
    {"stage": "wait_current_mp", "action": "Wait for existing MP controller and GPU children to finish"},
    {"stage": "mp_long", "algorithm": "MP full-state continuation", "total_epochs": 6,
     "target_iteration": 10824, "extra_epochs": 4, "preserve_optimizer": True, "gpus": 4, "batch": 8},
    {"stage": "frequency", "algorithm": "D + frequency-aware final FPN fusion", "epochs": 6,
     "iterations": 10824, "init": "frozen D", "gpus": 4, "batch": 8},
    {"stage": "mp_frequency", "algorithm": "D + MP + frequency", "epochs": 6,
     "iterations": 10824, "init": "selected long MP", "gpus": 4, "batch": 8,
     "condition": "Both single methods APs >= D + 0.2 points and AP >= D - 0.3 points"},
    {"stage": "best_algorithm_extend", "algorithm": "Best valid algorithm candidate", "total_epochs": 12,
     "target_iteration": 21648, "init": "selected algorithm's final full-state checkpoint", "gpus": 4, "batch": 8,
     "preserve_optimizer": True, "note": "Extended LR horizon, transition warmup; compare against retained parent best"},
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def json_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def verify_cpu_evidence():
    report = read(OUT / "cpu_tests.json")
    if not report["valid"] or report["exit_code"] != 0:
        raise RuntimeError("Required CPU algorithm tests not passed")
    if any(sha(ROOT/p) != expected for p, expected in report["source_sha256"].items()):
        raise RuntimeError("CPU tests are stale relative to queued source files")
    return report


def continuation_candidates(candidates):
    # The long MP branch retains the short parent's best. Do not restart the
    # short branch merely because those retained scores tie.
    return {k:v for k,v in candidates.items() if k != "best_algorithm_extend"
            and (k != "mp" or "mp_long" not in candidates)}


def protected_sources():
    paths = set(read(MP_MANAGER)["source_sha256"])
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "rgbd_seg/m2f").glob("*.py"))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "configs/mask2former").glob("*.yaml"))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "tools").glob("*algorithm*.py"))
    paths.update(["tools/test_frequency_fusion.py", "rgbd_seg/data.py", "run.sh",
                  "experiments/mask2former/d_continue6_results_20261026/config.yaml",
                  "third_party/Mask2Former/mask2former/modeling/pixel_decoder/msdeformattn.py"])
    return {p: sha(ROOT / p) for p in sorted(paths)}


def trainers():
    """Read-only process guard; never stops an unrelated/manual training job."""
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
            if not args or process_identity(path.name) is None:
                continue
            if any(Path(a).name.startswith("train_") and a.endswith(".py") for a in args) or \
                    any(Path(a).name == "torchrun" or a == "torch.distributed.run" for a in args):
                result.append({"pid": int(path.name), "command": args})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


class Queue:
    def __init__(self):
        self.path = OUT / "state.json"
        self.state = read(self.path) if self.path.exists() else {
            "version": 1, "status": "running", "stage": "wait_current_mp", "created_utc": now(),
            "plan": PLAN, "jobs": {}, "candidates": {}, "decisions": [],
            "protected_sha256": protected_sources(), "baseline_weights_sha256": sha(BASE_WEIGHTS),
            "baseline": {"APs": BASE_APS, "AP": BASE_AP},
            "policy": {"ablation": False, "serial_four_gpu": True, "full_val_images": 2584,
                       "full_val_size": 1536, "train_size": 1024, "seed": 20261020,
                       "finite_queue": True, "auto_retry_failed_jobs": False},
        }
        self.state.update(manager_pid=os.getpid(), manager_start_ticks=process_identity(os.getpid()))
        self.save()

    def save(self):
        self.state["updated_utc"] = now()
        atomic(self.path, self.state)

    def guard_sources(self):
        mismatches = [p for p, expected in self.state["protected_sha256"].items()
                      if not (ROOT / p).exists() or sha(ROOT / p) != expected]
        if mismatches:
            raise RuntimeError("Frozen source changed; inspect before resuming: " + repr(mismatches))
        if shutil.disk_usage(EXP).free < 8 * 1024**3:
            raise RuntimeError("Less than 8 GiB free in persistent experiment volume")

    def wait_idle(self):
        # Two observations reduce races at predecessor teardown.
        clear = 0
        while clear < 2:
            active = trainers()
            self.state["gpu_process_guard"] = {"checked_utc": now(), "active_trainers": active}
            self.save()
            clear = 0 if active else clear + 1
            if clear < 2:
                time.sleep(15 if active else 3)

    def wait_current(self):
        self.state["stage"] = "wait_current_mp"
        while True:
            old = read(MP_MANAGER)
            live = process_identity(old.get("pid", 0)) is not None
            self.state["predecessor"] = {k: old.get(k) for k in ("status", "stage", "pid", "error")}
            self.save()
            if old["status"] in ("completed", "failed") and not live:
                break
            jobs = old.get("jobs", [])
            child_live = jobs and process_identity(jobs[-1].get("pid", 0)) is not None
            if not live and not child_live:
                self.state["decisions"].append({"time": now(), "stage": "mp", "reason": "Predecessor disappeared without successful completion"})
                break
            time.sleep(15)
        self.wait_idle()
        if old["status"] == "completed":
            try:
                self.state["candidates"]["mp"] = self.validate_full(MP, 3608, MP_CFG)
            except Exception as error:
                self.state["decisions"].append({"time": now(), "stage": "mp", "error": repr(error)})
        self.save()

    def config(self, name, base, weights, output, iterations, smoke=False, frequency=False, lr=1e-5, start_iteration=0):
        content = {"_BASE_": str(base), "MODEL": {"WEIGHTS": str(weights)},
                   "PROJECT": {"EXPERIMENT": name.upper(), "EVAL_LIMIT": 0, "SKIP_FINAL_EVAL": False},
                   "SOLVER": {"IMS_PER_BATCH": 8, "BASE_LR": lr, "MAX_ITER": iterations,
                              "STEPS": [start_iteration+48, start_iteration+56] if smoke else [int(iterations*.8), int(iterations*.93)],
                              "WARMUP_ITERS": 5 if smoke else 100,
                              "CHECKPOINT_PERIOD": 60 if smoke else 902},
                   "TEST": {"EVAL_PERIOD": iterations if smoke else 1804}, "OUTPUT_DIR": str(output)}
        if frequency:
            content["MODEL"]["SEM_SEG_HEAD"] = {"PIXEL_DECODER_NAME": "FrequencyPixelDecoder"}
        target = OUT / "configs" / (name + ".yaml")
        target.parent.mkdir(exist_ok=True)
        serialized = yaml.safe_dump(content, sort_keys=False)
        if target.exists() and target.read_text() != serialized:
            raise RuntimeError("Refusing to change an existing job config: " + name)
        if not target.exists():
            target.write_text(serialized)
        return target

    def run_job(self, name, command, progress=None, timeout=43200):
        self.guard_sources()
        self.state["stage"] = name
        folder = OUT / "jobs" / name
        jobs = self.state["jobs"]
        if name not in jobs:
            self.wait_idle()
            folder.mkdir(parents=True, exist_ok=False)
            spec = {"command": command, "cwd": str(ROOT.parent), "timeout_seconds": timeout,
                    "stall_seconds": 5400}
            if progress is not None:
                spec["progress_path"] = str(progress)
            atomic(folder / "command.json", spec)
            with (folder / "worker.log").open("x") as stream:
                worker = subprocess.Popen([sys.executable, str(ROOT / "tools/algorithm_queue_worker.py"), str(folder)],
                                          cwd=ROOT.parent, stdin=subprocess.DEVNULL, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            jobs[name] = {"status": "running", "pid": worker.pid, "start_ticks": process_identity(worker.pid),
                          "started_utc": now(), "command": command, "folder": str(folder)}
            self.save()
        job = jobs[name]
        if job["status"] in ("completed", "failed"):
            return job["status"] == "completed"
        result = folder / "result.json"
        while not result.exists():
            if process_identity(job["pid"]) != job["start_ticks"]:
                job.update(status="failed", error="Worker disappeared without durable exit record", ended_utc=now())
                self.save()
                self.wait_idle()
                return False
            self.save()
            time.sleep(15)
        job.update(read(result))
        self.save()
        self.wait_idle()
        return job["status"] == "completed"

    @staticmethod
    def validate_full(folder, iterations, config):
        rows = json_rows(folder / "metrics.json")
        if not rows or max(row.get("iteration", -1) for row in rows) + 1 < iterations:
            raise RuntimeError("Incomplete formal training: " + str(folder))
        if not (folder / "model_final.pth").is_file():
            raise RuntimeError("Missing final checkpoint")
        diag = read(folder / "inference/round2_diagnostics.json")
        if diag["images"] != 2584 or diag["ranking"] != "mask-first":
            raise RuntimeError("Full-val geometry/ranking check failed")
        best = read(folder / "best_metrics.json")
        scores = best["metrics"]["segm"]
        if not all(math.isfinite(scores[k]) and 0 <= scores[k] <= 100 for k in ("APs", "AP")):
            raise RuntimeError("Invalid full-val metric units")
        weights = folder / "best_segm_APs.pth"
        if not weights.is_file() or weights.stat().st_size < 1000000:
            raise RuntimeError("Selected checkpoint missing")
        return {"APs": scores["APs"], "AP": scores["AP"], "iteration": best["iteration"],
                "config": str(config), "weights": str(weights), "weights_sha256": sha(weights),
                "full_val_images": 2584, "output": str(folder)}

    @staticmethod
    def validate_smoke(folder, frequency, mp, start_iteration=0):
        if read(folder / "inference_subset/round2_diagnostics.json")["images"] != 16:
            raise RuntimeError("Smoke did not evaluate all 16 images")
        if (folder / "best_metrics.json").exists():
            raise RuntimeError("Smoke subset selected a final checkpoint")
        rows = json_rows(folder / "metrics.json")
        if not rows or max(r.get("iteration", -1) for r in rows) < start_iteration+59:
            raise RuntimeError("Incomplete smoke training")
        if any(not math.isfinite(r["total_loss"]) for r in rows if "total_loss" in r):
            raise RuntimeError("Nonfinite smoke loss")
        if frequency:
            rows = json_rows(folder / "frequency_diagnostics.jsonl")
            if not rows or rows[-1]["completed_iterations"] != start_iteration+60 or len(rows[-1]["ranks"]) != 4:
                raise RuntimeError("Missing frequency diagnostics on four ranks")
            for rank in rows[-1]["ranks"]:
                if not math.isfinite(rank["alpha"]) or rank["alpha"] == 0:
                    raise RuntimeError("Frequency residual did not learn")
                if not all(g is not None and math.isfinite(g) for g in rank["gradient_norms"].values()):
                    raise RuntimeError("Missing/nonfinite frequency gradient")
                if not all(rank["gradient_norms"][name] > 0 for name in ("alpha", "low_encoder.weight", "high_encoder.weight")):
                    raise RuntimeError("Inactive frequency kernels")
        if mp:
            rows = json_rows(folder / "mp_diagnostics.jsonl")
            if not rows or rows[-1]["completed_iterations"] != start_iteration+60 or len(rows[-1]["ranks"]) != 4:
                raise RuntimeError("Missing four-rank MP diagnostics")
            if not all(r.get("enabled") and r.get("pilot_queries", 0) > 0 and
                       r.get("label_gradient_norm") is not None and math.isfinite(r["label_gradient_norm"]) and
                       r["label_gradient_norm"] > 0 for r in rows[-1]["ranks"]):
                raise RuntimeError("Inactive/nonfinite MP branch")

    def algorithm(self, name, base, weights, epochs, frequency, mp, lr=1e-5):
        if name in self.state["candidates"]:
            return
        folder = EXP / ("round3_" + name + "_20261026")
        smoke = EXP / ("round3_" + name + "_smoke_20261026")
        try:
            config = self.config(name, base, weights, folder, epochs*1804, frequency=frequency, lr=lr)
            reference_base = BASE_CFG if name == "frequency" else base
            reference = self.config(name+"_reference", reference_base, weights, OUT / "audits", epochs*1804, lr=lr)
            audit = OUT / (name + "_init_audit.json")
            if not self.run_job(name+"_init_audit", [str(ROOT/"run.sh"), "python", str(ROOT/"tools/verify_algorithm_init.py"),
                                                    "--config", str(config), "--reference-config", str(reference),
                                                    "--report", str(audit)], timeout=1800):
                raise RuntimeError("Initialization audit process failed")
            if not read(audit)["valid"]:
                raise RuntimeError("Initialization audit invalid")
            smoke_config = self.config(name+"_smoke", config, weights, smoke, 60, smoke=True, frequency=frequency, lr=lr)
            prefix = [str(ROOT/"run.sh"), "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                      str(ROOT/"tools/train_algorithm.py"), "--config-file"]
            if not self.run_job(name+"_smoke", prefix+[str(smoke_config), "--eval-limit", "16"], smoke/"metrics.json", timeout=3600):
                raise RuntimeError("Four-GPU smoke process failed")
            self.validate_smoke(smoke, frequency, mp)
            if not self.run_job(name+"_train", prefix+[str(config)], folder/"metrics.json", timeout=86400):
                raise RuntimeError("Formal training process failed")
            result = self.validate_full(folder, epochs*1804, config)
            result.update(epochs=epochs, has_frequency=frequency, has_mp=mp, learning_rate=lr,
                          initialization="weights-only; fresh AdamW", init_weights=str(weights), init_weights_sha256=sha(weights))
            self.state["candidates"][name] = result
            self.save()
        except Exception as error:
            self.state["decisions"].append({"time": now(), "stage": name, "status": "failed_or_skipped", "error": repr(error)})
            self.save()

    def resume_algorithm(self, name, parent, total_epochs):
        if name in self.state["candidates"]:
            return
        source = Path(parent["output"])
        folder = EXP/("round3_"+name+"_20261026")
        smoke = EXP/("round3_"+name+"_smoke_20261026")
        try:
            prepare = [str(ROOT/"run.sh"), "python", str(ROOT/"tools/prepare_algorithm_resume.py"),
                       "--source-output", str(source), "--output"]
            if not self.run_job(name+"_prepare_smoke", prepare+[str(smoke), "--smoke"], timeout=1800):
                raise RuntimeError("Full-state smoke checkpoint staging failed")
            start = read(smoke/"resume_seed.json")["completed_iterations"]
            if start >= total_epochs*1804:
                raise RuntimeError("Continuation target must exceed parent final iteration")
            smoke_cfg = self.config(name+"_smoke", Path(parent["config"]), source/"model_final.pth",
                                    smoke, start+60, smoke=True, frequency=parent["has_frequency"], start_iteration=start)
            command = [str(ROOT/"run.sh"), "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                       str(ROOT/"tools/train_algorithm_resume.py"), "--resume", "--config-file"]
            if not self.run_job(name+"_smoke", command+[str(smoke_cfg), "--eval-limit", "16"], smoke/"metrics.json", timeout=3600):
                raise RuntimeError("Four-GPU full-state resume smoke failed")
            self.validate_smoke(smoke, parent["has_frequency"], parent["has_mp"], start_iteration=start)
            audit = read(smoke/"optimizer_resume_audit.json")
            if not audit["valid"] or len(audit["ranks"]) != 4 or not all(r["valid"] and r["start_iter"]==start for r in audit["ranks"]):
                raise RuntimeError("Four-rank optimizer state preservation audit failed")
            if not self.run_job(name+"_prepare_train", prepare+[str(folder)], timeout=1800):
                raise RuntimeError("Formal full-state checkpoint staging failed")
            config = self.config(name, Path(parent["config"]), source/"model_final.pth", folder,
                                 total_epochs*1804, frequency=parent["has_frequency"])
            if not self.run_job(name+"_train", command+[str(config)], folder/"metrics.json", timeout=86400):
                raise RuntimeError("Full-state continuation failed")
            result = self.validate_full(folder, total_epochs*1804, config)
            result.update(total_epochs=total_epochs, extra_epochs=(total_epochs*1804-start)/1804,
                          has_frequency=parent["has_frequency"], has_mp=parent["has_mp"],
                          initialization="full final checkpoint; optimizer moments and iteration preserved; extended LR schedule", parent_output=str(source))
            self.state["candidates"][name] = result
            self.save()
        except Exception as error:
            self.state["decisions"].append({"time":now(), "stage":name, "status":"failed_or_skipped", "error":repr(error)})
            self.save()

    def execute(self):
        if self.state["status"] == "completed":
            return
        self.guard_sources()
        verify_cpu_evidence()
        if sha(BASE_WEIGHTS) != self.state["baseline_weights_sha256"]:
            raise RuntimeError("Frozen D checkpoint changed")
        self.state.update(status="running", error=None)
        self.wait_current()
        candidates = self.state["candidates"]
        # MP metadata is inferred only after full completion and validation.
        if "mp" in candidates:
            candidates["mp"].update(has_mp=True, has_frequency=False)
            self.resume_algorithm("mp_long", candidates["mp"], 6)
        self.algorithm("frequency", FREQ_CFG, BASE_WEIGHTS, 6, frequency=True, mp=False)
        candidates = self.state["candidates"]
        combine = all(k in candidates and candidates[k]["APs"] >= BASE_APS+.2 and
                      candidates[k]["AP"] >= BASE_AP-.3 for k in ("mp_long", "frequency"))
        if combine:
            self.algorithm("mp_frequency", MP_CFG, Path(candidates["mp_long"]["weights"]), 6, frequency=True, mp=True)
        else:
            self.state["decisions"].append({"time": now(), "stage": "mp_frequency", "status": "skipped",
                                            "reason": "Both single-method full-val improvement gates were not met"})
        candidates = self.state["candidates"]
        singles = continuation_candidates(candidates)
        if singles:
            winner = max(singles, key=lambda k: (singles[k]["APs"], singles[k]["AP"]))
            selected = singles[winner]
            self.state["continuation_parent"] = winner
            self.save()
            self.resume_algorithm("best_algorithm_extend", selected, 12)
        else:
            self.state["decisions"].append({"time": now(), "stage": "best_algorithm_extend", "status": "skipped",
                                            "reason": "No completed valid algorithm model to extend"})
        # Include frozen D in final selection. A longer run never replaces a better parent automatically.
        valid = self.state["candidates"]
        winner = max(valid, key=lambda k: (valid[k]["APs"], valid[k]["AP"])) if valid else None
        self.state["best_algorithm"] = winner
        eligible = {k:v for k,v in valid.items() if v["AP"] >= BASE_AP-.3}
        recommended = max(eligible, key=lambda k: (eligible[k]["APs"], eligible[k]["AP"])) if eligible else None
        beats = recommended is not None and (eligible[recommended]["APs"], eligible[recommended]["AP"]) > (BASE_APS, BASE_AP)
        self.state["recommended_model"] = dict(eligible[recommended], name=recommended) if beats else {
            "name": "frozen_D", "APs": BASE_APS, "AP": BASE_AP, "weights": str(BASE_WEIGHTS)}
        self.state.update(status="completed", stage="completed", completed_utc=now(),
                          completed_with_failures=any(j["status"] == "failed" for j in self.state["jobs"].values()) or
                          any(d.get("status") == "failed_or_skipped" for d in self.state["decisions"]))
        self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        state = read(OUT / "state.json")
        report = {k:state.get(k) for k in ("status", "stage", "manager_pid", "updated_utc", "predecessor",
                                          "jobs", "candidates", "continuation_parent", "recommended_model", "error")}
        report["manager_alive"] = process_identity(state.get("manager_pid", 0)) == state.get("manager_start_ticks")
        print(json.dumps(report, indent=2))
        return
    if args.preflight:
        old = read(MP_MANAGER)
        unchanged = all(sha(ROOT/p) == h for p,h in old["source_sha256"].items())
        verify_cpu_evidence()
        if not unchanged:
            raise RuntimeError("Preflight failed")
        print(json.dumps({"valid": True, "active_mp_sources_unchanged": unchanged,
                          "frozen_source_files": len(protected_sources()), "plan": PLAN}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = Queue()
        try:
            queue.execute()
        except BaseException as error:
            queue.state.update(status="failed", error=repr(error))
            queue.save()
            raise


if __name__ == "__main__":
    main()

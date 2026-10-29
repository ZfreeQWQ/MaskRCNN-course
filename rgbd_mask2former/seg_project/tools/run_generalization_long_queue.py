#!/usr/bin/env python
"""Convergence-aware extension of the retained two-epoch factorial experiment."""
import fcntl
import math
from pathlib import Path
import sys

import yaml
from algorithm_queue_worker import now
from run_algorithm_queue import read, sha, json_rows
from run_generalization_queue import Queue as ShortQueue, ROOT, EXP, OUT
from generalization_convergence import assess

EPOCH = 1804


class Queue(ShortQueue):
    def config(self, name, base, source, start, target, smoke=False):
        folder = EXP / ("round4_" + name + "_20261027")
        remaining = target - start
        content = {"_BASE_": str(base), "MODEL": {"WEIGHTS": str(source / "model_final.pth")},
                   "PROJECT": {"EXPERIMENT": name.upper(), "EVAL_LIMIT": 0, "SKIP_FINAL_EVAL": False},
                   "SOLVER": {"IMS_PER_BATCH": 8, "BASE_LR": 1e-5, "MAX_ITER": target,
                              "STEPS": [start + round(remaining * .8), start + round(remaining * .93)],
                              "WARMUP_ITERS": 5 if smoke else 100, "WARMUP_FACTOR": .1,
                              "CHECKPOINT_PERIOD": 60 if smoke else 902},
                   "TEST": {"EVAL_PERIOD": target if smoke else EPOCH}, "OUTPUT_DIR": str(folder)}
        path = OUT / "configs" / (name + ".yaml")
        serialized = yaml.safe_dump(content, sort_keys=False)
        if path.exists():
            if path.read_text() != serialized:
                raise RuntimeError("Refusing to rewrite an existing continuation configuration")
        else:
            path.write_text(serialized)
        self.state["protected_sha256"][str(path.relative_to(ROOT))] = sha(path)
        self.save()
        return path, folder

    def job(self, name, command, progress=None, timeout=43200):
        if "--config-file" in command:
            config = Path(command[command.index("--config-file") + 1])
            values = yaml.safe_load(config.read_text())
            self.state["current_training"] = {"name": name, "config": str(config),
                "output": values["OUTPUT_DIR"], "target_iterations": values["SOLVER"]["MAX_ITER"],
                "target_epochs": values["SOLVER"]["MAX_ITER"] / EPOCH}
        self.save()
        return super().job(name, command, progress, timeout)

    @staticmethod
    def curve(folder, target, previous=()):
        points = {row["epoch"]: row for row in previous}
        lr = None
        for row in json_rows(folder / "metrics.json"):
            lr = row.get("lr", lr)
            if "segm/APs" in row:
                completed = min(row["iteration"] + 1, target)
                epoch = completed / EPOCH
                if abs(epoch - round(epoch)) > 1e-6:
                    continue
                points[int(round(epoch))] = {"epoch": int(round(epoch)), "APs": row["segm/APs"],
                                            "AP": row["segm/AP"], "lr": lr if lr is not None else 1.}
        return [points[k] for k in sorted(points)]

    def collect(self, name, folder, config, target, previous=()):
        rows = [r for r in json_rows(folder / "metrics.json") if "total_loss" in r]
        diag = read(folder / "inference/round2_diagnostics.json")
        if not rows or rows[-1]["iteration"] + 1 != target or diag["images"] != 2584:
            raise RuntimeError("Incomplete formal training/full validation: " + name)
        if diag["ranking"] != "mask-first" or diag["refined_queries"]:
            raise RuntimeError("Changed validation inference")
        if not all(math.isfinite(r["total_loss"]) for r in rows):
            raise RuntimeError("Nonfinite training loss")
        if not (folder / "model_final.pth").exists():
            raise RuntimeError("Missing final full-state checkpoint")
        best = read(folder / "best_metrics.json")
        scores = best["metrics"]["segm"]
        if not all(math.isfinite(scores[k]) and 0 <= scores[k] <= 100 for k in ("AP", "APs")):
            raise RuntimeError("Invalid score units")
        slices = OUT / (name + "_slices.json")
        self.job(name + "_slices", [str(ROOT / "run.sh"), "python", str(ROOT / "tools/report_small_generalization.py"),
                 "--annotations", str(ROOT / "datasets/blender/annotations/instances_val.json"),
                 "--predictions", str(folder / "best_val_predictions.json"), "--output", str(slices)], timeout=3600)
        if abs(read(slices)["APs"] - scores["APs"]) > 1e-6:
            raise RuntimeError("Independent COCO verification disagrees")
        result = {"name": name, "APs": scores["APs"], "AP": scores["AP"], "full_val_images": 2584,
                  "config": str(config), "output": str(folder), "weights": str(folder / "best_segm_APs.pth"),
                  "weights_sha256": sha(folder / "best_segm_APs.pth"), "iteration": best["iteration"],
                  "slices": str(slices), "epochs": target / EPOCH,
                  "curve": self.curve(folder, target, previous)}
        self.state["candidates"][name] = result
        self.save()
        return result

    def short(self, arm):
        folder = EXP / ("round4_" + arm + "_20261027")
        config = OUT / "configs" / (arm + ".yaml")
        if arm not in self.state["candidates"]:
            command = [str(ROOT / "run.sh"), "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                       str(ROOT / "tools/train_generalization.py"), "--config-file", str(config)]
            self.job(arm + "_train", command, folder / "metrics.json")
            return self.collect(arm, folder, config, 2 * EPOCH)
        result = self.state["candidates"][arm]
        result["curve"] = self.curve(folder, 2 * EPOCH)
        self.save()
        return result

    def continuation(self, name, parent, epochs):
        if name in self.state["candidates"]:
            return self.state["candidates"][name]
        source = Path(parent["output"])
        start = int(round(parent["epochs"] * EPOCH))
        config, folder = self.config(name, Path(parent["config"]), source, start, epochs * EPOCH)
        prepare = [str(ROOT / "run.sh"), "python", str(ROOT / "tools/prepare_generalization_resume.py"),
                   "--source-output", str(source), "--output", str(folder)]
        self.job(name + "_prepare", prepare, timeout=1800)
        command = [str(ROOT / "run.sh"), "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=4",
                   str(ROOT / "tools/train_generalization_resume.py"), "--resume", "--config-file", str(config)]
        self.job(name + "_train", command, folder / "metrics.json", timeout=86400)
        audit = read(folder / "optimizer_resume_audit.json")
        if not audit["valid"] or len(audit["ranks"]) != 4 or not all(r["start_iter"] == start for r in audit["ranks"]):
            raise RuntimeError("Four-rank optimizer/iteration restoration audit failed")
        result = self.collect(name, folder, config, epochs * EPOCH, parent.get("curve", ()))
        result.update(initialization="full-state continuation; AdamW moments/global iteration retained; extended LR with 100-step warmup",
                      parent_output=str(source), parent_completed_epochs=parent["epochs"])
        self.save()
        return result

    def resume_smoke(self):
        source = EXP / "round4_control_20261027"
        start = 2 * EPOCH
        # Control checkpoint has the identical parameter/optimizer topology. Turn
        # on both recipe changes to test their actual continuation code together.
        config, folder = self.config("long_resume_smoke", OUT / "configs/crop_local.yaml", source,
                                     start, start + 60, smoke=True)
        self.job("long_resume_smoke_prepare", [str(ROOT / "run.sh"), "python",
                 str(ROOT / "tools/prepare_generalization_resume.py"), "--source-output", str(source),
                 "--output", str(folder), "--smoke"], timeout=1800)
        self.job("long_resume_smoke_train", [str(ROOT / "run.sh"), "torchrun", "--standalone", "--nnodes=1",
                 "--nproc_per_node=4", str(ROOT / "tools/train_generalization_resume.py"), "--resume", "--config-file",
                 str(config), "--eval-limit", "16"], folder / "metrics.json", timeout=3600)
        audit = read(folder / "optimizer_resume_audit.json")
        diag = read(folder / "inference_subset/round2_diagnostics.json")
        records = json_rows(folder / "generalization_diagnostics.jsonl")
        rows = [r for r in json_rows(folder / "metrics.json") if "total_loss" in r]
        if not audit["valid"] or len(audit["ranks"]) != 4 or not all(r["start_iter"] == start for r in audit["ranks"]):
            raise RuntimeError("Continuation smoke failed to preserve all four optimizers")
        if not rows or rows[-1]["iteration"] + 1 != start + 60 or diag["images"] != 16:
            raise RuntimeError("Incomplete continuation smoke")
        if (folder / "best_metrics.json").exists():
            raise RuntimeError("Smoke validation selected a final checkpoint")
        if not records or any(len(r["ranks"]) != 4 for r in records):
            raise RuntimeError("Missing continuation diagnostics")
        if not all(math.isfinite(r["total_loss"]) for r in rows):
            raise RuntimeError("Nonfinite continuation smoke loss")
        for record in records:
            for rank in record["ranks"]:
                local = rank["local_supervision"]
                if local["selected_pairs"] and (local["foreground_label_mean"] < .99 or local["sampled_logits_abs_mean"] <= 0):
                    raise RuntimeError("Local supervision is inactive after continuation")
        self.state["continuation_smoke"] = {"valid": True, "four_rank_optimizer_moments_preserved": True,
            "global_start_iteration": start, "completed_extra_steps": 60, "val_images": 16,
            "focus_crop_and_local_loss_enabled": True, "subset_used_for_selection": False}
        self.save()

    def execute(self):
        if self.state.get("long_protocol_completed"):
            return
        if not read(OUT / "long_cpu_tests.json")["valid"]:
            raise RuntimeError("Required continuation/policy tests not passed")
        self.guard()
        self.state.update(status="running", error=None, coordinator_transition=None)
        self.save()
        # Attach to the current durable crop job; never interrupt its last epoch/eval.
        crop_short = self.short("crop")
        control_short = self.short("control")
        self.resume_smoke()
        six = {}
        six["crop"] = self.continuation("crop_6ep", crop_short, 6)
        for arm in ("local", "crop_local"):
            six[arm] = self.continuation(arm + "_6ep", self.short(arm), 6)
        six["control"] = self.continuation("control_6ep", control_short, 6)
        baseline = self.state["baseline"]
        algorithms = {k: v for k, v in six.items() if k != "control"}
        acceptable = {k: v for k, v in algorithms.items() if v["AP"] >= baseline["AP"] - .3}
        pool = acceptable or algorithms
        arm = max(pool, key=lambda k: (pool[k]["APs"], pool[k]["AP"]))
        self.state["extended_algorithm"] = arm
        self.state["six_epoch_effects_APs"] = {k: v["APs"] - six["control"]["APs"] for k, v in algorithms.items()}
        self.save()
        selected, control = six[arm], six["control"]
        for target in (12, 18, 24):
            selected = self.continuation(arm + f"_{target}ep", selected, target)
            control = self.continuation(f"control_{target}ep", control, target)
            assessment = assess(selected["curve"])
            self.state.setdefault("convergence_checks", []).append(dict(assessment, algorithm=arm,
                target_epochs=target, selected_APs=selected["APs"], same_budget_control_APs=control["APs"],
                delta_vs_same_budget_control=selected["APs"] - control["APs"]))
            self.save()
            if assessment["plateau"]:
                break
        assessment = self.state["convergence_checks"][-1]
        self.state["stopping_reason"] = "observed_full_val_plateau" if assessment["plateau"] else "24_epoch_cap_reached_without_plateau"
        candidates = {k: v for k, v in self.state["candidates"].items() if v["epochs"] >= 6 and v["AP"] >= baseline["AP"] - .3}
        winner = max(candidates, key=lambda k: (candidates[k]["APs"], candidates[k]["AP"])) if candidates else None
        improved = winner is not None and candidates[winner]["APs"] > baseline["APs"] + .2
        self.state["recommended_model"] = candidates[winner] if improved else dict(baseline, name="retained_parent")
        self.save()
        if improved:
            self.job("selected_long_test_export", [str(ROOT / "run.sh"), "python", str(ROOT / "tools/export_generalization.py"),
                     "--selection", str(self.path), "--output-dir", str(OUT / "selected_test_submission_long")], timeout=7200)
        self.state.update(status="completed", stage="completed", long_protocol_completed=True, completed_utc=now(),
                          hidden_test_goal_verified=False, observed_validation_plateau=assessment["plateau"])
        self.save()


if __name__ == "__main__":
    with (OUT / "manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = Queue()
        try:
            queue.execute()
        except BaseException as error:
            queue.state.update(status="failed", error=repr(error))
            queue.save()
            raise

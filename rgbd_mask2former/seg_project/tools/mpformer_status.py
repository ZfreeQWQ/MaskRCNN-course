#!/usr/bin/env python
"""Dependency-free status of the active algorithm-priority pipeline."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments/mask2former"


def main():
    file = ROOT / "round2_mpformer_priority_20261026/state.json"
    if not file.exists():
        print(json.dumps({"algorithm_priority": "not_started"}))
        return
    state = json.loads(file.read_text())
    report = {key: state.get(key) for key in ("status", "stage", "pid", "started_utc", "error", "smoke_passed")}
    report["checked_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["algorithm"] = state["algorithm"]
    report["manager_alive"] = Path(f"/proc/{state['pid']}").exists()
    if state["jobs"]:
        job = state["jobs"][-1]
        report["job"] = {key: job.get(key) for key in ("stage", "status", "pid", "exit_code", "log")}
        report["job"]["process_alive"] = Path(f"/proc/{job.get('pid', 'missing')}").exists()
    stage = state["stage"]
    name = "round2_mpformer_smoke_20261026" if stage == "four_gpu_smoke" else "round2_mpformer_20261026"
    folder = ROOT / name
    for filename, key in (("metrics.json", "latest_training"), ("mp_diagnostics.jsonl", "pilot_diagnostics")):
        path = folder / filename
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if rows and filename == "metrics.json":
            maximum = 60 if stage == "four_gpu_smoke" else 3608
            report["completed_iterations"] = min(rows[-1]["iteration"] + 1, maximum)
            report["total_iterations"] = maximum
            report["progress_percent"] = round(report["completed_iterations"] / maximum * 100, 2)
            training = next((r for r in reversed(rows) if "total_loss" in r), {})
            report[key] = {k: training[k] for k in ("total_loss", "loss_ce_mp", "loss_mask_mp", "loss_dice_mp", "time", "lr") if k in training}
        elif rows:
            report[key] = rows[-1]
    best = folder / "best_metrics.json"
    if best.exists():
        report["best_full_val"] = json.loads(best.read_text())
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

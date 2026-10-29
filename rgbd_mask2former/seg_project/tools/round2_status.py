#!/usr/bin/env python
"""Read-only, dependency-free status report for the first round2 batch."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments/mask2former"
STATE = ROOT / "round2_dh_control_20261026/state.json"


def main():
    if not STATE.exists():
        print(json.dumps({"status": "not_started", "state_file": str(STATE)}))
        return
    state = json.loads(STATE.read_text())
    report = {key: state.get(key) for key in ("status", "active_job", "planned_jobs", "pid", "started_utc", "error")}
    report["checked_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report["manager_alive"] = Path("/proc/" + str(state["pid"])).exists()
    report["jobs"] = []
    for job in state["jobs"]:
        record = {key: job.get(key) for key in ("name", "status", "pid", "started_utc", "ended_utc", "exit_code")}
        folder = ROOT / job["name"]
        record["process_alive"] = Path("/proc/" + str(job.get("pid", "missing"))).exists()
        metrics = folder / "metrics.json"
        if metrics.exists():
            lines = metrics.read_text().splitlines()
            rows = []
            for line in lines:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # A concurrently written final line may be incomplete.
            if rows:
                last = rows[-1]
                record["completed_iterations"] = min(last["iteration"] + 1, 3608)
                record["total_iterations"] = 3608
                record["progress_percent"] = round(record["completed_iterations"] / 3608 * 100, 2)
                train = next((r for r in reversed(rows) if "total_loss" in r), {})
                record["latest_training"] = {k: train[k] for k in ("total_loss", "loss_mask_hires", "loss_dice_hires", "time", "lr") if k in train}
        best = folder / "best_metrics.json"
        if best.exists():
            record["best_full_validation"] = json.loads(best.read_text())
        report["jobs"].append(record)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

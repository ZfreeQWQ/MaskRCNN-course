#!/usr/bin/env python
"""Preserve a periodic checkpoint before releasing the user's deferred ablation GPUs."""
import datetime
import json
import os
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments/mask2former"
FOLDER = ROOT / "round2_dh_control_20261026"
TRAIN = ROOT / "round2_dh_20261026"


def main():
    state = json.loads((FOLDER / "state.json").read_text())
    if state["status"] != "deferring_at_checkpoint":
        raise RuntimeError("No user-requested deferral pending")
    pid = state["jobs"][0]["pid"]
    deadline = time.monotonic() + 1800
    checkpoint = None
    while time.monotonic() < deadline:
        tag = TRAIN / "last_checkpoint"
        if tag.exists():
            checkpoint = TRAIN / tag.read_text().strip()
            if checkpoint.is_file() and checkpoint.stat().st_size > 500_000_000:
                break
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError("Training exited before a periodic checkpoint was available")
        time.sleep(5)
    else:
        raise TimeoutError("No complete periodic checkpoint within 30 minutes")
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "torchrun" not in cmd or "train_round2.py" not in cmd or "round2_dh_20261026.yaml" not in cmd:
        raise RuntimeError("Training process identity mismatch; refusing to signal")
    print("Preserved", checkpoint, checkpoint.stat().st_size, "bytes", flush=True)
    os.kill(pid, signal.SIGTERM)
    for _ in range(36):
        proc = Path(f"/proc/{pid}/stat")
        if not proc.exists() or proc.read_text().split()[2] == "Z":
            break
        time.sleep(5)
    else:
        raise TimeoutError("torchrun did not exit after SIGTERM")
    state["status"] = "deferred_by_user"
    state["active_job"] = None
    state["deferred_checkpoint"] = str(checkpoint)
    state["deferred_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state["jobs"][0]["status"] = "deferred_by_user"
    state["jobs"][0]["resume_checkpoint"] = str(checkpoint)
    tmp = FOLDER / "state.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(FOLDER / "state.json")
    print(json.dumps({"status": state["status"], "checkpoint": str(checkpoint)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

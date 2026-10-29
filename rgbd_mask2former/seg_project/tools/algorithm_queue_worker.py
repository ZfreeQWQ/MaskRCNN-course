#!/usr/bin/env python
"""Durable one-job runner: records exit status even if the queue manager exits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_folder")
    args = parser.parse_args()
    folder = Path(args.job_folder)
    spec = json.loads((folder / "command.json").read_text())
    if (folder / "result.json").exists():
        raise FileExistsError("Never rerun an already finished job in place")
    record = {"started_utc": now(), "worker_pid": os.getpid(),
              "worker_start_ticks": process_identity(os.getpid()), "status": "starting"}
    proc = None
    def stop_child():
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    def handle_signal(signum, frame):
        raise InterruptedError("Worker received signal " + str(signum))
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        with (folder / "training.log").open("x") as log:
            proc = subprocess.Popen(spec["command"], cwd=spec["cwd"], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status="running", child_pid=proc.pid, child_start_ticks=process_identity(proc.pid))
            atomic(folder / "worker.json", record)
            started = time.monotonic()
            while proc.poll() is None:
                if time.monotonic() - started > spec.get("timeout_seconds", 43200):
                    raise TimeoutError("Job exceeded its finite time budget")
                progress = Path(spec.get("progress_path", folder / "training.log"))
                # Includes full-val CPU scoring time; allow 90 minutes without a new metric.
                mtime = progress.stat().st_mtime if progress.exists() else (folder / "training.log").stat().st_mtime
                if time.time() - mtime > spec.get("stall_seconds", 5400):
                    raise TimeoutError("No training progress for 90 minutes")
                record["heartbeat_utc"] = now()
                atomic(folder / "worker.json", record)
                time.sleep(spec.get("poll_seconds", 15))
            record.update(exit_code=proc.returncode, status="completed" if proc.returncode == 0 else "failed")
    except BaseException as error:
        stop_child()
        record.update(status="failed", exit_code=proc.returncode if proc is not None else None, error=repr(error))
    record["ended_utc"] = now()
    atomic(folder / "result.json", record)
    print(json.dumps(record, indent=2))
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

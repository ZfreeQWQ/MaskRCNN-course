"""One frozen-screen audit of retained P2 weights, no training or new panels."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile
from course.common import atomic_json


def main():
    root = Path(__file__).resolve().parent
    target = root/"runs/instance_audit_v1_r2"
    assert not target.exists(), "Never overwrite/repeat an existing audit"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    memory = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    assert int(memory.strip()) < 256, "GPU0 occupied; do not interrupt unrelated jobs"
    control_dir = root/"runs/screen_v2/j01_p2_queries"
    control = json.loads((control_dir/"status.json").read_text())
    assert control["status"] == "completed" and control["step"] == 3000
    weights = root/"runs/screen_v2_confirmation/j01_p2_queries_inference.pth"
    panel = root/"runs/screen_v1/panel_screen.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", commit], cwd=root)
    source = target/"source"
    source.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source/n).resolve().is_relative_to(source) for n in z.namelist())
        z.extractall(source)
    for name in ("weights", "data_cache"):
        (source/name).symlink_to(root/name, target_is_directory=True)
    state = dict(status="running", commit=commit, archive_sha256=hashlib.sha256(archive).hexdigest(),
                 weights=str(weights), weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                 panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(), started_unix=time.time(),
                 training=False, full_val=False, test_used=False, new_confirmation_used=False)
    atomic_json(target/"status.json", state)
    try:
        with (target/"evaluation.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "evaluate_rgbd.py", "--config", "configs/screen/j01_p2_queries.yaml",
                "--weights", str(weights), "--strict-load", "--image-ids", str(panel), "--diagnostics", "--instance-audit",
                "--predictions-output", str(target/"predictions.json"), "--output", str(target/"metrics.json")], cwd=source,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((target/"metrics.json").read_text())
        reference = control["final_metrics"]
        for key in ("mask_aps", "mask_ap", "mask_aps75", "mask_ars"):
            assert abs(result[key]-reference[key]) < 1e-10, (key, result[key], reference[key])
            if key in result["instance_audit"]["baseline"]:
                assert abs(result[key]-result["instance_audit"]["baseline"][key]) < 1e-10
        assert result["panel_sha256"] == state["panel_sha256"] and result["validation_images"] == 200
        state.update(status="completed", metric_reproduction_exact=True, finished_unix=time.time())
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        atomic_json(target/"status.json", state)


if __name__ == "__main__":
    main()

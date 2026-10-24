"""Two fixed-val400 inference comparisons from committed source. No training."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from course.common import atomic_json


def main():
    root = Path(__file__).resolve().parent
    out = root / "runs/optimization_v3_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise RuntimeError("Diagnostic output exists; inspect instead of overwriting")
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", commit])
    source = out / "source"
    source.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert all((source / p).resolve().is_relative_to(source) for p in z.namelist())
        z.extractall(source)
    for p in ("data_cache", "weights"):
        (source / p).symlink_to(root / p, target_is_directory=True)
    weight = root / "runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth"
    suite = json.loads((root / "runs/optimization_v2/suite.json").read_text())
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest == suite["jobs"]["d01_geometry_multiscale"]["inference_export"]["sha256"]
    state = dict(status="running", commit=commit, source=str(source), checkpoint=str(weight),
                 checkpoint_sha256=digest, fixed_samples=400, train_used=False, test_used=False,
                 started_unix=time.time(), jobs={})
    atomic_json(out / "status.json", state)
    try:
        for name in ("g00_fast_control", "g01_scale1280"):
            state["active"] = name
            atomic_json(out / "status.json", state)
            with (out / (name+".log")).open("w") as log:
                subprocess.run([sys.executable, "-u", "evaluate_rgbd.py", "--config", f"configs/v3/{name}.yaml",
                                "--weights", str(weight), "--samples", "400", "--diagnostics", "--strict-load",
                                "--output", str(out / (name+".json"))], cwd=source,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4"),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state["jobs"][name] = json.loads((out / (name+".json")).read_text())
        state.update(status="completed", active=None, finished_unix=time.time())
        atomic_json(out / "status.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Freeze source and small evidence on the server without reading bulk datasets."""
import argparse
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "seg_project"
SKIP = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "build", "dist", "node_modules",
        ".venv", ".ipynb_checkpoints", "snapshots", ".tox"}
SUFFIXES = {".py", ".yaml", ".yml", ".md", ".rst", ".txt", ".sh", ".c", ".cc", ".cpp",
            ".cu", ".h", ".hpp", ".cuh", ".cmake", ".toml", ".ini", ".cfg", ".json",
            ".html", ".css", ".js", ".svg"}
NAMES = {"LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "Makefile", "Dockerfile", ".gitignore", ".gitmodules"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if not out.is_relative_to(PROJECT / "reports"):
        raise ValueError("Persistent output must be inside seg_project/reports")
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "server_source_snapshot.zip"
    records, excluded = [], Counter()
    captured_start = now()
    seen = set()
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        def add(name, data, source=None, mode=0o644):
            if name in seen:
                raise ValueError("Duplicate member: " + name)
            seen.add(name)
            info = zipfile.ZipInfo(name, date_time=datetime.datetime.now().timetuple()[:6])
            info.create_system = 3
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, data)
            records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "source": str(source) if source else "generated_snapshot", "captured_utc": now()})

        def source(path):
            if not path.is_file() or not path.resolve().is_relative_to(ROOT):
                return
            relative = path.relative_to(ROOT).as_posix()
            if path.stat().st_size > 8 * 1024 * 1024:
                excluded["large_non_source"] += 1
                return
            add("project/" + relative, path.read_bytes(), path,
                0o755 if path.suffix == ".sh" else 0o644)

        for tree in (ROOT / "detectron2", PROJECT):
            for folder, directories, files in os.walk(tree):
                directories[:] = sorted(d for d in directories if d not in SKIP and not d.endswith(".egg-info"))
                if Path(folder) == PROJECT:
                    directories[:] = [d for d in directories if d not in {"datasets", "weights", "backups", "reports", "experiments"}]
                for filename in sorted(files):
                    path = Path(folder) / filename
                    if path.suffix.lower() in SUFFIXES or filename in NAMES:
                        if filename == ".env" or filename.startswith(("id_rsa", "id_ed25519")):
                            excluded["sensitive_filename"] += 1
                            continue
                        source(path)
                    else:
                        excluded["binary_cache_or_non_source"] += 1
        for path in (ROOT / "AGENTS.md", ROOT / ".gitignore", ROOT / "blender/README.md",
                     PROJECT / "weights/source.json",
                     PROJECT / "datasets/blender/annotations/depth_geometry_stats.json",
                     PROJECT / "datasets/blender/annotations/depth_stats.json"):
            source(path)
        # These files are needed by relative _BASE_ chains; never traverse snapshots/predictions.
        for family in ("mask2former", "maskrcnn"):
            parent = PROJECT / "experiments" / family
            if parent.exists():
                for run in sorted(parent.iterdir()):
                    if run.is_dir():
                        for path in sorted(run.glob("*.yaml")):
                            source(path)

        def evidence(relative, destination=None):
            path = PROJECT / relative
            if not path.is_file():
                return
            data = path.read_bytes()
            if len(data) > 8 * 1024 * 1024:
                excluded["large_evidence"] += 1
                return
            if path.suffix in (".jsonl",) or path.name == "metrics.json":
                lines = []
                for index, line in enumerate(data.decode("utf-8").splitlines()):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        excluded["incomplete_live_jsonl_line"] += 1
                        continue
                    lines.append(line)
                data = ("\n".join(lines) + "\n").encode()
            add("evidence/" + (destination or relative), data, path)

        completed = {
            "d_continue6_results_20261026": ["val_results.json", "best_metrics.json", "execution_audit.json",
                "test_complete.json", "submission.csv.validation.json", "workflow_state.json", "README.md"],
            "d_continue6_20261026": ["best_metrics.json", "latest_eval_metrics.json", "metrics.json", "manager_state.json"],
            "optimization_round1_20261026": ["val_results.json", "test_complete.json", "workflow_state.json"],
            "threshold_sweep_20261025": ["val_thresholds.json", "val_thresholds.md"],
            "round2_dh_control_20261026": ["state.json", "checkpoint_deferral.log", "deferred_checkpoint_audit.json"],
            "round2_mpformer_priority_20261026": ["state.json", "manager.log", "initial_inference_audit.exit_code",
                "four_gpu_smoke.exit_code", "algorithm_training.exit_code"],
            "round2_mpformer_checks_20261026": ["initial_inference_parity.json", "ddp_empty_targets.json", "ddp.exit_code"],
            "round2_mpformer_smoke_20261026": ["metrics.json", "mp_diagnostics.jsonl", "inference_subset/paired_metrics.json",
                "inference_subset/round2_diagnostics.json"],
            "round2_mpformer_20261026": ["metrics.json", "mp_diagnostics.jsonl", "best_metrics.json", "latest_eval_metrics.json"],
        }
        for run, names in completed.items():
            for name in names:
                evidence("experiments/mask2former/" + run + "/" + name)
        for name in ("d_continue6_coverage_20261026.json", "d_val_candidate_coverage_20261025.json"):
            evidence("reports/analysis/" + name)
        status = json.loads(subprocess.check_output(["python", str(PROJECT / "tools/mpformer_status.py")], text=True))
        add("evidence/current_status.json", json.dumps(status, ensure_ascii=False, indent=2).encode())
        versions = {}
        for name in ("torch", "torchvision", "mmcv", "numpy", "scipy", "Pillow", "opencv-python", "pycocotools",
                     "fvcore", "iopath", "timm", "PyYAML"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not a registered distribution; may be provided by source"
        env = {"captured_utc": now(), "python": platform.python_version(), "system": platform.platform(),
               "packages": versions, "project_root": str(ROOT), "gpu": "4 x Iluvatar BI-V150, 32GB each",
               "policy": "Inventory only; do not install/upgrade the underlying environment"}
        add("evidence/environment.json", json.dumps(env, ensure_ascii=False, indent=2).encode())
        gpu = subprocess.check_output([str(PROJECT / "run.sh"), "ixsmi"], text=True)
        add("evidence/ixsmi.txt", gpu.encode())
        metadata = {"captured_start_utc": captured_start, "captured_end_utc": now(), "server_root": str(ROOT),
                    "files": records, "excluded_counts": dict(excluded),
                    "scope": "Source, dependency source, configs, docs, train statistics and selected small evidence",
                    "excluded": "Raw datasets, full annotations, weights, predictions, submissions, caches, binaries, credentials"}
        stream.writestr("SERVER_MANIFEST.json", json.dumps(metadata, ensure_ascii=False, indent=2).encode())
    result = {"archive": str(archive), "bytes": archive.stat().st_size,
              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": len(records),
              "snapshot_status": {k: status.get(k) for k in ("checked_utc", "status", "stage", "completed_iterations", "total_iterations")}}
    (out / "server_archive.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

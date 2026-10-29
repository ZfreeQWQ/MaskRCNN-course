import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .config import PROJECT_ROOT


def snapshot(cfg, args):
    """Save every invocation separately, including resumes/evaluations and exact source hashes."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dest = Path(cfg.OUTPUT_DIR) / "snapshots" / stamp
    dest.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for folder in ("rgbd_seg", "configs", "tools"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix not in (".py", ".yaml", ".sh"):
                continue
            relative = path.relative_to(PROJECT_ROOT)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("run.sh", "weights/source.json"):
        path = PROJECT_ROOT / relative
        if path.exists():
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dest / "resolved_config.yaml").write_text(cfg.dump())
    # Mask R-CNN configs (rgbd_seg/config.py) use PROJECT.DEPTH_STATS; Mask2Former configs
    # (rgbd_seg/m2f/config.py) use PROJECT.DEPTH_GEOMETRY_STATS -- different encodings of depth
    # (per-pixel normalized depth vs. plane-fit geometry), so accept either field name here.
    stats_path = getattr(cfg.PROJECT, "DEPTH_STATS", None) or getattr(cfg.PROJECT, "DEPTH_GEOMETRY_STATS", None)
    stats = Path(stats_path) if stats_path else None
    if cfg.PROJECT.USE_DEPTH and stats and stats.exists():
        shutil.copy2(stats, dest / stats.name)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    vendor = PROJECT_ROOT.parent / "detectron2"
    vendor_revision = None
    try:
        vendor_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, text=True).strip()
        vendor_diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=vendor)
        (dest / "detectron2.patch").write_bytes(vendor_diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    import torch
    info = {"timestamp_utc": stamp, "command": sys.argv, "arguments": vars(args), "seed": cfg.SEED,
            "git_revision": revision, "detectron2_git_revision": vendor_revision, "source_sha256": hashes, "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda_compatibility": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            "resume_semantics": "model, optimizer, scheduler, iteration and best selection restored; dataloader worker RNG/order not bit-exact"}
    (dest / "manifest.json").write_text(json.dumps(info, indent=2))

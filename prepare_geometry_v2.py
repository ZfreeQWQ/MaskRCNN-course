"""Cache only supplied intrinsics; estimate normalization from train RGB-D only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from course.common import atomic_json
from course.geometry_v2 import CHANNELS, GEOMETRY_VERSION, encode_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data_cache")
    parser.add_argument("--dataset", default="/workspace/blender")
    parser.add_argument("--stats-images", type=int, default=256)
    args = parser.parse_args()
    cache, root = Path(args.cache), Path(args.dataset)
    index = json.loads((cache / "index.json").read_text())
    signature = hashlib.sha256((cache / "index.json").read_bytes()).hexdigest()
    metadata_path = cache / "geometry_v2_intrinsics.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert metadata["index_sha256"] == signature
    else:
        cameras = {}
        for split in ("train", "val"):
            paths = sorted((root / "annotations" / split).glob("*.json.gz"))
            expected = {r["image_id"] for r in index[split]}
            for i, path in enumerate(paths):
                with gzip.open(path, "rt") as stream:
                    payload = json.load(stream)
                for image in payload["images"]:
                    assert image["id"] in expected
                    camera = image["camera"]
                    K = np.asarray(camera["K"], np.float32).reshape(3, 3)
                    cameras[str(image["id"])] = dict(K=K.tolist(), split=split)
                if (i + 1) % 100 == 0:
                    print("INTRINSICS", split, i + 1, len(paths), flush=True)
        assert len(cameras) == sum(len(index[s]) for s in ("train", "val"))
        unique = {tuple(np.asarray(v["K"]).reshape(-1)) for v in cameras.values()}
        metadata = dict(index_sha256=signature, cameras=cameras, unique_intrinsics=len(unique),
                        source="supplied train/val image.camera.K only; no GT geometry/extrinsics")
        atomic_json(metadata_path, metadata)
    stats_path = cache / "geometry_v2_stats.json"
    if stats_path.exists():
        existing = json.loads(stats_path.read_text())
        assert existing["index_sha256"] == signature and existing["images"] == args.stats_images
        if existing.get("geometry_version") == GEOMETRY_VERSION and existing.get("accumulation_dtype") == "float64":
            print("EXISTING_STATS", json.dumps(existing), flush=True)
            return
        # Retain the preflight draft for audit; no formal run consumed it.
        backup = cache / "geometry_v2_stats.float32_draft.json"
        if backup.exists():
            raise FileExistsError("Inspect previous draft before regeneration: " + str(backup))
        atomic_json(backup, existing)
    records = index["train"]
    chosen = np.unique(np.linspace(0, len(records)-1, min(args.stats_images, len(records)), dtype=int))
    samples, diagnostics, ids = [], [], []
    started = time.monotonic()
    for i, position in enumerate(chosen):
        record = records[int(position)]
        assert record["split"] == "train"
        raw = np.asarray(Image.open(record["depth_path"]), np.uint16)
        depth = raw.astype(np.float32) * record["depth_scale_m"]
        depth[raw == record["depth_invalid_code"]] = 0
        geometry, audit = encode_geometry(depth, metadata["cameras"][str(record["image_id"])]["K"], record["image_id"])
        sampled = geometry[::8, ::8].reshape(-1, 7)
        samples.append(sampled[sampled[:, 6] > .5])
        diagnostics.append(audit)
        ids.append(record["image_id"])
        if (i+1) % 32 == 0:
            print("GEOMETRY_STATS", i+1, len(chosen), "seconds", time.monotonic()-started, flush=True)
    points = np.concatenate(samples)
    # Millions of nearly constant normal/depth samples need float64 reduction.
    mean, std = points.mean(0, dtype=np.float64), points.std(0, dtype=np.float64)
    mean[6], std[6] = 0., 1.
    std[:6] = np.maximum(std[:6], [.01, .002, .05, .05, .05, .005])
    result = dict(index_sha256=signature, channels=CHANNELS, images=len(ids), image_ids=ids,
                  geometry_version=GEOMETRY_VERSION, accumulation_dtype="float64",
                  mean=mean.tolist(), std=std.tolist(),
                  train_only=True, normalization="fixed train subset; no validation/test statistics",
                  plane_valid_fraction=float(np.mean([d["plane_valid"] for d in diagnostics])),
                  mean_plane_support=float(np.mean([d["plane_support"] for d in diagnostics])),
                  unique_intrinsics=metadata["unique_intrinsics"], elapsed_seconds=time.monotonic()-started)
    atomic_json(stats_path, result)
    print("GEOMETRY_READY", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

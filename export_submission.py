"""Inference on supplied test RGB-D only; never trains or evaluates test labels."""
import argparse
import csv
import json
import os
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from detectron2.modeling import build_model

from course.common import configuration, atomic_json
from course.experiments import mapper as build_mapper
from course.submission import test_records, attach_constant_test_intrinsics, validate_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/blender")
    parser.add_argument("--config", default="configs/blender_rgbd_r50.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--cache", default="data_cache")
    parser.add_argument("--metadata")
    parser.add_argument("--assume-constant-training-intrinsics", action="store_true")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; never upload an incomplete shard")
    args = parser.parse_args()
    assert 0 <= args.score_threshold <= 1
    assert args.shards > 0 and 0 <= args.shard_index < args.shards and args.limit >= 0
    torch.set_num_threads(4)
    root = Path(args.dataset)
    metadata = Path(args.metadata) if args.metadata else root / "test_images.json"
    records = test_records(root, metadata)[args.shard_index::args.shards]
    if args.limit:
        records = records[:args.limit]
    import fcntl
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "default")
    lock_name = "evaluation_gpu_" + hashlib.sha256(gpu.encode()).hexdigest()[:12] + ".lock"
    lock = (Path(args.cache).resolve() / lock_name).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = configuration(args.config)
    model = build_model(cfg).eval()
    state = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    del state
    mapper = build_mapper(cfg, training=False, cache=args.cache)
    camera_assumption = None
    if cfg.COURSE.V2.GEOMETRY:
        if not args.assume_constant_training_intrinsics:
            raise ValueError("Test metadata lacks K; explicitly acknowledge the fixed training-camera assumption")
        camera_assumption = attach_constant_test_intrinsics(mapper, records)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.with_suffix(".tmp.csv").exists():
        raise FileExistsError("Refusing to overwrite an existing submission: " + str(destination))
    temporary = destination.with_suffix(".tmp.csv")
    started = time.monotonic()
    progress = destination.with_suffix(".status.json")
    atomic_json(progress, dict(status="running", processed=0, total=len(records), gpu=gpu))
    with temporary.open("x", newline="", encoding="utf-8") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(["image_id", "PredictionString"])
        for n, record in enumerate(records):
            image = mapper(record)
            assert image["image"].shape[-2:] == (cfg.COURSE.TEST_SIZE, cfg.COURSE.TEST_SIZE)
            instances = model([image])[0]["instances"].to("cpu")
            assert len(instances) <= 100 and instances.pred_masks.shape[-2:] == (1024, 1024)
            predictions = []
            for j in range(len(instances)):
                assert int(instances.pred_classes[j]) == 0
                if float(instances.scores[j]) < args.score_threshold:
                    continue
                rle = mask_utils.encode(np.asfortranarray(instances.pred_masks[j].numpy().astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                predictions.append(dict(category_id=1, score=float(instances.scores[j]), segmentation=rle))
            validate_prediction(predictions, decode=n == 0)
            writer.writerow([record["image_id"], json.dumps(predictions, separators=(",", ":"), allow_nan=False)])
            if (n + 1) % 25 == 0 or n == 0:
                stream.flush()
                atomic_json(progress, dict(status="running", processed=n+1, total=len(records),
                                          gpu=gpu, elapsed_seconds=time.monotonic()-started))
                print("TEST_IMAGE", n + 1, "/", len(records), flush=True)
    os.replace(temporary, destination)
    atomic_json(progress, dict(status="completed", processed=len(records), total=len(records),
                gpu=gpu, elapsed_seconds=time.monotonic()-started,
                config=cfg.dump(), checkpoint_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                camera_assumption=camera_assumption, score_threshold=args.score_threshold,
                shard_index=args.shard_index, shards=args.shards, smoke_only=bool(args.limit)))
    print("SUBMISSION", destination, "ROWS", len(records), flush=True)


if __name__ == "__main__":
    main()

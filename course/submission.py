"""Machine-readable competition CSV contract, independent of training code."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def test_records(dataset, metadata):
    root = Path(dataset).resolve()
    images = json.loads(Path(metadata).read_text())["images"]
    ids = [im["id"] for im in images]
    if not images or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate test IDs")
    records = []
    for im in images:
        if im["height"] != 1024 or im["width"] != 1024:
            raise ValueError("Competition output canvas must be 1024 square")
        paths = [(root / im[k]).resolve() for k in ("file_name", "depth_file_name")]
        if any(not p.is_relative_to(root) or not p.is_file() for p in paths):
            raise ValueError("Missing test input or path outside dataset")
        if any(not p.is_relative_to(root / folder / "test")
               for p, folder in zip(paths, ("rgb", "depth"))):
            raise ValueError("Submission inference must use test RGB-D only")
        scale = float(im["depth_scale_m"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid supplied depth scale")
        records.append(dict(image_id=im["id"], file_name=str(paths[0]), depth_path=str(paths[1]),
                            height=1024, width=1024, depth_scale_m=scale,
                            depth_invalid_code=int(im.get("depth_invalid_code", 0)), split="test"))
    return records


def attach_constant_test_intrinsics(mapper, records):
    """Explicit inference assumption only; never mutate train/val cache or stats."""
    cameras = mapper.cameras["cameras"]
    train = [c["K"] for c in cameras.values() if c["split"] == "train"]
    unique = {tuple(np.asarray(k).reshape(-1)) for k in train}
    if len(unique) != 1 or not train:
        raise ValueError("Cannot infer test K: training intrinsics are not unique")
    K = np.asarray(train[0], dtype=float).tolist()
    if any(not np.array_equal(c["K"], K) for c in cameras.values()):
        raise ValueError("Train/val cameras differ; a test K assumption is unsafe")
    cameras.update({str(r["image_id"]): dict(K=K, split="test") for r in records})
    return dict(K=K, source="unique supplied training K; same in val",
                assumption="test uses the same fixed camera; test metadata supplies no K",
                statistics="unchanged train-only geometry normalization")


def validate_prediction(prediction, decode=False):
    if not isinstance(prediction, list) or len(prediction) > 100:
        raise ValueError("PredictionString must be a JSON list with at most100 instances")
    for item in prediction:
        if item.get("category_id") != 1:
            raise ValueError("Only category1 is allowed")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid confidence")
        rle = item["segmentation"]
        if rle["size"] != [1024, 1024] or not isinstance(rle["counts"], str):
            raise ValueError("Expected compressed COCO RLE on1024x1024 canvas")
        encoded = dict(size=rle["size"], counts=rle["counts"].encode("ascii"))
        area = int(mask_utils.area(encoded))
        if not 0 <= area <= 1024**2:
            raise ValueError("Invalid mask area")
        if decode:
            mask = mask_utils.decode(encoded)
            if mask.shape != (1024, 1024) or int(mask.sum()) != area:
                raise ValueError("RLE roundtrip failed")


def merge_shards(shards, metadata, output):
    """Check every JSON/RLE, IDs and order before publishing the final CSV."""
    csv.field_size_limit(64 * 1024 * 1024)
    expected = [str(im["id"]) for im in json.loads(Path(metadata).read_text())["images"]]
    rows = {}
    total = 0
    maximum = 0
    for path in shards:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_id", "PredictionString"]:
                raise ValueError("Wrong CSV columns")
            for row in reader:
                key = row["image_id"]
                if key in rows or key not in expected or None in row:
                    raise ValueError("Duplicate, extra ID or malformed CSV")
                prediction = json.loads(row["PredictionString"])
                validate_prediction(prediction, decode=(len(rows) % 100 == 0))
                total += len(prediction)
                maximum = max(maximum, len(prediction))
                rows[key] = row["PredictionString"]
    if set(rows) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("Missing test rows or duplicate metadata IDs")
    output = Path(output)
    temporary = output.with_suffix(".tmp.csv")
    if output.exists() or temporary.exists():
        raise FileExistsError("Never overwrite an earlier submission")
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "PredictionString"])
        writer.writerows((key, rows[key]) for key in expected)
    temporary.rename(output)
    return dict(rows=len(rows), instances=total, max_instances_per_image=maximum,
                every_json_and_rle_area_checked=True, decoded_every_100th_image=True,
                mask_size=[1024, 1024], image_ids_exact_metadata_order=True)

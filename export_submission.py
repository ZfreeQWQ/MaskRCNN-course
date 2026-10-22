"""Inference on supplied test RGB-D only; never trains or evaluates test labels."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from course.common import configuration
from course.data import RGBDMapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/workspace/blender")
    parser.add_argument("--config", default="configs/blender_rgbd_r50.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()
    assert 0 <= args.score_threshold <= 1
    torch.set_num_threads(4)
    root = Path(args.dataset)
    metadata = json.loads((root / "test_images.json").read_text())["images"]
    assert len(metadata) == len({item["id"] for item in metadata}), "Duplicate test image IDs"
    cfg = configuration(args.config)
    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(args.weights)
    mapper = RGBDMapper(cfg, training=False)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing submission: " + str(destination))
    temporary = destination.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        writer.writerow(["image_id", "PredictionString"])
        for n, image in enumerate(metadata):
            assert image["width"] == image["height"] == 1024
            record = dict(image_id=image["id"], file_name=str(root / image["file_name"]),
                          depth_path=str(root / image["depth_file_name"]), height=1024, width=1024,
                          depth_scale_m=float(image["depth_scale_m"]),
                          depth_invalid_code=int(image.get("depth_invalid_code", 0)))
            instances = model([mapper(record)])[0]["instances"].to("cpu")
            assert len(instances) <= 100 and instances.pred_masks.shape[-2:] == (1024, 1024)
            predictions = []
            for j in range(len(instances)):
                assert int(instances.pred_classes[j]) == 0
                if float(instances.scores[j]) < args.score_threshold:
                    continue
                rle = mask_utils.encode(np.asfortranarray(instances.pred_masks[j].numpy().astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                predictions.append(dict(category_id=1, score=float(instances.scores[j]), segmentation=rle))
            writer.writerow([image["id"], json.dumps(predictions, separators=(",", ":"))])
            if (n + 1) % 100 == 0:
                print("TEST_IMAGE", n + 1, "/", len(metadata), flush=True)
    os.replace(temporary, destination)
    print("SUBMISSION", destination, "ROWS", len(metadata), flush=True)


if __name__ == "__main__":
    main()

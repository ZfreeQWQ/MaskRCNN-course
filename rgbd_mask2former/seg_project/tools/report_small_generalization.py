#!/usr/bin/env python
"""COCO small-object AP and one-to-one matched recall slices, val only."""
import argparse
from collections import defaultdict
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def report(annotations, predictions):
    if Path(annotations).name != "instances_val.json":
        raise ValueError("This analysis only accepts instances_val.json; no test labels")
    coco = COCO(str(annotations))
    pred = coco.loadRes(str(predictions))
    # Follow COCO mask AP: remove bbox from result objects before area-based evaluation.
    for ann in pred.anns.values():
        ann.pop("bbox", None)
        ann["area"] = float(__import__("pycocotools.mask", fromlist=["area"]).area(ann["segmentation"]))
    evaluation = COCOeval(coco, pred, "segm")
    evaluation.params.maxDets = [1, 10, 100]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    # Each row uses COCO's score-ordered, one-to-one matching, not oracle coverage.
    groups = defaultdict(lambda: {"n": 0, "hits": np.zeros(3, dtype=np.int64)})
    image_info = coco.imgs
    for image in evaluation.evalImgs:
        if image is None or image["aRng"] != [0, 1024]:
            continue
        for col, gt_id in enumerate(image["gtIds"]):
            if image["gtIgnore"][col]:
                continue
            ann = coco.anns[gt_id]
            area = float(ann["area"])
            box = ann["bbox"]
            fill = area / max(float(box[2] * box[3]), 1.)
            names = ["all_small", "area_le_256" if area <= 256 else "area_257_1024",
                     "fill_lt_0.25" if fill < .25 else "fill_ge_0.25",
                     "semantic:" + str(ann.get("semantic_type", "unknown")),
                     "density:" + str(image_info[ann["image_id"]].get("density", "unknown"))]
            hits = image["gtMatches"][[0, 5, 8], col] > 0  # .50, .75, .90
            for name in names:
                groups[name]["n"] += 1
                groups[name]["hits"] += hits
    sliced = {name: {"small_gt": g["n"], "matched_recall_max100": {
        str(iou): round(float(value / g["n"]), 6) for iou, value in zip((.5, .75, .9), g["hits"])}}
        for name, g in sorted(groups.items()) if g["n"]}
    return {"split": "val", "images": len(coco.imgs),
            "AP": float(evaluation.stats[0] * 100), "APs": float(evaluation.stats[3] * 100),
            "ARs": float(evaluation.stats[9] * 100), "slices": sliced,
            "annotations_sha256": sha(annotations), "predictions_sha256": sha(predictions),
            "matching": "COCO score-ordered one-to-one; mask IoU; maxDets=100",
            "test_annotations_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = report(args.annotations, args.predictions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))

"""Validation-only small-mask diagnostics. No labels enter model inference."""
import numpy as np
from pycocotools import mask as mask_utils


def mean_valid(array):
    values = np.asarray(array)
    values = values[values > -1]
    return float(values.mean()) if values.size else -1.


def coco_small_metrics(evaluator):
    p = evaluator.params
    a = p.areaRngLbl.index("small")
    m = p.maxDets.index(100)
    result = {"mask_ars": float(evaluator.stats[9]),
              "mask_ar100": float(evaluator.stats[8])}
    for threshold in (.5, .75):
        t = int(np.flatnonzero(np.isclose(p.iouThrs, threshold))[0])
        result["mask_aps" + str(round(threshold*100))] = mean_valid(evaluator.eval["precision"][t, :, :, a, m])
    return result


def diagnose_small(coco, predictions, ids):
    by_image = {}
    for p in predictions:
        by_image.setdefault(p["image_id"], []).append(p)
    rows = []
    for image_id in ids:
        gt = [a for a in coco.imgToAnns[image_id] if not a.get("iscrowd", 0) and 0 < a["area"] <= 1024]
        if not gt:
            continue
        pred = by_image.get(image_id, [])
        ious = (mask_utils.iou([p["segmentation"] for p in pred], [coco.annToRLE(a) for a in gt], [0]*len(gt))
                if pred else np.zeros((0, len(gt))))
        for j, ann in enumerate(gt):
            i = int(ious[:, j].argmax()) if len(pred) else None
            best = float(ious[i, j]) if i is not None else 0.
            _, _, w, h = ann["bbox"]
            rows.append(dict(image_id=image_id, annotation_id=ann["id"], area=ann["area"],
                             aspect_ratio=max(w/max(h, 1), h/max(w, 1)),
                             fill_ratio=ann["area"]/max(w*h, 1), best_iou=best,
                             best_prediction_score=float(pred[i]["score"]) if i is not None else 0.))

    def summarize(selected):
        values = np.array([r["best_iou"] for r in selected])
        return dict(instances=len(selected), mean_best_iou=float(values.mean()) if len(values) else None,
                    fraction_best_iou_ge50=float((values >= .5).mean()) if len(values) else None,
                    fraction_best_iou_ge75=float((values >= .75).mean()) if len(values) else None,
                    fraction_best_iou_lt10=float((values < .1).mean()) if len(values) else None)

    groups = {"all_small": rows,
              "area_le64": [r for r in rows if r["area"] <= 64],
              "area65_256": [r for r in rows if 64 < r["area"] <= 256],
              "area257_1024": [r for r in rows if 256 < r["area"] <= 1024],
              "elongated_gt3": [r for r in rows if r["aspect_ratio"] > 3],
              "sparse_fill_lt25": [r for r in rows if r["fill_ratio"] < .25]}
    return dict(note="Independent best IoU over predicted masks is an oracle upper-bound diagnostic, NOT COCO recall/AP; a prediction can be best for several GTs.",
                groups={k: summarize(v) for k, v in groups.items()},
                lowest_overlap_examples=sorted(rows, key=lambda r: (r["best_iou"], -r["area"]))[:30],
                partial_boundary_examples=[r for r in rows if .5 <= r["best_iou"] < .75][:30])

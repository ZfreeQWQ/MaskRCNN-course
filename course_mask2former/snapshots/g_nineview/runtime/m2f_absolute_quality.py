"""Fixed CPU absolute-IoU regressor. Inference accepts predictions only."""
import copy
import math
import numpy as np
from pycocotools import mask as mu
import m2f_quality_selector as ridge
import m2f_cached_geometry_refine as geometry

FEATURES = ("score", "is_local", "log_area", "fill", "log_aspect",
            "center_x", "center_y", "global_iou", "global_score",
            "large_containment", "large_score", "log_area_ratio", "image_edge")


def features(base, rows):
    if not rows:
        return np.empty((0, 13), dtype=np.float64)
    masks = [r["segmentation"] for r in rows]
    areas = np.asarray(mu.area(masks), dtype=float)
    boxes = np.asarray(mu.toBbox(masks), dtype=float)
    assert (areas > 0).all() and all(m["size"] == [1024, 1024] for m in masks)
    keys = {geometry.key(r) for r in base}
    bm = [r["segmentation"] for r in base]
    ba = np.asarray(mu.area(bm), dtype=float) if base else np.empty(0)
    overlap = mu.iou(masks, bm, [0]*len(bm)) if base else np.zeros((len(rows), 0))
    result = []
    for k, row in enumerate(rows):
        same = [j for j, b in enumerate(base) if b["category_id"] == row["category_id"]]
        best = max(same, key=lambda j: (overlap[k, j], -j)) if same else None
        large = [j for j in same if ba[j] > 1024 and base[j]["score"] >= .3
                 and geometry.key(base[j]) != geometry.key(row)]
        ratios = {j: float(overlap[k,j]*(areas[k]+ba[j])/(1+overlap[k,j])/areas[k])
                  for j in large}
        container = max(large, key=lambda j: (ratios[j], -j)) if large else None
        x, y, w, h = boxes[k]
        assert w > 0 and h > 0 and math.isfinite(row["score"]) and 0 <= row["score"] <= 1
        result.append([row["score"], float(geometry.key(row) not in keys), math.log1p(areas[k]),
            areas[k]/(w*h), math.log(w/h), (x+w/2)/1024, (y+h/2)/1024,
            float(overlap[k,best]) if best is not None else 0.,
            base[best]["score"] if best is not None else 0.,
            ratios[container] if container is not None else 0.,
            base[container]["score"] if container is not None else 0.,
            math.log1p(ba[container]/areas[k]) if container is not None else 0.,
            float(x <= 0 or y <= 0 or x+w >= 1024 or y+h >= 1024)])
    result = np.asarray(result, dtype=np.float64)
    assert result.shape == (len(rows), 13) and np.isfinite(result).all()
    assert (result[:,9] <= 1+1e-9).all()
    return result


def labels(rows, annotations, coco):
    valid = sorted((a for a in annotations if not a.get("iscrowd", 0)
                    and not a.get("ignore", 0)), key=lambda a: a["id"])
    overlap = mu.iou([r["segmentation"] for r in rows],
                    [coco.annToRLE(a) for a in valid], [0]*len(valid)) if valid and rows else None
    result = []
    for k, row in enumerate(rows):
        same = [j for j,a in enumerate(valid) if a["category_id"] == row["category_id"]]
        j = max(same, key=lambda j: (overlap[k,j], -j)) if same else None
        value = float(overlap[k,j]) if j is not None else 0.
        matched = valid[j] if value > 0 else None
        result.append(dict(target_iou=value, gt_id=matched["id"] if matched else None,
                           gt_area=matched["area"] if matched else None))
    return result


def fit(x, y):
    model = ridge.fit(x, y)
    model.update(feature_names=list(FEATURES), kind="absolute_mask_iou_quadratic_ridge",
                 target="best_same_category_nonignored_gt_mask_iou", threshold=None)
    return model


def predict(x, model):
    assert model["feature_names"] == list(FEATURES) and model["lambda_l2"] == 1
    raw = ridge.design(x, model["mean"], model["scale"]) @ np.asarray(model["coefficients"])
    raw += model["intercept"]
    assert np.isfinite(raw).all()
    return np.clip(raw, 0., 1.)


def rescore(base, rows, model):
    scores = predict(features(base, rows), model) if rows else []
    result = copy.deepcopy(rows)
    for row, score in zip(result, scores):
        row["score"] = float(score)
    order = sorted(range(len(result)), key=lambda i: (-result[i]["score"], i))
    result = [result[i] for i in order]
    for row, i in zip(result, order):
        assert {k:v for k,v in row.items() if k != "score"} == {
            k:v for k,v in rows[i].items() if k != "score"}
    assert len(result) == len(rows) <= 100
    return result, order

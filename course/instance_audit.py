"""Validation-only diagnostic counterfactuals; never inference features or scores.

Density strata are declared before looking at model scores. Oracle overlap may
reuse predictions and is not detector recall. Oracle ranking is label-informed,
not a deployable method or a mathematical upper bound on AP.
"""
import contextlib
import copy
import io
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from course.diagnostics_v3 import coco_small_metrics


def evaluate(coco, predictions, ids):
    if not predictions:
        return dict(mask_aps=0., mask_aps50=0., mask_aps75=0., mask_ars=0.), None
    with contextlib.redirect_stdout(io.StringIO()):
        # COCO.loadRes mutates its input with ndarray bbox/area/id fields. A
        # repeated load then takes the bbox branch and some versions raise on
        # ndarray == []. Rebuild the original segmentation-only schema.
        clean = [{k: copy.deepcopy(p[k]) for k in ("image_id", "category_id", "score", "segmentation")}
                 for p in predictions]
        evaluator = COCOeval(coco, coco.loadRes(clean), "segm")
        evaluator.params.imgIds = list(ids)
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return dict(mask_aps=float(evaluator.stats[3]), **coco_small_metrics(evaluator)), evaluator


def overlap_summary(rows):
    values = np.array([r["best_iou"] for r in rows])
    return dict(instances=len(rows), counts_best_iou={str(t): int((values >= t).sum())
                for t in (.1, .5, .75, .85, .9, .95)},
                adequate_iou50_but_best_score_lt01=sum(r["best_iou"] >= .5 and r["adequate_score50"] < .1 for r in rows))


def audit_instances(coco, predictions, ids):
    ids = list(ids)
    assert len(ids) == len(set(ids)) and set(ids) <= set(coco.imgs)
    assert all(p["image_id"] in set(ids) and p["category_id"] == 1 for p in predictions)
    by_image = {i: [] for i in ids}
    for p in predictions:
        by_image[p["image_id"]].append(p)
    rows, image_rows, ranked = [], [], []
    strata = {"instances_le20": [], "instances21_40": [], "instances_gt40": []}
    for image_id in ids:
        gt = [a for a in coco.imgToAnns[image_id] if not a.get("iscrowd", 0) and a["area"] > 0]
        pred = by_image[image_id]
        assert len(pred) <= 100
        ious = (np.asarray(mask_utils.iou([p["segmentation"] for p in pred],
                     [coco.annToRLE(a) for a in gt], [0]*len(gt)))
                if pred and gt else np.zeros((len(pred), len(gt))))
        areas = np.array([float(mask_utils.area(p["segmentation"])) for p in pred])
        scores = np.array([p["score"] for p in pred])
        gt_areas = np.array([a["area"] for a in gt])
        density = "instances_le20" if len(gt) <= 20 else "instances21_40" if len(gt) <= 40 else "instances_gt40"
        strata[density].append(image_id)
        duplicate_extras = 0
        merge_like = 0
        if pred and gt:
            best_gt = ious.argmax(1)
            best_iou = ious.max(1)
            for g in range(len(gt)):
                duplicate_extras += max(0, int(((best_gt == g) & (best_iou >= .5) & (scores >= .1)).sum())-1)
            intersections = ious*(areas[:, None]+gt_areas[None, :])/(1+ious)
            coverage = intersections/gt_areas[None, :]
            merge_like = int(((coverage >= .5).sum(1) >= 2)[(scores >= .1) & (best_iou < .5)].sum())
        for j, ann in enumerate(gt):
            if ann["area"] > 1024:
                continue
            _, _, w, h = ann["bbox"]
            adequate = scores[ious[:, j] >= .5]
            rows.append(dict(image_id=image_id, annotation_id=ann["id"], area=ann["area"], density=density,
                             fill_ratio=ann["area"]/max(w*h, 1),
                             best_iou=float(ious[:, j].max()) if pred else 0.,
                             adequate_score50=float(adequate.max()) if adequate.size else 0.))
        image_rows.append(dict(image_id=image_id, gt_instances=len(gt), small_instances=sum(a["area"] <= 1024 for a in gt),
                               nonempty_predictions=int((areas > 0).sum()), duplicate_extras_score_ge01=duplicate_extras,
                               merge_like_score_ge01=merge_like))
        # Same predictions and top100, only replace scores using validation GT.
        # All GT sizes used, not just small, to avoid deliberately suppressing
        # valid medium/large detections. This remains a non-deployable probe.
        for j, p in enumerate(pred):
            item = copy.deepcopy(p)
            item["score"] = float(ious[j].max()) if gt else 0.
            ranked.append(item)
    baseline, evaluator = evaluate(coco, predictions, ids)
    oracle_ranked, _ = evaluate(coco, ranked, ids)
    matched = {}
    if evaluator is not None:
        p = evaluator.params
        small_range = p.areaRng[p.areaRngLbl.index("small")]
        for threshold in (.5, .75, .9, .95):
            t = int(np.flatnonzero(np.isclose(p.iouThrs, threshold))[0])
            total, count, tp_scores = 0, 0, []
            for record in evaluator.evalImgs:
                if record is None or list(record["aRng"]) != list(small_range):
                    continue
                eligible = ~np.asarray(record["gtIgnore"], dtype=bool)
                total += int(eligible.sum())
                count += int(((record["gtMatches"][t] > 0) & eligible).sum())
                matched_dt = (record["dtMatches"][t] > 0) & ~record["dtIgnore"][t]
                tp_scores.extend(np.array(record["dtScores"])[matched_dt].tolist())
            matched[str(threshold)] = dict(gt_small=total, matched_gt=count,
                matched_score_median=float(np.median(tp_scores)) if tp_scores else None,
                matched_score_lt01=sum(v < .1 for v in tp_scores))
    density_metrics = {}
    for name, selected in strata.items():
        subset_predictions = [p for p in predictions if p["image_id"] in set(selected)]
        metrics, _ = evaluate(coco, subset_predictions, selected)
        density_metrics[name] = dict(images=len(selected), **metrics,
                oracle_overlap=overlap_summary([r for r in rows if r["density"] == name]))
    return dict(note="Fixed val diagnostic only. Oracle overlaps reuse queries; merge-like and duplicate heuristics are not causal labels. Oracle ranking uses GT, is not an AP bound and must never be used for test/inference.",
        baseline=baseline, label_informed_ranking_probe=oracle_ranked,
        ranking_aps_delta=oracle_ranked["mask_aps"]-baseline["mask_aps"],
        groups={"all_small": overlap_summary(rows),
                "area_le64": overlap_summary([r for r in rows if r["area"] <= 64]),
                "area65_256": overlap_summary([r for r in rows if 64 < r["area"] <= 256]),
                "area257_1024": overlap_summary([r for r in rows if r["area"] > 256]),
                "sparse_fill_lt25": overlap_summary([r for r in rows if r["fill_ratio"] < .25])},
        coco_one_to_one_small_matches=matched, density=density_metrics,
        per_image=image_rows, per_small_gt=rows)

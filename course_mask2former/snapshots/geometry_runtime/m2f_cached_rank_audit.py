"""CPU-only paired audit of frozen final predictions. No torch/model/inference.

Fixed 256-image manifest, three diagnostic IoUs, no counterfactual filtering or
threshold optimization. The audit is NOT a new full-val AP endpoint.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

START = 1792405504
DEADLINE = 1792409104
TILES = "outputs/course_m2f_tiles_20261019_v1"
BASE = "outputs/course_m2f_copypaste1536_20261018_v1/evaluate_1536"
MANIFEST = "outputs/course_mrcnn_stages_20261018_v2/manifest.json"
ANN = "data/course_blender_20261013/prepared/annotations/instances_val.json"
IOUS = (.5, .75, .9)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_time(*_):
    if time.time() >= DEADLINE:
        raise TimeoutError("Original 60-minute preparation-inclusive deadline; no extension")


def read(path):
    check_time()
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b""):
            check_time()
            h.update(data)
    return h.hexdigest()


def save(path, data):
    check_time()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def bucket(area):
    return "tiny" if area <= 256 else "small_257_1024" if area <= 1024 else "medium" if area <= 9216 else "large"


def rankbin(rank):
    return "01_10" if rank <= 10 else "11_25" if rank <= 25 else "26_50" if rank <= 50 else "51_100"


def match_all(ious, scores, threshold):
    """All-GT no-crowd matching, independently checked against COCOeval."""
    import numpy as np
    used = set()
    result = [None] * len(scores)
    for d in np.argsort(-np.asarray(scores), kind="stable"):
        best = threshold
        chosen = None
        for g in range(ious.shape[1]):
            if g not in used and ious[d, g] >= best:
                chosen, best = g, float(ious[d, g])
        if chosen is not None:
            used.add(chosen)
            result[d] = chosen
    return result


def fp_kind(best_iou, matched, threshold):
    if matched:
        return "tp"
    if best_iou >= threshold:
        return "duplicate"
    return "localization" if best_iou >= .1 else "background"


def intersection_from_iou(iou, area_d, area_g):
    return iou * (area_d + area_g) / (1.0 + iou)


def self_test():
    import numpy as np
    tests = 0
    for areas, expected in [([1, 256, 257, 1024, 1025, 9216, 9217], ["tiny", "tiny", "small_257_1024", "small_257_1024", "medium", "medium", "large"])]:
        assert list(map(bucket, areas)) == expected
        tests += 1
    assert match_all(np.array([[.8, .0], [.9, .6], [.0, .7]]), [.9, .8, .7], .5) == [0, 1, None]; tests += 1
    assert match_all(np.zeros((2, 0)), [.9, .8], .5) == [None, None]; tests += 1
    assert match_all(np.array([[.5], [.6]]), [.5, .5], .5) == [0, None]; tests += 1
    assert fp_kind(.6, False, .5) == "duplicate" and fp_kind(.2, False, .5) == "localization" and fp_kind(0, False, .5) == "background"; tests += 1
    assert abs(intersection_from_iou(.1, 100, 1000) - 100) < 1e-9; tests += 1
    assert [rankbin(x) for x in (1, 10, 11, 25, 26, 50, 51, 100)] == ["01_10", "01_10", "11_25", "11_25", "26_50", "26_50", "51_100", "51_100"]; tests += 1
    for area in (np.float64(100), np.float64(2000)):
        fragment = bool(area <= 1024 and np.float64(.9) >= .8)
        assert json.loads(json.dumps({"large_fragment": fragment}))["large_fragment"] is (area <= 1024).item()
    tests += 1
    return dict(status="passed", tests=tests, skipped=0)


def evaluate_cached(coco, predictions, ids):
    from pycocotools.cocoeval import COCOeval
    # Match official segmentation evaluation: do not use bbox area for small
    # unmatched-detection ignore semantics. Do not change masks/scores/order.
    rows = [{k: v for k, v in x.items() if k != "bbox"} for x in predictions]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(rows)
        ev = COCOeval(coco, dt, "segm")
        ev.params.imgIds = ids
        ev.params.iouThrs = __import__("numpy").array(IOUS)
        ev.evaluate()
        ev.accumulate()
    out = {}
    for area_idx, label in enumerate(ev.params.areaRngLbl):
        out[label] = {}
        for entry in ev.evalImgs:
            if entry and entry["aRng"] == ev.params.areaRng[area_idx]:
                out[label][entry["image_id"]] = entry
    precision = ev.eval["precision"]
    metrics = {}
    for area_idx, label in enumerate(ev.params.areaRngLbl):
        metrics[label] = {}
        for ti, threshold in enumerate(IOUS):
            valid = precision[ti, :, 0, area_idx, -1]
            valid = valid[valid >= 0]
            metrics[label][str(threshold)] = float(valid.mean()) if len(valid) else None
    return out, dt, metrics


def run(workspace, output):
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    began = time.monotonic()
    assert not any(Path(f"/dev/{x}").exists() for x in ("nvidia0", "nvidia1", "nvidiactl")), "GPU device exposed"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU visibility not disabled"
    tests = self_test()
    save(output / "cpu_tests.json", tests)
    bound = {}

    def bind(relative, expected=None):
        digest = sha(workspace / relative)
        if expected is not None:
            assert digest == expected, relative
        bound[relative] = digest

    bind(TILES + "/evaluate/report.json", "dad09b10fc61467a3068e9aef058af2b8818c89509454a5889411b154a837d92")
    bind(BASE + "/report.json", "b4e4c3dc6ebfbe3a63e88be36be6065388d63987e92323f7dbde0e57100451ee")
    report = read(workspace / TILES / "evaluate/report.json")
    bind(ANN, report["val_annotations_sha256"])
    bind(MANIFEST, report["manifest_sha256"])
    bind(TILES + "/analysis/per_gt.json", "ddbe1d4ad4cb63dbabe84fa573cf63764eb8d312a504188d852ff2ab66c69fa8")
    ids = read(workspace / MANIFEST)["image_ids"]
    assert len(ids) == len(set(ids)) == 256
    input_hashes = read(workspace / TILES / "evaluate/input_hashes.json")
    assert input_hashes == read(workspace / BASE / "input_hashes.json")
    assert [x["image_id"] for x in input_hashes] == report["image_ids"]
    for rel in (TILES + "/evaluate/input_hashes.json", BASE + "/input_hashes.json", BASE + "/predictions.segm.json", TILES + "/evaluate/predictions.segm.json"):
        bind(rel)
    subset = set(ids)
    arms = {}
    origins = {}
    for name, relative in (("baseline", BASE), ("tiles", TILES + "/evaluate")):
        allrows = read(workspace / relative / "predictions.segm.json")
        assert {x["image_id"] for x in allrows} == set(report["image_ids"])
        selected = [x for x in allrows if x["image_id"] in subset]
        del allrows
        by_image = defaultdict(list)
        for x in selected:
            assert x["category_id"] == 1 and 0 <= x["score"] <= 1 and math.isfinite(x["score"])
            assert x["segmentation"]["size"] == [1024, 1024]
            by_image[x["image_id"]].append(x)
        assert set(by_image) == subset and all(len(x) <= 100 for x in by_image.values())
        arms[name] = by_image
    for identity in ids:
        rel = f"{TILES}/evaluate/subset/{identity}.json"
        bind(rel, report["capture_sha256"][f"subset/{identity}.json"])
        capture = read(workspace / rel)
        cached = arms["tiles"][identity]
        assert len(cached) == len(capture["predictions"]) == len(capture["origins"])
        for a, b in zip(cached, capture["predictions"]):
            assert a["segmentation"] == b["segmentation"] and abs(a["score"] - b["score"]) < 1e-7
        origins[identity] = capture["origins"]
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(workspace / ANN))
    assert not any(a.get("iscrowd", 0) or a.get("ignore", 0) for a in coco.anns.values())
    all_records, per_gt, metrics = [], {}, {}
    global_identity = Counter()
    for arm, by_image in arms.items():
        check_time()
        ev, dt, metrics[arm] = evaluate_cached(coco, [x for i in ids for x in by_image[i]], ids)
        arm_gt = []
        for n, identity in enumerate(ids):
            check_time()
            rows = by_image[identity]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[identity]))
            ann_ids = [a["id"] for a in anns]
            gt_areas = np.array([a["area"] for a in anns])
            rles = [x["segmentation"] for x in rows]
            areas = np.array(mu.area(rles), dtype=float)
            ious = mu.iou(rles, [coco.annToRLE(a) for a in anns], [0] * len(anns)) if anns else np.zeros((len(rows), 0))
            scores = np.array([x["score"] for x in rows])
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(rows), dtype=int); ranks[order] = np.arange(1, len(rows) + 1)
            eval_all, eval_small = ev["all"][identity], ev["small"][identity]
            assert np.allclose(eval_all["dtScores"], scores[order], atol=0, rtol=0)
            assert len(eval_all["dtIds"]) == len(rows)
            matchers = [match_all(ious, scores, t) for t in IOUS]
            for ti in range(len(IOUS)):
                expected = [ann_ids[matchers[ti][d]] if matchers[ti][d] is not None else 0 for d in order]
                assert expected == eval_all["dtMatches"][ti].tolist(), (arm, identity, ti)
            for d, row in enumerate(rows):
                pos = int(ranks[d]) - 1
                best = int(ious[d].argmax()) if anns else None
                best_iou = float(ious[d, best]) if best is not None else 0.
                intersections = intersection_from_iou(ious[d], areas[d], gt_areas)
                iop = intersections / max(areas[d], 1.)
                large_indices = np.where(gt_areas > 1024)[0]
                fragment_g = int(large_indices[iop[large_indices].argmax()]) if len(large_indices) else None
                origin = origins[identity][d] if arm == "tiles" else dict(view=0, index=d, area=int(areas[d]), tile_edge=False)
                if arm == "tiles" and origin["view"] == 0:
                    old = arms["baseline"][identity][origin["index"]]
                    same = old["segmentation"] == row["segmentation"]
                    global_identity["same_mask" if same else "different_mask"] += 1
                    global_identity["same_score" if abs(old["score"] - row["score"]) <= 1e-6 else "different_score"] += 1
                record = dict(arm=arm, image_id=identity, index=d, rank=int(ranks[d]), rank_bin=rankbin(int(ranks[d])),
                    score=float(scores[d]), area=int(areas[d]), predicted_size=bucket(areas[d]), source="global" if origin["view"] == 0 else "local",
                    view=origin["view"], source_index=origin["index"], tile_edge=origin["tile_edge"],
                    best_iou=best_iou, best_gt_id=ann_ids[best] if best is not None else None,
                    best_gt_area=int(gt_areas[best]) if best is not None else None,
                    large_fragment=bool(areas[d] <= 1024 and fragment_g is not None and iop[fragment_g] >= .8 and best_iou < .5),
                    large_fragment_gt_id=ann_ids[fragment_g] if fragment_g is not None else None,
                    large_fragment_iop=float(iop[fragment_g]) if fragment_g is not None else 0., thresholds={})
                for ti, t in enumerate(IOUS):
                    g = matchers[ti][d]
                    sm = int(eval_small["dtMatches"][ti, pos]); ignored = bool(eval_small["dtIgnore"][ti, pos])
                    record["thresholds"][str(t)] = dict(kind=fp_kind(best_iou, g is not None, t),
                        matched_gt_id=ann_ids[g] if g is not None else None,
                        small_status="ignored" if ignored else "tp" if sm else "fp", small_matched_gt_id=sm or None)
                all_records.append(record)
            for g, ann in enumerate(anns):
                if ann["area"] > 1024:
                    continue
                best = int(ious[:, g].argmax()) if rows else None
                qualifying = [int(d) for d in order if ious[d, g] >= .5]
                matched = [int(d) for d in order if matchers[0][d] == g]
                small_pos = np.where(eval_small["dtMatches"][0] == ann["id"])[0]
                arm_gt.append(dict(annotation_id=ann["id"], image_id=identity, area=ann["area"],
                    best_iou=float(ious[best, g]) if best is not None else 0.,
                    best_rank=int(ranks[best]) if best is not None else None,
                    first_geometric_rank=int(ranks[qualifying[0]]) if qualifying else None,
                    all_tp_rank=int(ranks[matched[0]]) if matched else None,
                    small_tp_rank=int(small_pos[0] + 1) if len(small_pos) else None))
            if (n + 1) % 64 == 0:
                save(output / "status.json", dict(status="running", phase="audit", arm=arm, completed_images=n + 1, updated_at_utc=now()))
                print(json.dumps(dict(arm=arm, completed=n + 1)), flush=True)
        per_gt[arm] = arm_gt
    expected = {x["annotation_id"]: x for x in read(workspace / TILES / "analysis/per_gt.json")}
    paired = []
    gt_maps = {arm: {r["annotation_id"]: r for r in rows} for arm, rows in per_gt.items()}
    assert all(set(x) == set(expected) for x in gt_maps.values()) and len(expected) == 1187
    for gid, ex in expected.items():
        old, new = gt_maps["baseline"][gid], gt_maps["tiles"][gid]
        assert abs(old["best_iou"] - ex["old_final_iou"]) < 1e-8
        assert abs(new["best_iou"] - ex["new_iou"]) < 1e-8
        cohort = "rescued" if ex["new_covered"] and not ex["old_covered"] else "lost" if ex["old_covered"] and not ex["new_covered"] else "both_covered" if ex["new_covered"] else "both_missing"
        paired.append(dict(annotation_id=gid, cohort=cohort, baseline=old, tiles=new))
    summary = summarize(all_records)
    save(output / "per_prediction.json", all_records)
    save(output / "paired_gt.json", paired)
    save(output / "summary.json", summary)
    for relative, digest in bound.items():
        assert sha(workspace / relative) == digest, "Input mutated"
    result = dict(status="passed", finished_at_utc=now(), seconds=time.monotonic() - began,
        preparation_unix=START, deadline_unix=DEADLINE, cpu_tests=tests, image_count=256, small_gt_count=1187,
        subset_diagnostic_AP_at_fixed_IoUs=metrics, global_retained_identity=dict(global_identity),
        cohorts=dict(Counter(x["cohort"] for x in paired)), input_sha256=bound, source_sha256=sha(Path(__file__)),
        artifacts_sha256={f: sha(output / f) for f in ("per_prediction.json", "paired_gt.json", "summary.json")},
        gpu_devices_exposed=False, gpu_calls=0, model_loaded=False, weight_updates=0, parameter_search=False,
        training_started=False, test_inputs_used=False, test_annotations_used=False,
        limitations=["Only fixed256 cached final outputs; diagnostic AP at three IoUs is NOT official full2584 APs@[.50:.95].",
                    "Fragment is geometric IoP>=.8 within a larger GT and IoU<.5, not semantic proof or a deployment filter.",
                    "Discarded local candidates were not saved; exact NMS/top100 cause is unknown.",
                    "No counterfactual filtering, reranking, threshold fitting, new GPU or training was performed."])
    save(output / "report.json", result)
    save(output / "status.json", dict(status="complete", phase="complete", finished_at_utc=now(), report_sha256=sha(output / "report.json")))


def summarize(records):
    """Fixed bins, descriptive only: never select or optimize a threshold."""
    result = {}
    for arm in ("baseline", "tiles"):
        rows = [r for r in records if r["arm"] == arm]
        result[arm] = {}
        scopes = {"all": rows, "score_ge03": [r for r in rows if r["score"] >= .3],
                  "score_ge07": [r for r in rows if r["score"] >= .7], "score_ge09": [r for r in rows if r["score"] >= .9]}
        scopes.update({"source_" + s: [r for r in rows if r["source"] == s] for s in ("global", "local")})
        scopes.update({"rank_" + b: [r for r in rows if r["rank_bin"] == b] for b in ("01_10", "11_25", "26_50", "51_100")})
        scopes.update({"local_rank_" + b: [r for r in rows if r["source"] == "local" and r["rank_bin"] == b] for b in ("01_10", "11_25", "26_50", "51_100")})
        scopes.update({"predicted_" + b: [r for r in rows if r["predicted_size"] == b] for b in ("tiny", "small_257_1024", "medium", "large")})
        for scope, subset in scopes.items():
            result[arm][scope] = dict(count=len(subset), mean_score=sum(r["score"] for r in subset) / len(subset) if subset else None,
                edge=sum(r["tile_edge"] for r in subset), large_fragment=sum(r["large_fragment"] for r in subset),
                thresholds={str(t): dict(kinds=dict(Counter(r["thresholds"][str(t)]["kind"] for r in subset)),
                    small_status=dict(Counter(r["thresholds"][str(t)]["small_status"] for r in subset))) for t in IOUS})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Absolute deadline")))
    signal.alarm(max(1, math.ceil(DEADLINE - time.time())))
    if (args.output / "status.json").exists():
        raise ValueError("No replay or overwrite of existing audit")
    save(args.output / "status.json", dict(status="running", phase="binding", started_at_utc=now()))
    try:
        run(args.workspace, args.output)
    except BaseException as exc:
        data = dict(status="failed", error=repr(exc), traceback=traceback.format_exc(), failed_at_utc=now())
        # Failure evidence remains writeable even at the hard deadline.
        for name in ("first_error.json", "status.json"):
            (args.output / name).write_text(json.dumps(data, indent=2))
        raise

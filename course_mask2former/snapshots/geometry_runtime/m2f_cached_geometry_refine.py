"""One fixed CPU-only geometry refinement of accepted prediction caches."""
import argparse
import contextlib
import copy
from collections import Counter
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

import m2f_cached_rank_audit as audit

# Preparation-inclusive window, conservatively starting before first file read.
START = 1792469220  # 2026-10-20 12:07:00 Asia/Shanghai
DEADLINE = START + 3600
audit.START, audit.DEADLINE = START, DEADLINE
read, save, sha, check, now = audit.read, audit.save, audit.sha, audit.check_time, audit.now
PRIOR = "outputs/course_m2f_rank_audit_20261019_v2"
CONTAIN = "outputs/course_m2f_containment_20261019_v1"
NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl")
KEYS = ("course/segm_mAP", "course/segm_mAP_50", "course/segm_mAP_75",
        "course/segm_mAP_s", "course/segm_mAP_m", "course/segm_mAP_l")


def key(row):
    rle = row["segmentation"]
    return (row["category_id"], float(row["score"]), tuple(rle["size"]), rle["counts"])


def identify_local(base, merged):
    """Exact global provenance excludes ambiguous identical geometries.

    Every retained global prediction must be identical to its original full-view
    cache entry. This invariant is checked against all saved origin captures
    before applying the same rule to the rest of the frozen final cache.
    """
    full_keys = {key(row) for row in base}
    return [i for i, row in enumerate(merged) if key(row) not in full_keys]


def refine(base, merged):
    import numpy as np
    from pycocotools import mask as mu
    result = copy.deepcopy(base)
    if not base or not merged:
        return result, []
    sizes = np.asarray(mu.area([r["segmentation"] for r in base]))
    local_indices = identify_local(base, merged)
    local_indices = [i for i in local_indices if 0 < int(mu.area(merged[i]["segmentation"])) <= 1024]
    small = [i for i, area in enumerate(sizes) if 0 < int(area) <= 1024]
    changes = []
    if small and local_indices:
        overlaps = mu.iou([base[i]["segmentation"] for i in small],
                          [merged[i]["segmentation"] for i in local_indices], [0] * len(local_indices))
        for row, i in enumerate(small):
            choices = [j for j in range(len(local_indices))
                       if float(overlaps[row, j]) >= .5 and merged[local_indices[j]]["category_id"] == base[i]["category_id"]]
            if not choices:
                continue
            j = min(choices, key=lambda k: (-float(overlaps[row, k]), -merged[local_indices[k]]["score"], local_indices[k]))
            source = local_indices[j]
            result[i]["segmentation"] = copy.deepcopy(merged[source]["segmentation"])
            if "bbox" in result[i]:
                result[i]["bbox"] = mu.toBbox(result[i]["segmentation"]).tolist()
            changes.append(dict(index=i, local_index=source, overlap=float(overlaps[row, j])))
    assert len(result) == len(base)
    for old, new in zip(base, result):
        assert {k: v for k, v in old.items() if k not in ("segmentation", "bbox")} == {
            k: v for k, v in new.items() if k not in ("segmentation", "bbox")}
    return result, changes


def tests():
    import numpy as np
    from pycocotools import mask as mu
    n = audit.self_test()["tests"]
    def row(x, y, w, h, score=.8):
        a = np.zeros((80, 80), np.uint8); a[y:y+h, x:x+w] = 1
        rle = mu.encode(np.asfortranarray(a)); rle["counts"] = rle["counts"].decode()
        return dict(image_id=1, category_id=1, score=score, segmentation=rle)
    a, b, c = row(0, 0, 10, 10), row(1, 0, 10, 10, .99), row(60, 60, 10, 10)
    assert refine([], [b]) == ([], []); n += 1
    assert refine([a], []) == ([a], []); n += 1
    assert refine([a], [a]) == ([a], []); n += 1
    assert refine([a], [c]) == ([a], []); n += 1
    old = copy.deepcopy([a, c]); out, changes = refine(old, [b])
    assert old == [a, c] and out[0]["score"] == a["score"] and out[0]["segmentation"] == b["segmentation"] and out[1] == c
    assert changes[0]["index"] == 0 and len(out) == 2; n += 1
    big = row(0, 0, 40, 40)
    assert refine([big], [b]) == ([big], []); n += 1
    assert refine([a], [big]) == ([a], []); n += 1
    low = copy.deepcopy(b); low["score"] = .2
    assert refine([a], [low, b])[1][0]["local_index"] == 1; n += 1
    assert refine([a], [b, copy.deepcopy(b)])[1][0]["local_index"] == 0; n += 1
    assert identify_local([a], [a, b]) == [1]; n += 1
    no_overlap = row(5, 0, 10, 10, .9)
    assert refine([a], [no_overlap]) == ([a], []); n += 1
    exact_half = row(0, 0, 10, 5, .9)
    assert refine([a], [exact_half])[1][0]["overlap"] == .5; n += 1
    empty = row(0, 0, 0, 0)
    assert refine([empty], [b]) == ([empty], []); n += 1
    return dict(status="passed", tests=n, skipped=0)


def run(workspace, output):
    import numpy as np
    from pycocotools import mask as mu
    begun = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not any(Path("/dev/" + x).exists() for x in ("nvidia0", "nvidia1", "nvidiactl"))
    cpu = tests(); save(output / "cpu_tests.json", cpu)
    assert time.time() < START + 600, "CPU preparation gate missed"
    bound = {}
    def bind(rel, expected):
        actual = sha(workspace / rel); assert actual == expected, rel
        bound[rel] = actual
        return read(workspace / rel)
    prior = bind(PRIOR + "/report.json", "975f926808429fde13effa5bf4c69d46eb8f8ab2c01ea50b1df56da292bcf81b")
    base_report = bind(audit.BASE + "/report.json", "b4e4c3dc6ebfbe3a63e88be36be6065388d63987e92323f7dbde0e57100451ee")
    tiles_report = bind(audit.TILES + "/evaluate/report.json", "dad09b10fc61467a3068e9aef058af2b8818c89509454a5889411b154a837d92")
    ids = tiles_report["image_ids"]
    assert len(ids) == len(set(ids)) == 2584
    hashes = bind(audit.BASE + "/input_hashes.json", prior["input_sha256"][audit.BASE + "/input_hashes.json"])
    assert hashes == bind(audit.TILES + "/evaluate/input_hashes.json", prior["input_sha256"][audit.TILES + "/evaluate/input_hashes.json"])
    assert [r["image_id"] for r in hashes] == ids
    arms = {}
    for name, rel in (("base", audit.BASE), ("tiles", audit.TILES + "/evaluate")):
        rows = bind(rel + "/predictions.segm.json", prior["input_sha256"][rel + "/predictions.segm.json"])
        arms[name] = {i: [] for i in ids}
        for r in rows:
            assert r["image_id"] in arms[name] and r["category_id"] == 1
            assert r["segmentation"]["size"] == [1024, 1024] and isinstance(r["segmentation"]["counts"], str)
            assert math.isfinite(r["score"]) and 0 <= r["score"] <= 1
            arms[name][r["image_id"]].append(r)
        assert all(len(v) <= 100 for v in arms[name].values())
    manifest = bind(audit.MANIFEST, tiles_report["manifest_sha256"])["image_ids"]
    assert len(set(manifest)) == len(manifest) == 256
    origin_counts = Counter()
    for relative, digest in tiles_report["capture_sha256"].items():
        if not relative.startswith("subset/"): continue
        captured = bind(audit.TILES + "/evaluate/" + relative, digest)
        ident = captured["image_id"]; merged = arms["tiles"][ident]
        assert len(merged) == len(captured["predictions"]) == len(captured["origins"])
        inferred = set(identify_local(arms["base"][ident], merged))
        for i, (r, p, origin) in enumerate(zip(merged, captured["predictions"], captured["origins"])):
            assert key(r) == key(p), "Capture/cache mismatch"
            source = origin["view"]
            assert source in (0, 1, 2, 3, 4)
            is_local = source != 0
            assert (i in inferred) == is_local, ("Ambiguous source", ident, i, source)
            origin_counts[source] += 1
    assert time.time() < START + 600, "Source binding preparation gate missed"
    save(output / "status.json", dict(status="running", phase="refine", updated_at_utc=now()))
    started = time.monotonic()
    refined, changes = {}, {}
    for ident in ids:
        check()
        refined[ident], changes[ident] = refine(arms["base"][ident], arms["tiles"][ident])
    after = [row for i in ids for row in refined[i]]
    save(output / "predictions.segm.json", after); save(output / "changes.json", changes)
    prediction_sha = sha(output / "predictions.segm.json")
    refinement_seconds = time.monotonic() - started
    for _ in range(5): refine(arms["base"][201960001], arms["tiles"][201960001])
    seconds = []
    for _ in range(20):
        t = time.perf_counter(); refine(arms["base"][201960001], arms["tiles"][201960001]); seconds.append(time.perf_counter()-t)
    timing = dict(warmup=5, repeats=20, mean_ms=statistics.mean(seconds)*1000, seconds=seconds,
                  full_cache_refinement_seconds=refinement_seconds,
                  scope="CPU RLE matching, selection, copy and bbox update only; excludes IO/model/GPU/previous five-view work",
                  new_gpu_memory_measured=False)
    save(output / "timing.json", timing)
    # All geometric choices and prediction bytes are immutable before any GT.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    assert sha(workspace / audit.ANN) == tiles_report["val_annotations_sha256"]
    bound[audit.ANN] = tiles_report["val_annotations_sha256"]
    save(output / "status.json", dict(status="running", phase="full_val_evaluate", updated_at_utc=now()))
    ev_start = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(workspace / audit.ANN)); assert set(coco.getImgIds()) == set(ids)
        dt = coco.loadRes([{k: v for k, v in r.items() if k != "bbox"} for r in after])
        ev = COCOeval(coco, dt, "segm"); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    check(); evaluation_seconds = time.monotonic() - ev_start
    assert refinement_seconds + evaluation_seconds <= 2100
    assert ev.params.maxDets == [1, 10, 100] and np.allclose(ev.params.iouThrs, np.linspace(.5, .95, 10))
    metrics = dict(zip(NAMES, map(float, ev.stats[:6])))
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in metrics.values())
    save(output / "full_val_metrics.json", dict(metrics=metrics, images=2584, prediction_sha256=prediction_sha,
         ious=ev.params.iouThrs.tolist(), max_dets=ev.params.maxDets, evaluation_seconds=evaluation_seconds))
    print(json.dumps(dict(phase="full_val_passed", metrics=metrics)), flush=True)
    pairs = bind(PRIOR + "/paired_gt.json", prior["artifacts_sha256"]["paired_gt.json"])
    old = {r["annotation_id"]: r for r in pairs}
    containment = bind(CONTAIN + "/report.json", "4b9d44124c1b42ac375cabd07bc49316e0442849d17361dd44b2b9877adcc165")
    containment_pairs = bind(CONTAIN + "/paired_gt.json", containment["artifacts_sha256"]["paired_gt.json"])
    deleted50 = {r["annotation_id"] for r in containment_pairs if r["lost_coverage"]}
    assert len(deleted50) == 50
    small_eval = {e["image_id"]: e for e in ev.evalImgs if e and e["aRng"] == ev.params.areaRng[1]}
    paired = []
    for ident in manifest:
        check()
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
        ious = mu.iou([r["segmentation"] for r in refined[ident]], [coco.annToRLE(a) for a in anns], [0]*len(anns))
        for j, ann in enumerate(anns):
            if ann["area"] > 1024: continue
            prior_row = old[ann["id"]]
            new_iou = float(ious[:, j].max()) if len(refined[ident]) else 0.
            paired.append(dict(annotation_id=ann["id"], image_id=ident, area=ann["area"],
                before_iou=prior_row["baseline"]["best_iou"], after_iou=new_iou,
                tiles_iou=prior_row["tiles"]["best_iou"], original_cohort=prior_row["cohort"],
                before_small_tp=prior_row["baseline"]["small_tp_rank"] is not None,
                after_small_tp=bool(np.any(small_eval[ident]["dtMatches"][0] == ann["id"])),
                containment_deleted=ann["id"] in deleted50))
    assert len(paired) == len({r["annotation_id"] for r in paired}) == 1187
    def stats(selected):
        return dict(count=len(selected),
            gained=sum(r["before_iou"] < .5 <= r["after_iou"] for r in selected),
            lost=sum(r["after_iou"] < .5 <= r["before_iou"] for r in selected),
            coverage={str(t): dict(before=sum(r["before_iou"] >= t for r in selected),
                after=sum(r["after_iou"] >= t for r in selected)) for t in (.5, .75, .9)},
            before_tp=sum(r["before_small_tp"] for r in selected), after_tp=sum(r["after_small_tp"] for r in selected))
    groups = {"all_small": stats(paired), "tiny": stats([r for r in paired if r["area"] <= 256]),
              "small_257_1024": stats([r for r in paired if r["area"] > 256]),
              "containment_deleted50": stats([r for r in paired if r["containment_deleted"]])}
    for cohort in ("rescued", "lost", "both_covered", "both_missing"):
        groups["tiles_" + cohort] = stats([r for r in paired if r["original_cohort"] == cohort])
    comparison = dict(status="passed", metrics=metrics, groups=groups,
        baseline_metrics=dict(zip(NAMES, [base_report["metrics"][k] for k in KEYS])),
        delta_vs_baseline_points={n: 100*(metrics[n]-base_report["metrics"][k]) for n, k in zip(NAMES, KEYS)},
        baseline_precision="Report rounded to 3 decimals; no finer baseline delta inference",
        full_val_instances=len(after), changed=sum(map(len, changes.values())), source_capture_counts=dict(origin_counts),
        positive_examples=sorted(paired, key=lambda r: r["after_iou"]-r["before_iou"], reverse=True)[:15],
        negative_examples=sorted(paired, key=lambda r: r["after_iou"]-r["before_iou"])[:15],
        original_five_view_cost_persists=True, timing=timing)
    save(output / "comparison.json", comparison); save(output / "paired_gt.json", paired)
    assert sha(output / "predictions.segm.json") == prediction_sha
    for rel, digest in bound.items(): assert sha(workspace / rel) == digest, rel
    assert "torch" not in sys.modules
    artifacts = ("cpu_tests.json", "predictions.segm.json", "changes.json", "timing.json", "full_val_metrics.json", "comparison.json", "paired_gt.json")
    save(output / "report.json", dict(status="passed", finished_at_utc=now(), image_ids=ids,
        completed_images=2584, paired_small=1187, metrics=metrics, preparation_unix=START, deadline_unix=DEADLINE,
        elapsed_worker_seconds=time.monotonic()-begun, gpu_calls=0, model_loads=0, weight_updates=0,
        new_model_predictions=0, full_val_evaluations=1, gt_used_only_after_refinement=True,
        test_inputs_used=False, test_annotations_used=False, parameter_search=False, cpu_tests=cpu,
        input_sha256=bound, source_sha256={Path(__file__).name: sha(Path(__file__)), Path(audit.__file__).name: sha(Path(audit.__file__))},
        artifacts_sha256={f: sha(output/f) for f in artifacts}))
    save(output / "status.json", dict(status="complete", phase="complete", report_sha256=sha(output / "report.json"), updated_at_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); check()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Absolute 1h deadline; no extension")))
    signal.alarm(max(1, math.ceil(DEADLINE-time.time())))
    if (args.output / "status.json").exists(): raise ValueError("No duplicate execution")
    save(args.output / "status.json", dict(status="running", phase="cpu_and_binding", updated_at_utc=now()))
    try:
        run(args.workspace, args.output)
    except BaseException as exc:
        failure = dict(status="failed", error=repr(exc), traceback=traceback.format_exc(), failed_at_utc=now())
        for name in ("first_error.json", "status.json"):
            (args.output / name).write_text(json.dumps(failure, indent=2))
        raise

"""Bounded CPU-only fit/evaluate/analyze; no GPU, new predictions, or model load."""
import contextlib
import io
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

from run_m2f_quality_select import read, save, sha, analysis_manifest_ids

START = 1792485726
PREP_END, FIT_END, EVAL_END, DEADLINE = (START+s for s in (600, 1200, 2400, 3600))
NAME = "course_m2f_balanced_quality_20261020_v1"
PARENT = "outputs/course_m2f_quality_select_20261020_v1"
ACCEPTED = "outputs/course_m2f_quality_analysis_20261020_v1"
GEOM = "outputs/course_m2f_geometry_20261020_v1"
DATA = "data/course_blender_20261013/prepared"
OWNER = "bd6654ab1597439381cc6582d3b9b10f5"
ROOT = Path("/workspace")
OUT = ROOT/"outputs"/NAME
VAL_SHA = "71612ebf1d548eb85e146e97de4d211578d2f87d5273f9b0f5e2d48b1af2ea2d"


def check(end=DEADLINE):
    assert time.time() < end, "Approved fixed phase budget expired; no extension"


def mark(phase, **kw):
    save(OUT/"status.json", dict(status="running", phase=phase, unix=time.time(), **kw))
    print(phase, kw, flush=True)


def worker():
    import unittest
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import m2f_quality_selector as original
    import m2f_balanced_quality as balanced
    import m2f_cached_geometry_refine as refine
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU deadline")))
    signal.alarm(max(1, math.ceil(DEADLINE-time.time())))
    check(PREP_END); mark("cpu_contracts")
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_balanced_quality"))
    assert tests.wasSuccessful() and tests.testsRun == 7 and not tests.skipped
    parent, accepted = ROOT/PARENT, ROOT/ACCEPTED
    accepted_report = read(accepted/"report.json")
    assert sha(accepted/"report.json") == "2ce384d3a0a6c3719706058a2b72d5caae385592cb2dd08e2f40d56378c692cd"
    assert read(accepted/"status.json")["status"] == "complete"
    assert read(accepted/"terminal_acceptance.json")["report_sha256"] == sha(accepted/"report.json")
    inputs = {}
    def bound(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None: assert digest == expected, str(path)
        inputs[str(path.relative_to(ROOT))] = digest
        return read(path)
    old_selector = bound(parent/"selector.json", accepted_report["selector_sha256"])
    gpu = bound(parent/"gpu_report.json", accepted_report["artifacts_sha256"]["gpu_report.json"])
    fit_pairs = bound(parent/"fit_pairs.json", gpu["artifacts"]["fit_pairs.json"])
    check_pairs = bound(parent/"check_pairs.json", gpu["artifacts"]["check_pairs.json"])
    manifest = bound(parent/"train_manifest.json", gpu["train_manifest_sha256"])
    assert len(manifest["fit_ids"]) == 384 and len(manifest["check_ids"]) == 128
    assert not (set(manifest["fit_scenes"]) & set(manifest["check_scenes"]))
    assert {p["image_id"] for p in fit_pairs} <= set(manifest["fit_ids"])
    assert {p["image_id"] for p in check_pairs} <= set(manifest["check_ids"])
    parent_plan = read(parent/"plan.json")
    assert sha(Path(original.__file__)) == parent_plan["source_sha256"]["m2f_quality_selector.py"]
    geom = bound(ROOT/GEOM/"report.json",
                 "7dbb3f1d4d624814e43c36c1464ea413095e9d26591d1d30853d0c49819878b6")
    assert sha(refine.__file__) == geom["source_sha256"]["m2f_cached_geometry_refine.py"]
    ids = geom["image_ids"]
    assert len(ids) == len(set(ids)) == 2584
    binding = bound(parent/"source_binding.json")
    arms = []
    for rel in (refine.audit.BASE+"/predictions.segm.json",
                refine.audit.TILES+"/evaluate/predictions.segm.json"):
        rows = bound(ROOT/rel, binding["full_val_cache_sha256"][rel])
        by_image = {i: [] for i in ids}
        for row in rows:
            assert row["segmentation"]["size"] == [1024, 1024] and row["category_id"] == 1
            assert math.isfinite(row["score"]) and 0 <= row["score"] <= 1
            by_image[row["image_id"]].append(row)
        assert all(len(v) <= 100 for v in by_image.values())
        arms.append(by_image)
    save(OUT/"cpu_acceptance.json", dict(status="passed", tests=7, skipped=0,
         sources_and_train_cache_verified=True, input_sha256=inputs.copy(), unix=time.time()))
    check(PREP_END); mark("fit")
    begun = time.time()
    selector = balanced.fit(fit_pairs)
    assert selector["mean"] == old_selector["mean"] and selector["scale"] == old_selector["scale"]
    selector.update(fit_image_ids=manifest["fit_ids"], check_image_ids=manifest["check_ids"],
                    train_manifest_sha256=gpu["train_manifest_sha256"],
                    m2f_weights_updated=False, validation_labels_used=False)
    save(OUT/"selector.json", selector); selector_sha = sha(OUT/"selector.json")
    fit_seconds = time.time()-begun
    diagnostics = {}
    for name, pairs in (("fit", fit_pairs), ("check", check_pairs)):
        diagnostics[name] = {}
        for group, subset in (("all", pairs), ("tiny", [p for p in pairs if p["gt_area"] is not None and p["gt_area"] <= 256]),
            ("small_257_1024", [p for p in pairs if p["gt_area"] is not None and 256 < p["gt_area"] <= 1024]),
            ("large_or_unmatched", [p for p in pairs if p["gt_area"] is None or p["gt_area"] > 1024])):
            diagnostics[name][group] = dict(
                previous=original.diagnostic(subset, old_selector),
                balanced=original.diagnostic(subset, selector))
    save(OUT/"fit_diagnostics.json", diagnostics)
    timings = {"previous": [], "balanced": []}
    ident = 201960001
    for n in range(25):
        order = (("previous", old_selector), ("balanced", selector))
        if n % 2: order = order[::-1]
        for name, model in order:
            begin = time.perf_counter()
            original.choose(arms[0][ident], arms[1][ident], model)
            elapsed = (time.perf_counter()-begin)*1000
            if n >= 5: timings[name].append(elapsed)
    save(OUT/"cpu_timing.json", dict(warmup=5, repeats=20, milliseconds=timings,
        mean_ms={k: statistics.mean(v) for k,v in timings.items()},
        scope="cached rows: candidate matching, geometry features, quality selector and output copy only",
        new_gpu_end_to_end_timing=False, historical_full_ms=gpu["timing"]["mean_ms"],
        historical_peak_allocated_mib=gpu["timing"]["peak_allocated_mib"]))
    check(FIT_END); mark("cache_select")
    selected, decisions = {}, {}
    for n, ident in enumerate(ids):
        check(EVAL_END)
        selected[ident], decisions[ident] = original.choose(arms[0][ident], arms[1][ident], selector)
        if n % 256 == 0: mark("cache_select", images=n+1, expected=2584)
    rows = [r for ident in ids for r in selected[ident]]
    save(OUT/"predictions.segm.json", rows); save(OUT/"decisions.json", decisions)
    prediction_sha = sha(OUT/"predictions.segm.json")
    save(OUT/"predictions_frozen.json", dict(prediction_sha256=prediction_sha,
        selector_sha256=selector_sha, unix=time.time(), images=2584))
    assert sha(ROOT/DATA/"annotations/instances_val.json") == VAL_SHA
    mark("full_val"); begun = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ROOT/DATA/"annotations/instances_val.json"))
        assert set(coco.getImgIds()) == set(ids)
        dt = coco.loadRes([{k:v for k,v in r.items() if k != "bbox"} for r in rows])
        ev = COCOeval(coco, dt, "segm"); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    check(EVAL_END)
    assert ev.params.maxDets == [1,10,100] and np.allclose(ev.params.iouThrs, np.linspace(.5,.95,10))
    metrics = dict(zip(refine.NAMES, map(float, ev.stats[:6])))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values())
    save(OUT/"full_val_metrics.json", dict(status="passed", metrics=metrics, image_ids=ids,
        images=2584, prediction_sha256=prediction_sha, selector_sha256=selector_sha,
        evaluation_seconds=time.time()-begun, full_val_evaluations=1, finished_unix=time.time()))
    mark("analyze")
    prior = bound(accepted/"paired_gt.json", accepted_report["artifacts_sha256"]["paired_gt.json"])
    fixed = bound(ROOT/refine.audit.MANIFEST, geom["input_sha256"][refine.audit.MANIFEST])
    subset_ids = analysis_manifest_ids(fixed, prior)
    by_gt = {p["annotation_id"]: p for p in prior}
    paired = []
    for ident in subset_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
        small = [a for a in anns if a["area"] <= 1024]
        if not small: continue
        overlap = mu.iou([r["segmentation"] for r in selected[ident]],
                         [coco.annToRLE(a) for a in small], [0]*len(small))
        for j, ann in enumerate(small):
            old = by_gt[ann["id"]]
            paired.append(dict(annotation_id=ann["id"], image_id=ident, area=ann["area"],
                global_iou=old["global_iou"], geometry_iou=old["previous_iou"],
                previous_iou=old["selected_iou"],
                selected_iou=float(overlap[:, j].max()) if selected[ident] else 0.,
                geometry_gain=old["previous_gain"], geometry_loss=old["previous_loss"],
                selector_gain=old["previous_iou"] < .5 <= old["selected_iou"],
                selector_loss=old["selected_iou"] < .5 <= old["previous_iou"]))
    assert len(paired) == len(by_gt) == 1187
    def summarize(values):
        return dict(count=len(values), gained=sum(p["previous_iou"] < .5 <= p["selected_iou"] for p in values),
                    lost=sum(p["selected_iou"] < .5 <= p["previous_iou"] for p in values),
                    coverage={str(t): dict(before=sum(p["previous_iou"] >= t for p in values),
                        after=sum(p["selected_iou"] >= t for p in values)) for t in (.5,.75,.9)})
    groups = {"all_small": summarize(paired),
              "tiny": summarize([p for p in paired if p["area"] <= 256]),
              "small_257_1024": summarize([p for p in paired if p["area"] > 256])}
    for name in ("geometry_gain", "geometry_loss", "selector_gain", "selector_loss"):
        groups[name] = summarize([p for p in paired if p[name]])
    delta = {k:100*(v-accepted_report["metrics"][k]) for k,v in metrics.items()}
    comparison = dict(status="passed", metrics=metrics, baseline_metrics=accepted_report["metrics"],
        delta_points=delta, groups=groups, diagnostic_images=256, paired_small=1187,
        accepted_replacements=sum(d["accepted"] for v in decisions.values() for d in v),
        rejected_replacements=sum(not d["accepted"] for v in decisions.values() for d in v),
        positive_examples=sorted(paired,key=lambda p:p["selected_iou"]-p["previous_iou"], reverse=True)[:15],
        negative_examples=sorted(paired,key=lambda p:p["selected_iou"]-p["previous_iou"])[:15],
        geometric_coverage_is_not_ap=True, single_run_development_val=True,
        any_positive_aps_retained=delta["APs"]>0)
    save(OUT/"comparison.json", comparison); save(OUT/"paired_gt.json", paired)
    assert all(sha(ROOT/p) == h for p,h in inputs.items())
    assert sha(OUT/"selector.json") == selector_sha and sha(OUT/"predictions.segm.json") == prediction_sha
    check()
    artifacts = ("cpu_acceptance.json","selector.json","fit_diagnostics.json","cpu_timing.json",
                 "predictions_frozen.json","predictions.segm.json","decisions.json",
                 "full_val_metrics.json","comparison.json","paired_gt.json")
    save(OUT/"report.json", dict(status="passed", start_unix=START, deadline_unix=DEADLINE,
        finished_unix=time.time(), metrics=metrics, delta_points=delta, input_sha256=inputs,
        full_val_evaluations=1, complete_val_images=2584, fit_images=384, check_images=128,
        fit_pairs=len(fit_pairs), check_pairs=len(check_pairs), group_counts=selector["group_counts"],
        selector_sha256=selector_sha, selector_parameters=105, fit_seconds=fit_seconds,
        gpu_used=False, model_loaded=False, m2f_weights_updated=False, new_model_predictions=0,
        hyperparameter_search=False, test_inputs_used=False, test_annotations_used=False,
        new_gpu_timing=False, artifacts_sha256={n:sha(OUT/n) for n in artifacts}))


def host():
    import launch_m2f_rle_passthrough as reused
    host_root = Path("/mnt/nfs/fenghaibin/AI_Class")
    host_out = host_root/"outputs"/NAME
    assert not (host_out/"plan.json").exists(), "No duplicate launch"
    check(PREP_END)
    plan = dict(start_unix=START, preparation_end=PREP_END, fit_end=FIT_END,
        evaluation_end=EVAL_END, deadline_unix=DEADLINE, owner=OWNER, source_commit=sys.argv[2],
        source_sha256={p.name:sha(p) for p in (host_out/"source").glob("*.py")},
        gpu=False, cpu=4, memory_gib=8, parent=ACCEPTED, only_change="three-group balanced fit loss")
    save(host_out/"plan.json", plan)
    reused.ROOT, reused.OUT, reused.OWNER = host_root, host_out, OWNER
    name, cmd = reused.command("run")
    name = "ai-m2f-balanced-"+OWNER[:10]
    cmd[cmd.index("--name")+1] = name
    cmd[cmd.index("--memory")+1] = "8g"
    cmd[cmd.index("m2f_rle_passthrough.py"):] = ["run_m2f_balanced_quality.py", "worker"]
    assert "--gpus" not in cmd and cmd[cmd.index("--runtime")+1] == "runc"
    try:
        with (host_out/"worker.log").open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=DEADLINE-time.time(), check=True)
        report = read(host_out/"report.json")
        assert report["status"] == "passed" and report["finished_unix"] < DEADLINE
        assert all(sha(host_out/n) == h for n,h in report["artifacts_sha256"].items())
        assert all(sha(host_out/"source"/n) == h for n,h in plan["source_sha256"].items())
        save(host_out/"terminal_acceptance.json", dict(status="passed", report_sha256=sha(host_out/"report.json"),
            original_deadline_preserved=True, all_artifacts_verified=True, unix=time.time()))
        save(host_out/"status.json", dict(status="complete", phase="complete", unix=time.time()))
    except BaseException as exc:
        save(host_out/"first_error.json", dict(error=repr(exc), unix=time.time()))
        # Only stop this CPU container if it is still running with the exact owner.
        import json
        try:
            item = json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
            assert item["Config"]["Labels"]["ai-course-directions-owner"] == OWNER
            subprocess.run(["docker","stop","-t","5",name],timeout=12,check=True)
        except subprocess.CalledProcessError:
            pass
        save(host_out/"status.json", dict(status="failed", phase="worker", unix=time.time()))
        raise


if __name__ == "__main__":
    if sys.argv[1] == "host":
        host()
    else:
        try:
            worker()
        except BaseException as exc:
            save(OUT/"worker_first_error.json", dict(error=repr(exc), traceback=traceback.format_exc(), unix=time.time()))
            raise

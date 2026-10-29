"""One fixed 90-minute, zero-GPU absolute-quality rescoring experiment."""
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
from run_m2f_quality_select import read, save, sha, analysis_manifest_ids, TRAIN_SHA, VAL_SHA

START = 1792488107
PREP_END, FIT_END, EVAL_END, DEADLINE = (START+s for s in (1200,2100,3600,5400))
NAME = "course_m2f_quality_rescore_20261020_v1"
OWNER = "9631186a551a4a6ebec3bc57356e46ae"
ROOT = Path("/workspace")
OUT = ROOT/"outputs"/NAME
PARENT = "outputs/course_m2f_quality_select_20261020_v1"
ACCEPTED = "outputs/course_m2f_quality_analysis_20261020_v1"
DATA = "data/course_blender_20261013/prepared"
GEOM = "outputs/course_m2f_geometry_20261020_v1"


def check(end=DEADLINE):
    assert time.time() < end, "Original preparation-inclusive phase budget expired"


def mark(phase, **kw):
    check()
    save(OUT/"status.json", dict(status="running", phase=phase, unix=time.time(), **kw))
    print(phase, kw, flush=True)


def worker():
    import unittest
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import m2f_absolute_quality as q
    import m2f_cached_geometry_refine as geometry
    audit = geometry.audit
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not any(Path("/dev"/Path(n)).exists() for n in ("nvidia0","nvidia1","nvidiactl"))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Absolute CPU deadline")))
    signal.alarm(max(1, math.ceil(DEADLINE-time.time())))
    check(PREP_END); mark("contracts")
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_absolute_quality"))
    assert tests.wasSuccessful() and tests.testsRun == 9 and not tests.skipped
    inputs = {}
    def bound(path, expected=None):
        path = Path(path); digest = sha(path)
        if expected is not None: assert digest == expected, str(path)
        inputs[str(path.relative_to(ROOT))] = digest
        return read(path)
    parent, accepted = ROOT/PARENT, ROOT/ACCEPTED
    previous = bound(accepted/"report.json",
        "2ce384d3a0a6c3719706058a2b72d5caae385592cb2dd08e2f40d56378c692cd")
    assert read(accepted/"status.json")["status"] == "complete"
    assert read(accepted/"terminal_acceptance.json")["report_sha256"] == sha(accepted/"report.json")
    gpu = bound(parent/"gpu_report.json", previous["artifacts_sha256"]["gpu_report.json"])
    manifest = bound(parent/"train_manifest.json", gpu["train_manifest_sha256"])
    hashes = bound(parent/"train_input_hashes.json", gpu["artifacts"]["train_input_hashes.json"])
    by_input = {v["image_id"]:v for v in hashes}
    assert len(manifest["fit_ids"]) == 384 and len(manifest["check_ids"]) == 128
    assert not set(manifest["fit_scenes"]) & set(manifest["check_scenes"])
    assert len(set(manifest["fit_ids"]+manifest["check_ids"])) == len(by_input) == 512
    geom = bound(ROOT/GEOM/"report.json",
        "7dbb3f1d4d624814e43c36c1464ea413095e9d26591d1d30853d0c49819878b6")
    assert sha(geometry.__file__) == geom["source_sha256"]["m2f_cached_geometry_refine.py"]
    assert sha(q.ridge.__file__) == read(parent/"plan.json")["source_sha256"]["m2f_quality_selector.py"]
    ids = geom["image_ids"]
    assert len(ids) == len(set(ids)) == 2584
    assert not set(ids) & set(by_input)
    trainpath = ROOT/DATA/"annotations/instances_train.json"
    assert sha(trainpath) == TRAIN_SHA
    inputs[str(trainpath.relative_to(ROOT))] = TRAIN_SHA
    with contextlib.redirect_stdout(io.StringIO()):
        train = COCO(str(trainpath))
    assert set(by_input) <= set(train.getImgIds())
    # Freeze every existing capture before fitting. Upstream reports bind image
    # inputs, not these complete JSON blobs; record that distinction explicitly.
    captures = {}
    for ident in manifest["fit_ids"]+manifest["check_ids"]:
        c = bound(parent/"train_predictions"/f"{ident}.json")
        assert c["image_id"] == ident and c["input_hashes"] == by_input[ident]
        assert all(r["image_id"] == ident for r in c["global"]+c["merged"])
        assert len(c["global"]) <= 100 and len(c["merged"]) <= 100
        captures[ident] = c
    binding = bound(parent/"source_binding.json")
    arms = {}
    for name, rel in (("global",audit.BASE+"/predictions.segm.json"),
                      ("raw",audit.TILES+"/evaluate/predictions.segm.json")):
        values = bound(ROOT/rel,binding["full_val_cache_sha256"][rel])
        groups = {i:[] for i in ids}
        for r in values:
            assert r["segmentation"]["size"] == [1024,1024] and r["category_id"] == 1
            assert math.isfinite(r["score"]) and 0 <= r["score"] <= 1
            groups[r["image_id"]].append(r)
        assert all(len(v) <= 100 for v in groups.values())
        arms[name] = groups
    del values
    check(PREP_END)
    save(OUT/"cpu_acceptance.json",dict(status="passed",tests=9,skipped=0,
        capture_hashes_frozen_now=True, upstream_capture_blobs_not_prehashed=True,
        upstream_input_hashes_verified=True, train_val_disjoint=True,unix=time.time()))
    mark("train_features")
    sets = {}
    for arm in ("fit","check"):
        xx, yy, records = [], [], []
        for ident in manifest[arm+"_ids"]:
            check(FIT_END)
            c = captures[ident]
            x = q.features(c["global"],c["merged"])
            labels = q.labels(c["merged"], train.loadAnns(train.getAnnIds(imgIds=[ident])), train)
            xx.extend(x.tolist()); yy.extend(v["target_iou"] for v in labels)
            records.extend(dict(image_id=ident,index=j,**v) for j,v in enumerate(labels))
        sets[arm] = (np.asarray(xx), np.asarray(yy), records)
    del captures, train
    begun = time.time()
    model = q.fit(sets["fit"][0],sets["fit"][1])
    model.update(fit_image_ids=manifest["fit_ids"],check_image_ids=manifest["check_ids"],
        train_manifest_sha256=gpu["train_manifest_sha256"],train_annotations_sha256=TRAIN_SHA,
        validation_labels_used=False,hyperparameter_search=False,fitted_unix=time.time())
    save(OUT/"quality_model.json",model); model_sha = sha(OUT/"quality_model.json")
    fit_seconds = time.time()-begun
    diagnostics = {}
    for arm,(x,y,records) in sets.items():
        predictions = q.predict(x,model)
        groups = {"all":list(range(len(y)))}
        groups["tiny"] = [i for i,v in enumerate(records) if v["gt_area"] is not None and v["gt_area"] <= 256]
        groups["small_257_1024"] = [i for i,v in enumerate(records) if v["gt_area"] is not None and 256 < v["gt_area"] <= 1024]
        groups["large_or_unmatched"] = [i for i,v in enumerate(records) if v["gt_area"] is None or v["gt_area"] > 1024]
        diagnostics[arm] = {name:dict(count=len(index),mse=float(np.mean((predictions[index]-y[index])**2)),
            mean_label=float(y[index].mean()), mean_prediction=float(predictions[index].mean()),
            no_overlap=sum(records[i]["gt_id"] is None for i in index)) for name,index in groups.items()}
        save(OUT/(arm+"_labels.json"),records)
    save(OUT/"fit_diagnostics.json", diagnostics)
    timings = {"raw_sort":[], "features_quality_sort_copy":[]}
    for n in range(25):
        for name in (list(timings) if n%2 == 0 else list(timings)[::-1]):
            t = time.perf_counter()
            if name == "raw_sort": sorted(arms["raw"][201960001],key=lambda r:-r["score"])
            else: q.rescore(arms["global"][201960001],arms["raw"][201960001],model)
            if n >= 5: timings[name].append((time.perf_counter()-t)*1000)
    save(OUT/"cpu_timing.json",dict(warmup=5,repeats=20,milliseconds=timings,
        mean_ms={k:statistics.mean(v) for k,v in timings.items()},
        scope="cached prediction-only feature extraction, quality score, sort and copy; no model/IO",
        new_gpu_measurement=False, historical_full_ms=gpu["timing"]["mean_ms"],
        historical_peak_allocated_mib=gpu["timing"]["peak_allocated_mib"]))
    check(FIT_END); mark("freeze_full_predictions")
    selected, orders = {}, {}
    for n,ident in enumerate(ids):
        check(EVAL_END)
        selected[ident], orders[ident] = q.rescore(arms["global"][ident],arms["raw"][ident],model)
        if n%512 == 0: mark("freeze_full_predictions",images=n+1)
    rows = [r for i in ids for r in selected[i]]
    assert len(rows) == sum(map(len,arms["raw"].values()))
    save(OUT/"predictions.segm.json",rows); save(OUT/"original_indices.json",orders)
    prediction_sha = sha(OUT/"predictions.segm.json")
    frozen_time = time.time()
    save(OUT/"predictions_frozen.json",dict(unix=frozen_time,model_sha256=model_sha,
        prediction_sha256=prediction_sha,images=2584,instances=len(rows),
        all_masks_categories_counts_unchanged=True,stable_ties=True))
    # First validation-label read happens only after model AND outputs freeze.
    valpath = ROOT/DATA/"annotations/instances_val.json"
    assert sha(valpath) == VAL_SHA
    inputs[str(valpath.relative_to(ROOT))] = VAL_SHA
    mark("full_val"); began = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(valpath))
        assert set(coco.getImgIds()) == set(ids)
        dt = coco.loadRes([{k:v for k,v in r.items() if k != "bbox"} for r in rows])
        ev = COCOeval(coco,dt,"segm"); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    check(EVAL_END)
    assert ev.params.maxDets == [1,10,100] and np.allclose(ev.params.iouThrs,np.linspace(.5,.95,10))
    metrics = dict(zip(geometry.NAMES,map(float,ev.stats[:6])))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values())
    save(OUT/"full_val_metrics.json",dict(status="passed",metrics=metrics,image_ids=ids,
        full_val_evaluations=1,evaluation_seconds=time.time()-began,
        prediction_sha256=prediction_sha,model_sha256=model_sha,finished_unix=time.time()))
    del ev,dt,rows
    mark("paired_analysis")
    prior = bound(accepted/"paired_gt.json", previous["artifacts_sha256"]["paired_gt.json"])
    fixed = bound(ROOT/audit.MANIFEST,geom["input_sha256"][audit.MANIFEST])
    subset_ids = analysis_manifest_ids(fixed,prior); subset = set(subset_ids)
    bestrows = bound(parent/"predictions.segm.json",
        "b6ecd63df06de91692a745c77bde98f81d7f730b7e2a442d89a1e2fb5c92a8fe")
    arms["best"] = {i:[] for i in subset_ids}
    for r in bestrows:
        if r["image_id"] in subset: arms["best"][r["image_id"]].append(r)
    del bestrows
    arms["rescored"] = selected
    # Diagnostic-only 256-image matching at three declared IoUs, not repeated
    # full-val AP. Use COCO's actual area/ignore/one-to-one matching.
    rank_records, subset_metrics = {}, {}
    for arm in ("raw","best","rescored"):
        check()
        entries, _, subset_metrics[arm] = audit.evaluate_cached(coco,
            [r for i in subset_ids for r in arms[arm][i]],subset_ids)
        rank_records[arm] = {}
        for ti, threshold in enumerate(audit.IOUS):
            records = []
            for ident in subset_ids:
                e = entries["small"].get(ident)
                if e is None: continue
                for rank,(score,gt,ignore) in enumerate(zip(e["dtScores"],e["dtMatches"][ti],e["dtIgnore"][ti]),1):
                    if not ignore: records.append(dict(image_id=ident,rank=rank,score=float(score),
                        gt_id=int(gt),tp=bool(gt)))
            rank_records[arm][str(threshold)] = records
    comparisons = {}
    for threshold in map(str,audit.IOUS):
        common_tp = min(sum(v["tp"] for v in rank_records[a][threshold]) for a in rank_records)
        comparisons[threshold] = {}
        for arm in rank_records:
            r = rank_records[arm][threshold]
            ordered = sorted(r,key=lambda v:-v["score"])
            seen=fp=0
            for value in ordered:
                seen += value["tp"]; fp += not value["tp"]
                if seen >= common_tp: break
            comparisons[threshold][arm] = dict(total_tp=sum(v["tp"] for v in r),
                total_fp=sum(not v["tp"] for v in r),equal_tp_target=common_tp,
                fp_until_equal_tp=fp,rank_cutoffs={str(k):dict(
                    tp=sum(v["tp"] for v in r if v["rank"]<=k),
                    fp=sum(not v["tp"] for v in r if v["rank"]<=k)) for k in (10,25,50,100)})
    by_gt = {p["annotation_id"]:p for p in prior}; paired=[]
    for ident in subset_ids:
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident])) if a["area"]<=1024]
        if not anns: continue
        raw = arms["raw"][ident]; new = selected[ident]
        overlaps = mu.iou([r["segmentation"] for r in new],[coco.annToRLE(a) for a in anns],[0]*len(anns))
        raw_overlap = mu.iou([r["segmentation"] for r in raw],[coco.annToRLE(a) for a in anns],[0]*len(anns))
        for j,a in enumerate(anns):
            value = float(overlaps[:,j].max()) if new else 0.
            original = float(raw_overlap[:,j].max()) if raw else 0.
            assert value == original
            p = by_gt[a["id"]]
            paired.append(dict(annotation_id=a["id"],image_id=ident,area=a["area"],
                best_iou=p["selected_iou"],raw_iou=original,rescored_iou=value))
    assert len(paired) == 1187
    changes = {}
    for threshold in map(str,audit.IOUS):
        matched = {arm:{v["gt_id"] for v in rank_records[arm][threshold] if v["tp"]} for arm in rank_records}
        changes[threshold] = {}
        for reference in ("best","raw"):
            gained = sorted(matched["rescored"]-matched[reference])
            lost = sorted(matched[reference]-matched["rescored"])
            changes[threshold][reference] = dict(gained=gained,lost=lost,
                tiny_gained=[i for i in gained if coco.anns[i]["area"]<=256],
                tiny_lost=[i for i in lost if coco.anns[i]["area"]<=256])
    delta = {k:100*(metrics[k]-previous["metrics"][k]) for k in metrics}
    comparison = dict(status="passed",metrics=metrics,baseline_metrics=previous["metrics"],
        delta_points=delta,rank_comparisons=comparisons,matched_changes=changes,
        subset_metrics=subset_metrics,paired_images=256,paired_small=1187,
        geometry_unchanged_from_raw=True,subset_not_official_ap=True,
        any_positive_aps_retained=delta["APs"]>0,single_run_development_val=True)
    save(OUT/"comparison.json",comparison);save(OUT/"paired_gt.json",paired)
    save(OUT/"rank_records.json",rank_records)
    assert all(sha(ROOT/p)==h for p,h in inputs.items())
    assert sha(OUT/"quality_model.json")==model_sha and sha(OUT/"predictions.segm.json")==prediction_sha
    check()
    artifacts = ("cpu_acceptance.json","quality_model.json","fit_labels.json","check_labels.json",
        "fit_diagnostics.json","cpu_timing.json","predictions_frozen.json","predictions.segm.json",
        "original_indices.json","full_val_metrics.json","comparison.json","paired_gt.json","rank_records.json")
    save(OUT/"report.json",dict(status="passed",start_unix=START,deadline_unix=DEADLINE,
        finished_unix=time.time(),metrics=metrics,delta_points=delta,input_sha256=inputs,
        fit_images=384,check_images=128,fit_candidates=len(sets["fit"][1]),
        check_candidates=len(sets["check"][1]),full_val_evaluations=1,complete_val_images=2584,
        quality_model_sha256=model_sha,quality_parameters=105,fit_seconds=fit_seconds,
        predictions_frozen_unix=frozen_time,validation_gt_read_after_freeze=True,
        gpu_used=False,model_loaded=False,m2f_weights_updated=False,new_model_predictions=0,
        test_inputs_used=False,test_annotations_used=False,hyperparameter_search=False,
        new_gpu_timing=False,artifacts_sha256={n:sha(OUT/n) for n in artifacts}))


def host():
    import launch_m2f_rle_passthrough as reused
    root=Path("/mnt/nfs/fenghaibin/AI_Class"); out=root/"outputs"/NAME
    assert not (out/"plan.json").exists(), "No duplicate launch"
    check(PREP_END)
    plan=dict(start_unix=START,preparation_end=PREP_END,fit_end=FIT_END,evaluation_end=EVAL_END,
        deadline_unix=DEADLINE,owner=OWNER,source_commit=sys.argv[2],
        source_sha256={p.name:sha(p) for p in (out/"source").glob("*.py")},
        gpu=False,cpu=4,memory_gib=8,only_change="absolute quality score of fixed merged candidate pool")
    save(out/"plan.json",plan)
    reused.ROOT,reused.OUT,reused.OWNER=root,out,OWNER
    _,cmd=reused.command("run");name="ai-m2f-rescore-"+OWNER[:10]
    cmd[cmd.index("--name")+1]=name;cmd[cmd.index("--memory")+1]="8g"
    cmd[cmd.index("m2f_rle_passthrough.py"):]=["run_m2f_absolute_quality.py","worker"]
    assert "--gpus" not in cmd and cmd[cmd.index("--runtime")+1]=="runc"
    save(out/"container_command.json",cmd)
    try:
        with (out/"worker.log").open("x") as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=DEADLINE-time.time(),check=True)
        report=read(out/"report.json")
        assert report["status"]=="passed" and report["finished_unix"]<DEADLINE
        assert all(sha(out/n)==h for n,h in report["artifacts_sha256"].items())
        assert all(sha(out/"source"/n)==h for n,h in plan["source_sha256"].items())
        check()
        save(out/"terminal_acceptance.json",dict(status="passed",report_sha256=sha(out/"report.json"),
            all_artifacts_verified=True,original_deadline_preserved=True,unix=time.time()))
        save(out/"status.json",dict(status="complete",phase="complete",unix=time.time()))
    except BaseException as exc:
        save(out/"first_error.json",dict(error=repr(exc),unix=time.time()))
        import json
        try:
            item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
            assert item["Config"]["Labels"]["ai-course-directions-owner"]==OWNER
            subprocess.run(["docker","stop","-t","5",name],timeout=12,check=True)
        except subprocess.CalledProcessError:
            pass
        save(out/"status.json",dict(status="failed",phase="worker",unix=time.time()))
        raise


if __name__=="__main__":
    if sys.argv[1]=="host": host()
    else:
        try: worker()
        except BaseException as exc:
            save(OUT/"worker_first_error.json",dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
            raise

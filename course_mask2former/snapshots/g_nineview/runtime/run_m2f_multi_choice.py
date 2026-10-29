"""One CPU-only 60-minute experiment, with a binding train-only oracle gate."""
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

START=1792489328
PREP_END,FIT_END,EVAL_END,DEADLINE=(START+s for s in (900,1500,2700,3600))
NAME="course_m2f_multichoice_20261020_v1"
OWNER="a18cc943e862452e9e37858aca6a2dc6"
ROOT=Path("/workspace");OUT=ROOT/"outputs"/NAME
PARENT="outputs/course_m2f_quality_select_20261020_v1"
ACCEPTED="outputs/course_m2f_quality_analysis_20261020_v1"
RESCORE="outputs/course_m2f_quality_rescore_20261020_v1"
GEOM="outputs/course_m2f_geometry_20261020_v1"
DATA="data/course_blender_20261013/prepared"


def check(end=DEADLINE):
    assert time.time()<end,"Original preparation-inclusive phase deadline expired"


def mark(phase,**kw):
    check()
    save(OUT/"status.json",dict(status="running",phase=phase,unix=time.time(),**kw))
    print(phase,kw,flush=True)


def worker():
    import unittest
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import m2f_multi_choice as multiple
    original=multiple.quality;geometry=multiple.geometry
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not any(
        Path("/dev",n).exists() for n in ("nvidia0","nvidia1","nvidiactl"))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("CPU deadline")))
    signal.alarm(max(1,math.ceil(DEADLINE-time.time())))
    check(PREP_END);mark("contracts")
    tests=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_multi_choice"))
    assert tests.wasSuccessful() and tests.testsRun==9 and not tests.skipped
    inputs={}
    def bound(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None:assert digest==expected,str(path)
        inputs[str(path.relative_to(ROOT))]=digest
        return read(path)
    parent,accepted=ROOT/PARENT,ROOT/ACCEPTED
    previous=bound(accepted/"report.json",
        "2ce384d3a0a6c3719706058a2b72d5caae385592cb2dd08e2f40d56378c692cd")
    assert read(accepted/"status.json")["status"]=="complete"
    assert read(accepted/"terminal_acceptance.json")["report_sha256"]==sha(accepted/"report.json")
    upstream=bound(ROOT/RESCORE/"report.json",
        "11df572401d7d5758dbfc593fa2fc865b003548c217db2b689e5cc7c305db9e5")
    assert read(ROOT/RESCORE/"status.json")["status"]=="complete"
    old_model=bound(parent/"selector.json",previous["selector_sha256"])
    gpu=bound(parent/"gpu_report.json",previous["artifacts_sha256"]["gpu_report.json"])
    manifest=bound(parent/"train_manifest.json",gpu["train_manifest_sha256"])
    hashes=bound(parent/"train_input_hashes.json",gpu["artifacts"]["train_input_hashes.json"])
    by_input={v["image_id"]:v for v in hashes}
    assert len(manifest["fit_ids"])==384 and len(manifest["check_ids"])==128
    assert not set(manifest["fit_scenes"])&set(manifest["check_scenes"])
    assert len(set(manifest["fit_ids"]+manifest["check_ids"]))==len(by_input)==512
    geom=bound(ROOT/GEOM/"report.json",
        "7dbb3f1d4d624814e43c36c1464ea413095e9d26591d1d30853d0c49819878b6")
    assert sha(geometry.__file__)==geom["source_sha256"]["m2f_cached_geometry_refine.py"]
    assert sha(original.__file__)==read(parent/"plan.json")["source_sha256"]["m2f_quality_selector.py"]
    ids=geom["image_ids"]
    assert len(ids)==len(set(ids))==2584 and not set(ids)&set(by_input)
    trainpath=ROOT/DATA/"annotations/instances_train.json"
    assert sha(trainpath)==TRAIN_SHA
    inputs[str(trainpath.relative_to(ROOT))]=TRAIN_SHA
    with contextlib.redirect_stdout(io.StringIO()):train=COCO(str(trainpath))
    assert set(by_input)<=set(train.getImgIds())
    mark("train_headroom")
    groups={};gates={}
    for arm in ("fit","check"):
        pairs=[];oracles=[]
        for ident in manifest[arm+"_ids"]:
            check(PREP_END)
            rel=PARENT+f"/train_predictions/{ident}.json"
            c=bound(ROOT/rel,upstream["input_sha256"][rel])
            assert c["image_id"]==ident and c["input_hashes"]==by_input[ident]
            assert all(r["image_id"]==ident for r in c["global"]+c["merged"])
            assert len(c["global"])<=100 and len(c["merged"])<=100
            p,o=multiple.supervised(c["global"],c["merged"],
                train.loadAnns(train.getAnnIds(imgIds=[ident])),train)
            pairs.extend(p);oracles.extend(o)
        old_pairs=bound(parent/(arm+"_pairs.json"),gpu["artifacts"][arm+"_pairs.json"])
        retained=[{k:v for k,v in p.items() if k!="original_single"} for p in pairs if p["original_single"]]
        assert retained==old_pairs,"Original single-candidate pairs must reproduce exactly"
        groups[arm]=pairs;gates[arm]=multiple.oracle_summary(oracles)
        save(OUT/(arm+"_pairs.json"),pairs)
    del train
    gate=dict(status="passed",gate_passed=gates["check"]["gate_passed"],
        fit=gates["fit"],check=gates["check"],unix=time.time(),
        rule="check tiny extra threshold crossing AND mean unique-small oracle IoU increment > 0",
        validation_used=False,original_single_pairs_exact=True)
    save(OUT/"headroom_gate.json",gate)
    check(PREP_END)
    save(OUT/"cpu_acceptance.json",dict(status="passed",tests=9,skipped=0,
        upstream_512_capture_hashes_verified=True,original_single_pairs_exact=True,
        train_val_disjoint=True,unix=time.time()))
    if not gate["gate_passed"]:
        assert all(sha(ROOT/p)==h for p,h in inputs.items())
        artifacts=("headroom_gate.json","cpu_acceptance.json","fit_pairs.json","check_pairs.json")
        save(OUT/"report.json",dict(status="passed",outcome="stopped_no_train_headroom",
            start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),input_sha256=inputs,
            fitted=False,full_val_evaluations=0,gpu_used=False,model_loaded=False,
            m2f_weights_updated=False,new_model_predictions=0,test_inputs_used=False,
            test_annotations_used=False,gate_passed=False,
            artifacts_sha256={n:sha(OUT/n) for n in artifacts}))
        return
    mark("fit_once");begin=time.time()
    model=original.fit([p["features"] for p in groups["fit"]],[p["target_delta"] for p in groups["fit"]])
    model.update(fit_image_ids=manifest["fit_ids"],check_image_ids=manifest["check_ids"],
        train_manifest_sha256=gpu["train_manifest_sha256"],train_annotations_sha256=TRAIN_SHA,
        validation_labels_used=False,hyperparameter_search=False,fitted_unix=time.time(),
        policy="maximum positive predicted delta among all original-compatible local masks")
    save(OUT/"selector.json",model);model_sha=sha(OUT/"selector.json")
    fit_seconds=time.time()-begin
    diagnostics={arm:dict(previous=multiple.realized(p,old_model,False),
                          multiple=multiple.realized(p,model,True)) for arm,p in groups.items()}
    save(OUT/"fit_diagnostics.json",diagnostics)
    check(FIT_END)
    binding=bound(parent/"source_binding.json")
    arms=[]
    for rel in (geometry.audit.BASE+"/predictions.segm.json",
                geometry.audit.TILES+"/evaluate/predictions.segm.json"):
        rows=bound(ROOT/rel,binding["full_val_cache_sha256"][rel])
        by_image={i:[] for i in ids}
        for row in rows:
            assert row["segmentation"]["size"]==[1024,1024] and row["category_id"]==1
            assert math.isfinite(row["score"]) and 0<=row["score"]<=1
            by_image[row["image_id"]].append(row)
        assert all(len(v)<=100 for v in by_image.values())
        arms.append(by_image)
    timings={"previous":[],"multiple":[]}
    for n in range(25):
        for name in (list(timings) if n%2==0 else list(timings)[::-1]):
            begin=time.perf_counter()
            if name=="previous":original.choose(arms[0][201960001],arms[1][201960001],old_model)
            else:multiple.choose(arms[0][201960001],arms[1][201960001],model)
            if n>=5:timings[name].append((time.perf_counter()-begin)*1000)
    save(OUT/"cpu_timing.json",dict(warmup=5,repeats=20,milliseconds=timings,
        mean_ms={k:statistics.mean(v) for k,v in timings.items()},
        scope="cached matching/features/quality selection/output copy; no model or IO",
        new_gpu_measurement=False,historical_full_ms=gpu["timing"]["mean_ms"],
        historical_peak_allocated_mib=gpu["timing"]["peak_allocated_mib"]))
    check(FIT_END);mark("freeze_predictions")
    selected={};decisions={}
    for n,ident in enumerate(ids):
        check(EVAL_END)
        selected[ident],decisions[ident]=multiple.choose(arms[0][ident],arms[1][ident],model)
        if n%512==0:mark("freeze_predictions",images=n+1)
    rows=[r for ident in ids for r in selected[ident]]
    save(OUT/"predictions.segm.json",rows);save(OUT/"decisions.json",decisions)
    prediction_sha=sha(OUT/"predictions.segm.json")
    save(OUT/"predictions_frozen.json",dict(unix=time.time(),images=2584,
        selector_sha256=model_sha,prediction_sha256=prediction_sha,
        global_counts_scores_order_categories_unchanged=True))
    valpath=ROOT/DATA/"annotations/instances_val.json"
    assert sha(valpath)==VAL_SHA
    inputs[str(valpath.relative_to(ROOT))]=VAL_SHA
    mark("full_val");begin=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(valpath));assert set(coco.getImgIds())==set(ids)
        dt=coco.loadRes([{k:v for k,v in r.items() if k!="bbox"} for r in rows])
        ev=COCOeval(coco,dt,"segm");ev.params.imgIds=ids
        ev.evaluate();ev.accumulate();ev.summarize()
    check(EVAL_END)
    assert ev.params.maxDets==[1,10,100] and np.allclose(ev.params.iouThrs,np.linspace(.5,.95,10))
    metrics=dict(zip(geometry.NAMES,map(float,ev.stats[:6])))
    assert all(math.isfinite(v) and 0<=v<=1 for v in metrics.values())
    save(OUT/"full_val_metrics.json",dict(status="passed",metrics=metrics,image_ids=ids,
        full_val_evaluations=1,evaluation_seconds=time.time()-begin,
        prediction_sha256=prediction_sha,selector_sha256=model_sha,finished_unix=time.time()))
    del ev,dt,rows
    mark("paired_analysis")
    prior=bound(accepted/"paired_gt.json",previous["artifacts_sha256"]["paired_gt.json"])
    fixed=bound(ROOT/geometry.audit.MANIFEST,geom["input_sha256"][geometry.audit.MANIFEST])
    subset_ids=analysis_manifest_ids(fixed,prior);by_gt={p["annotation_id"]:p for p in prior}
    paired=[]
    for ident in subset_ids:
        check()
        small=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident])) if a["area"]<=1024]
        if not small:continue
        overlap=mu.iou([r["segmentation"] for r in selected[ident]],[coco.annToRLE(a) for a in small],[0]*len(small))
        for j,a in enumerate(small):
            old=by_gt[a["id"]]
            paired.append(dict(annotation_id=a["id"],image_id=ident,area=a["area"],
                global_iou=old["global_iou"],geometry_iou=old["previous_iou"],
                previous_iou=old["selected_iou"],
                selected_iou=float(overlap[:,j].max()) if selected[ident] else 0.,
                geometry_gain=old["previous_gain"],geometry_loss=old["previous_loss"]))
    assert len(paired)==len(by_gt)==1187
    def summarize(values):
        return dict(count=len(values),gained=sum(p["previous_iou"]<.5<=p["selected_iou"] for p in values),
            lost=sum(p["selected_iou"]<.5<=p["previous_iou"] for p in values),
            coverage={str(float(t)):dict(before=int(sum(p["previous_iou"]>=float(t) for p in values)),
                after=int(sum(p["selected_iou"]>=float(t) for p in values))) for t in np.linspace(.5,.95,10)})
    summaries={name:summarize(values) for name,values in (
        ("all_small",paired),("tiny",[p for p in paired if p["area"]<=256]),
        ("small_257_1024",[p for p in paired if p["area"]>256]),
        ("geometry_gain",[p for p in paired if p["geometry_gain"]]),
        ("geometry_loss",[p for p in paired if p["geometry_loss"]]))}
    delta={k:100*(v-previous["metrics"][k]) for k,v in metrics.items()}
    comparison=dict(status="passed",metrics=metrics,baseline_metrics=previous["metrics"],delta_points=delta,
        groups=summaries,diagnostic_images=256,paired_small=1187,
        accepted=sum(d["accepted"] for v in decisions.values() for d in v),
        changed_candidate=sum(d["accepted"] and d["different_from_max_overlap"] for v in decisions.values() for d in v),
        positive_examples=sorted(paired,key=lambda p:p["selected_iou"]-p["previous_iou"],reverse=True)[:15],
        negative_examples=sorted(paired,key=lambda p:p["selected_iou"]-p["previous_iou"])[:15],
        subset_not_ap=True,single_run_development_val=True,any_positive_aps_retained=delta["APs"]>0)
    save(OUT/"comparison.json",comparison);save(OUT/"paired_gt.json",paired)
    assert all(sha(ROOT/p)==h for p,h in inputs.items())
    assert sha(OUT/"selector.json")==model_sha and sha(OUT/"predictions.segm.json")==prediction_sha
    check()
    artifacts=("headroom_gate.json","cpu_acceptance.json","fit_pairs.json","check_pairs.json",
        "selector.json","fit_diagnostics.json","cpu_timing.json","predictions_frozen.json",
        "predictions.segm.json","decisions.json","full_val_metrics.json","comparison.json","paired_gt.json")
    save(OUT/"report.json",dict(status="passed",outcome="evaluated",start_unix=START,deadline_unix=DEADLINE,
        finished_unix=time.time(),metrics=metrics,delta_points=delta,input_sha256=inputs,
        gate_passed=True,fitted=True,fit_images=384,check_images=128,fit_pairs=len(groups["fit"]),
        check_pairs=len(groups["check"]),selector_parameters=105,selector_sha256=model_sha,
        fit_seconds=fit_seconds,complete_val_images=2584,full_val_evaluations=1,
        gpu_used=False,model_loaded=False,m2f_weights_updated=False,new_model_predictions=0,
        test_inputs_used=False,test_annotations_used=False,hyperparameter_search=False,new_gpu_timing=False,
        artifacts_sha256={n:sha(OUT/n) for n in artifacts}))


def host():
    import launch_m2f_rle_passthrough as reused
    root=Path("/mnt/nfs/fenghaibin/AI_Class");out=root/"outputs"/NAME
    assert not (out/"plan.json").exists(),"No duplicate launch"
    check(PREP_END)
    plan=dict(start_unix=START,preparation_end=PREP_END,fit_end=FIT_END,evaluation_end=EVAL_END,
        deadline_unix=DEADLINE,owner=OWNER,source_commit=sys.argv[2],
        source_sha256={p.name:sha(p) for p in (out/"source").glob("*.py")},
        gpu=False,cpu=4,memory_gib=8,only_change="all original-compatible local candidates")
    save(out/"plan.json",plan)
    reused.ROOT,reused.OUT,reused.OWNER=root,out,OWNER
    _,cmd=reused.command("run");name="ai-m2f-multichoice-"+OWNER[:10]
    cmd[cmd.index("--name")+1]=name;cmd[cmd.index("--memory")+1]="8g"
    cmd[cmd.index("m2f_rle_passthrough.py"):]=["run_m2f_multi_choice.py","worker"]
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
        save(out/"status.json",dict(status="complete",phase=report["outcome"],unix=time.time()))
    except BaseException as exc:
        save(out/"first_error.json",dict(error=repr(exc),unix=time.time()))
        import json
        try:
            item=json.loads(subprocess.check_output(["docker","inspect",name],timeout=6))[0]
            assert item["Config"]["Labels"]["ai-course-directions-owner"]==OWNER
            subprocess.run(["docker","stop","-t","5",name],timeout=12,check=True)
        except subprocess.CalledProcessError:pass
        save(out/"status.json",dict(status="failed",phase="worker",unix=time.time()))
        raise


if __name__=="__main__":
    if sys.argv[1]=="host":host()
    else:
        try:worker()
        except BaseException as exc:
            save(OUT/"worker_first_error.json",dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
            raise

"""Four-hour, train-gated, single-GPU RGB-D refinement experiment."""
import argparse
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time
import traceback

from run_m2f_quality_select import read, save, sha, TRAIN_SHA, VAL_SHA, FIXED8

START = 1792490362
PREP_END, TRAIN_END, EVAL_END, DEADLINE = (START+s for s in (3600, 9000, 12600, 14400))
NAME = "course_m2f_boundary_refine_20261020_v1"
OWNER = "2f8b60d1a45642d39c8e571ad3f74b06"
ROOT = Path("/workspace")
OUT = ROOT/"outputs"/NAME
PARENT = "outputs/course_m2f_quality_select_20261020_v1"
ACCEPTED = "outputs/course_m2f_quality_analysis_20261020_v1"
MULTI = "outputs/course_m2f_multichoice_20261020_v1"
DATA = "data/course_blender_20261013/prepared"
BEST_SHA = "2ce384d3a0a6c3719706058a2b72d5caae385592cb2dd08e2f40d56378c692cd"


def check(end=DEADLINE):
    assert time.time() < end, "Original preparation-inclusive deadline expired"


def mark(phase, **kw):
    check()
    value = dict(phase=phase, unix=time.time(), **kw)
    save(OUT/"progress.json", value)
    print(json.dumps(value), flush=True)


def bounded(path, inputs, expected=None):
    path = Path(path)
    value = sha(path)
    if expected is not None:
        assert value == expected, str(path)
    inputs[str(path.relative_to(ROOT))] = value
    return read(path)


def image_request(image, split):
    root = ROOT/DATA
    rgb, depth = root/image["file_name"], root/image["depth_file_name"]
    assert rgb.resolve().is_relative_to(root/"rgb"/split)
    assert depth.resolve().is_relative_to(root/"depth"/split)
    assert (image["height"], image["width"]) == (1024, 1024)
    return dict(img_id=image["id"], img_path=str(rgb), depth_file=str(depth),
                depth_scale_m=image["depth_scale_m"],
                depth_invalid_code=image["depth_invalid_code"], scene=image["scene_id"])


def prep():
    import unittest
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    import m2f_boundary_refiner as ref
    import m2f_quality_selector as quality
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    mark("contracts")
    tests = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_boundary_refiner"))
    assert tests.wasSuccessful() and not tests.skipped and tests.testsRun == 9
    inputs = {}
    accepted = bounded(ROOT/ACCEPTED/"report.json", inputs, BEST_SHA)
    assert read(ROOT/ACCEPTED/"status.json")["status"] == "complete"
    assert read(ROOT/ACCEPTED/"terminal_acceptance.json")["report_sha256"] == BEST_SHA
    assert abs(accepted["metrics"]["APs"]-.41095135423262297) < 1e-14
    prior = bounded(ROOT/MULTI/"report.json", inputs,
        "e20b0a3f7a9957419d26239e557915129ff8ba83d624e9511643ec8b666f5041")
    upstream = prior["input_sha256"]
    parent = ROOT/PARENT
    model = bounded(parent/"selector.json", inputs, accepted["selector_sha256"])
    manifest = bounded(parent/"train_manifest.json", inputs,
                       upstream[PARENT+"/train_manifest.json"])
    captured_hashes = bounded(parent/"train_input_hashes.json", inputs,
                             upstream[PARENT+"/train_input_hashes.json"])
    by_hash = {h["image_id"]:h for h in captured_hashes}
    assert len(manifest["fit_ids"]) == 384 and len(manifest["check_ids"]) == 128
    assert not set(manifest["fit_scenes"]) & set(manifest["check_scenes"])
    assert len(set(manifest["fit_ids"]+manifest["check_ids"])) == 512
    assert sha(quality.__file__) == read(parent/"plan.json")["source_sha256"]["m2f_quality_selector.py"]
    trainpath = ROOT/DATA/"annotations/instances_train.json"
    assert sha(trainpath) == TRAIN_SHA
    inputs[str(trainpath.relative_to(ROOT))] = TRAIN_SHA
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(trainpath))
    (OUT/"prepared").mkdir()
    (OUT/"train_selected").mkdir()
    metadata = {}
    gate_groups = {}
    files = {}
    for arm in ("fit", "check"):
        records = []
        for n, ident in enumerate(manifest[arm+"_ids"]):
            check(PREP_END)
            rel = PARENT+f"/train_predictions/{ident}.json"
            cached = bounded(ROOT/rel, inputs, upstream[rel])
            request = manifest["requests"][str(ident)]
            assert cached["image_id"] == ident and cached["input_hashes"] == by_hash[ident]
            actual = dict(image_id=ident, rgb_sha256=sha(request["img_path"]),
                          depth_sha256=sha(request["depth_file"]))
            assert actual == by_hash[ident]
            selected, _ = quality.choose(cached["global"], cached["merged"], model)
            save(OUT/"train_selected"/f"{ident}.json", selected)
            files[f"train_selected/{ident}.json"] = sha(OUT/"train_selected"/f"{ident}.json")
            annotations = sorted((a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
                                  if not a.get("iscrowd", 0) and not a.get("ignore", 0)), key=lambda a:a["id"])
            rles = [coco.annToRLE(a) for a in annotations]
            overlap = mu.iou([r["segmentation"] for r in selected], rles,
                             [0]*len(rles)) if selected and rles else np.zeros((len(selected),len(rles)))
            before = overlap.max(0) if selected else np.zeros(len(rles))
            oracle = before.copy()
            rgb, depth, valid = ref.load_rgbd(request)
            arrays = {k:[] for k in ("x", "coarse", "band", "target")}
            details = []
            for i, row in enumerate(selected):
                xy = ref.window(row)
                if xy is None:
                    continue
                x, y = xy
                coarse = mu.decode(row["segmentation"])[y:y+128, x:x+128].astype(bool)
                changeable = ref.band(coarse)
                same = [j for j,a in enumerate(annotations) if a["category_id"] == row["category_id"]]
                j = max(same, key=lambda j:(overlap[i,j],-j)) if same else None
                # Positive-overlap same-instance supervision; unmatched predictions
                # use identity labels, never GT-conditioned patch selection.
                match = j is not None and overlap[i,j] > 0
                target = (mu.decode(rles[j])[y:y+128, x:x+128].astype(bool)
                          if match else coarse.copy())
                ideal, _ = ref.compose(row, xy, coarse, changeable, target)
                if match:
                    value = float(mu.iou([ideal["segmentation"]],[rles[j]],[0])[0,0])
                    oracle[j] = max(oracle[j], value)
                details.append(dict(index=i, xy=list(xy), gt_id=annotations[j]["id"] if match else None,
                                    gt_area=annotations[j]["area"] if match else None,
                                    before_iou=float(overlap[i,j]) if match else 0.))
                arrays["x"].append(ref.patch_input(rgb[y:y+128,x:x+128], depth[y:y+128,x:x+128],
                                                   valid[y:y+128,x:x+128], coarse))
                arrays["coarse"].append(coarse);arrays["band"].append(changeable);arrays["target"].append(target)
            if details:
                np.savez_compressed(OUT/"prepared"/f"{ident}.npz",
                                    **{k:np.stack(v) for k,v in arrays.items()})
                files[f"prepared/{ident}.npz"] = sha(OUT/"prepared"/f"{ident}.npz")
            metadata[str(ident)] = dict(arm=arm, records=details, request=request, input_hashes=actual)
            for j,a in enumerate(annotations):
                if a["area"] <= 1024:
                    records.append(dict(image_id=ident, annotation_id=a["id"], area=a["area"],
                                        before=float(before[j]), oracle=float(oracle[j])))
            if n % 16 == 0:
                mark("train_oracle", arm=arm, images=n+1, expected=len(manifest[arm+"_ids"]))
        assert len({v["annotation_id"] for v in records}) == len(records)
        thresholds = [float(t) for t in np.linspace(.5,.95,10)]
        mean = float(np.mean([p["oracle"]-p["before"] for p in records]))
        tiny_cross = sum(sum(p["before"] < t <= p["oracle"] for t in thresholds)
                         for p in records if p["area"] <= 256)
        gate_groups[arm] = dict(small_count=len(records),
            tiny_count=sum(p["area"]<=256 for p in records), mean_small_iou_increment=mean,
            tiny_extra_threshold_crossings=tiny_cross, records=records,
            coverage={str(t):dict(before=sum(p["before"]>=t for p in records),
                                  oracle=sum(p["oracle"]>=t for p in records)) for t in thresholds})
    gate = dict(status="passed", gate_passed=bool(gate_groups["check"]["mean_small_iou_increment"]>0
                 and gate_groups["check"]["tiny_extra_threshold_crossings"]>0),
                groups=gate_groups, oracle_not_learned_or_ap=True, validation_labels_used=False,
                rule="check128 mean unique-small oracle IoU positive AND tiny extra .50:.05:.95 crossing",
                finished_unix=time.time())
    save(OUT/"headroom_gate.json", gate)
    save(OUT/"prepared_manifest.json", dict(images=metadata, fit_ids=manifest["fit_ids"],
        check_ids=manifest["check_ids"], scene_disjoint=True, files=files))
    # No official val labels used by training preparation or train gate.
    save(OUT/"binding.json", dict(input_sha256=inputs, parent=PARENT, best=ACCEPTED,
        best_report_sha256=BEST_SHA, selector_sha256=accepted["selector_sha256"],
        baseline_metrics=accepted["metrics"], prepared_manifest_sha256=sha(OUT/"prepared_manifest.json")))
    check(PREP_END)
    save(OUT/"cpu_acceptance.json", dict(status="passed", tests=9, skipped=0, gate_passed=gate["gate_passed"],
        native_rgbd_checked=512, gpu_used=False, validation_labels_used=False,
        finished_unix=time.time()))


def configure_torch():
    import torch
    import numpy as np
    import m2f_boundary_refiner as ref
    torch.manual_seed(ref.SEED);np.random.seed(ref.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    return torch


def train():
    import numpy as np
    import m2f_boundary_refiner as ref
    from run_m2f_quality_select import tensor_digest
    torch = configure_torch()
    assert torch.cuda.device_count() == 1
    assert read(OUT/"headroom_gate.json")["gate_passed"]
    manifest = read(OUT/"prepared_manifest.json")
    assert sha(OUT/"prepared_manifest.json") == read(OUT/"binding.json")["prepared_manifest_sha256"]
    ids = manifest["fit_ids"]
    n = sum(len(manifest["images"][str(i)]["records"]) for i in ids)
    assert n > 0 and n*6*128*128*4 < 6*2**30, "Refuse unbounded in-memory fit cache"
    inputs = torch.empty((n,6,128,128))
    targets = torch.empty((n,128,128),dtype=torch.bool)
    bands = torch.empty_like(targets)
    offset = 0
    for ident in ids:
        check(TRAIN_END)
        count = len(manifest["images"][str(ident)]["records"])
        if not count:continue
        p = OUT/"prepared"/f"{ident}.npz"
        assert sha(p) == manifest["files"][f"prepared/{ident}.npz"]
        with np.load(p) as z:
            inputs[offset:offset+count].copy_(torch.from_numpy(z["x"]))
            targets[offset:offset+count].copy_(torch.from_numpy(z["target"]))
            bands[offset:offset+count].copy_(torch.from_numpy(z["band"]))
        offset += count
        mark("fit_cache_load", images=ids.index(ident)+1, patches=offset, expected=n)
    assert offset == n
    model = ref.Refiner().cuda().train()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    assert not optimizer.state
    rng = torch.Generator().manual_seed(ref.SEED)
    history = []
    begun = time.time()
    phase_end = read(OUT/"phase_deadlines.json")["train_end"]
    for step in range(1,ref.STEPS+1):
        check(min(TRAIN_END,phase_end))
        chosen = torch.randint(n,(ref.BATCH,),generator=rng)
        x = inputs[chosen].cuda();y = targets[chosen].cuda();b = bands[chosen].cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = ref.band_loss(model(x),y,b)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert bool(torch.isfinite(norm))
        optimizer.step()
        history.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(norm)))
        if step==1 or step%50==0:
            mark("fit_refiner", step=step, expected=ref.STEPS, loss=history[-1]["loss"])
    torch.cuda.synchronize()
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    assert all(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
    native = sorted({int(v["step"].item()) for v in optimizer.state.values()})
    assert native == [ref.STEPS]
    assert all(bool(torch.isfinite(v).all()) for s in optimizer.state.values()
               for v in s.values() if torch.is_tensor(v))
    weight = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=weight, optimizer=optimizer.state_dict(),
                    steps=ref.STEPS, fit_image_ids=ids, seed=ref.SEED),OUT/"refiner_3000.pth")
    save(OUT/"fit_history.json",history)
    save(OUT/"fit_report.json",dict(status="passed",native_steps=native,finite_steps=len(history),
        consumed_patches=ref.STEPS*ref.BATCH,available_patches=n,fit_images=384,
        check_images_used_for_fit=0,validation_used=False,model="RGBD mask residual UNet16/32/64",
        parameters=sum(p.numel() for p in model.parameters()),state_bytes=sum(v.numel()*v.element_size() for v in weight.values()),
        source_state_initial=tensor_digest(initial),source_state_final=tensor_digest(weight),
        checkpoint_sha256=sha(OUT/"refiner_3000.pth"),m2f_loaded=False,m2f_updated=False,
        all_head_tensors_changed=True,model_optimizer_finite=True,fit_seconds=time.time()-begun,
        finished_unix=time.time()))


def load_refiner():
    import m2f_boundary_refiner as ref
    torch = configure_torch()
    fit = read(OUT/"fit_report.json")
    assert fit["status"] == "passed" and fit["native_steps"] == [3000]
    assert sha(OUT/"refiner_3000.pth") == fit["checkpoint_sha256"]
    ckpt = torch.load(OUT/"refiner_3000.pth",map_location="cpu")
    assert ckpt["steps"] == 3000
    assert sorted({int(s["step"].item()) for s in ckpt["optimizer"]["state"].values()}) == [3000]
    assert all(bool(torch.isfinite(v).all()) for s in ckpt["optimizer"]["state"].values()
               for v in s.values() if torch.is_tensor(v))
    model = ref.Refiner()
    model.load_state_dict(ckpt["state_dict"],strict=True)
    del ckpt
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    return model.cuda().eval()


def predict():
    """Only new refiner on full val; frozen full M2F evaluated on8+25 images."""
    import numpy as np
    import m2f_boundary_refiner as ref
    import m2f_quality_selector as quality
    import run_m2f_quality_select as original
    torch = configure_torch()
    model = load_refiner()
    phase_end = read(OUT/"phase_deadlines.json")["predict_end"]
    parent = ROOT/PARENT
    binding = read(OUT/"binding.json")
    best = read(ROOT/ACCEPTED/"report.json")
    source_binding = read(parent/"source_binding.json")
    assert sha(parent/"predictions.segm.json") == best["artifacts_sha256"]["predictions.segm.json"]
    rows = read(parent/"predictions.segm.json")
    # Existing accepted metadata only; official labels are not passed to inference.
    valfile = ROOT/DATA/"annotations/instances_val.json"
    assert sha(valfile) == VAL_SHA
    images = read(valfile)["images"]
    by_image = {i["id"]:[] for i in images}
    ids = read(parent/"full_val_metrics.json")["image_ids"]
    assert len(ids)==len(set(ids))==len(images)==2584 and set(ids)==set(by_image)
    requests = {i["id"]:image_request(i,"val") for i in images}
    for row in rows:by_image[row["image_id"]].append(row)
    del rows,images
    hashfile = next(p for p in source_binding["full_val_cache_sha256"] if p.endswith("input_hashes.json"))
    assert sha(ROOT/hashfile) == source_binding["full_val_cache_sha256"][hashfile]
    expected_hashes = {v["image_id"]:v for v in read(ROOT/hashfile)}
    save(OUT/"val_requests.json", requests)
    (OUT/"val_predictions").mkdir()
    stats=[];hashes=[];all_rows=[]
    begun=time.time()
    for n,ident in enumerate(ids):
        check(min(EVAL_END,phase_end))
        request=requests[ident]
        h=dict(image_id=ident,rgb_sha256=sha(request["img_path"]),depth_sha256=sha(request["depth_file"]))
        assert h==expected_hashes[ident]
        final,counts=ref.infer(by_image[ident],ref.load_rgbd(request),model,"cuda")
        save(OUT/"val_predictions"/f"{ident}.json",final)
        all_rows.extend(final);hashes.append(h);stats.append(dict(image_id=ident,**counts))
        if n%16==0:mark("refine_full_val",completed=n+1,expected=2584)
    save(OUT/"predictions.segm.json",all_rows)
    save(OUT/"val_input_hashes.json",hashes);save(OUT/"refinement_activity.json",stats)
    save(OUT/"predictions_frozen.json",dict(prediction_sha256=sha(OUT/"predictions.segm.json"),
        refiner_sha256=sha(OUT/"refiner_3000.pth"),selector_sha256=binding["selector_sha256"],
        images=2584,image_ids=ids,global_counts_scores_order_unchanged=True,unix=time.time()))
    del all_rows
    cache_seconds=time.time()-begun
    # Eight actual end-to-end parity checks and paired complete-scope timing.
    original.START,original.DEADLINE=START,DEADLINE
    accepted,previous,tiles,_=original.modules()
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    compact=read(parent/"compact.json")
    cfg=Config.fromfile(str(parent/"live_config.py"))
    register_all_modules(init_default_scope=True);import_modules_from_strings(**cfg.custom_imports)
    assert sha(compact["weight"])==tiles.SPEC["weight_sha"]
    cp=torch.load(compact["weight"],map_location="cpu")
    m2f=MODELS.build(cfg.model)
    m2f.load_state_dict(cp["state_dict"],strict=True);del cp
    before=original.tensor_digest(m2f.state_dict())
    m2f.cuda().eval()
    assert m2f.active_parameter_count()==44400414
    pipeline=previous.typed_pipeline(cfg,compact,tiles)
    loader,tail=Compose(pipeline[:1]),Compose(pipeline[1:])
    selector=read(parent/"selector.json")
    assert sha(parent/"selector.json")==binding["selector_sha256"]
    refs=[{int(k):v for k,v in arm.items()} for arm in compact["references"]]
    calls=dict(eval_calls=0,train_calls=0)
    def full(ident,enabled):
        check(min(EVAL_END,phase_end))
        torch.cuda.synchronize();start=time.perf_counter()
        # Match accepted frozen M2F's PyTorch defaults; the new refiner has
        # separate deterministic FP32 convolution settings, including cache run.
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.deterministic=False
        views=tiles.pack_views(requests[ident],loader,tail)
        encoder=accepted.CandidateEncoder(tiles);pool=[];global_rows=[]
        for index,view in enumerate(views):
            predicted=m2f.test_step(pseudo_collate([view]))[0].cpu()
            captured,candidates=encoder.encode(predicted.pred_instances,index,ident)
            calls["eval_calls"]+=1
            if index==0:global_rows=captured
            pool.extend(candidates);del predicted
        merged=accepted.base.direct_rows(tiles.merge_candidates(pool),ident)
        base=quality.choose(global_rows,merged,selector)[0]
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.deterministic=True
        final=ref.infer(base,ref.load_rgbd(requests[ident]),model,"cuda")[0] if enabled else base
        json.dumps(final,separators=(",",":"),allow_nan=False)
        torch.cuda.synchronize()
        assert encoder.encoded==encoder.roundtrips
        return base,final,(time.perf_counter()-start)*1000
    parity=[]
    with torch.inference_mode():
        for ident in FIXED8:
            base,final,_=full(ident,True)
            assert base==by_image[ident]
            assert base==quality.choose(refs[0][ident],refs[1][ident],selector)[0]
            assert final==read(OUT/"val_predictions"/f"{ident}.json")
            parity.append(dict(image_id=ident,baseline_exact=True,refined_exact=True))
            mark("live_parity",images=len(parity),expected=8)
        timings={k:dict(milliseconds=[],peak_allocated_mib=[],peak_reserved_mib=[]) for k in ("baseline","refined")}
        ident=FIXED8[0]
        for repeat in range(25):
            for key in (("baseline","refined") if repeat%2==0 else ("refined","baseline")):
                model.to("cuda" if key=="refined" else "cpu")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base,final,milliseconds=full(ident,key=="refined")
                assert base==by_image[ident]
                assert final==(read(OUT/"val_predictions"/f"{ident}.json") if key=="refined" else base)
                if repeat>=5:
                    timings[key]["milliseconds"].append(milliseconds)
                    timings[key]["peak_allocated_mib"].append(torch.cuda.max_memory_allocated()/2**20)
                    timings[key]["peak_reserved_mib"].append(torch.cuda.max_memory_reserved()/2**20)
            mark("paired_timing",completed=repeat+1,expected=25)
    assert calls==dict(eval_calls=290,train_calls=0)
    assert original.tensor_digest(m2f.state_dict())==before
    assert sha(parent/"selector.json")==binding["selector_sha256"]
    for v in timings.values():
        v.update(mean_ms=statistics.mean(v["milliseconds"]),median_ms=statistics.median(v["milliseconds"]),
                 peak_allocated_max_mib=max(v["peak_allocated_mib"]),peak_reserved_max_mib=max(v["peak_reserved_mib"]))
    save(OUT/"timing.json",dict(warmup=5,repeats=20,paired=timings,
        scope="read/decode/five-view preprocessing/model/D2H/RLE/merge/selection/RGBD correction/bbox/JSON; output write excluded",
        baseline_head_resident=False,baseline_head_not_executed=True,
        head_resident_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values())))
    save(OUT/"parity.json",parity)
    save(OUT/"gpu_report.json",dict(status="passed",full_val_refiner_images=2584,
        m2f_calls=calls,m2f_state_before=before,m2f_state_after=before,m2f_updated=False,
        selector_unchanged=True,strict_parity_images=8,full_m2f_val_repeated=False,
        full_refinement_seconds=cache_seconds,finished_unix=time.time()))


def evaluate():
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    assert not Path("/dev/nvidiactl").exists()
    frozen=read(OUT/"predictions_frozen.json")
    assert sha(OUT/"predictions.segm.json")==frozen["prediction_sha256"]
    assert sha(OUT/"refiner_3000.pth")==frozen["refiner_sha256"]
    assert sha(ROOT/DATA/"annotations/instances_val.json")==VAL_SHA
    rows=read(OUT/"predictions.segm.json");begin=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(ROOT/DATA/"annotations/instances_val.json"))
        dt=coco.loadRes([{k:v for k,v in r.items() if k!="bbox"} for r in rows])
        ev=COCOeval(coco,dt,"segm");ev.params.imgIds=frozen["image_ids"]
        ev.evaluate();ev.accumulate();ev.summarize()
    assert ev.params.maxDets==[1,10,100] and np.allclose(ev.params.iouThrs,np.linspace(.5,.95,10))
    metrics=dict(zip(("AP","AP50","AP75","APs","APm","APl"),map(float,ev.stats[:6])))
    assert all(math.isfinite(v) and 0<=v<=1 for v in metrics.values())
    check(EVAL_END)
    save(OUT/"full_val_metrics.json",dict(status="passed",metrics=metrics,images=2584,
        full_val_evaluations=1,evaluation_seconds=time.time()-begin,finished_unix=time.time(),
        prediction_sha256=frozen["prediction_sha256"]))


def analyze():
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    import m2f_cached_geometry_refine as geometry
    from run_m2f_quality_select import analysis_manifest_ids
    binding=read(OUT/"binding.json")
    previous=read(ROOT/ACCEPTED/"report.json")
    assert sha(ROOT/ACCEPTED/"report.json")==BEST_SHA
    prior=read(ROOT/ACCEPTED/"paired_gt.json")
    assert sha(ROOT/ACCEPTED/"paired_gt.json")==previous["artifacts_sha256"]["paired_gt.json"]
    ids=analysis_manifest_ids(read(ROOT/geometry.audit.MANIFEST),prior)
    by_gt={p["annotation_id"]:p for p in prior}
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(ROOT/DATA/"annotations/instances_val.json"))
    pairs=[]
    for ident in ids:
        check()
        anns=[a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident])) if a["area"]<=1024]
        if not anns:continue
        rows=read(OUT/"val_predictions"/f"{ident}.json")
        iou=mu.iou([r["segmentation"] for r in rows],[coco.annToRLE(a) for a in anns],[0]*len(anns))
        for j,a in enumerate(anns):
            p=by_gt[a["id"]]
            pairs.append(dict(image_id=ident,annotation_id=a["id"],area=a["area"],
                before=p["selected_iou"],after=float(iou[:,j].max()) if rows else 0.,
                geometry_gain=p["previous_gain"],geometry_loss=p["previous_loss"]))
    assert len(pairs)==1187
    def group(values):
        return dict(count=len(values),gained=sum(v["before"]<.5<=v["after"] for v in values),
            lost=sum(v["after"]<.5<=v["before"] for v in values),
            coverage={str(float(t)):dict(before=sum(v["before"]>=t for v in values),
                after=sum(v["after"]>=t for v in values)) for t in np.linspace(.5,.95,10)})
    full=read(OUT/"full_val_metrics.json")
    delta={k:100*(v-binding["baseline_metrics"][k]) for k,v in full["metrics"].items()}
    compare=dict(status="passed",metrics=full["metrics"],baseline_metrics=binding["baseline_metrics"],
        delta_points=delta,groups={k:group(v) for k,v in (
            ("all_small",pairs),("tiny",[p for p in pairs if p["area"]<=256]),
            ("small_257_1024",[p for p in pairs if p["area"]>256]),
            ("old_geometry_gains",[p for p in pairs if p["geometry_gain"]]),
            ("old_geometry_losses",[p for p in pairs if p["geometry_loss"]]))},
        positive_examples=sorted(pairs,key=lambda p:p["after"]-p["before"],reverse=True)[:20],
        negative_examples=sorted(pairs,key=lambda p:p["after"]-p["before"])[:20],
        any_positive_aps_retained=delta["APs"]>0,subset_not_ap=True,single_development_val=True)
    save(OUT/"comparison.json",compare);save(OUT/"paired_gt.json",pairs)
    assert all(sha(ROOT/p)==h for p,h in binding["input_sha256"].items())
    check()
    artifacts=("cpu_acceptance.json","headroom_gate.json","prepared_manifest.json","binding.json",
        "fit_report.json","fit_history.json","refiner_3000.pth","gpu_report.json","timing.json",
        "parity.json","predictions_frozen.json","predictions.segm.json","val_input_hashes.json",
        "refinement_activity.json","full_val_metrics.json","comparison.json","paired_gt.json")
    save(OUT/"report.json",dict(status="passed",outcome="evaluated",metrics=full["metrics"],
        delta_points=delta,start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),
        full_val_evaluations=1,complete_val_images=2584,refiner_trained=True,m2f_updated=False,
        test_inputs_used=False,test_annotations_used=False,maskrcnn_paused=True,
        artifacts_sha256={n:sha(OUT/n) for n in artifacts}))


def main():
    phase=argparse.ArgumentParser()
    phase.add_argument("phase",choices=("prep","train","predict","evaluate","analyze"))
    args=phase.parse_args()
    limits={"prep":PREP_END,"train":TRAIN_END,"predict":EVAL_END,"evaluate":EVAL_END,"analyze":DEADLINE}
    check(limits[args.phase])
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Immutable stage deadline")))
    signal.alarm(max(1,math.ceil(limits[args.phase]-time.time())))
    try:
        globals()[args.phase]()
    except BaseException as exc:
        save(OUT/(args.phase+"_first_error.json"),dict(error=repr(exc),traceback=traceback.format_exc(),
                                                     phase=args.phase,unix=time.time()))
        raise


if __name__=="__main__":
    main()

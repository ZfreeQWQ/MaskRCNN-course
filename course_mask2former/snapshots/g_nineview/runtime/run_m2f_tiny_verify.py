"""Fixed tiny-entry experiment: CPU space -> 2000 steps -> actual gate -> one val."""
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
import run_m2f_boundary_refine as base
import run_m2f_supervision_correct as previous

START=1792509627
PREP_END,TRAIN_END,GPU_END,DEADLINE=(START+s for s in (1800,3900,5700,7200))
NAME="course_m2f_tiny_verify_20261020_v1"
OWNER="0b026b4e5a1543b69bf72eef6470382c"
ROOT=Path("/workspace")
OUT=ROOT/"outputs"/NAME
BEST=ROOT/"outputs/course_m2f_supervision_correct_20261020_v1"
OLD=ROOT/"outputs/course_m2f_boundary_refine_20261020_v1"
PARENT=ROOT/base.PARENT
BEST_SHA="f811ce35aa8a9b3f8d4cc45ee30e2c11bb6d4b11fe9fb7e19e34dcc70dbede63"
HEAD_SHA="cc9531891e9a59303cc866c3e1d3ef36f80012180f3155c74bba64f00cf257d8"
BEST_APS=.4128089961706839
read,save,sha=base.read,previous.save,base.sha


def check(end=None):
    assert time.time()<min(end or DEADLINE,DEADLINE),"Immutable deadline"


def mark(phase,**kwargs):
    check()
    data=dict(phase=phase,unix=time.time(),**kwargs)
    save(OUT/"progress.json",data);print(json.dumps(data),flush=True)


def install():
    base.OUT,base.ROOT,base.START,base.DEADLINE=OUT,ROOT,START,DEADLINE
    base.check,base.mark,base.save=check,mark,save


def load_current():
    import m2f_boundary_refiner as ref
    torch=base.configure_torch()
    assert sha(BEST/"refiner_3000.pth")==HEAD_SHA
    state=torch.load(BEST/"refiner_3000.pth",map_location="cpu")
    assert state["steps"]==3000
    model=ref.Refiner();model.load_state_dict(state["state_dict"],strict=True)
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    return model.eval()


def reconstruct(ident,manifest,model,device):
    """Use the original prepared x/band/order and padded32 inference."""
    import numpy as np
    import torch
    import m2f_boundary_refiner as ref
    meta=manifest["images"][str(ident)]
    path=OLD/"train_selected"/f"{ident}.json"
    assert sha(path)==manifest["files"][f"train_selected/{ident}.json"]
    rows=read(path);final=copy.deepcopy(rows);records=meta["records"]
    if records:
        path=OLD/"prepared"/f"{ident}.npz"
        assert sha(path)==manifest["files"][f"prepared/{ident}.npz"]
        with np.load(path) as z,torch.inference_mode():
            for start in range(0,len(records),32):
                group=records[start:start+32]
                data=np.zeros((32,6,128,128),np.float32)
                data[:len(group)]=z["x"][start:start+len(group)]
                predicted=(model(torch.from_numpy(data).to(device))>0).cpu().numpy()
                for j,record in enumerate(group):
                    k=start+j;index=record["index"]
                    assert tuple(record["xy"])==ref.window(rows[index])
                    final[index],_=ref.compose(rows[index],record["xy"],z["coarse"][k],
                                               z["band"][k],predicted[j])
    return final


def coco_load(split):
    from pycocotools.coco import COCO
    path=ROOT/base.DATA/"annotations"/f"instances_{split}.json"
    assert sha(path)==(base.TRAIN_SHA if split=="train" else base.VAL_SHA)
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def metrics(coco,ids,rows):
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    assert rows
    with contextlib.redirect_stdout(io.StringIO()):
        dt=coco.loadRes([{k:v for k,v in r.items() if k!="bbox"} for r in rows])
        ev=COCOeval(coco,dt,"segm");ev.params.imgIds=list(ids)
        ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
    result={k:float(v) for k,v in zip(("AP","AP50","AP75","APs","APm","APl"),ev.stats[:6])}
    assert all(np.isfinite(v) and 0<=v<=1 for v in result.values())
    return result


def matched(coco,ids,rows):
    """COCO small-area .5 matching; filter tiny only after one-to-one assignment."""
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        dt=coco.loadRes([{k:v for k,v in r.items() if k!="bbox"} for r in rows])
        ev=COCOeval(coco,dt,"segm");ev.params.imgIds=list(ids)
        ev.params.maxDets=[100];ev.params.iouThrs=np.array([.5])
        ev.params.areaRng=[[0,1024]];ev.params.areaRngLbl=["small"]
        ev.evaluate()
    hits=set();false_positives=[];true_predictions=[]
    for item in ev.evalImgs:
        if item is None:continue
        for j,gtid in enumerate(item["gtIds"]):
            if not item["gtIgnore"][j] and item["gtMatches"][0,j]>0:hits.add(int(gtid))
        for j,dtid in enumerate(item["dtIds"]):
            if item["dtIgnore"][0,j]:continue
            entry=dict(image_id=item["image_id"],detection_id=int(dtid),
                       score=float(item["dtScores"][j]))
            if item["dtMatches"][0,j]>0:
                entry["annotation_id"]=int(item["dtMatches"][0,j]);true_predictions.append(entry)
            else:false_positives.append(entry)
    return dict(small=sorted(hits),tiny=sorted(i for i in hits if coco.anns[i]["area"]<=256),
                false_positives=false_positives,true_predictions=true_predictions)


def prep():
    import unittest
    import numpy as np
    from pycocotools import mask as mu
    import m2f_boundary_refiner as ref
    import m2f_tiny_verifier as tiny
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not Path("/dev/nvidiactl").exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName("test_m2f_tiny_verifier"))
    assert result.wasSuccessful() and not result.skipped
    assert sha(BEST/"report.json")==BEST_SHA
    best=read(BEST/"report.json")
    assert best["status"]=="passed" and best["metrics"]["APs"]==BEST_APS
    assert read(BEST/"terminal_acceptance.json")["report_sha256"]==BEST_SHA
    for name,h in read(BEST/"plan.json")["source_sha256"].items():
        assert sha(OUT/"source"/name)==h,name
    manifest=read(BEST/"prepared_manifest.json")
    assert sha(BEST/"prepared_manifest.json")==best["artifacts_sha256"]["prepared_manifest.json"]
    assert len(manifest["fit_ids"])==384 and len(manifest["check_ids"])==128
    assert len(set(manifest["fit_ids"]+manifest["check_ids"]))==512 and manifest["scene_disjoint"]
    scenes=[{manifest["images"][str(i)]["request"]["scene"] for i in manifest[a+"_ids"]} for a in ("fit","check")]
    assert not scenes[0]&scenes[1]
    parent_manifest=read(PARENT/"train_manifest.json")
    assert all(parent_manifest[a+"_ids"]==manifest[a+"_ids"] for a in ("fit","check"))
    expected={p["annotation_id"]:p for p in read(BEST/"corrected_check128_paired_gt.json")}
    assert sha(BEST/"corrected_check128_paired_gt.json")==best["artifacts_sha256"]["corrected_check128_paired_gt.json"]
    coco=coco_load("train");model=load_current()
    for name in ("current_train","candidate_patches","train_metadata"):(OUT/name).mkdir()
    input_sha={str((BEST/n).relative_to(ROOT)):sha(BEST/n) for n in
               ("report.json","refiner_3000.pth","prepared_manifest.json","corrected_check128_paired_gt.json")}
    upstream=read(BEST/"binding.json")["input_sha256"]
    counts={"positive":0,"negative":0,"ambiguous":0};opportunities=[];records={};matched_old=0
    # Check images first, but all 384 fit images must also pass source and label checks.
    ids=manifest["check_ids"]+manifest["fit_ids"]
    for n,ident in enumerate(ids):
        check(PREP_END)
        meta=manifest["images"][str(ident)];current=reconstruct(ident,manifest,model,"cpu")
        cachepath=PARENT/"train_predictions"/f"{ident}.json"
        rel=str(cachepath.relative_to(ROOT));digest=sha(cachepath)
        assert rel in upstream and digest==upstream[rel],rel
        input_sha[rel]=digest
        cache=read(cachepath);assert cache["image_id"]==ident
        merged=cache["merged"];anns=sorted(coco.loadAnns(coco.getAnnIds(imgIds=[ident])),key=lambda a:a["id"])
        small=[a for a in anns if not a.get("iscrowd",0) and a["area"]<=1024]
        matrix=mu.iou([r["segmentation"] for r in current],[coco.annToRLE(a) for a in small],[0]*len(small))
        for j,a in enumerate(small):
            if a["id"] in expected:
                value=float(matrix[:,j].max()) if current else 0.
                assert abs(value-expected[a["id"]]["after"])<1e-12,(a["id"],value,expected[a["id"]]["after"])
                matched_old+=1
        jobs=tiny.eligible(current,merged)
        request=meta["request"]
        actual=dict(image_id=ident,rgb_sha256=sha(request["img_path"]),depth_sha256=sha(request["depth_file"]))
        assert actual==meta["input_hashes"] and actual==cache["input_hashes"]
        rgbd=ref.load_rgbd(request);patches=[]
        for job in jobs:
            row=merged[job["index"]]
            same=[a for a in anns if a["category_id"]==row["category_id"] and not a.get("iscrowd",0)]
            values=mu.iou([row["segmentation"]],[coco.annToRLE(a) for a in same],[0]*len(same))[0] if same else []
            k=int(np.argmax(values)) if len(same) else None
            value=float(values[k]) if k is not None else 0.
            ann=same[k] if k is not None and value>0 else None
            label=tiny.target(value,ann["area"] if ann else None)
            job.update(label=label,gt_id=ann["id"] if ann else None,gt_area=ann["area"] if ann else None,max_gt_iou=value)
            patches.append(tiny.patch(row,job["xy"],rgbd))
            if ident in manifest["fit_ids"]:
                counts[{1:"positive",0:"negative",-1:"ambiguous"}[label]]+=1
            if ident in manifest["check_ids"] and label==1 and expected[ann["id"]]["after"]<.5:
                opportunities.append(dict(image_id=ident,annotation_id=ann["id"],candidate_index=job["index"],
                    current_iou=expected[ann["id"]]["after"],candidate_iou=value,max_current_iou=job["max_current_iou"]))
        np.savez_compressed(OUT/"candidate_patches"/f"{ident}.npz",
                            x=np.stack(patches) if patches else np.empty((0,6,128,128),np.float32))
        save(OUT/"current_train"/f"{ident}.json",current)
        save(OUT/"train_metadata"/f"{ident}.json",dict(image_id=ident,arm=meta["arm"],jobs=jobs,input_hashes=actual))
        records[str(ident)]=dict(count=len(jobs),labels=[j["label"] for j in jobs])
        if n%8==0:mark("cpu_space",images=n+1,expected=512,fit_counts=counts,opportunities=len({p["annotation_id"] for p in opportunities}))
    assert matched_old==435
    # At least a full balanced batch is required; no repetition can invent missing classes.
    passed=bool(opportunities and counts["positive"]>=16 and counts["negative"]>=16)
    save(OUT/"prepared_manifest.json",dict(fit_ids=manifest["fit_ids"],check_ids=manifest["check_ids"],
         records=records,source_manifest_sha256=sha(BEST/"prepared_manifest.json"),scene_disjoint=True))
    save(OUT/"binding.json",dict(input_sha256=input_sha,baseline_metrics=best["metrics"],
        baseline_report_sha256=BEST_SHA,baseline_head_sha256=HEAD_SHA,
        selector_sha256=sha(PARENT/"selector.json"),source_best28_unchanged=True))
    save(OUT/"headroom_gate.json",dict(status="passed",gate_passed=passed,fit_counts=counts,
        opportunities=opportunities,unique_check_tiny_opportunities=len({p["annotation_id"] for p in opportunities}),
        accepted_check_gt_exact=matched_old,gt_only_labels_and_audit=True,validation_used=False,
        current_train_reconstructed_cpu=True,current_model_updated=False,finished_unix=time.time()))
    save(OUT/"cpu_acceptance.json",dict(status="passed",tests=result.testsRun,skipped=0,
        gpu_used=False,scene_disjoint=True,gate_passed=passed,finished_unix=time.time()))
    check(PREP_END)


def traincheck():
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    from run_m2f_quality_select import tensor_digest
    torch=base.configure_torch()
    assert torch.cuda.device_count()==1 and read(OUT/"headroom_gate.json")["gate_passed"]
    end=read(OUT/"phase_deadlines.json")["train_end"]
    manifest=read(OUT/"prepared_manifest.json");original=read(BEST/"prepared_manifest.json")
    # Exact per-instance CPU/GPU reconstruction parity, not just the GT summaries.
    old=load_current().cuda()
    for n,ident in enumerate(manifest["check_ids"]+manifest["fit_ids"]):
        check(end)
        assert reconstruct(ident,original,old,"cuda")==read(OUT/"current_train"/f"{ident}.json"),ident
        if n%16==0:mark("current_train_gpu_parity",images=n+1,expected=512)
    old.cpu();del old;torch.cuda.empty_cache()
    save(OUT/"current_train_parity.json",dict(status="passed",images=512,instances_exact=True,old_head_updated=False))
    arrays=[];labels=[]
    for ident in manifest["fit_ids"]:
        check(end)
        with np.load(OUT/"candidate_patches"/f"{ident}.npz") as z:
            labs=np.asarray(manifest["records"][str(ident)]["labels"])
            keep=labs>=0
            arrays.append(z["x"][keep]);labels.extend(labs[keep].tolist())
    x=torch.from_numpy(np.concatenate(arrays));del arrays
    y=torch.tensor(labels,dtype=torch.float32)
    assert len(x)==len(y) and x.numel()*x.element_size()<10*2**30
    pos=torch.where(y==1)[0];neg=torch.where(y==0)[0]
    assert len(pos)>=16 and len(neg)>=16
    # Reset RNG after frozen-head reconstruction so initialization is independent of it.
    torch.manual_seed(20261013);np.random.seed(20261013)
    model=tiny.Verifier().cuda().train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(20261013);history=[];begun=time.time()
    for step in range(1,2001):
        check(end)
        inds=torch.cat((pos[torch.randint(len(pos),(16,),generator=rng)],
                        neg[torch.randint(len(neg),(16,),generator=rng)]))
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[inds].cuda()),y[inds].cuda())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        # No extra gradient clipping or schedule is introduced.
        grad=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters())))
        assert math.isfinite(grad)
        opt.step();history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad))
        if step==1 or step%50==0:mark("fit_tiny_verifier",step=step,expected=2000,loss=history[-1]["loss"])
    torch.cuda.synchronize();fit_seconds=time.time()-begun
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    native=sorted({int(v["step"].item()) for v in opt.state.values()});assert native==[2000]
    assert all(bool(torch.isfinite(v).all()) for v in state.values())
    assert all(not torch.equal(initial[k],v) for k,v in state.items())
    assert all(bool(torch.isfinite(v).all()) for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    torch.save(dict(state_dict=state,optimizer=opt.state_dict(),steps=2000,fit_ids=manifest["fit_ids"],seed=20261013),
               OUT/"verifier_2000.pth")
    save(OUT/"fit_history.json",history)
    save(OUT/"fit_report.json",dict(status="passed",native_steps=native,finite_steps=2000,consumed_patches=64000,
        positive_patches=int(len(pos)),negative_patches=int(len(neg)),fit_images=384,check_used_for_fit=0,
        parameters=sum(p.numel() for p in model.parameters()),state_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        initial_state_sha256=tensor_digest(initial),final_state_sha256=tensor_digest(state),
        all_parameters_updated=True,model_optimizer_finite=True,fit_seconds=fit_seconds,
        checkpoint_sha256=sha(OUT/"verifier_2000.pth"),m2f_loaded=False,m2f_updated=False,finished_unix=time.time()))
    del x,y,opt;model.eval()
    before=[];after=[];activity=[]
    for n,ident in enumerate(manifest["check_ids"]):
        check(end);current=read(OUT/"current_train"/f"{ident}.json")
        cache=read(PARENT/"train_predictions"/f"{ident}.json")
        meta=read(OUT/"train_metadata"/f"{ident}.json")
        jobs=meta["jobs"];probabilities=[]
        with np.load(OUT/"candidate_patches"/f"{ident}.npz") as z,torch.inference_mode():
            for start in range(0,len(jobs),32):
                batch=np.zeros((32,6,128,128),np.float32);num=min(32,len(jobs)-start)
                batch[:num]=z["x"][start:start+num]
                probabilities.extend(map(float,torch.sigmoid(model(torch.from_numpy(batch).cuda())).cpu().numpy()[:num]))
        final,detail=tiny.merge(current,cache["merged"],jobs,probabilities)
        before.extend(current);after.extend(final)
        detail.update(image_id=ident,candidates=[dict(**j,probability=p) for j,p in zip(jobs,probabilities)])
        activity.append(detail)
        if n%16==0:mark("internal_actual",images=n+1,expected=128)
    coco=coco_load("train")
    bm=metrics(coco,manifest["check_ids"],before);am=metrics(coco,manifest["check_ids"],after)
    bp=matched(coco,manifest["check_ids"],before);ap=matched(coco,manifest["check_ids"],after)
    gained=sorted(set(ap["tiny"])-set(bp["tiny"]));lost=sorted(set(bp["tiny"])-set(ap["tiny"]))
    passed=am["APs"]>bm["APs"] and len(gained)>len(lost)
    save(OUT/"check_before.json",before);save(OUT/"check_after.json",after);save(OUT/"check_activity.json",activity)
    save(OUT/"check_matches.json",dict(before=bp,after=ap,tiny_gained=gained,tiny_lost=lost))
    save(OUT/"learned_gate.json",dict(status="passed",gate_passed=passed,baseline_metrics=bm,metrics=am,
        delta_points={k:100*(am[k]-bm[k]) for k in bm},tiny_gained=gained,tiny_lost=lost,
        tiny_net=len(gained)-len(lost),rule="small AP strict positive AND tiny .5 one-to-one net>0",
        fixed_checkpoint=2000,val_used=False,check_not_fit=True,
        repeated_development_check_selection_bias=True,finished_unix=time.time()))
    check(end)


def load_verifier():
    import m2f_tiny_verifier as tiny
    torch=base.configure_torch();state=torch.load(OUT/"verifier_2000.pth",map_location="cpu")
    assert state["steps"]==2000
    model=tiny.Verifier();model.load_state_dict(state["state_dict"],strict=True)
    return model.cuda().eval()


def predict():
    import m2f_boundary_refiner as ref
    import m2f_tiny_verifier as tiny
    import m2f_quality_selector as quality
    import run_m2f_quality_select as original
    torch=base.configure_torch();model=load_verifier()
    end=read(OUT/"phase_deadlines.json")["predict_end"]
    binding=read(OUT/"binding.json")
    inputs=read(PARENT/"source_binding.json")["full_val_cache_sha256"]
    path=ROOT/"outputs/course_m2f_tiles_20261019_v1/evaluate/predictions.segm.json"
    assert sha(path)==inputs[str(path.relative_to(ROOT))]
    merged_rows=read(path)
    assert sha(BEST/"predictions.segm.json")==read(BEST/"report.json")["artifacts_sha256"]["predictions.segm.json"]
    best_rows=read(BEST/"predictions.segm.json")
    coco=coco_load("val");ids=read(PARENT/"full_val_metrics.json")["image_ids"]
    assert len(ids)==len(set(ids))==2584 and set(ids)==set(coco.imgs)
    requests={i:base.image_request(coco.imgs[i],"val") for i in ids}
    by_image={i:[] for i in ids};by_merged={i:[] for i in ids}
    for r in best_rows:by_image[r["image_id"]].append(r)
    for r in merged_rows:by_merged[r["image_id"]].append(r)
    del best_rows,merged_rows
    expected={v["image_id"]:v for v in read(BEST/"val_input_hashes.json")}
    assert sha(BEST/"val_input_hashes.json")==read(BEST/"report.json")["artifacts_sha256"]["val_input_hashes.json"]
    (OUT/"val_predictions").mkdir();all_rows=[];hashes=[];activity=[];begun=time.time()
    for n,ident in enumerate(ids):
        check(end);request=requests[ident]
        h=dict(image_id=ident,rgb_sha256=sha(request["img_path"]),depth_sha256=sha(request["depth_file"]))
        assert h==expected[ident]
        final,detail=tiny.infer(by_image[ident],by_merged[ident],ref.load_rgbd(request),model,"cuda")
        save(OUT/"val_predictions"/f"{ident}.json",final)
        all_rows.extend(final);hashes.append(h);activity.append(dict(image_id=ident,**detail))
        if n%16==0:mark("new_head_full_val",images=n+1,expected=2584)
    save(OUT/"predictions.segm.json",all_rows);del all_rows
    save(OUT/"val_input_hashes.json",hashes);save(OUT/"val_activity.json",activity)
    save(OUT/"predictions_frozen.json",dict(prediction_sha256=sha(OUT/"predictions.segm.json"),
        verifier_sha256=sha(OUT/"verifier_2000.pth"),image_ids=ids,images=2584,unix=time.time()))
    cache_seconds=time.time()-begun
    old_model=load_current().cuda()
    original.START,original.DEADLINE=START,DEADLINE
    accepted,earlier,tiles,_=original.modules()
    from mmengine.config import Config
    from mmengine.dataset import Compose,pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    compact=read(PARENT/"compact.json");cfg=Config.fromfile(str(PARENT/"live_config.py"))
    register_all_modules(init_default_scope=True);import_modules_from_strings(**cfg.custom_imports)
    assert sha(compact["weight"])==tiles.SPEC["weight_sha"]
    cp=torch.load(compact["weight"],map_location="cpu")
    m2f=MODELS.build(cfg.model);m2f.load_state_dict(cp["state_dict"],strict=True);del cp
    before=original.tensor_digest(m2f.state_dict());m2f.cuda().eval()
    pipeline=earlier.typed_pipeline(cfg,compact,tiles);loader,tail=Compose(pipeline[:1]),Compose(pipeline[1:])
    selector=read(PARENT/"selector.json");assert sha(PARENT/"selector.json")==binding["selector_sha256"]
    calls=dict(eval_calls=0,train_calls=0)
    def full(ident,enabled):
        check(end);torch.cuda.synchronize();start=time.perf_counter()
        torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.deterministic=False
        views=tiles.pack_views(requests[ident],loader,tail)
        encoder=accepted.CandidateEncoder(tiles);pool=[];global_rows=[]
        for index,view in enumerate(views):
            predicted=m2f.test_step(pseudo_collate([view]))[0].cpu()
            captured,candidates=encoder.encode(predicted.pred_instances,index,ident)
            calls["eval_calls"]+=1
            if index==0:global_rows=captured
            pool.extend(candidates)
        merged=accepted.base.direct_rows(tiles.merge_candidates(pool),ident)
        chosen=quality.choose(global_rows,merged,selector)[0]
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
        rgbd=ref.load_rgbd(requests[ident])
        current=ref.infer(chosen,rgbd,old_model,"cuda")[0]
        final=tiny.infer(current,merged,rgbd,model,"cuda")[0] if enabled else current
        json.dumps(final,separators=(",",":"),allow_nan=False);torch.cuda.synchronize()
        assert encoder.encoded==encoder.roundtrips
        return current,merged,final,(time.perf_counter()-start)*1000
    parity=[]
    with torch.inference_mode():
        for ident in base.FIXED8:
            current,merged,final,_=full(ident,True)
            assert current==by_image[ident] and merged==by_merged[ident]
            assert final==read(OUT/"val_predictions"/f"{ident}.json")
            parity.append(dict(image_id=ident,current_exact=True,merged_exact=True,final_exact=True))
            mark("live_parity",images=len(parity),expected=8)
        timing={k:dict(milliseconds=[],peak_allocated_mib=[],peak_reserved_mib=[]) for k in ("baseline","tiny_entry")}
        ident=base.FIXED8[0]
        for repeat in range(25):
            for key in (("baseline","tiny_entry") if repeat%2==0 else ("tiny_entry","baseline")):
                enabled=key=="tiny_entry";model.to("cuda" if enabled else "cpu")
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                current,merged,final,ms=full(ident,enabled)
                assert current==by_image[ident]
                assert final==(read(OUT/"val_predictions"/f"{ident}.json") if enabled else by_image[ident])
                if repeat>=5:
                    timing[key]["milliseconds"].append(ms)
                    timing[key]["peak_allocated_mib"].append(torch.cuda.max_memory_allocated()/2**20)
                    timing[key]["peak_reserved_mib"].append(torch.cuda.max_memory_reserved()/2**20)
            mark("paired_timing",completed=repeat+1,expected=25)
    assert calls==dict(eval_calls=290,train_calls=0) and original.tensor_digest(m2f.state_dict())==before
    assert sha(BEST/"refiner_3000.pth")==HEAD_SHA
    for v in timing.values():
        v.update(mean_ms=statistics.mean(v["milliseconds"]),median_ms=statistics.median(v["milliseconds"]),
            peak_allocated_max_mib=max(v["peak_allocated_mib"]),peak_reserved_max_mib=max(v["peak_reserved_mib"]))
    save(OUT/"timing.json",dict(warmup=5,repeats=20,paired=timing,
        scope="read/decode/five-view/M2F/D2H/RLE/merge/selector/correction/tiny verifier/NMS/bbox/JSON; excludes output write"))
    save(OUT/"parity.json",parity)
    save(OUT/"gpu_report.json",dict(status="passed",new_head_val_images=2584,m2f_calls=calls,
        m2f_state_before=before,m2f_state_after=before,m2f_updated=False,old_head_unchanged=True,
        strict_parity_images=8,full_m2f_val_repeated=False,new_head_val_seconds=cache_seconds,finished_unix=time.time()))


def finish(stopped=None):
    trained=(OUT/"fit_report.json").exists()
    compare={}
    if not stopped:
        check(GPU_END)
        assert read(OUT/"learned_gate.json")["gate_passed"]
        ids=read(OUT/"predictions_frozen.json")["image_ids"]
        coco=coco_load("val");rows=read(OUT/"predictions.segm.json")
        assert sha(OUT/"predictions.segm.json")==read(OUT/"predictions_frozen.json")["prediction_sha256"]
        current=metrics(coco,ids,rows)
        save(OUT/"full_val_metrics.json",dict(metrics=current,image_ids=ids,images=2584,evaluations=1))
        best=read(BEST/"report.json")["metrics"]
        subset_ids=sorted({p["image_id"] for p in read(BEST/"paired_gt.json")});assert len(subset_ids)==256
        before=[r for r in read(BEST/"predictions.segm.json") if r["image_id"] in set(subset_ids)]
        after=[r for r in rows if r["image_id"] in set(subset_ids)]
        bm=matched(coco,subset_ids,before);am=matched(coco,subset_ids,after)
        save(OUT/"val_subset_matches.json",dict(before=bm,after=am,
            tiny_gained=sorted(set(am["tiny"])-set(bm["tiny"])),tiny_lost=sorted(set(bm["tiny"])-set(am["tiny"]))))
        activity=read(OUT/"val_activity.json")
        compare=dict(metrics=current,baseline_metrics=best,delta_points={k:100*(current[k]-best[k]) for k in best},
            tiny_one_to_one_gain=len(set(am["tiny"])-set(bm["tiny"])),
            tiny_one_to_one_loss=len(set(bm["tiny"])-set(am["tiny"])),
            subset_small_fp_before=len(bm["false_positives"]),subset_small_fp_after=len(am["false_positives"]),
            original_nms_removed=sum(sum(p["origin"]=="original" for p in a["suppressed"]) for a in activity),
            original_top100_removed=sum(sum(p["origin"]=="original" for p in a["top100_dropped"]) for a in activity),
            single_development_val=True,subset_not_full_ap=True,any_positive_retained=current["APs"]>BEST_APS)
        save(OUT/"comparison.json",compare)
    binding=read(OUT/"binding.json")
    assert all(sha(ROOT/p)==h for p,h in binding["input_sha256"].items())
    names=[p.name for p in OUT.glob("*.json") if p.name not in
           ("status.json","progress.json","report.json","terminal_acceptance.json","plan.json","launch.json",
            "phases.json","phase_deadlines.json") and not p.name.endswith(("_command.json","_container.json"))]
    if trained:names+=["verifier_2000.pth"]
    check()
    save(OUT/"report.json",dict(status="passed",outcome="stopped_failed_"+stopped if stopped else "evaluated",
        start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),baseline_aps=BEST_APS,
        full_val_evaluations=int(not stopped),new_aps=compare.get("metrics",{}).get("APs"),
        verifier_trained=trained,m2f_updated=False,old_refiner_updated=False,maskrcnn_paused=True,
        artifacts_sha256={n:sha(OUT/n) for n in names},comparison=compare))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prep","traincheck","predict","finish"))
    phase=parser.parse_args().phase;install()
    end=dict(prep=PREP_END,traincheck=TRAIN_END,predict=GPU_END,finish=DEADLINE)[phase]
    if phase in ("traincheck","predict"):
        end=min(end,read(OUT/"phase_deadlines.json")["train_end" if phase=="traincheck" else "predict_end"])
    check(end)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Immutable stage deadline")))
    signal.alarm(max(1,math.ceil(end-time.time())))
    try:
        if phase=="finish":
            stopped="headroom_gate" if not read(OUT/"headroom_gate.json")["gate_passed"] else (
                "learned_gate" if not read(OUT/"learned_gate.json")["gate_passed"] else None)
            finish(stopped)
        else:globals()[phase]()
    except BaseException as exc:
        save(OUT/(phase+"_first_error.json"),dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise


if __name__=="__main__":main()

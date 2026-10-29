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

START=1792513278
PREP_END,TRAIN_END,GPU_END,DEADLINE=(START+s for s in (900,1800,2700,3600))
NAME="course_m2f_tiny_positive_20261021_v1"
OWNER="aadc2288f49b44ceab83470421786f20"
ROOT=Path("/workspace")
OUT=ROOT/"outputs"/NAME
BEST=ROOT/"outputs/course_m2f_supervision_correct_20261020_v1"
OLD=ROOT/"outputs/course_m2f_boundary_refine_20261020_v1"
PARENT=ROOT/base.PARENT
BEST_SHA="f811ce35aa8a9b3f8d4cc45ee30e2c11bb6d4b11fe9fb7e19e34dcc70dbede63"
HEAD_SHA="cc9531891e9a59303cc866c3e1d3ef36f80012180f3155c74bba64f00cf257d8"
BEST_APS=.4131279352535198
REFERENCE=ROOT/"outputs/course_m2f_tiny_verify_20261020_v1"
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
    import m2f_tiny_positive_helpers as h
    h.prep(__import__("sys").modules[__name__])


def traincheck():
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    from run_m2f_quality_select import tensor_digest
    torch=base.configure_torch()
    assert torch.cuda.device_count()==1 and read(OUT/"headroom_gate.json")["gate_passed"]
    end=read(OUT/"phase_deadlines.json")["train_end"]
    manifest=read(OUT/"prepared_manifest.json")
    # Reuse accepted exact512 old-head parity; no frozen-head inference is repeated.
    assert read(REFERENCE/"current_train_parity.json")["instances_exact"]
    save(OUT/"current_train_parity.json",dict(status="passed",images=512,instances_exact=True,
        reused=True,source_sha256=sha(REFERENCE/"current_train_parity.json"),new_gpu_calls=0))
    arrays=[];labels=[]
    for ident in manifest["fit_ids"]:
        check(end)
        with np.load(REFERENCE/"candidate_patches"/f"{ident}.npz") as z:
            labs=np.asarray(manifest["records"][str(ident)]["labels"])
            keep=labs>=0
            arrays.append(z["x"][keep]);labels.extend(labs[keep].tolist())
    for ident in manifest["fit_ids"]:
        path=OUT/"extra_positive"/f"{ident}.npz"
        if path.exists():
            with np.load(path) as z:
                arrays.append(z["x"]);labels.extend([1]*len(z["x"]))
    x=torch.from_numpy(np.concatenate(arrays));del arrays
    y=torch.tensor(labels,dtype=torch.float32)
    assert len(x)==len(y) and x.numel()*x.element_size()<10*2**30
    pos=torch.where(y==1)[0];neg=torch.where(y==0)[0]
    assert len(pos)>=16 and len(neg)>=16
    # Reset RNG after frozen-head reconstruction so initialization is independent of it.
    torch.manual_seed(20261013);np.random.seed(20261013)
    model=tiny.Verifier().cuda().train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert tensor_digest(initial)==read(REFERENCE/"fit_report.json")["initial_state_sha256"]
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
    before=read(REFERENCE/"check_after.json");after=[];activity=[]
    for n,ident in enumerate(manifest["check_ids"]):
        check(end);current=read(REFERENCE/"current_train"/f"{ident}.json")
        cache=read(PARENT/"train_predictions"/f"{ident}.json")
        meta=read(REFERENCE/"train_metadata"/f"{ident}.json")
        jobs=meta["jobs"];probabilities=[]
        with np.load(REFERENCE/"candidate_patches"/f"{ident}.npz") as z,torch.inference_mode():
            for start in range(0,len(jobs),32):
                batch=np.zeros((32,6,128,128),np.float32);num=min(32,len(jobs)-start)
                batch[:num]=z["x"][start:start+num]
                probabilities.extend(map(float,torch.sigmoid(model(torch.from_numpy(batch).cuda())).cpu().numpy()[:num]))
        final,detail=tiny.merge(current,cache["merged"],jobs,probabilities)
        after.extend(final)
        detail.update(image_id=ident,candidates=[dict(**j,probability=p) for j,p in zip(jobs,probabilities)])
        activity.append(detail)
        if n%16==0:mark("internal_actual",images=n+1,expected=128)
    coco=coco_load("train")
    bm=read(REFERENCE/"learned_gate.json")["metrics"];am=metrics(coco,manifest["check_ids"],after)
    bp=read(REFERENCE/"check_matches.json")["after"];ap=matched(coco,manifest["check_ids"],after)
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
    ref_state=torch.load(REFERENCE/"verifier_2000.pth",map_location="cpu")
    reference_model=tiny.Verifier();reference_model.load_state_dict(ref_state["state_dict"],strict=True)
    reference_model.eval().cpu();del ref_state
    reference_rows={i:[] for i in ids}
    for row in read(REFERENCE/"predictions.segm.json"):reference_rows[row["image_id"]].append(row)
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
    def full(ident,enabled,verify_old=False):
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
        final=tiny.infer(current,merged,rgbd,model if enabled else reference_model,"cuda")[0]
        if verify_old:
            reference_model.cuda()
            assert tiny.infer(current,merged,rgbd,reference_model,"cuda")[0]==reference_rows[ident]
            reference_model.cpu()
        json.dumps(final,separators=(",",":"),allow_nan=False);torch.cuda.synchronize()
        assert encoder.encoded==encoder.roundtrips
        return current,merged,final,(time.perf_counter()-start)*1000
    parity=[]
    with torch.inference_mode():
        for ident in base.FIXED8:
            current,merged,final,_=full(ident,True,verify_old=True)
            assert current==by_image[ident] and merged==by_merged[ident]
            assert final==read(OUT/"val_predictions"/f"{ident}.json")
            parity.append(dict(image_id=ident,current_exact=True,merged_exact=True,final_exact=True))
            mark("live_parity",images=len(parity),expected=8)
        timing={k:dict(milliseconds=[],peak_allocated_mib=[],peak_reserved_mib=[]) for k in ("baseline","tiny_entry")}
        ident=base.FIXED8[0]
        for repeat in range(25):
            for key in (("baseline","tiny_entry") if repeat%2==0 else ("tiny_entry","baseline")):
                enabled=key=="tiny_entry";model.to("cuda" if enabled else "cpu")
                reference_model.to("cpu" if enabled else "cuda")
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                current,merged,final,ms=full(ident,enabled)
                assert current==by_image[ident]
                assert final==(read(OUT/"val_predictions"/f"{ident}.json") if enabled else reference_rows[ident])
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
    import m2f_tiny_positive_helpers as h
    h.finish(__import__("sys").modules[__name__],stopped)


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


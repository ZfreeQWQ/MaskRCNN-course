"""One final-convolution stride change, fixed data/initialization/steps/gates."""
import argparse
import collections
import math
import os
from pathlib import Path
import signal
import time
import traceback
import unittest
import run_m2f_tiny_scene as scene
import run_m2f_tiny_positive as w
import m2f_tiny_positive_helpers as helpers

START=1792519800
PREP_END,TRAIN_END,GPU_END,EVAL_END,DEADLINE=(START+s for s in (1200,2400,3600,4800,5400))
NAME="course_m2f_tiny_stride4_20261021_v1"
OWNER="f7a1eb1731be4763a105c14fc5b09b06"
ROOT=Path("/workspace")
OUT=ROOT/"outputs"/NAME
REFERENCE=ROOT/"outputs/course_m2f_tiny_scene_20261021_v1"
OLD_POS=ROOT/"outputs/course_m2f_tiny_positive_20261021_v1"
DATA_REF=ROOT/"outputs/course_m2f_tiny_verify_20261020_v1"
BEST_APS=.41396866834940094
read,save,sha=w.read,w.save,w.sha


def new_model():
    import m2f_tiny_verifier as tiny
    model=tiny.Verifier()
    assert model.features[6].stride==(2,2)
    model.features[6].stride=(1,1)
    return model


def load_verifier():
    import torch
    state=torch.load(OUT/"verifier_2000.pth",map_location="cpu")
    assert state["steps"]==2000 and state["final_stride"]==1
    model=new_model();model.load_state_dict(state["state_dict"],strict=True)
    return model.cuda().eval()


def install():
    for key in ("START","PREP_END","TRAIN_END","GPU_END","DEADLINE","NAME","OWNER",
                "ROOT","OUT","REFERENCE","BEST_APS"):
        setattr(w,key,globals()[key])
    w.install()
    w.load_verifier=load_verifier
    # check_arm reads accepted patches from REFERENCE, never writes there.
    scene.OUT=REFERENCE
    scene.DATA_REF=DATA_REF


class Contracts(unittest.TestCase):
    def test_same_initial_tensors_and_rng(self):
        import torch
        import m2f_tiny_verifier as tiny
        torch.manual_seed(20261013);a=tiny.Verifier();ar=torch.get_rng_state()
        torch.manual_seed(20261013);b=new_model();br=torch.get_rng_state()
        self.assertTrue(torch.equal(ar,br))
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))
        self.assertEqual(sum(p.numel() for p in b.parameters()),24369)

    def test_unique_stride_difference(self):
        import m2f_tiny_verifier as tiny
        a,b=tiny.Verifier(),new_model()
        diffs=[(n,x.stride,y.stride) for (n,x),(_,y) in zip(a.named_modules(),b.named_modules())
               if hasattr(x,"stride") and x.stride!=y.stride]
        self.assertEqual(diffs,[("features.6",(2,2),(1,1))])

    def test_actual_grid_and_finite_backward(self):
        import torch
        m=new_model();x=torch.zeros(2,6,128,128);x[:,5,63,63]=1
        self.assertEqual(tuple(m.features(x).shape),(2,64,32,32))
        y=m(x);self.assertTrue(bool(torch.isfinite(y).all()))
        y.sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))

    def test_area_mask_not_erased(self):
        import torch
        x=torch.zeros(1,1,128,128);x[:,:,63,63]=1
        for size in (16,32):
            self.assertGreater(float(torch.nn.functional.interpolate(x,size=(size,size),mode="area").sum()),0)

    def test_bounds(self):
        self.assertEqual(DEADLINE-START,5400)
        self.assertEqual(PREP_END-START,1200)


def prep():
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    from run_m2f_quality_select import tensor_digest
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not Path("/dev/nvidiactl").exists()
    torch=w.base.configure_torch()
    suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromName("test_m2f_tiny_verifier"),
                             unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)))
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful() and not tests.skipped
    assert sha(REFERENCE/"report.json")=="0ff473e602ed3c929b23d8a238cdb73e7f8c94527d489b63a24c4f464f3830da"
    assert read(REFERENCE/"status.json")["status"]=="complete"
    assert read(REFERENCE/"report.json")["new_aps"]==BEST_APS
    plan=read(REFERENCE/"plan.json")
    assert all(sha(OUT/"source"/n)==h for n,h in plan["source_sha256"].items())
    inputs={}
    def bind(path,expected=None):
        w.check(PREP_END);h=sha(path)
        if expected is not None:assert h==expected,str(path)
        inputs[str(path.relative_to(ROOT))]=h
        return h
    for n in ("prepared_manifest.json","sample_manifest.json","fit_report.json","check_after.json",
              "learned_gate.json","check_matches.json","predictions.segm.json","full_val_metrics.json"):
        bind(REFERENCE/n,read(REFERENCE/"report.json")["artifacts_sha256"][n])
    bind(REFERENCE/"verifier_2000.pth","ca2bbf549d1c5ddd9c4e140c5718e90185f295d0850c1d60865acb7d1d5b8827")
    m=read(REFERENCE/"prepared_manifest.json");sm=read(REFERENCE/"sample_manifest.json")
    old=read(DATA_REF/"prepared_manifest.json")
    assert len(m["old_fit_ids"]+m["fit_ids"])==640 and len(m["check_ids"])==128
    assert not set(m["old_fit_ids"]+m["fit_ids"])&set(m["check_ids"])
    # Validate all accepted input links, not a copy or regenerated sample pool.
    upstream=read(REFERENCE/"binding.json")["input_sha256"]
    for rel,h in upstream.items():bind(ROOT/rel,h)
    for rel,h in sm["files"].items():bind(REFERENCE/rel,h)
    capture=read(REFERENCE/"capture_report.json")
    for item in capture["files"]:bind(REFERENCE/"train_predictions"/(str(item["image_id"])+".json"),item["sha256"])
    expansion=read(OLD_POS/"positive_expansion.json")
    extra=collections.defaultdict(list)
    for item in expansion["added"]:extra[item["image_id"]].append(item)
    model=tiny.Verifier()
    model.load_state_dict(torch.load(REFERENCE/"verifier_2000.pth",map_location="cpu")["state_dict"],strict=True)
    model.eval();coco=w.coco_load("train")
    positive_records=[];counts=collections.Counter()
    def inspect(path,jobs,ident,origin):
        labs=np.asarray([j["label"] for j in jobs]);counts.update(labs.tolist())
        chosen=[k for k,j in enumerate(jobs) if j["label"]==1 and j["gt_area"]<=64]
        if not chosen:return
        with np.load(path) as z:
            assert len(z["x"])==len(jobs)
            for start in range(0,len(chosen),32):
                inds=chosen[start:start+32];data=np.zeros((32,6,128,128),np.float32)
                data[:len(inds)]=z["x"][inds]
                with torch.inference_mode():
                    probs=torch.sigmoid(model(torch.from_numpy(data))).numpy()
                for q,k in enumerate(inds):
                    j=jobs[k];mask=torch.from_numpy(data[q:q+1,5:6])
                    support={}
                    for size in (16,32):
                        pooled=torch.nn.functional.interpolate(mask,size=(size,size),mode="area")
                        support[str(size)]=dict(nonzero=int((pooled>0).sum()),
                            effective_cells=float(pooled.sum().square()/pooled.square().sum().clamp_min(1e-12)))
                    positive_records.append(dict(image_id=ident,gt_id=j["gt_id"],gt_area=j["gt_area"],
                        scene=coco.imgs[ident]["scene_id"],origin=origin,probability=float(probs[q]),
                        support=support))
    for n,ident in enumerate(m["old_fit_ids"]+m["fit_ids"]):
        w.check(PREP_END)
        root=DATA_REF if ident in m["old_fit_ids"] else REFERENCE
        meta=root/"train_metadata"/f"{ident}.json";bind(meta)
        jobs=read(meta)["jobs"]
        inspect(root/"candidate_patches"/f"{ident}.npz",jobs,ident,"ordinary_or_scene")
        if ident in extra:
            inspect(OLD_POS/"extra_positive"/f"{ident}.npz",extra[ident],ident,"old_extra")
        if n%16==0:w.mark("fit_false_rejection_audit",images=n+1,expected=640)
    assert counts[1]==617 and counts[0]==13805,counts
    grouped=collections.defaultdict(list)
    for r in positive_records:grouped[r["gt_id"]].append(r)
    # Conservative instance-level evidence: none of its available positive patches is accepted.
    rejected=[dict(gt_id=k,image_id=v[0]["image_id"],scene=v[0]["scene"],area=v[0]["gt_area"],
                   max_probability=max(r["probability"] for r in v),patches=len(v))
              for k,v in grouped.items() if max(r["probability"] for r in v)<.5]
    scenes={r["scene"] for r in rejected}
    gate=len(rejected)>=16 and len(scenes)>=8
    save(OUT/"prepared_manifest.json",m)
    save(OUT/"headroom_gate.json",dict(status="passed",gate_passed=gate,fit_only=True,
        check_used=False,val_used=False,unique_rejected_gt=len(rejected),scenes=len(scenes),
        required_gt=16,required_scenes=8,rejected=rejected,positive_patch_records=positive_records,
        examined_unique_positive_le64=len(grouped),sample_counts=dict(counts),
        rule="area<=64 true tiny GT with all available fit positive patches p<.5",
        baseline_state_sha256=tensor_digest(model.state_dict()),new_gpu_calls=0,finished_unix=time.time()))
    save(OUT/"binding.json",dict(input_sha256=inputs,selector_sha256=bind(w.PARENT/"selector.json"),
        baseline_report_sha256=sha(REFERENCE/"report.json"),frozen37_unchanged=True))
    save(OUT/"cpu_acceptance.json",dict(status="passed",tests=tests.testsRun,skipped=0,
        gate_passed=gate,gpu_used=False,only_architecture_difference="features.6.stride (2,2)->(1,1)",
        parameters=24369,finished_unix=time.time()))
    w.check(PREP_END)


def traincheck():
    import numpy as np
    import torch
    from run_m2f_quality_select import tensor_digest
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    end=read(OUT/"phase_deadlines.json")["train_end"]
    assert read(OUT/"headroom_gate.json")["gate_passed"]
    m=read(OUT/"prepared_manifest.json");old=read(DATA_REF/"prepared_manifest.json")
    sm=read(REFERENCE/"sample_manifest.json");arrays=[];labels=[]
    for root,ids,records in ((DATA_REF,m["old_fit_ids"],old["records"]),
                             (REFERENCE,m["fit_ids"],sm["records"])):
        for ident in ids:
            w.check(end)
            with np.load(root/"candidate_patches"/f"{ident}.npz") as z:
                labs=np.asarray(records[str(ident)]["labels"],dtype=np.int64)
                arrays.append(z["x"][labs>=0]);labels.extend(labs[labs>=0].tolist())
    for ident in m["old_fit_ids"]:
        path=OLD_POS/"extra_positive"/f"{ident}.npz"
        if path.exists():
            with np.load(path) as z:arrays.append(z["x"]);labels.extend([1]*len(z["x"]))
    x=torch.from_numpy(np.concatenate(arrays));del arrays
    y=torch.tensor(labels,dtype=torch.float32);pos=torch.where(y==1)[0];neg=torch.where(y==0)[0]
    assert len(pos)==617 and len(neg)==13805
    torch.manual_seed(20261013);np.random.seed(20261013)
    model=new_model().cuda().train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert tensor_digest(initial)==read(REFERENCE/"fit_report.json")["initial_state_sha256"]
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(20261013);history=[];begun=time.time()
    for step in range(1,2001):
        w.check(end)
        inds=torch.cat((pos[torch.randint(len(pos),(16,),generator=rng)],neg[torch.randint(len(neg),(16,),generator=rng)]))
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[inds].cuda()),y[inds].cuda())
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        grad=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters())))
        assert math.isfinite(grad)
        opt.step();history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad))
        if step==1 or step%50==0:w.mark("fit_stride4",step=step,expected=2000)
    torch.cuda.synchronize();state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    native=sorted({int(v["step"].item()) for v in opt.state.values()})
    assert native==[2000] and all(not torch.equal(initial[k],v) for k,v in state.items())
    assert all(bool(torch.isfinite(v).all()) for v in state.values())
    assert all(bool(torch.isfinite(v).all()) for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    torch.save(dict(state_dict=state,optimizer=opt.state_dict(),steps=2000,final_stride=1,
        fit_ids=m["old_fit_ids"]+m["fit_ids"],seed=20261013),OUT/"verifier_2000.pth")
    save(OUT/"fit_history.json",history)
    save(OUT/"fit_report.json",dict(status="passed",native_steps=native,finite_steps=2000,
        consumed_patches=64000,positive_patches=len(pos),negative_patches=len(neg),fit_images=640,
        check_used_for_fit=0,initial_state_sha256=tensor_digest(initial),final_state_sha256=tensor_digest(state),
        parameters=sum(p.numel() for p in model.parameters()),state_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        all_parameters_updated=True,model_optimizer_finite=True,fit_seconds=time.time()-begun,
        feature_grid=[32,32],checkpoint_sha256=sha(OUT/"verifier_2000.pth"),finished_unix=time.time()))
    del x,y,opt
    after,activity=scene.check_arm(model.eval(),m["check_ids"],True,end)
    before=read(REFERENCE/"check_after.json");coco=w.coco_load("train")
    bm=read(REFERENCE/"learned_gate.json")["metrics"];am=w.metrics(coco,m["check_ids"],after)
    bp=read(REFERENCE/"check_matches.json")["after"];ap=w.matched(coco,m["check_ids"],after)
    gained=sorted(set(ap["tiny"])-set(bp["tiny"]));lost=sorted(set(bp["tiny"])-set(ap["tiny"]))
    save(OUT/"check_before.json",before);save(OUT/"check_after.json",after);save(OUT/"check_activity.json",activity)
    save(OUT/"check_matches.json",dict(before=bp,after=ap,tiny_gained=gained,tiny_lost=lost))
    save(OUT/"learned_gate.json",dict(status="passed",gate_passed=am["APs"]>bm["APs"] and len(gained)>=len(lost),
        baseline_metrics=bm,metrics=am,delta_points={k:100*(am[k]-bm[k]) for k in bm},
        tiny_gained=gained,tiny_lost=lost,tiny_net=len(gained)-len(lost),check_not_fit=True,val_used=False,
        reused_internal_check=True,rule="same128 APs strict positive AND tiny one-to-one net>=0",finished_unix=time.time()))


def finish():
    stopped="headroom_gate" if not read(OUT/"headroom_gate.json")["gate_passed"] else (
        "learned_gate" if not read(OUT/"learned_gate.json")["gate_passed"] else None)
    comparison={}
    if (OUT/"learned_gate.json").exists():
        m=read(OUT/"prepared_manifest.json")
        summary,_=helpers.paired_summary(w,w.coco_load("train"),m["check_ids"],read(OUT/"check_before.json"),read(OUT/"check_after.json"))
        save(OUT/"internal_analysis.json",summary)
    if not stopped:
        assert not (OUT/"full_val_metrics.json").exists()
        frozen=read(OUT/"predictions_frozen.json")
        assert sha(OUT/"predictions.segm.json")==frozen["prediction_sha256"]
        ids=frozen["image_ids"];assert len(ids)==len(set(ids))==2584
        coco=w.coco_load("val");rows=read(OUT/"predictions.segm.json")
        actual=w.metrics(coco,ids,rows);w.check(EVAL_END)
        save(OUT/"full_val_metrics.json",dict(metrics=actual,image_ids=ids,images=2584,evaluations=1))
        baseline=read(REFERENCE/"full_val_metrics.json")["metrics"]
        summary,hits=helpers.paired_summary(w,coco,ids,read(REFERENCE/"predictions.segm.json"),rows)
        old=read(REFERENCE/"full_val_error_analysis.json")
        summary["previous31_kept"]=sorted(set(old["tiny_gained"])&set(hits["tiny"]))
        summary["previous31_lost"]=sorted(set(old["tiny_gained"])-set(hits["tiny"]))
        summary["previous18_recovered"]=sorted(set(old["tiny_lost"])&set(hits["tiny"]))
        save(OUT/"full_val_error_analysis.json",summary)
        comparison=dict(metrics=actual,baseline_metrics=baseline,delta_points={k:100*(actual[k]-baseline[k]) for k in baseline},
            any_positive_retained=actual["APs"]>BEST_APS,single_development_val=True)
        save(OUT/"comparison.json",comparison)
    assert all(sha(ROOT/p)==h for p,h in read(OUT/"binding.json")["input_sha256"].items())
    names=[p.name for p in OUT.glob("*.json") if p.name not in
        ("status.json","progress.json","report.json","terminal_acceptance.json","plan.json","launch.json","phases.json","phase_deadlines.json")
        and not p.name.endswith(("_command.json","_container.json"))]
    if (OUT/"verifier_2000.pth").exists():names.append("verifier_2000.pth")
    w.check(EVAL_END)
    save(OUT/"report.json",dict(status="passed",outcome="stopped_failed_"+stopped if stopped else "evaluated",
        start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),baseline_aps=BEST_APS,
        full_val_evaluations=int(not stopped),new_aps=comparison.get("metrics",{}).get("APs"),
        verifier_trained=(OUT/"fit_report.json").exists(),m2f_updated=False,old_refiner_updated=False,
        maskrcnn_paused=True,only_change="verifier.features.6.stride 2->1; last feature grid16->32",
        artifacts_sha256={n:sha(OUT/n) for n in names},comparison=comparison))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prep","traincheck","predict","finish"))
    phase=parser.parse_args().phase;install()
    end=dict(prep=PREP_END,traincheck=TRAIN_END,predict=GPU_END,finish=EVAL_END)[phase]
    if phase in ("traincheck","predict"):
        end=min(end,read(OUT/"phase_deadlines.json")["train_end" if phase=="traincheck" else "predict_end"])
    w.check(end)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Immutable stage cutoff")))
    signal.alarm(max(1,math.ceil(end-time.time())))
    try:
        if phase=="predict":w.predict()
        else:globals()[phase]()
    except BaseException as exc:
        save(OUT/(phase+"_first_error.json"),dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise


if __name__=="__main__":main()

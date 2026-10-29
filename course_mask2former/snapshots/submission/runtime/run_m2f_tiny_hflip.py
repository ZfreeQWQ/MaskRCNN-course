"""One training-only six-channel horizontal flip, fixed scene data and inference."""
import argparse
import collections
import hashlib
import math
import os
from pathlib import Path
import signal
import time
import traceback
import unittest
import run_m2f_tiny_stride4 as shared
import run_m2f_tiny_scene as scene
import run_m2f_tiny_positive as w
import m2f_tiny_positive_helpers as helpers

START=1792521000
PREP_END,TRAIN_END,GPU_END,EVAL_END,DEADLINE=(START+s for s in (900,1800,2700,3000,3600))
NAME="course_m2f_tiny_hflip_20261021_v1"
OWNER="b5ca78268cb044e89ee19d24ab4fda56"
ROOT=Path("/workspace")
OUT=ROOT/"outputs"/NAME
REFERENCE=ROOT/"outputs/course_m2f_tiny_scene_20261021_v1"
PRIOR=ROOT/"outputs/course_m2f_tiny_stride4_20261021_v1"
OLD_POS=ROOT/"outputs/course_m2f_tiny_positive_20261021_v1"
DATA_REF=ROOT/"outputs/course_m2f_tiny_verify_20261020_v1"
BEST_APS=.41396866834940094
read,save,sha=w.read,w.save,w.sha


def augment(x,flags):
    """Apply the same exact width permutation to RGB, depth, validity and mask."""
    assert x.ndim==4 and x.shape[1:]==(6,128,128) and len(flags)==len(x)
    result=x.clone()
    result[flags]=x[flags].flip(-1)
    return result


def new_model():
    import m2f_tiny_verifier as tiny
    model=tiny.Verifier()
    assert model.features[6].stride==(2,2)
    return model


def install():
    for key in ("START","PREP_END","TRAIN_END","GPU_END","DEADLINE","NAME","OWNER",
                "ROOT","OUT","REFERENCE","BEST_APS"):
        setattr(w,key,globals()[key])
    w.install()
    scene.OUT=REFERENCE
    scene.DATA_REF=DATA_REF


class Contracts(unittest.TestCase):
    def test_six_channel_alignment_and_no_mutation(self):
        import torch
        x=torch.arange(2*6*128*128,dtype=torch.float32).reshape(2,6,128,128)
        old=x.clone();y=augment(x,torch.tensor([True,False]))
        self.assertTrue(torch.equal(y[0],x[0].flip(-1)))
        self.assertTrue(torch.equal(y[1],x[1]))
        self.assertTrue(torch.equal(old,x))

    def test_exact_involution(self):
        import torch
        x=torch.randn(3,6,128,128);flags=torch.tensor([True,False,True])
        self.assertTrue(torch.equal(augment(augment(x,flags),flags),x))

    def test_mask_area_iou_label_invariant(self):
        import torch
        x=torch.zeros(1,6,128,128);x[:,5,10:18,1:7]=1
        gt=torch.zeros_like(x[:,5]);gt[:,11:19,2:8]=1
        before=(x[:,5]*gt).sum()/((x[:,5]+gt)>0).sum()
        y=augment(x,torch.tensor([True]));z=gt.flip(-1)
        after=(y[:,5]*z).sum()/((y[:,5]+z)>0).sum()
        self.assertEqual(float(before),float(after))
        self.assertEqual(float(x[:,5].sum()),float(y[:,5].sum()))
        self.assertTrue(set(y[:,5].unique().tolist())<={0.,1.})

    def test_sampling_rng_unchanged(self):
        import torch
        a=torch.Generator().manual_seed(20261013)
        b=torch.Generator().manual_seed(20261013)
        aug=torch.Generator().manual_seed(20261021)
        for _ in range(100):
            for size in (617,13805):
                self.assertTrue(torch.equal(torch.randint(size,(16,),generator=a),
                                            torch.randint(size,(16,),generator=b)))
            torch.rand(32,generator=aug)
        self.assertTrue(torch.equal(a.get_state(),b.get_state()))

    def test_original_initialization_and_grid(self):
        import torch
        import m2f_tiny_verifier as tiny
        torch.manual_seed(20261013);a=tiny.Verifier();ar=torch.get_rng_state()
        torch.manual_seed(20261013);b=new_model();br=torch.get_rng_state()
        self.assertTrue(torch.equal(ar,br))
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))
        self.assertEqual(tuple(b.features(torch.zeros(2,6,128,128)).shape),(2,64,16,16))
        self.assertEqual(sum(p.numel() for p in b.parameters()),24369)

    def test_augmented_backward(self):
        import torch
        model=new_model();x=torch.randn(2,6,128,128);x[:,5]=0;x[:,5,60:66,60:66]=1
        output=model(augment(x,torch.tensor([True,False])))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(output,torch.tensor([1.,0.]))
        self.assertTrue(bool(torch.isfinite(loss)));loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_budget_and_gate(self):
        self.assertEqual(DEADLINE-START,3600)
        self.assertEqual(PREP_END-START,900)
        self.assertEqual(EVAL_END-START,3000)


def prep():
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not Path("/dev/nvidiactl").exists()
    w.base.configure_torch()
    suite=unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromName("test_m2f_tiny_verifier"),
                             unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)))
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful() and not tests.skipped
    assert sha(PRIOR/"report.json")=="f200c01bb38f75d76b082d6be55d31f56ffe50d9c2bd18dee056499a7ace064a"
    prior=read(PRIOR/"plan.json")
    assert all(sha(OUT/"source"/n)==h for n,h in prior["source_sha256"].items())
    binding=read(PRIOR/"binding.json");inputs=dict(binding["input_sha256"])
    for rel,h in inputs.items():
        w.check(PREP_END);assert sha(ROOT/rel)==h,rel
    for path in (PRIOR/"report.json",PRIOR/"terminal_acceptance.json",REFERENCE/"report.json",
                 REFERENCE/"plan.json",REFERENCE/"check_activity.json",REFERENCE/"full_val_error_analysis.json"):
        inputs[str(path.relative_to(ROOT))]=sha(path)
    assert sha(REFERENCE/"report.json")=="0ff473e602ed3c929b23d8a238cdb73e7f8c94527d489b63a24c4f464f3830da"
    assert read(REFERENCE/"report.json")["new_aps"]==BEST_APS
    assert inputs[str((REFERENCE/"check_activity.json").relative_to(ROOT))]==read(REFERENCE/"report.json")["artifacts_sha256"]["check_activity.json"]
    m=read(REFERENCE/"prepared_manifest.json")
    assert len(m["old_fit_ids"]+m["fit_ids"])==640 and len(m["check_ids"])==128
    assert not set(m["old_fit_ids"]+m["fit_ids"])&set(m["check_ids"])
    save(OUT/"prepared_manifest.json",m)
    save(OUT/"binding.json",dict(input_sha256=inputs,selector_sha256=binding["selector_sha256"],
        baseline_report_sha256=sha(REFERENCE/"report.json"),frozen39_unchanged=True))
    save(OUT/"cpu_acceptance.json",dict(status="passed",tests=tests.testsRun,skipped=0,
        gate_passed=True,gpu_used=False,parameters=24369,
        only_change="training six-channel horizontal flip p=.5; independent augmentation RNG",
        architecture_unchanged=True,finished_unix=time.time()))
    w.check(PREP_END)


def traincheck():
    import numpy as np
    import torch
    from run_m2f_quality_select import tensor_digest
    torch=w.base.configure_torch();assert torch.cuda.device_count()==1
    end=read(OUT/"phase_deadlines.json")["train_end"]
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
    model=new_model().cuda().train();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert tensor_digest(initial)==read(REFERENCE/"fit_report.json")["initial_state_sha256"]
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(20261013);control=torch.Generator().manual_seed(20261013)
    flip_rng=torch.Generator().manual_seed(20261021)
    sample_hash=hashlib.sha256();flip_hash=hashlib.sha256();flipped=0;history=[];begun=time.time()
    for step in range(1,2001):
        w.check(end)
        inds=torch.cat((pos[torch.randint(len(pos),(16,),generator=rng)],neg[torch.randint(len(neg),(16,),generator=rng)]))
        reference=torch.cat((pos[torch.randint(len(pos),(16,),generator=control)],neg[torch.randint(len(neg),(16,),generator=control)]))
        assert torch.equal(inds,reference)
        flags=torch.rand(32,generator=flip_rng)<.5
        sample_hash.update(inds.numpy().tobytes());flip_hash.update(flags.numpy().tobytes());flipped+=int(flags.sum())
        opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(augment(x[inds],flags).cuda()),y[inds].cuda())
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        grad=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters())))
        assert math.isfinite(grad)
        opt.step();history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad,flipped=int(flags.sum())))
        if step==1 or step%50==0:w.mark("fit_hflip",step=step,expected=2000)
    torch.cuda.synchronize();state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    native=sorted({int(v["step"].item()) for v in opt.state.values()})
    assert native==[2000] and all(not torch.equal(initial[k],v) for k,v in state.items())
    assert all(bool(torch.isfinite(v).all()) for v in state.values())
    assert all(bool(torch.isfinite(v).all()) for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    torch.save(dict(state_dict=state,optimizer=opt.state_dict(),steps=2000,
        fit_ids=m["old_fit_ids"]+m["fit_ids"],seed=20261013,training_flip_probability=.5),OUT/"verifier_2000.pth")
    save(OUT/"fit_history.json",history)
    save(OUT/"fit_report.json",dict(status="passed",native_steps=native,finite_steps=2000,
        consumed_patches=64000,positive_patches=len(pos),negative_patches=len(neg),fit_images=640,
        check_used_for_fit=0,initial_state_sha256=tensor_digest(initial),final_state_sha256=tensor_digest(state),
        parameters=sum(p.numel() for p in model.parameters()),state_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        all_parameters_updated=True,model_optimizer_finite=True,fit_seconds=time.time()-begun,
        feature_grid=[16,16],sample_indices_sha256=sample_hash.hexdigest(),sample_indices_exact_reference=True,
        flip_decisions_sha256=flip_hash.hexdigest(),flipped_patches=flipped,flip_seed=20261021,
        checkpoint_sha256=sha(OUT/"verifier_2000.pth"),finished_unix=time.time()))
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


def decision_summary(rows):
    gt=collections.defaultdict(list);negative=0;ambiguous=0;negative_total=0
    for image in rows:
        for candidate in image["candidates"]:
            p=candidate["probability"];label=candidate["label"]
            if label==1 and candidate["gt_area"]<=64:gt[candidate["gt_id"]].append(p)
            if label==0:
                negative_total+=1;negative+=int(p>=.5)
            if label==-1:ambiguous+=int(p>=.5)
    return dict(positive_le64_gt=sorted(gt),accepted_le64_gt=sorted(k for k,v in gt.items() if max(v)>=.5),
        rejected_le64_gt=sorted(k for k,v in gt.items() if max(v)<.5),
        negative_accepted=negative,negative_total=negative_total,ambiguous_accepted=ambiguous)


def finish():
    stopped=not read(OUT/"learned_gate.json")["gate_passed"]
    m=read(OUT/"prepared_manifest.json")
    summary,_=helpers.paired_summary(w,w.coco_load("train"),m["check_ids"],read(OUT/"check_before.json"),read(OUT/"check_after.json"))
    before=decision_summary(read(REFERENCE/"check_activity.json"));after=decision_summary(read(OUT/"check_activity.json"))
    summary["candidate_decisions"]=dict(before=before,after=after,
        previous12_now_accepted=sorted(set(before["rejected_le64_gt"])&set(after["accepted_le64_gt"])),
        new_rejected=sorted(set(before["accepted_le64_gt"])-set(after["accepted_le64_gt"])))
    save(OUT/"internal_analysis.json",summary);comparison={}
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
    names.append("verifier_2000.pth");w.check(EVAL_END)
    save(OUT/"report.json",dict(status="passed",outcome="stopped_failed_learned_gate" if stopped else "evaluated",
        start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),baseline_aps=BEST_APS,
        full_val_evaluations=int(not stopped),new_aps=comparison.get("metrics",{}).get("APs"),
        verifier_trained=True,m2f_updated=False,old_refiner_updated=False,maskrcnn_paused=True,
        only_change="training six-channel horizontal flip p=.5; same architecture and inference",
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

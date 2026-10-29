"""One immutable 60-minute supervision experiment; learned train gate before val."""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
import run_m2f_boundary_refine as base

START = 1792493372
PREP_END, TRAIN_END, GPU_END, EVAL_END, DEADLINE = (
    START+s for s in (900, 1800, 2700, 3000, 3600))
NAME = "course_m2f_supervision_correct_20261020_v1"
OWNER = "95a4683b00954bc59a38756c6de96549"
ROOT = Path("/workspace")
OUT = ROOT/"outputs"/NAME
OLD = ROOT/"outputs/course_m2f_boundary_refine_20261020_v1"
RECOVERY = ROOT/"outputs/course_m2f_boundary_analysis_20261020_v1"
RECOVERY_SHA = "13710b76954e2dbf4a462c73fe647c50e1d463f7c7a91374a5ccbe46b0a86246"
read, sha = base.read, base.sha


def native(value):
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return native(value.item())
    if isinstance(value, dict):
        return {k:native(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Unsupported or nonfinite JSON scalar: "+type(value).__name__)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(native(value), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def check(end=None):
    assert time.time() < (DEADLINE if end is None else min(end, DEADLINE)), "Immutable deadline"


def install():
    for k in ("START", "PREP_END", "TRAIN_END", "EVAL_END", "DEADLINE", "NAME", "OWNER", "ROOT", "OUT"):
        setattr(base, k, globals()[k])
    base.check, base.save = check, save


def corrected_target(target, coarse, records):
    """Change ONLY targets linked to a >1024-area train GT; retain all examples."""
    import numpy as np
    result = target.copy()
    change = np.asarray([r["gt_area"] is not None and r["gt_area"] > 1024 for r in records])
    result[change] = coarse[change]
    assert np.array_equal(result[~change], target[~change])
    assert np.array_equal(result[change], coarse[change])
    return result


def threshold_summary(pairs, field):
    thresholds = [.5+.05*k for k in range(10)]
    def group(values):
        old = [sum(p["before"] >= t for p in values) for t in thresholds]
        new = [sum(p[field] >= t for p in values) for t in thresholds]
        return dict(count=len(values), before=old, after=new,
                    net_crossings=sum(new)-sum(old),
                    gained_05=sum(p["before"] < .5 <= p[field] for p in values),
                    lost_05=sum(p[field] < .5 <= p["before"] for p in values),
                    mean_iou_increment=sum(p[field]-p["before"] for p in values)/max(1,len(values)))
    return dict(small=group(pairs), tiny=group([p for p in pairs if p["area"] <= 256]))


def learned_gate(summary):
    return bool(summary["small"]["net_crossings"] > 0 and summary["tiny"]["net_crossings"] >= 0)


def prep():
    import unittest
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    tests = unittest.defaultTestLoader.loadTestsFromNames(
        ["test_m2f_boundary_refiner", "test_m2f_supervision_correct"])
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful() and not result.skipped and result.testsRun == 16
    assert sha(RECOVERY/"report.json") == RECOVERY_SHA
    accepted = read(RECOVERY/"report.json")
    assert accepted["status"] == "passed"
    original_plan = read(OLD/"plan.json")
    source = {}
    for name, digest in original_plan["source_sha256"].items():
        assert sha(OUT/"source"/name) == digest
        source[name] = digest
    original_binding = read(OLD/"binding.json")
    manifest = read(OLD/"prepared_manifest.json")
    assert sha(OLD/"prepared_manifest.json") == original_binding["prepared_manifest_sha256"]
    assert len(manifest["fit_ids"]) == 384 and len(manifest["check_ids"]) == 128
    assert not set(manifest["fit_ids"]) & set(manifest["check_ids"])
    assert manifest["scene_disjoint"]
    assert sha(ROOT/base.ACCEPTED/"report.json") == base.BEST_SHA
    for name in ("fit_report.json", "refiner_3000.pth", "headroom_gate.json",
                 "prepared_manifest.json", "binding.json"):
        assert sha(OLD/name) == accepted["artifacts_sha256"][name]
    assert read(OLD/"fit_report.json")["native_steps"] == [3000]
    # Read-only symlinks; no full RGB-D recapture and no rewritten old targets.
    for name in ("prepared", "train_selected"):
        (OUT/name).symlink_to(OLD/name, target_is_directory=True)
    save(OUT/"prepared_manifest.json", manifest)
    save(OUT/"headroom_gate.json", read(OLD/"headroom_gate.json"))
    original_binding["prepared_manifest_sha256"] = sha(OUT/"prepared_manifest.json")
    original_binding["input_sha256"].update({
        str((RECOVERY/"report.json").relative_to(ROOT)):RECOVERY_SHA,
        str((OLD/"refiner_3000.pth").relative_to(ROOT)):sha(OLD/"refiner_3000.pth")})
    save(OUT/"binding.json", original_binding)
    counts = dict(tiny=0, small=0, large=0, unmatched=0)
    for ident in manifest["fit_ids"]:
        for record in manifest["images"][str(ident)]["records"]:
            area = record["gt_area"]
            counts["unmatched" if area is None else "tiny" if area<=256 else "small" if area<=1024 else "large"] += 1
    assert counts == dict(tiny=1426, small=1906, large=7976, unmatched=356)
    save(OUT/"cpu_acceptance.json", dict(status="passed",tests=result.testsRun,skipped=0,
        old_source_unchanged=source,target_groups=counts,only_large_target_changed=True,
        scene_disjoint=True,check_images=128,validation_labels_used=False,gpu_used=False,unix=time.time()))
    check(PREP_END)


def audit_head(model, ids, arm, label):
    """Inference on official TRAIN only; same original per-image predictions."""
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    import m2f_boundary_refiner as ref
    manifest = read(OUT/"prepared_manifest.json")
    annotation = ROOT/base.DATA/"annotations/instances_train.json"
    assert sha(annotation) == base.TRAIN_SHA
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation))
    pairs = []
    for n, ident in enumerate(ids):
        check(TRAIN_END)
        path = OUT/"train_selected"/f"{ident}.json"
        assert sha(path) == manifest["files"][f"train_selected/{ident}.json"]
        rows = read(path)
        request = manifest["images"][str(ident)]["request"]
        expected = manifest["images"][str(ident)]["input_hashes"]
        assert sha(request["img_path"]) == expected["rgb_sha256"]
        assert sha(request["depth_file"]) == expected["depth_sha256"]
        refined, _ = ref.infer(rows, ref.load_rgbd(request), model, "cuda")
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
                if a["area"] <= 1024 and not a.get("iscrowd",0) and not a.get("ignore",0)]
        if anns:
            gt = [coco.annToRLE(a) for a in anns]
            before = mu.iou([r["segmentation"] for r in rows],gt,[0]*len(gt)) if rows else np.zeros((0,len(gt)))
            after = mu.iou([r["segmentation"] for r in refined],gt,[0]*len(gt)) if refined else np.zeros((0,len(gt)))
            for j,a in enumerate(anns):
                # Class-specific geometry, not a claim about one-to-one AP.
                selected = [i for i,r in enumerate(rows) if r["category_id"] == a["category_id"]]
                pairs.append(dict(image_id=ident,annotation_id=a["id"],area=a["area"],
                    before=float(before[selected,j].max()) if selected else 0.,
                    after=float(after[selected,j].max()) if selected else 0.))
        base.mark("internal_audit",arm=arm,model=label,completed=n+1,expected=len(ids))
    summary = threshold_summary(pairs, "after")
    save(OUT/f"{label}_{arm}_paired_gt.json",pairs)
    save(OUT/f"{label}_{arm}_summary.json",summary)
    return summary


def traincheck():
    import numpy as np
    import m2f_boundary_refiner as ref
    from run_m2f_quality_select import tensor_digest
    torch = base.configure_torch()
    assert torch.cuda.device_count() == 1
    manifest = read(OUT/"prepared_manifest.json")
    assert sha(OUT/"prepared_manifest.json") == read(OUT/"binding.json")["prepared_manifest_sha256"]
    ids = manifest["fit_ids"]
    fit_probe = sorted(ids)[:64]  # fixed before outcomes; never used to choose parameters
    old_ckpt = torch.load(OLD/"refiner_3000.pth",map_location="cpu")
    assert sha(OLD/"refiner_3000.pth") == read(OLD/"fit_report.json")["checkpoint_sha256"]
    old = ref.Refiner()
    old.load_state_dict(old_ckpt["state_dict"],strict=True)
    old.cuda().eval()
    audit_head(old,fit_probe,"fit64","old")
    audit_head(old,manifest["check_ids"],"check128","old")
    del old, old_ckpt
    torch.cuda.empty_cache()
    # Reset ALL RNGs after audit: identical original initializer and minibatches.
    torch = base.configure_torch()
    n = sum(len(manifest["images"][str(i)]["records"]) for i in ids)
    assert n == 11664 and n*6*128*128*4 < 6*2**30
    inputs = torch.empty((n,6,128,128))
    targets = torch.empty((n,128,128),dtype=torch.bool)
    bands = torch.empty_like(targets)
    offset = 0
    actual_changes = 0
    for ident in ids:
        check(TRAIN_END)
        records = manifest["images"][str(ident)]["records"]
        count = len(records)
        if not count: continue
        path = OUT/"prepared"/f"{ident}.npz"
        assert sha(path) == manifest["files"][f"prepared/{ident}.npz"]
        with np.load(path) as z:
            target = corrected_target(z["target"],z["coarse"],records)
            actual_changes += int(np.count_nonzero(np.any(target != z["target"],axis=(1,2))))
            inputs[offset:offset+count].copy_(torch.from_numpy(z["x"]))
            targets[offset:offset+count].copy_(torch.from_numpy(target))
            bands[offset:offset+count].copy_(torch.from_numpy(z["band"]))
        offset += count
        base.mark("fit_cache_load",images=ids.index(ident)+1,patches=offset,expected=n)
    assert offset == n
    model = ref.Refiner().cuda().train()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert tensor_digest(initial) == read(OLD/"fit_report.json")["source_state_initial"]
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng = torch.Generator().manual_seed(ref.SEED)
    history = []
    begun = time.time()
    for step in range(1,ref.STEPS+1):
        check(TRAIN_END)
        chosen = torch.randint(n,(ref.BATCH,),generator=rng)
        x=inputs[chosen].cuda();y=targets[chosen].cuda();b=bands[chosen].cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = ref.band_loss(model(x),y,b)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert bool(torch.isfinite(norm))
        optimizer.step()
        history.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(norm)))
        if step==1 or step%50==0:base.mark("fit_refiner",step=step,expected=ref.STEPS,loss=history[-1]["loss"])
    torch.cuda.synchronize()
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    assert all(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.state_dict().items())
    native_steps=sorted({int(v["step"].item()) for v in optimizer.state.values()})
    assert native_steps == [3000]
    assert all(bool(torch.isfinite(v).all()) for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
    weight={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(state_dict=weight,optimizer=optimizer.state_dict(),steps=3000,fit_image_ids=ids,seed=ref.SEED),OUT/"refiner_3000.pth")
    save(OUT/"fit_history.json",history)
    save(OUT/"fit_report.json",dict(status="passed",native_steps=native_steps,finite_steps=3000,
        consumed_patches=96000,available_patches=n,fit_images=384,check_images_used_for_fit=0,validation_used=False,
        parameters=sum(p.numel() for p in model.parameters()),state_bytes=sum(v.numel()*v.element_size() for v in weight.values()),
        source_state_initial=tensor_digest(initial),source_state_final=tensor_digest(weight),
        checkpoint_sha256=sha(OUT/"refiner_3000.pth"),m2f_loaded=False,m2f_updated=False,
        all_head_tensors_changed=True,model_optimizer_finite=True,fit_seconds=time.time()-begun,
        large_labels_changed_instances=actual_changes,large_label_eligible=7976,finished_unix=time.time()))
    del inputs,targets,bands,optimizer
    model.eval()
    audit_head(model,fit_probe,"fit64","corrected")
    summary=audit_head(model,manifest["check_ids"],"check128","corrected")
    assert summary["small"]["count"] == 435 and summary["tiny"]["count"] == 141
    save(OUT/"learned_gate.json",dict(status="passed",gate_passed=learned_gate(summary),
        rule="check128 small ten-IoU net crossings > 0 AND tiny net crossings >= 0 vs identity",
        summary=summary,check_images=128,model_frozen_sha256=sha(OUT/"refiner_3000.pth"),
        check_not_fit=True,checkpoint_selection=False,val_used=False,finished_unix=time.time()))


def stopped_report():
    gate=read(OUT/"learned_gate.json")
    assert not gate["gate_passed"]
    names=["cpu_acceptance.json","binding.json","prepared_manifest.json","fit_report.json",
           "fit_history.json","refiner_3000.pth","learned_gate.json"]
    names += [f"{label}_{arm}_{suffix}.json" for label in ("old","corrected")
              for arm in ("fit64","check128") for suffix in ("summary","paired_gt")]
    save(OUT/"report.json",dict(status="passed",outcome="stopped_failed_learned_gate",
        start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),gate=gate,
        full_val_evaluations=0,complete_val_images=0,new_aps=None,baseline_aps=.41095135423262297,
        refiner_trained=True,m2f_loaded=False,m2f_updated=False,maskrcnn_paused=True,
        artifacts_sha256={n:sha(OUT/n) for n in names}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prep","traincheck","predict","evaluate","analyze"))
    args=parser.parse_args()
    install()
    limits=dict(prep=PREP_END,traincheck=TRAIN_END,predict=GPU_END,evaluate=EVAL_END,analyze=DEADLINE)
    check(limits[args.phase])
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Immutable phase deadline")))
    signal.alarm(max(1,math.ceil(limits[args.phase]-time.time())))
    try:
        if args.phase=="prep":prep()
        elif args.phase=="traincheck":traincheck()
        else:
            assert read(OUT/"learned_gate.json")["gate_passed"],"Never evaluate after failed gate"
            if args.phase=="predict":base.predict()
            elif args.phase=="evaluate":base.evaluate()
            else:
                base.analyze()
                report=read(OUT/"report.json")
                extra=["learned_gate.json"]+[f"{l}_{a}_{s}.json" for l in ("old","corrected")
                    for a in ("fit64","check128") for s in ("summary","paired_gt")]
                report["artifacts_sha256"].update({n:sha(OUT/n) for n in extra})
                save(OUT/"report.json",report)
    except BaseException as exc:
        save(OUT/(args.phase+"_first_error.json"),dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise


if __name__=="__main__":
    main()

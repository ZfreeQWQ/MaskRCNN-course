"""Scene-disjoint data expansion; frozen production inference and hard gates."""
import argparse
import collections
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import unittest
import run_m2f_tiny_positive as w
import m2f_tiny_positive_helpers as helpers

START = 1792516726
PREP_END, CAPTURE_END, TRAIN_END, GPU_END, DEADLINE = (
    START + n for n in (1200, 3000, 4200, 6000, 7200))
NAME = "course_m2f_tiny_scene_20261021_v1"
OWNER = "840fe68b93914cb993de57d3ea87e18a"
ROOT = Path("/workspace")
OUT = ROOT / "outputs" / NAME
REFERENCE = ROOT / "outputs/course_m2f_tiny_positive_20261021_v1"
DATA_REF = ROOT / "outputs/course_m2f_tiny_verify_20261020_v1"
BEST_APS = .4134619806403432
read, save, sha = w.read, w.save, w.sha


def install():
    for name in ("START", "PREP_END", "TRAIN_END", "GPU_END", "DEADLINE",
                 "NAME", "OWNER", "ROOT", "OUT", "REFERENCE", "BEST_APS"):
        setattr(w, name, globals()[name])
    w.install()


def select_manifest(images, annotations, old_scenes, seed=20261013):
    """Select one image per unseen scene; freeze before inference or labels."""
    tiny_ids = {a["image_id"] for a in annotations
                if 0 < a["area"] <= 256 and not a.get("iscrowd", 0)
                and not a.get("ignore", 0)}
    by_scene = collections.defaultdict(list)
    for im in images:
        if im["id"] in tiny_ids and im["scene_id"] not in old_scenes:
            by_scene[im["scene_id"]].append(im["id"])
    scenes = sorted(by_scene)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    if len(scenes) < 384:
        return dict(gate_passed=False, available_scenes=len(scenes),
                    reason="insufficient unseen tiny-positive scenes")
    chosen = [(s, rng.choice(sorted(by_scene[s]))) for s in scenes[:384]]
    return dict(gate_passed=True, available_scenes=len(scenes),
                fit_ids=[i for _, i in chosen[:256]], check_ids=[i for _, i in chosen[256:]],
                scenes={str(i): s for s, i in chosen}, seed=seed,
                rule="one seeded tiny-positive official train image per unseen scene")


def positive_gate(positives, scenes):
    gt = {p["gt_id"]: p for p in positives}
    return dict(gate_passed=len(gt) >= 32
                and len({scenes[str(p["image_id"])] for p in gt.values()}) >= 16
                and sum(p["gt_area"] <= 64 for p in gt.values()) >= 8,
                unique_positive_tiny=len(gt),
                positive_scenes=len({scenes[str(p["image_id"])] for p in gt.values()}),
                area_le64=sum(p["gt_area"] <= 64 for p in gt.values()),
                required_unique=32, required_scenes=16, required_le64=8)


class Contracts(unittest.TestCase):
    def test_fixed_split_and_disjoint(self):
        ims = [dict(id=i, scene_id=str(i)) for i in range(500)]
        anns = [dict(image_id=i, area=20) for i in range(500)]
        a = select_manifest(ims, anns, {"0"})
        self.assertEqual(a, select_manifest(ims, anns, {"0"}))
        self.assertEqual(len(a["fit_ids"]), 256)
        self.assertEqual(len(a["check_ids"]), 128)
        self.assertFalse(set(a["fit_ids"]) & set(a["check_ids"]))
        self.assertNotIn("0", a["scenes"].values())
        self.assertEqual(len(set(a["scenes"].values())), 384)

    def test_scene_not_image_shortage(self):
        ims = [dict(id=i, scene_id=str(i // 2)) for i in range(600)]
        anns = [dict(image_id=i, area=20) for i in range(600)]
        self.assertFalse(select_manifest(ims, anns, set())["gate_passed"])

    def test_no_large_or_ignored(self):
        ims = [dict(id=i, scene_id=str(i)) for i in range(500)]
        anns = [dict(image_id=i, area=300) for i in range(500)]
        self.assertFalse(select_manifest(ims, anns, set())["gate_passed"])

    def test_gate_unique_not_patch(self):
        p = [dict(gt_id=i, gt_area=64 if i < 8 else 200, image_id=i) for i in range(32)]
        scenes = {str(i): str(i) for i in range(32)}
        self.assertTrue(positive_gate(p, scenes)["gate_passed"])
        self.assertFalse(positive_gate(p[:31] * 2, scenes)["gate_passed"])
        self.assertFalse(positive_gate(p, {str(i): "same" for i in range(32)})["gate_passed"])

    def test_budget(self):
        self.assertEqual(DEADLINE - START, 7200)
        self.assertEqual(PREP_END - START, 1200)


def prep():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    suite = unittest.TestSuite((
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_tiny_verifier"),
        unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)))
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful() and not tests.skipped
    assert sha(REFERENCE / "report.json") == "ded4ec17b5a94bc603534e191b93c901d5b31caa8b2e92d396efa3f8b805de4c"
    assert read(REFERENCE / "status.json")["status"] == "complete"
    assert read(REFERENCE / "report.json")["new_aps"] == BEST_APS
    for name, digest in read(REFERENCE / "plan.json")["source_sha256"].items():
        assert sha(OUT / "source" / name) == digest, name
    inputs = {}
    def bind(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        inputs[str(path.relative_to(ROOT))] = digest
        return digest
    bind(REFERENCE / "verifier_2000.pth",
         "889a3c0e0e6b2c76945ad8588a1af27cff697c5ea1feab0911042704cb78f65a")
    for name in ("report.json", "prepared_manifest.json", "positive_expansion.json",
                 "fit_report.json", "check_after.json", "predictions.segm.json"):
        bind(REFERENCE / name)
    old = read(DATA_REF / "prepared_manifest.json")
    assert old == read(REFERENCE / "prepared_manifest.json")
    source = read(w.OLD / "prepared_manifest.json")
    old_ids = old["fit_ids"] + old["check_ids"]
    old_scenes = {source["images"][str(i)]["request"]["scene"] for i in old_ids}
    coco = w.coco_load("train")
    bind(ROOT / w.base.DATA / "annotations/instances_train.json", w.base.TRAIN_SHA)
    manifest = select_manifest(list(coco.imgs.values()), list(coco.anns.values()), old_scenes)
    manifest.update(old_fit_ids=old["fit_ids"], old_check_ids=old["check_ids"],
                    excluded_scenes=sorted(old_scenes), old_images=len(old_ids))
    if manifest["gate_passed"]:
        assert not set(old_ids) & set(manifest["fit_ids"] + manifest["check_ids"])
        manifest["requests"] = {str(i): w.base.image_request(coco.imgs[i], "train")
                               for i in manifest["fit_ids"] + manifest["check_ids"]}
        manifest["input_hashes"] = {
            str(i): dict(image_id=i, rgb_sha256=bind(Path(q["img_path"])),
                         depth_sha256=bind(Path(q["depth_file"])))
            for i, q in ((int(k), q) for k, q in manifest["requests"].items())}
    # Preserve the entire accepted old fit sample pool with exact patch hashes.
    old_binding = read(REFERENCE / "binding.json")["input_sha256"]
    counts = collections.Counter()
    for i in old["fit_ids"]:
        w.check(PREP_END)
        p = DATA_REF / "candidate_patches" / f"{i}.npz"
        bind(p, old_binding[str(p.relative_to(ROOT))])
        counts.update(old["records"][str(i)]["labels"])
        p = REFERENCE / "extra_positive" / f"{i}.npz"
        if p.exists():
            bind(p, read(REFERENCE / "binding.json")["extra_patches_sha256"][p.name])
    assert counts == {1: 47, 0: 7947, -1: 188}
    save(OUT / "prepared_manifest.json", manifest)
    save(OUT / "binding.json", dict(input_sha256=inputs,
        selector_sha256=bind(w.PARENT / "selector.json"),
        baseline_report_sha256=sha(REFERENCE / "report.json"),
        frozen35_unchanged=True, old_positive=226, old_negative=7947))
    save(OUT / "cpu_acceptance.json", dict(status="passed", tests=tests.testsRun, skipped=0,
        gate_passed=manifest["gate_passed"], gpu_used=False, old_samples_preserved=True,
        scene_disjoint=True, finished_unix=time.time()))
    w.check(PREP_END)


def capture():
    import torch
    import m2f_boundary_refiner as ref
    import m2f_quality_selector as quality
    import run_m2f_quality_select as original
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    torch = w.base.configure_torch()
    assert torch.cuda.device_count() == 1
    end = read(OUT / "phase_deadlines.json")["capture_end"]
    m = read(OUT / "prepared_manifest.json")
    assert m["gate_passed"]
    original.START, original.DEADLINE = START, DEADLINE
    accepted, earlier, tiles, _ = original.modules()
    compact = read(w.PARENT / "compact.json")
    cfg = Config.fromfile(str(w.PARENT / "live_config.py"))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    assert sha(compact["weight"]) == tiles.SPEC["weight_sha"]
    cp = torch.load(compact["weight"], map_location="cpu")
    model = MODELS.build(cfg.model); model.load_state_dict(cp["state_dict"], strict=True); del cp
    state_sha = original.tensor_digest(model.state_dict())
    model.cuda().eval(); correction = w.load_current().cuda()
    pipeline = earlier.typed_pipeline(cfg, compact, tiles)
    loader, tail = Compose(pipeline[:1]), Compose(pipeline[1:])
    selector = read(w.PARENT / "selector.json")
    assert sha(w.PARENT / "selector.json") == read(OUT / "binding.json")["selector_sha256"]
    (OUT / "train_predictions").mkdir()
    records = []
    with torch.inference_mode():
        for n, ident in enumerate(m["fit_ids"] + m["check_ids"]):
            w.check(end)
            request = m["requests"][str(ident)]
            actual = dict(image_id=ident, rgb_sha256=sha(request["img_path"]),
                          depth_sha256=sha(request["depth_file"]))
            assert actual == m["input_hashes"][str(ident)]
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            views = tiles.pack_views(request, loader, tail)
            encoder = accepted.CandidateEncoder(tiles)
            pool, global_rows = [], []
            for index, view in enumerate(views):
                predicted = model.test_step(pseudo_collate([view]))[0].cpu()
                captured, candidates = encoder.encode(predicted.pred_instances, index, ident)
                if index == 0: global_rows = captured
                pool.extend(candidates)
            merged = accepted.base.direct_rows(tiles.merge_candidates(pool), ident)
            chosen = quality.choose(global_rows, merged, selector)[0]
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            current = ref.infer(chosen, ref.load_rgbd(request), correction, "cuda")[0]
            assert encoder.encoded == encoder.roundtrips
            path = OUT / "train_predictions" / f"{ident}.json"
            save(path, dict(image_id=ident, input_hashes=actual, global_rows=global_rows,
                            merged=merged, current=current))
            records.append(dict(image_id=ident, sha256=sha(path)))
            if n % 4 == 0: w.mark("capture_new_train", images=n + 1, expected=384)
    assert original.tensor_digest(model.state_dict()) == state_sha
    save(OUT / "capture_report.json", dict(status="passed", images=384, eval_calls=1920,
        train_calls=0, state_before=state_sha, state_after=state_sha, files=records,
        gt_used_for_prediction=False, finished_unix=time.time()))


def samples():
    import numpy as np
    from pycocotools import mask as mu
    import m2f_boundary_refiner as ref
    import m2f_tiny_verifier as tiny
    m = read(OUT / "prepared_manifest.json")
    end = read(OUT / "phase_deadlines.json")["capture_end"]
    coco = w.coco_load("train")
    (OUT / "candidate_patches").mkdir()
    (OUT / "train_metadata").mkdir()
    positives, records, files = [], {}, {}
    frozen = {v["image_id"]: v["sha256"] for v in read(OUT / "capture_report.json")["files"]}
    for n, ident in enumerate(m["fit_ids"] + m["check_ids"]):
        w.check(end)
        path = OUT / "train_predictions" / f"{ident}.json"
        assert sha(path) == frozen[ident]
        cache = read(path); merged, current = cache["merged"], cache["current"]
        jobs = tiny.eligible(current, merged)
        anns = sorted((a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
                       if not a.get("iscrowd", 0) and not a.get("ignore", 0)), key=lambda a: a["id"])
        def label(job):
            row = merged[job["index"]]
            same = [a for a in anns if a["category_id"] == row["category_id"]]
            values = mu.iou([row["segmentation"]], [coco.annToRLE(a) for a in same],
                            [0] * len(same))[0] if same else []
            j = int(np.argmax(values)) if len(values) else None
            value = float(values[j]) if j is not None else 0.
            ann = same[j] if j is not None and value > 0 else None
            return dict(**job, label=tiny.target(value, ann["area"] if ann else None),
                        gt_id=ann["id"] if ann else None, gt_area=ann["area"] if ann else None,
                        max_gt_iou=value)
        jobs = [label(j) for j in jobs]
        ordinary_count = len(jobs)
        if ident in m["fit_ids"]:
            keys = {helpers.mask_key(merged[j["index"]]) for j in jobs}
            for j in tiny.eligible([], merged):
                row = merged[j["index"]]
                if helpers.mask_key(row) in keys: continue
                item = label(j)
                same = [r["segmentation"] for r in current if r["category_id"] == row["category_id"]]
                overlap = float(mu.iou([row["segmentation"]], same, [0]*len(same)).max()) if same else 0.
                if helpers.is_new_positive(row, keys, item["label"], overlap):
                    item["max_current_iou"] = overlap
                    jobs.append(item); keys.add(helpers.mask_key(row))
            positives.extend(dict(image_id=ident, **j) for j in jobs if j["label"] == 1)
        rgbd = ref.load_rgbd(m["requests"][str(ident)])
        x = np.stack([tiny.patch(merged[j["index"]], j["xy"], rgbd) for j in jobs]) if jobs else (
            np.empty((0, 6, 128, 128), np.float32))
        path = OUT / "candidate_patches" / f"{ident}.npz"
        np.savez_compressed(path, x=x); files[str(path.relative_to(OUT))] = sha(path)
        path = OUT / "train_metadata" / f"{ident}.json"
        save(path, dict(image_id=ident, jobs=jobs, ordinary_count=ordinary_count,
                        input_hashes=cache["input_hashes"]))
        files[str(path.relative_to(OUT))] = sha(path)
        records[str(ident)] = dict(count=len(jobs), labels=[j["label"] for j in jobs])
        if n % 16 == 0: w.mark("new_train_samples", images=n + 1, expected=384)
    gate = positive_gate(positives, m["scenes"])
    save(OUT / "sample_manifest.json", dict(records=records, files=files,
        fit_ids=m["fit_ids"], check_ids=m["check_ids"], new_positive_records=positives))
    save(OUT / "headroom_gate.json", dict(status="passed", **gate,
        fit_only=True, check_used=False, val_used=False, finished_unix=time.time()))


def check_arm(model, ids, new, end):
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    rows, activities = [], []
    for n, ident in enumerate(ids):
        w.check(end)
        if new:
            cache = read(OUT / "train_predictions" / f"{ident}.json")
            current, merged = cache["current"], cache["merged"]
            root = OUT
        else:
            current = read(DATA_REF / "current_train" / f"{ident}.json")
            merged = read(w.PARENT / "train_predictions" / f"{ident}.json")["merged"]
            root = DATA_REF
        meta = read(root / "train_metadata" / f"{ident}.json")
        jobs = meta["jobs"][:meta.get("ordinary_count", len(meta["jobs"]))]
        probabilities = []
        with np.load(root / "candidate_patches" / f"{ident}.npz") as z, torch.inference_mode():
            for start in range(0, len(jobs), 32):
                size = min(32, len(jobs) - start)
                batch = np.zeros((32,6,128,128), np.float32)
                batch[:size] = z["x"][start:start+size]
                probabilities.extend(map(float, torch.sigmoid(model(torch.from_numpy(batch).cuda())).cpu().numpy()[:size]))
        final, detail = tiny.merge(current, merged, jobs, probabilities)
        rows.extend(final)
        activities.append(dict(image_id=ident, **detail,
            candidates=[dict(**j, probability=p) for j, p in zip(jobs, probabilities)]))
        if n % 16 == 0: w.mark("fresh_check" if new else "old_continuity", images=n+1, expected=len(ids))
    return rows, activities


def traincheck():
    import numpy as np
    import torch
    import m2f_tiny_verifier as tiny
    from run_m2f_quality_select import tensor_digest
    torch = w.base.configure_torch()
    end = read(OUT / "phase_deadlines.json")["train_end"]
    assert read(OUT / "headroom_gate.json")["gate_passed"]
    m = read(OUT / "prepared_manifest.json")
    old = read(DATA_REF / "prepared_manifest.json")
    sm = read(OUT / "sample_manifest.json")
    arrays, labels = [], []
    for root, ids, records in ((DATA_REF, m["old_fit_ids"], old["records"]),
                               (OUT, m["fit_ids"], sm["records"])):
        for ident in ids:
            w.check(end)
            with np.load(root / "candidate_patches" / f"{ident}.npz") as z:
                labs = np.asarray(records[str(ident)]["labels"], dtype=np.int64)
                arrays.append(z["x"][labs >= 0]); labels.extend(labs[labs >= 0].tolist())
    for ident in m["old_fit_ids"]:
        path = REFERENCE / "extra_positive" / f"{ident}.npz"
        if path.exists():
            with np.load(path) as z:
                arrays.append(z["x"]); labels.extend([1] * len(z["x"]))
    x = torch.from_numpy(np.concatenate(arrays)); del arrays
    y = torch.tensor(labels, dtype=torch.float32)
    pos, neg = torch.where(y == 1)[0], torch.where(y == 0)[0]
    assert len(pos) >= 226 + 32 and len(neg) >= 7947
    torch.manual_seed(20261013); np.random.seed(20261013)
    model = tiny.Verifier().cuda().train()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    assert tensor_digest(initial) == read(REFERENCE / "fit_report.json")["initial_state_sha256"]
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = torch.Generator().manual_seed(20261013)
    history = []; begun = time.time()
    for step in range(1, 2001):
        w.check(end)
        inds = torch.cat((pos[torch.randint(len(pos),(16,),generator=rng)],
                          neg[torch.randint(len(neg),(16,),generator=rng)]))
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[inds].cuda()), y[inds].cuda())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        grad = float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters())))
        assert math.isfinite(grad)
        opt.step(); history.append(dict(step=step, loss=float(loss.detach()), gradient_norm=grad))
        if step == 1 or step % 50 == 0: w.mark("fit_expanded_scene", step=step, expected=2000)
    torch.cuda.synchronize()
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    native = sorted({int(v["step"].item()) for v in opt.state.values()})
    assert native == [2000] and all(not torch.equal(initial[k],v) for k,v in state.items())
    assert all(bool(torch.isfinite(v).all()) for v in state.values())
    assert all(bool(torch.isfinite(v).all()) for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    torch.save(dict(state_dict=state, optimizer=opt.state_dict(), steps=2000,
        fit_ids=m["old_fit_ids"]+m["fit_ids"], seed=20261013), OUT/"verifier_2000.pth")
    save(OUT/"fit_history.json", history)
    save(OUT/"fit_report.json", dict(status="passed", native_steps=native, finite_steps=2000,
        consumed_patches=64000, positive_patches=len(pos), negative_patches=len(neg), fit_images=640,
        old_positive_preserved=226, old_negative_preserved=7947, check_used_for_fit=0,
        initial_state_sha256=tensor_digest(initial), final_state_sha256=tensor_digest(state),
        parameters=sum(p.numel() for p in model.parameters()), state_bytes=sum(v.numel()*v.element_size() for v in state.values()),
        all_parameters_updated=True, model_optimizer_finite=True, fit_seconds=time.time()-begun,
        checkpoint_sha256=sha(OUT/"verifier_2000.pth"), finished_unix=time.time()))
    del x, y, opt
    model.eval()
    reference = tiny.Verifier().cuda()
    reference.load_state_dict(torch.load(REFERENCE/"verifier_2000.pth", map_location="cpu")["state_dict"], strict=True)
    reference.eval()
    before, before_activity = check_arm(reference, m["check_ids"], True, end)
    after, activity = check_arm(model, m["check_ids"], True, end)
    old_after, _ = check_arm(model, m["old_check_ids"], False, end)
    coco = w.coco_load("train")
    bm, am = w.metrics(coco,m["check_ids"],before), w.metrics(coco,m["check_ids"],after)
    bp, ap = w.matched(coco,m["check_ids"],before), w.matched(coco,m["check_ids"],after)
    gained, lost = sorted(set(ap["tiny"])-set(bp["tiny"])), sorted(set(bp["tiny"])-set(ap["tiny"]))
    for name, value in (("check_before",before),("check_after",after),("check_activity",activity),
                         ("check_before_activity",before_activity),("old_check_after",old_after)):
        save(OUT/(name+".json"),value)
    save(OUT/"check_matches.json",dict(before=bp,after=ap,tiny_gained=gained,tiny_lost=lost))
    save(OUT/"old_continuity.json",dict(metrics=w.metrics(coco,m["old_check_ids"],old_after),
        baseline_metrics=read(REFERENCE/"learned_gate.json")["metrics"],not_gate=True))
    save(OUT/"learned_gate.json",dict(status="passed",
        gate_passed=am["APs"]>bm["APs"] and len(gained)>=len(lost),
        baseline_metrics=bm,metrics=am,delta_points={k:100*(am[k]-bm[k]) for k in bm},
        tiny_gained=gained,tiny_lost=lost,tiny_net=len(gained)-len(lost),
        rule="fresh128 small AP strict positive AND tiny one-to-one net>=0",
        fresh_scenes=True,check_not_fit=True,val_used=False,finished_unix=time.time()))


def finish():
    comparison = {}
    stopped = ("manifest_gate" if not read(OUT/"cpu_acceptance.json")["gate_passed"] else
               "headroom_gate" if not read(OUT/"headroom_gate.json")["gate_passed"] else
               "learned_gate" if not read(OUT/"learned_gate.json")["gate_passed"] else None)
    if (OUT/"fit_report.json").exists():
        m = read(OUT/"prepared_manifest.json")
        summary,_ = helpers.paired_summary(w,w.coco_load("train"),m["check_ids"],
                                          read(OUT/"check_before.json"),read(OUT/"check_after.json"))
        save(OUT/"internal_analysis.json",summary)
    if not stopped:
        assert not (OUT/"full_val_metrics.json").exists()
        frozen = read(OUT/"predictions_frozen.json")
        assert sha(OUT/"predictions.segm.json") == frozen["prediction_sha256"]
        ids = frozen["image_ids"]; assert len(ids)==len(set(ids))==2584
        coco=w.coco_load("val"); rows=read(OUT/"predictions.segm.json")
        actual=w.metrics(coco,ids,rows)
        save(OUT/"full_val_metrics.json",dict(metrics=actual,image_ids=ids,images=2584,evaluations=1))
        baseline=read(REFERENCE/"full_val_metrics.json")["metrics"]
        summary,hits=helpers.paired_summary(w,coco,ids,read(REFERENCE/"predictions.segm.json"),rows)
        old=read(REFERENCE/"full_val_error_analysis.json")
        summary["previous45_kept"]=sorted(set(old["tiny_gained"])&set(hits["tiny"]))
        summary["previous45_lost"]=sorted(set(old["tiny_gained"])-set(hits["tiny"]))
        summary["previous4_recovered"]=sorted(set(old["tiny_lost"])&set(hits["tiny"]))
        save(OUT/"full_val_error_analysis.json",summary)
        comparison=dict(metrics=actual,baseline_metrics=baseline,
            delta_points={k:100*(actual[k]-baseline[k]) for k in baseline},
            any_positive_retained=actual["APs"]>BEST_APS,single_development_val=True)
        save(OUT/"comparison.json",comparison)
    binding=read(OUT/"binding.json")
    assert all(sha(ROOT/p)==h for p,h in binding["input_sha256"].items())
    names=[p.name for p in OUT.glob("*.json") if p.name not in (
        "status.json","progress.json","report.json","terminal_acceptance.json","plan.json",
        "launch.json","phases.json","phase_deadlines.json")
        and not p.name.endswith(("_command.json","_container.json"))]
    if (OUT/"verifier_2000.pth").exists():names.append("verifier_2000.pth")
    w.check()
    save(OUT/"report.json",dict(status="passed",outcome="stopped_failed_"+stopped if stopped else "evaluated",
        start_unix=START,deadline_unix=DEADLINE,finished_unix=time.time(),baseline_aps=BEST_APS,
        full_val_evaluations=int(not stopped),new_aps=comparison.get("metrics",{}).get("APs"),
        verifier_trained=(OUT/"fit_report.json").exists(),m2f_updated=False,old_refiner_updated=False,
        maskrcnn_paused=True,only_change="new scene-disjoint official train data; uniform sampler",
        artifacts_sha256={n:sha(OUT/n) for n in names},comparison=comparison))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prep","capture","samples","traincheck","predict","finish"))
    phase=parser.parse_args().phase; install()
    end=dict(prep=PREP_END,capture=CAPTURE_END,samples=CAPTURE_END,
             traincheck=TRAIN_END,predict=GPU_END,finish=DEADLINE)[phase]
    if phase not in ("prep","finish"):
        limits=read(OUT/"phase_deadlines.json")
        end=min(end,limits["capture_end" if phase in ("capture","samples") else
                           "train_end" if phase=="traincheck" else "predict_end"])
    w.check(end)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("Immutable stage deadline")))
    signal.alarm(max(1,math.ceil(end-time.time())))
    try:
        if phase=="predict":w.predict()
        else:globals()[phase]()
    except BaseException as exc:
        save(OUT/(phase+"_first_error.json"),dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        raise


if __name__=="__main__":main()

"""Only expand fit positives. No GT enters inference and no check/val is fitted."""
import contextlib
import io
import json
import os
from pathlib import Path
import time
import unittest


def mask_key(row):
    r = row["segmentation"]
    return (row["category_id"], tuple(r["size"]), r["counts"])


def is_new_positive(row, old_keys, label, max_current_iou):
    return label == 1 and max_current_iou >= .5 and mask_key(row) not in old_keys


def analysis_ids(manifest):
    ids = manifest["image_ids"]
    assert len(ids) == len(set(ids)) == 256
    return ids


class Contracts(unittest.TestCase):
    def test_already_present_not_added(self):
        r = dict(category_id=1, segmentation=dict(size=[1024, 1024], counts="a"))
        self.assertFalse(is_new_positive(r, {mask_key(r)}, 1, .9))
    def test_only_positive(self):
        r = dict(category_id=1, segmentation=dict(size=[1024, 1024], counts="a"))
        self.assertFalse(is_new_positive(r, set(), 0, .9))
        self.assertFalse(is_new_positive(r, set(), -1, .9))
    def test_only_currently_excluded(self):
        r = dict(category_id=1, segmentation=dict(size=[1024, 1024], counts="a"))
        self.assertFalse(is_new_positive(r, set(), 1, .499))
        self.assertTrue(is_new_positive(r, set(), 1, .5))
    def test_fixed256_includes_no_small_images(self):
        self.assertEqual(analysis_ids({"image_ids": list(range(256))}), list(range(256)))
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(AssertionError):
            analysis_ids({"image_ids": [1] * 256})


def prep(w):
    import numpy as np
    from pycocotools import mask as mu
    import m2f_boundary_refiner as ref
    import m2f_tiny_verifier as tiny
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    suite = unittest.TestSuite((
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_tiny_verifier"),
        unittest.defaultTestLoader.loadTestsFromTestCase(Contracts),
    ))
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.wasSuccessful() and not tests.skipped
    prior = w.read(w.REFERENCE / "report.json")
    assert w.sha(w.REFERENCE / "report.json") == "2a1785e49614964afa398c6638232d26564cd3ec42f8b1ad85d49977a95e1065"
    assert prior["new_aps"] == w.BEST_APS and prior["status"] == "passed"
    assert w.read(w.REFERENCE / "recovery_status.json")["status"] == "complete"
    assert w.read(w.REFERENCE / "terminal_acceptance.json")["report_sha256"] == w.sha(w.REFERENCE / "report.json")
    for name, h in w.read(w.REFERENCE / "plan.json")["source_sha256"].items():
        assert w.sha(w.OUT / "source" / name) == h, name
    manifest = w.read(w.REFERENCE / "prepared_manifest.json")
    source_manifest = w.read(w.BEST / "prepared_manifest.json")
    assert manifest["fit_ids"] == source_manifest["fit_ids"]
    assert manifest["check_ids"] == source_manifest["check_ids"]
    assert len(manifest["fit_ids"]) == 384 and len(manifest["check_ids"]) == 128
    assert len(set(manifest["fit_ids"] + manifest["check_ids"])) == 512
    scenes = [{source_manifest["images"][str(i)]["request"]["scene"] for i in manifest[k]}
              for k in ("fit_ids", "check_ids")]
    assert not scenes[0] & scenes[1]
    inputs = {}
    def bind(p):
        rel = str(p.relative_to(w.ROOT))
        value = w.sha(p)
        inputs[rel] = value
        return value
    for name in ("report.json", "prepared_manifest.json", "check_after.json", "learned_gate.json",
                 "check_matches.json", "fit_report.json", "verifier_2000.pth",
                 "current_train_parity.json", "predictions.segm.json", "changed_image_audit.json"):
        value = bind(w.REFERENCE / name)
        if name in prior["artifacts_sha256"]:
            assert value == prior["artifacts_sha256"][name]
    assert inputs[str((w.REFERENCE / "verifier_2000.pth").relative_to(w.ROOT))] == (
        "e7e4da8db1281b197f9bc11b0a325ec25f546ff492818561bd7c0b89315973b6")
    upstream = w.read(w.REFERENCE / "binding.json")["input_sha256"]
    coco = w.coco_load("train")
    old_pos, old_neg, old_ambiguous, added = [], 0, 0, []
    (w.OUT / "extra_positive").mkdir()
    for n, ident in enumerate(manifest["fit_ids"] + manifest["check_ids"]):
        w.check(w.PREP_END)
        meta_path = w.REFERENCE / "train_metadata" / f"{ident}.json"
        current_path = w.REFERENCE / "current_train" / f"{ident}.json"
        patches_path = w.REFERENCE / "candidate_patches" / f"{ident}.npz"
        for p in (meta_path, current_path, patches_path):
            bind(p)
        meta = w.read(meta_path)
        assert [j["label"] for j in meta["jobs"]] == manifest["records"][str(ident)]["labels"]
        if ident not in manifest["fit_ids"]:
            continue
        old_pos += [dict(image_id=ident, **j) for j in meta["jobs"] if j["label"] == 1]
        old_neg += sum(j["label"] == 0 for j in meta["jobs"])
        old_ambiguous += sum(j["label"] == -1 for j in meta["jobs"])
        path = w.PARENT / "train_predictions" / f"{ident}.json"
        assert bind(path) == upstream[str(path.relative_to(w.ROOT))]
        cache = w.read(path)
        current = w.read(current_path)
        assert tiny.eligible(current, cache["merged"]) == [
            {k: j[k] for k in ("index", "xy", "max_current_iou")} for j in meta["jobs"]
        ]
        old_keys = {mask_key(cache["merged"][j["index"]]) for j in meta["jobs"]}
        anns = sorted((a for a in coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
                       if not a.get("iscrowd", 0)), key=lambda a: a["id"])
        positives = []
        for job in tiny.eligible([], cache["merged"]):
            row = cache["merged"][job["index"]]
            if mask_key(row) in old_keys:
                continue
            same = [a for a in anns if a["category_id"] == row["category_id"]]
            if not same:
                continue
            values = mu.iou([row["segmentation"]], [coco.annToRLE(a) for a in same], [0] * len(same))[0]
            k = int(np.argmax(values)); value = float(values[k]); ann = same[k]
            label = tiny.target(value, ann["area"] if value > 0 else None)
            existing = [r["segmentation"] for r in current if r["category_id"] == row["category_id"]]
            overlap = float(mu.iou([row["segmentation"]], existing, [0] * len(existing)).max()) if existing else 0.
            if is_new_positive(row, old_keys, label, overlap):
                positives.append(dict(**job, gt_id=ann["id"], gt_area=ann["area"],
                                      max_gt_iou=value, max_old_overlap=overlap, label=1))
                old_keys.add(mask_key(row))
        if positives:
            request = source_manifest["images"][str(ident)]["request"]
            actual = dict(image_id=ident, rgb_sha256=bind(Path(request["img_path"])),
                          depth_sha256=bind(Path(request["depth_file"])))
            assert actual == meta["input_hashes"] == cache["input_hashes"]
            rgbd = ref.load_rgbd(request)
            x = np.stack([tiny.patch(cache["merged"][j["index"]], j["xy"], rgbd) for j in positives])
            np.savez_compressed(w.OUT / "extra_positive" / f"{ident}.npz", x=x)
            added += [dict(image_id=ident, **j) for j in positives]
        if n % 16 == 0:
            w.mark("positive_coverage_cpu", images=n + 1, expected=512, extra_positive=len(added))
    assert len(old_pos) == 47 and old_neg == 7947 and old_ambiguous == 188
    old_gt = {p["gt_id"] for p in old_pos}
    new_gt = old_gt | {p["gt_id"] for p in added}
    assert len(old_gt) == 42
    passed = len(new_gt) > len(old_gt)
    w.save(w.OUT / "prepared_manifest.json", manifest)
    w.save(w.OUT / "positive_expansion.json", dict(original=old_pos, added=added,
        original_positive=47, expanded_positive=47 + len(added), original_unique_gt=42,
        expanded_unique_gt=len(new_gt), negative_unchanged=7947, ambiguous_unchanged=188,
        fit_images=384, check_added=0, inference_eligibility_unchanged=True))
    inputs[str((w.PARENT / "selector.json").relative_to(w.ROOT))] = w.sha(w.PARENT / "selector.json")
    w.save(w.OUT / "binding.json", dict(input_sha256=inputs, baseline_report_sha256=w.sha(w.REFERENCE / "report.json"),
        baseline_metrics=w.read(w.REFERENCE / "full_val_metrics.json")["metrics"],
        selector_sha256=w.sha(w.PARENT / "selector.json"), original32_source_unchanged=True,
        extra_patches_sha256={p.name: w.sha(p) for p in (w.OUT / "extra_positive").glob("*.npz")}))
    w.save(w.OUT / "headroom_gate.json", dict(status="passed", gate_passed=passed,
        original_unique_positive_gt=42, expanded_unique_positive_gt=len(new_gt), extra_positive=len(added),
        original_positive_preserved=True, negatives_unchanged=True, scene_disjoint=True,
        only_fit_data=True, validation_used=False, finished_unix=time.time()))
    w.save(w.OUT / "cpu_acceptance.json", dict(status="passed", tests=tests.testsRun, skipped=0,
        gpu_used=False, gate_passed=passed, scene_disjoint=True, finished_unix=time.time()))
    w.check(w.PREP_END)


def paired_summary(w, coco, ids, before, after):
    """One-to-one IoU.5 statistics only; no AP accumulate here."""
    from pycocotools import mask as mu
    bp, ap = w.matched(coco, ids, before), w.matched(coco, ids, after)
    result = {}
    for size in ("small", "tiny"):
        result[size + "_gained"] = sorted(set(ap[size]) - set(bp[size]))
        result[size + "_lost"] = sorted(set(bp[size]) - set(ap[size]))
    true_after = {p["detection_id"]: p["annotation_id"] for p in ap["true_predictions"]}
    false_after = {p["detection_id"] for p in ap["false_positives"]}
    by_new = {i: set() for i in ids}
    by_old = {i: set() for i in ids}
    for r in before: by_old[r["image_id"]].add(mask_key(r))
    for r in after: by_new[r["image_id"]].add(mask_key(r))
    removed = []
    for p in bp["true_predictions"]:
        r = before[p["detection_id"] - 1]
        if mask_key(r) not in by_new[r["image_id"]]:
            removed.append(dict(**p, gt_still_matched=p["annotation_id"] in ap["small"]))
    novel_tp, novel_fp = [], []
    for index, r in enumerate(after, 1):
        if mask_key(r) in by_old[r["image_id"]]:
            continue
        if index in true_after:
            novel_tp.append(dict(image_id=r["image_id"], annotation_id=true_after[index], score=r["score"]))
        elif index in false_after:
            large = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[r["image_id"]]))
                     if a["category_id"] == r["category_id"] and a["area"] > 1024 and not a.get("iscrowd", 0)]
            contain = float(mu.iou([r["segmentation"]], [coco.annToRLE(a) for a in large],
                                   [1] * len(large)).max()) if large else 0.
            novel_fp.append(dict(image_id=r["image_id"], score=r["score"], large_gt_contains_fraction=contain))
    result.update(images=len(ids), before_tiny=len(bp["tiny"]), after_tiny=len(ap["tiny"]),
        before_small=len(bp["small"]), after_small=len(ap["small"]),
        before_fp=len(bp["false_positives"]), after_fp=len(ap["false_positives"]),
        removed_old_correct_small=removed, novel_true_predictions=novel_tp,
        novel_fp_count=len(novel_fp), novel_fp_large_containment=sum(p["large_gt_contains_fraction"] >= .8 for p in novel_fp),
        novel_fp_examples=sorted(novel_fp, key=lambda p: -p["score"])[:20],
        scope="small-area one-to-one IoU .5; novel geometry relative to previous verifier; not AP")
    return result, ap


def finish(w, stopped):
    w.check()
    compare = {}
    if (w.OUT / "fit_report.json").exists():
        manifest = w.read(w.OUT / "prepared_manifest.json")
        internal, _ = paired_summary(w, w.coco_load("train"), manifest["check_ids"],
            w.read(w.REFERENCE / "check_after.json"), w.read(w.OUT / "check_after.json"))
        act = w.read(w.OUT / "check_activity.json")
        internal["candidate_labels"] = {str(label): dict(
            total=sum(p["label"] == label for a in act for p in a["candidates"]),
            accepted=sum(p["label"] == label and p["probability"] >= .5 for a in act for p in a["candidates"]))
            for label in (-1, 0, 1)}
        w.save(w.OUT / "internal_analysis.json", internal)
    if not stopped:
        w.check(w.GPU_END)
        assert w.read(w.OUT / "learned_gate.json")["gate_passed"]
        assert not (w.OUT / "full_val_metrics.json").exists(), "Never repeat full AP"
        frozen = w.read(w.OUT / "predictions_frozen.json")
        assert w.sha(w.OUT / "predictions.segm.json") == frozen["prediction_sha256"]
        ids = frozen["image_ids"]; assert len(ids) == len(set(ids)) == 2584
        coco = w.coco_load("val")
        rows = w.read(w.OUT / "predictions.segm.json")
        current = w.metrics(coco, ids, rows)
        w.save(w.OUT / "full_val_metrics.json", dict(metrics=current, image_ids=ids, images=2584, evaluations=1))
        baseline = w.read(w.REFERENCE / "full_val_metrics.json")["metrics"]
        before = w.read(w.REFERENCE / "predictions.segm.json")
        summary, matched = paired_summary(w, coco, ids, before, rows)
        previous12 = set(w.read(w.REFERENCE / "changed_image_audit.json")["tiny_gained"])
        summary["previous12_kept"] = sorted(previous12 & set(matched["tiny"]))
        summary["previous12_lost"] = sorted(previous12 - set(matched["tiny"]))
        w.save(w.OUT / "full_val_error_analysis.json", summary)
        compare = dict(metrics=current, baseline_metrics=baseline,
            delta_points={k: 100 * (current[k] - baseline[k]) for k in baseline},
            any_positive_retained=current["APs"] > w.BEST_APS, single_development_val=True)
        w.save(w.OUT / "comparison.json", compare)
    binding = w.read(w.OUT / "binding.json")
    assert all(w.sha(w.ROOT / p) == h for p, h in binding["input_sha256"].items())
    assert all(w.sha(w.OUT / "extra_positive" / p) == h for p, h in binding["extra_patches_sha256"].items())
    names = [p.name for p in w.OUT.glob("*.json") if p.name not in (
        "status.json", "progress.json", "report.json", "terminal_acceptance.json", "plan.json",
        "launch.json", "phases.json", "phase_deadlines.json")
        and not p.name.endswith(("_command.json", "_container.json"))]
    trained = (w.OUT / "fit_report.json").exists()
    if trained: names.append("verifier_2000.pth")
    w.check()
    w.save(w.OUT / "report.json", dict(status="passed", outcome="stopped_failed_" + stopped if stopped else "evaluated",
        start_unix=w.START, deadline_unix=w.DEADLINE, finished_unix=time.time(), baseline_aps=w.BEST_APS,
        full_val_evaluations=int(not stopped), new_aps=compare.get("metrics", {}).get("APs"),
        verifier_trained=trained, m2f_updated=False, old_refiner_updated=False, maskrcnn_paused=True,
        only_change="fit positive coverage", artifacts_sha256={n: w.sha(w.OUT / n) for n in names},
        comparison=compare))

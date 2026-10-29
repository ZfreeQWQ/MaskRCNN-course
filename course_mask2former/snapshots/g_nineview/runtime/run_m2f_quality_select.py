"""One approved 120-minute train-only selector experiment, frozen M2F."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import time
import traceback

import m2f_quality_selector as quality

START = 1792482558  # 2026-10-20 15:49:18 Beijing; includes implementation.
CPU_END, GPU_END, EVAL_END, DEADLINE = (START + s for s in (1800, 4500, 5700, 7200))
PARENT = "outputs/course_m2f_candidate_reuse_20261020_v1"
PARENT_SHA = "5f2800a5dd9228791416f92b767db8d631faedf5bec6d5397be58ee6acbd694a"
GEOM = "outputs/course_m2f_geometry_20261020_v1"
GEOM_SHA = "7dbb3f1d4d624814e43c36c1464ea413095e9d26591d1d30853d0c49819878b6"
DATA = "data/course_blender_20261013/prepared"
TRAIN_SHA = "1f72e6564579f9257ab1ef54044207eccaeb452affd33f05b19a6ef00105e882"
VAL_SHA = "71612ebf1d548eb85e146e97de4d211578d2f87d5273f9b0f5e2d48b1af2ea2d"
FIXED8 = list(range(201960001, 201967002, 1000))
ROOT = Path("/workspace")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def modules():
    import m2f_candidate_reuse as accepted
    previous = accepted.previous
    previous.START, previous.DEADLINE = START, DEADLINE
    tiles, refine = previous.modules()
    return accepted, previous, tiles, refine


def progress(out, phase, **kwargs):
    data = dict(phase=phase, unix=time.time(), **kwargs)
    save(out/"progress.json", data)
    print(json.dumps(data), flush=True)


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu(out):
    import unittest
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    from mmdet.utils import register_all_modules
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not Path("/dev/nvidiactl").exists()
    accepted, previous, tiles, refine = modules()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_quality_selector"))
    assert result.wasSuccessful() and not result.skipped and result.testsRun == 20
    parent = ROOT/PARENT
    assert sha(parent/"report.json") == PARENT_SHA
    assert read(parent/"status.json")["status"] == "complete"
    assert read(parent/"terminal_acceptance.json")["report_sha256"] == PARENT_SHA
    parent_plan = read(parent/"plan.json")
    assert all(sha(parent/"source"/n) == h for n, h in parent_plan["source_sha256"].items())
    # Reused executable modules are the accepted exact files, not reimplemented.
    for module in (accepted, accepted.base, previous):
        assert sha(module.__file__) == parent_plan["source_sha256"][Path(module.__file__).name]
    parent_manifest = read(parent/"compact_manifest.json")
    assert all(sha(parent/n) == h for n, h in parent_manifest["files"].items())
    for name in ("compact.json", "live_config.py", "binding.json"):
        shutil.copyfile(parent/name, out/name)
    binding, compact = read(out/"binding.json"), read(out/"compact.json")
    assert all(sha(ROOT/n) == h for n, h in binding["files"].items())
    assert sha(compact["weight"]) == binding["checkpoint_sha256"] == tiles.SPEC["weight_sha"]
    assert sha(ROOT/GEOM/"report.json") == GEOM_SHA
    geometry = read(ROOT/GEOM/"report.json")
    assert geometry["status"] == "passed" and geometry["completed_images"] == 2584
    assert abs(geometry["metrics"]["APs"] - .40779742930530144) < 1e-14
    parents = {p: h for p, h in geometry["input_sha256"].items()
               if p.endswith(("predictions.segm.json", "input_hashes.json"))}
    assert len([p for p in parents if p.endswith("predictions.segm.json")]) == 2
    assert all(sha(ROOT/p) == h for p, h in parents.items())
    assert sha(ROOT/DATA/"annotations/instances_train.json") == TRAIN_SHA
    assert sha(ROOT/DATA/"annotations/instances_val.json") == VAL_SHA
    # Only image metadata is used here. GT shapes/quality never choose the manifest.
    images = read(ROOT/DATA/"annotations/instances_train.json")["images"]
    assert len(images) == 14432
    chosen = quality.select_manifest(images)
    val_ids = set(geometry["image_ids"])
    assert len(val_ids) == 2584 and not val_ids.intersection(i["id"] for i in images)
    requests = {}
    for image in chosen["fit"] + chosen["check"]:
        r = quality.train_request(ROOT/DATA, image)
        assert Path(r["img_path"]).is_file() and Path(r["depth_file"]).is_file()
        requests[image["id"]] = r
    manifest = dict(seed=quality.SEED, fit_ids=[i["id"] for i in chosen["fit"]],
                    check_ids=[i["id"] for i in chosen["check"]],
                    fit_scenes=sorted({str(i["scene_id"]) for i in chosen["fit"]}),
                    check_scenes=sorted({str(i["scene_id"]) for i in chosen["check"]}),
                    requests=requests, train_annotations_sha256=TRAIN_SHA,
                    quality_labels_used_for_selection=False, unix=time.time())
    save(out/"train_manifest.json", manifest)
    cfg = Config.fromfile(str(out/"live_config.py"))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    pipeline = previous.typed_pipeline(cfg, compact, tiles)
    assert all(p["type"] != "LoadAnnotations" for p in pipeline)
    loader, tail = Compose(pipeline[:1]), Compose(pipeline[1:])
    for r in (requests[manifest["fit_ids"][0]], compact["requests"][str(FIXED8[0])]):
        views = tiles.pack_views(r, loader, tail)
        assert len(views) == 5 and all(tuple(v["inputs"].shape) == (5, 1536, 1536) for v in views)
        assert all("gt_instances" not in v["data_samples"] for v in views)
    # Cheap exact binding check of the unchanged original candidate choice.
    references = [{int(k): v for k, v in x.items()} for x in compact["references"]]
    for ident in FIXED8:
        assert refine.refine(references[0][ident], references[1][ident])[0] == references[2][ident]
    save(out/"cpu_tests.json", dict(status="passed", tests=20, skipped=0,
        actual_train_and_val_preprocessing=True, original_eight_cache_refinement_exact=True,
        model_loaded=False, gpu_used=False))
    save(out/"source_binding.json", dict(parent=PARENT, parent_report_sha256=PARENT_SHA,
        geometry_report_sha256=GEOM_SHA, parent_sources=parent_plan["source_sha256"],
        full_val_cache_sha256=parents, checkpoint_sha256=binding["checkpoint_sha256"]))
    assert time.time() < CPU_END
    save(out/"cpu_acceptance.json", dict(status="passed", unix=time.time(), files={
        n: sha(out/n) for n in ("compact.json", "live_config.py", "binding.json", "train_manifest.json",
                                "cpu_tests.json", "source_binding.json")}))


def fit_train(out):
    import numpy as np
    from pycocotools.coco import COCO
    manifest = read(out/"train_manifest.json")
    assert sha(ROOT/DATA/"annotations/instances_train.json") == TRAIN_SHA
    # GT labels are consumed only after all 512 predictions are immutable.
    assert len(list((out/"train_predictions").glob("*.json"))) == 512
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ROOT/DATA/"annotations/instances_train.json"))
    groups = {}
    for arm in ("fit", "check"):
        pairs = []
        for ident in manifest[arm+"_ids"]:
            cached = read(out/"train_predictions"/f"{ident}.json")
            pairs.extend(quality.supervised_pairs(cached["global"], cached["merged"],
                coco.loadAnns(coco.getAnnIds(imgIds=[ident])), coco))
        groups[arm] = pairs
        save(out/(arm+"_pairs.json"), pairs)
    selector = quality.fit([p["features"] for p in groups["fit"]],
                           [p["target_delta"] for p in groups["fit"]])
    selector.update(fit_image_ids=manifest["fit_ids"], check_image_ids=manifest["check_ids"],
                    train_manifest_sha256=sha(out/"train_manifest.json"),
                    train_annotations_sha256=TRAIN_SHA, fitted_unix=time.time(),
                    model_weights_updated=False, validation_labels_used=False)
    save(out/"selector.json", selector)
    selector_sha = sha(out/"selector.json")
    stats = {}
    for arm, pairs in groups.items():
        stats[arm] = dict(all=quality.diagnostic(pairs, selector))
        for name, choose in (("tiny", lambda p: p["gt_area"] is not None and p["gt_area"] <= 256),
                             ("small_257_1024", lambda p: p["gt_area"] is not None and 256 < p["gt_area"] <= 1024),
                             ("large_or_unmatched", lambda p: p["gt_area"] is None or p["gt_area"] > 1024)):
            stats[arm][name] = quality.diagnostic([p for p in pairs if choose(p)], selector)
    save(out/"fit_diagnostics.json", dict(status="passed", groups=stats, selector_sha256=selector_sha,
        fitted_parameters=105, lambda_search=False, threshold_search=False, check_used_for_fit=False))
    assert sha(out/"selector.json") == selector_sha
    return selector


def gpu(out):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    accepted, previous, tiles, refine = modules()
    cpu_report = read(out/"cpu_acceptance.json")
    assert cpu_report["status"] == "passed" and cpu_report["unix"] < CPU_END
    assert all(sha(out/n) == h for n, h in cpu_report["files"].items())
    phase_end = min(GPU_END, read(out/"phase_deadlines.json")["gpu_end"])
    compact = read(out/"compact.json")
    train = read(out/"train_manifest.json")
    requests = {int(k): v for k, v in train["requests"].items()}
    requests.update({i: compact["requests"][str(i)] for i in FIXED8})
    refs = [{int(k): v for k, v in x.items()} for x in compact["references"]]
    cfg = Config.fromfile(str(out/"live_config.py"))
    pipeline = previous.typed_pipeline(cfg, compact, tiles)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    assert sha(compact["weight"]) == tiles.SPEC["weight_sha"]
    checkpoint = torch.load(compact["weight"], map_location="cpu")
    trained = Config.fromstring(checkpoint["meta"].get("cfg", checkpoint["meta"].get("config")), ".py")
    assert trained.model.to_dict() == cfg.model.to_dict()
    assert all(bool(torch.isfinite(t).all()) for t in checkpoint["state_dict"].values() if t.is_floating_point())
    model = MODELS.build(cfg.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    before = tensor_digest(model.state_dict())
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    assert model.active_parameter_count() == 44400414
    net = sum(t.numel()*t.element_size() for t in model.state_dict().values())
    assert net == 177814584
    loader, tail = Compose(pipeline[:1]), Compose(pipeline[1:])
    calls = dict(eval_calls=0, train_calls=0)
    hashes = []
    (out/"train_predictions").mkdir()
    started = time.time()

    def full_call(ident, selector=None):
        assert time.time() < phase_end and not model.training
        torch.cuda.synchronize()
        begun = time.perf_counter()
        views = tiles.pack_views(requests[ident], loader, tail)
        batches = [pseudo_collate([v]) for v in views]
        encoder = accepted.CandidateEncoder(tiles)
        captured, pool = [], []
        for view, batch in enumerate(batches):
            assert time.time() < phase_end
            pred = model.test_step(batch)[0].cpu()
            calls["eval_calls"] += 1
            rows, candidates = encoder.encode(pred.pred_instances, view, ident)
            if view == 0: captured = rows
            pool.extend(candidates)
            del pred
        merged = accepted.base.direct_rows(tiles.merge_candidates(pool), ident)
        final, choices = (quality.choose(captured, merged, selector) if selector is not None
                          else refine.refine(captured, merged))
        json.dumps(final, separators=(",", ":"), allow_nan=False)
        torch.cuda.synchronize()
        assert encoder.encoded == encoder.roundtrips
        return (captured, merged, final), choices, (time.perf_counter()-begun)*1000

    with torch.inference_mode():
        for n, ident in enumerate(train["fit_ids"] + train["check_ids"]):
            r = requests[ident]
            h = dict(image_id=ident, rgb_sha256=sha(r["img_path"]), depth_sha256=sha(r["depth_file"]))
            actual, _, _ = full_call(ident)
            save(out/"train_predictions"/f"{ident}.json", dict(image_id=ident, input_hashes=h,
                **{"global": actual[0]}, merged=actual[1]))
            hashes.append(h)
            progress(out, "train_capture", completed=n+1, expected=512, **calls)
        save(out/"train_input_hashes.json", hashes)
        capture_seconds = time.time()-started
        progress(out, "fit_selector", captured=512, **calls)
        fit_started = time.time()
        selector = fit_train(out)
        fit_seconds = time.time()-fit_started
        selector_sha = sha(out/"selector.json")
        parity = []
        for ident in FIXED8:
            r = requests[ident]
            assert dict(image_id=ident, rgb_sha256=sha(r["img_path"]), depth_sha256=sha(r["depth_file"])) == compact["input_hashes"][str(ident)]
            actual, decisions, _ = full_call(ident, selector)
            expected = quality.choose(refs[0][ident], refs[1][ident], selector)[0]
            assert actual[0] == refs[0][ident] and actual[1] == refs[1][ident] and actual[2] == expected
            parity.append(dict(image_id=ident, global_exact=True, merged_exact=True,
                               selected_exact=True, selector_sha256=selector_sha))
            save(out/f"case_{ident}.json", dict(predictions=actual[2], decisions=decisions))
        save(out/"parity.json", parity)
        reference = quality.choose(refs[0][FIXED8[0]], refs[1][FIXED8[0]], selector)[0]
        for _ in range(5):
            assert full_call(FIXED8[0], selector)[0][2] == reference
        times, peaks, reserved = [], [], []
        for _ in range(20):
            torch.cuda.reset_peak_memory_stats()
            actual, _, milliseconds = full_call(FIXED8[0], selector)
            assert actual[2] == reference
            times.append(milliseconds)
            peaks.append(torch.cuda.max_memory_allocated()/2**20)
            reserved.append(torch.cuda.max_memory_reserved()/2**20)
            progress(out, "timing", completed=len(times), expected=20, **calls)
    assert calls == dict(eval_calls=2725, train_calls=0)
    after = tensor_digest(model.state_dict())
    assert after == before and sha(out/"selector.json") == selector_sha
    assert sha(compact["weight"]) == tiles.SPEC["weight_sha"]
    timing = dict(warmup=5, repeats=20, milliseconds=times, mean_ms=statistics.mean(times),
        median_ms=statistics.median(times), peak_allocated_mib=max(peaks), peak_reserved_mib=max(reserved),
        scope="warm fixed image: read/decode, five-view preprocessing/forward/D2H/RLE, merge, quality gate, bbox and JSON; excludes output write/network",
        historical_reference_mean_ms=1938.3251087798271, historical_same_scope_not_paired=True)
    save(out/"timing.json", timing)
    save(out/"gpu_report.json", dict(status="passed", calls=calls, captured_train_images=512,
        train_manifest_sha256=sha(out/"train_manifest.json"), selector_sha256=selector_sha,
        checkpoint_sha256=tiles.SPEC["weight_sha"], model_state_before=before, model_state_after=after,
        m2f_weights_updated=False, selector_fitted_on_cpu=True, active_parameters=44400414,
        state_tensor_bytes=net, selector_parameters=105, parity_images=8, strict_parity=True,
        capture_seconds=capture_seconds, fit_seconds=fit_seconds, timing=timing,
        finished_unix=time.time(), artifacts={n: sha(out/n) for n in (
            "selector.json", "fit_pairs.json", "check_pairs.json", "fit_diagnostics.json",
            "train_input_hashes.json", "parity.json", "timing.json")}))


def evaluate(out):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    _, _, _, refine = modules()
    audit = refine.audit
    gpu = read(out/"gpu_report.json")
    assert gpu["status"] == "passed" and gpu["calls"]["train_calls"] == 0
    assert sha(out/"selector.json") == gpu["selector_sha256"]
    selector = read(out/"selector.json")
    frozen = read(out/"source_binding.json")
    geometry = read(ROOT/GEOM/"report.json")
    ids = geometry["image_ids"]
    assert len(ids) == len(set(ids)) == 2584
    paths = ((audit.BASE+"/predictions.segm.json"), (audit.TILES+"/evaluate/predictions.segm.json"))
    arms = []
    for p in paths:
        assert sha(ROOT/p) == frozen["full_val_cache_sha256"][p]
        by_image = {i: [] for i in ids}
        for row in read(ROOT/p):
            assert row["category_id"] == 1 and row["segmentation"]["size"] == [1024, 1024]
            assert math.isfinite(row["score"]) and 0 <= row["score"] <= 1
            by_image[row["image_id"]].append(row)
        assert all(len(rows) <= 100 for rows in by_image.values())
        arms.append(by_image)
    selected, decisions = {}, {}
    for n, ident in enumerate(ids):
        assert time.time() < EVAL_END
        selected[ident], decisions[ident] = quality.choose(arms[0][ident], arms[1][ident], selector)
        if n % 128 == 0: progress(out, "cache_select", completed=n+1, expected=2584)
    rows = [row for i in ids for row in selected[i]]
    save(out/"predictions.segm.json", rows)
    save(out/"decisions.json", decisions)
    prediction_sha = sha(out/"predictions.segm.json")
    # Predictions and selector are frozen before official val GT evaluation.
    save(out/"predictions_frozen.json", dict(prediction_sha256=prediction_sha,
        selector_sha256=sha(out/"selector.json"), images=2584, unix=time.time()))
    assert sha(ROOT/DATA/"annotations/instances_val.json") == VAL_SHA
    progress(out, "full_val")
    begun = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ROOT/DATA/"annotations/instances_val.json"))
        assert set(coco.getImgIds()) == set(ids)
        dt = coco.loadRes([{k: v for k, v in r.items() if k != "bbox"} for r in rows])
        ev = COCOeval(coco, dt, "segm"); ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    assert time.time() < EVAL_END
    assert ev.params.maxDets == [1, 10, 100] and np.allclose(ev.params.iouThrs, np.linspace(.5, .95, 10))
    metrics = dict(zip(refine.NAMES, map(float, ev.stats[:6])))
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in metrics.values())
    save(out/"full_val_metrics.json", dict(status="passed", metrics=metrics, images=2584,
        image_ids=ids, prediction_sha256=prediction_sha, selector_sha256=gpu["selector_sha256"],
        max_dets=[1,10,100], evaluation_seconds=time.time()-begun, finished_unix=time.time(),
        full_val_evaluations=1))


def analysis_manifest_ids(manifest, pairs):
    """Image membership comes from the frozen manifest, not filtered GT rows."""
    ids = manifest["image_ids"]
    assert len(ids) == len(set(ids)) == 256
    assert {p["image_id"] for p in pairs}.issubset(set(ids))
    assert len({p["annotation_id"] for p in pairs}) == len(pairs)
    return list(ids)


def analyze(out):
    import numpy as np
    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not Path("/dev/nvidiactl").exists()
    _, _, _, refine = modules()
    old_report = read(ROOT/GEOM/"report.json")
    old_pairs_path = ROOT/GEOM/"paired_gt.json"
    assert sha(old_pairs_path) == old_report["artifacts_sha256"]["paired_gt.json"]
    old_pairs = read(old_pairs_path)
    prior = {p["annotation_id"]: p for p in old_pairs}
    assert len(prior) == len(old_pairs) == 1187
    manifest_path = ROOT/"outputs/course_mrcnn_stages_20261018_v2/manifest.json"
    assert sha(manifest_path) == old_report["input_sha256"][str(manifest_path.relative_to(ROOT))]
    ids = analysis_manifest_ids(read(manifest_path), old_pairs)
    full = read(out/"full_val_metrics.json")
    assert full["status"] == "passed" and full["finished_unix"] < EVAL_END
    assert sha(out/"predictions.segm.json") == full["prediction_sha256"]
    subset = {i: [] for i in ids}
    for row in read(out/"predictions.segm.json"):
        if row["image_id"] in subset: subset[row["image_id"]].append(row)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ROOT/DATA/"annotations/instances_val.json"))
    paired = []
    for ident in ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[ident]))
        # Empty/no-small images remain part of the manifest, but have no pairs.
        if not any(a["area"] <= 1024 for a in anns):
            continue
        overlaps = mu.iou([r["segmentation"] for r in subset[ident]],
                          [coco.annToRLE(a) for a in anns], [0]*len(anns))
        for j, ann in enumerate(anns):
            if ann["area"] > 1024: continue
            old = prior[ann["id"]]
            new_iou = float(overlaps[:, j].max()) if len(subset[ident]) else 0.
            paired.append(dict(annotation_id=ann["id"], image_id=ident, area=ann["area"],
                global_iou=old["before_iou"], previous_iou=old["after_iou"], selected_iou=new_iou,
                previous_gain=bool(old["before_iou"] < .5 <= old["after_iou"]),
                previous_loss=bool(old["after_iou"] < .5 <= old["before_iou"])))
    assert len(paired) == 1187
    assert sum(p["previous_gain"] for p in paired) == 17
    assert sum(p["previous_loss"] for p in paired) == 9
    def summarize(values):
        return dict(count=len(values),
            gained=sum(p["previous_iou"] < .5 <= p["selected_iou"] for p in values),
            lost=sum(p["selected_iou"] < .5 <= p["previous_iou"] for p in values),
            coverage={str(t): dict(before=sum(p["previous_iou"] >= t for p in values),
                after=sum(p["selected_iou"] >= t for p in values)) for t in (.5,.75,.9)})
    groups = {name: summarize(values) for name, values in (
        ("all_small", paired), ("tiny", [p for p in paired if p["area"] <= 256]),
        ("small_257_1024", [p for p in paired if p["area"] > 256]),
        ("previous17_gained", [p for p in paired if p["previous_gain"]]),
        ("previous9_lost", [p for p in paired if p["previous_loss"]]))}
    metrics = full["metrics"]; base = old_report["metrics"]
    decisions = read(out/"decisions.json")
    gpu = read(out/"gpu_report.json")
    comparison = dict(status="passed", metrics=metrics, baseline_metrics=base,
        diagnostic_images=len(ids), images_with_small_gt=len({p["image_id"] for p in paired}),
        delta_points={k:100*(metrics[k]-base[k]) for k in metrics}, groups=groups,
        accepted_replacements=sum(d["accepted"] for rows in decisions.values() for d in rows),
        rejected_replacements=sum(not d["accepted"] for rows in decisions.values() for d in rows),
        positive_examples=sorted(paired,key=lambda p:p["selected_iou"]-p["previous_iou"],reverse=True)[:15],
        negative_examples=sorted(paired,key=lambda p:p["selected_iou"]-p["previous_iou"])[:15],
        timing=gpu["timing"], geometric_coverage_is_not_ap=True,
        single_run_development_val=True, any_positive_aps_retained=metrics["APs"]>base["APs"])
    save(out/"paired_gt.json", paired); save(out/"comparison.json", comparison)
    assert sha(out/"selector.json") == full["selector_sha256"] == gpu["selector_sha256"]
    files = ("cpu_acceptance.json", "train_manifest.json", "gpu_report.json", "selector.json",
             "full_val_metrics.json", "predictions_frozen.json", "predictions.segm.json",
             "comparison.json", "paired_gt.json", "decisions.json", "fit_diagnostics.json", "timing.json")
    save(out/"report.json", dict(status="passed", start_unix=START, deadline_unix=DEADLINE,
        finished_unix=time.time(), metrics=metrics, delta_points=comparison["delta_points"],
        selector_sha256=gpu["selector_sha256"], checkpoint_sha256=gpu["checkpoint_sha256"],
        train_images=512, fit_images=384, internal_check_images=128, complete_val_images=2584,
        full_val_evaluations=1, model_train_calls=0, m2f_weights_updated=False,
        selector_fitted=True, test_inputs_used=False, test_annotations_used=False,
        hyperparameter_search=False, paired_small=1187, diagnostic_images=len(ids),
        images_with_small_gt=len({p["image_id"] for p in paired}), calls=gpu["calls"],
        active_parameters=44400414, state_tensor_bytes=177814584, selector_parameters=105,
        capture_seconds=gpu["capture_seconds"], fit_seconds=gpu["fit_seconds"], timing=gpu["timing"],
        artifacts_sha256={n:sha(out/n) for n in files}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("cpu", "gpu", "evaluate", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    limit = dict(cpu=CPU_END, gpu=GPU_END, evaluate=EVAL_END, analyze=DEADLINE)[args.phase]
    if args.phase == "gpu":
        limit = min(limit, read(args.output/"phase_deadlines.json")["gpu_end"])
    def expired(*_):
        raise TimeoutError("Approved quality-selector phase expired; no extension")
    signal.signal(signal.SIGALRM, expired)
    if time.time() >= limit: expired()
    signal.alarm(max(1, math.ceil(limit-time.time())))
    try:
        globals()[args.phase](args.output)
    except BaseException as exc:
        save(args.output/("worker_first_error_"+args.phase+".json"),
             dict(error=repr(exc), traceback=traceback.format_exc(), unix=time.time()))
        raise

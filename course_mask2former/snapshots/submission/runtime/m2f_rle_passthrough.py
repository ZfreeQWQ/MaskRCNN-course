"""One bounded, exact-output RLE transport optimization. No training or AP."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import statistics
import time
import traceback

import m2f_geometry_compact_check as previous

START = 1792474893  # First preparation, 2026-10-20 13:41:33 Beijing.
CPU_END = START + 900
GATE_END = START + 1200
GPU_END = START + 2700
DEADLINE = START + 3600
previous.START, previous.DEADLINE = START, DEADLINE
read, save, sha = previous.read, previous.save, previous.sha
FIXED8 = previous.IDS


def direct_rows(kept, identity):
    """Exactly mimic as_instances -> serialize, without materializing masks."""
    import numpy as np
    from pycocotools import mask as mu
    assert len(kept) <= 100
    rows = []
    for candidate in kept:
        rle = copy.deepcopy(candidate["rle"])
        assert list(rle["size"]) == [1024, 1024]
        box = np.asarray(mu.toBbox(rle), dtype=np.float32)
        box[2:] += box[:2]  # Preserve the original float32 XYWH -> XYXY path.
        box = box.tolist()
        score = float(np.float32(candidate["score"]))
        assert all(math.isfinite(x) for x in box) and math.isfinite(score) and 0 <= score <= 1
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode()
        assert isinstance(rle["counts"], str)
        rows.append(dict(image_id=identity, bbox=[box[0], box[1], box[2]-box[0], box[3]-box[1]],
                         score=score, category_id=1, segmentation=rle))
    return rows


def cpu_tests(tiles):
    import numpy as np
    from pycocotools import mask as mu
    passed = []
    def verify(name, masks, scores):
        kept = [dict(rle=mu.encode(np.asfortranarray(mask)), score=score,
                     view=i % 5, index=i, area=int(mask.sum()), tile_edge=False)
                for i, (mask, score) in enumerate(zip(masks, scores))]
        before = copy.deepcopy(kept)
        slow = previous.serialize(tiles.as_instances(kept), 123)
        fast = direct_rows(kept, 123)
        assert slow == fast and before == kept, name
        # Explicit canonical compressed-RLE roundtrip, including empty masks.
        for row in fast:
            rec = mu.encode(np.asfortranarray(mu.decode(row["segmentation"])))
            rec["counts"] = rec["counts"].decode()
            assert rec == row["segmentation"]
        passed.append(name)
    empty = np.zeros((1024, 1024), dtype=np.uint8)
    one = empty.copy(); one[1023, 1023] = 1
    rect = empty.copy(); rect[12:47, 9:64] = 1
    hole = rect.copy(); hole[20:30, 20:30] = 0
    verify("empty_list", [], [])
    verify("empty_mask_zero_score", [empty], [0.0])
    verify("single_corner_pixel", [one], [1.0])
    verify("rectangle_bbox", [rect], [.7])
    verify("holes", [hole], [.3])
    verify("float32_score_rounding", [one, rect], [.1234567891234, .987654321123])
    verify("native_unsorted_and_duplicate", [hole, rect, hole, empty], [.2, .9, .2, 0])
    rng = np.random.default_rng(20261020)
    sparse = (rng.random((1024, 1024)) < .003).astype(np.uint8)
    verify("disconnected_mask", [sparse], [.85])
    verify("max100_instances", [rect, hole, one, empty]*25, [0., .2, .9, 1.]*25)
    try:
        direct_rows([dict(rle=mu.encode(np.asfortranarray(one)), score=float("nan"))], 1)
    except AssertionError:
        passed.append("reject_nonfinite")
    else:
        raise AssertionError("Nonfinite score accepted")
    return dict(status="passed", tests=len(passed), skipped=0, cases=passed)


def cpu(out):
    import os
    import unittest
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    from mmdet.utils import register_all_modules
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    tiles, _ = previous.modules()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName("test_m2f_rle_contract"))
    assert result.wasSuccessful() and result.testsRun == 8 and not result.skipped
    tests = cpu_tests(tiles)
    tests["entry_contract_tests"] = result.testsRun
    cfg, weight, pipeline, images = previous.bind(out)
    ref_report = read(Path("/workspace")/previous.OLD/"evaluate/report.json")
    subset = ref_report["subset_ids"]
    assert len(subset) == len(set(subset)) == 256
    ids = sorted(set(subset) | set(FIXED8))
    assert len(ids) == 264
    refs = []
    for parent in (previous.BASE, previous.OLD+"/evaluate", previous.GEOM):
        rows = read(Path("/workspace")/parent/"predictions.segm.json")
        selected = previous.select_rows(rows, ids)
        assert set(selected) == set(ids) and all(len(x) <= 100 for x in selected.values())
        refs.append(selected)
        del rows
    hashes = {x["image_id"]: x for x in read(Path("/workspace")/previous.BASE/"input_hashes.json")}
    requests = {i["id"]: tiles.safe_request(cfg.val_dataloader.dataset.data_root, i)
                for i in images if i["id"] in ids}
    assert set(requests) == set(ids)
    cfg.dump(str(out/"live_config.py"))
    compact = dict(image_ids=ids, subset_ids=subset, fixed_ids=FIXED8, references=refs,
        requests=requests, weight=str(weight), pipeline=pipeline,
        input_hashes={i: hashes[i] for i in ids})
    save(out/"compact.json", compact)
    # Test exactly the entry reconstruction used by the GPU phase, including types.
    from mmengine.config import Config
    reloaded = Config.fromfile(str(out/"live_config.py"))
    loaded = read(out/"compact.json")
    pipeline = previous.typed_pipeline(reloaded, loaded, tiles)
    assert reloaded.model.to_dict() == cfg.model.to_dict()
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    views = tiles.pack_views(requests[FIXED8[0]], Compose(pipeline[:1]), Compose(pipeline[1:]))
    assert len(views) == 5 and all(tuple(v["inputs"].shape) == (5,1536,1536) for v in views)
    tests["actual_config_preprocess_checks"] = 3
    save(out/"cpu_tests.json", tests)
    assert time.time() < CPU_END, "CPU preparation deadline missed"
    save(out/"compact_manifest.json", dict(status="passed", prepared_unix=time.time(),
        files={n: sha(out/n) for n in ("live_config.py", "compact.json", "binding.json", "cpu_tests.json")},
        image_ids=ids, no_model_loaded=True, no_gpu=True, no_ap_evaluation=True))


def verify_rows(actual, references, identity):
    stages = []
    for name, observed, reference in zip(("global", "merged", "refined"), actual, references):
        expected = reference[identity]
        stages.append(dict(name=name, exact=observed == expected, count=len(observed),
            expected_count=len(expected),
            mismatch_indices=[i for i,(a,b) in enumerate(zip(observed, expected)) if a != b]))
    return dict(image_id=identity, stages=stages)


def summarize_times(rows, allocated, reserved):
    return dict(warmup=5, repeats=20, image_id=FIXED8[0], all_calls=rows,
        mean_ms=statistics.mean(x["total_ms"] for x in rows),
        median_ms=statistics.median(x["total_ms"] for x in rows),
        min_ms=min(x["total_ms"] for x in rows), max_ms=max(x["total_ms"] for x in rows),
        peak_allocated_mib=allocated, peak_reserved_mib=reserved,
        scope="Full warm-cache fixed-image file read/decode, five-view preprocessing, five forwards, D2H/RLE/candidates/merge, output conversion, geometry refinement, JSON serialization; excludes result file writes and transport.")


def gpu(out):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    manifest = read(out/"compact_manifest.json")
    assert manifest["status"] == "passed" and manifest["prepared_unix"] < CPU_END
    assert all(sha(out/n) == h for n,h in manifest["files"].items())
    compact = read(out/"compact.json")
    ids = compact["image_ids"]
    assert len(ids) == len(set(ids)) == 264
    refs = [{int(k):v for k,v in x.items()} for x in compact["references"]]
    requests = {int(k):v for k,v in compact["requests"].items()}
    cfg = Config.fromfile(str(out/"live_config.py"))
    tiles, refine = previous.modules()
    pipeline = previous.typed_pipeline(cfg, compact, tiles)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    state = torch.load(compact["weight"], map_location="cpu")
    trained = Config.fromstring(state["meta"].get("cfg", state["meta"].get("config")), ".py")
    assert trained.model.to_dict() == cfg.model.to_dict()
    assert all(bool(torch.isfinite(t).all()) for t in state["state_dict"].values() if t.is_floating_point())
    model = MODELS.build(cfg.model)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    assert model.active_parameter_count() == 44400414
    net = sum(t.numel()*t.element_size() for t in model.state_dict().values())
    assert net == 177814584
    loader, tail = Compose(pipeline[:1]), Compose(pipeline[1:])
    counter = dict(eval_calls=0, train_calls=0)
    profiles = []
    model.eval()

    def full_call(identity, fast, profile=False):
        captured, pool = [], []
        torch.cuda.synchronize()
        begun = time.perf_counter()
        views = tiles.pack_views(requests[identity], loader, tail)
        batches = [pseudo_collate([v]) for v in views]
        prep_done = time.perf_counter()
        components = dict(forward_ms=0., d2h_ms=0., global_serialization_ms=0., candidates_ms=0.)
        prototype = None
        for view, batch in enumerate(batches):
            assert time.time() < GPU_END, "GPU phase cutoff"
            t = time.perf_counter()
            pred = model.test_step(batch)[0]
            if profile: torch.cuda.synchronize()
            t1 = time.perf_counter()
            pred = pred.cpu()
            t2 = time.perf_counter()
            counter["eval_calls"] += 1
            if view == 0:
                prototype = pred
                captured = previous.serialize(pred.pred_instances, identity)
            t3 = time.perf_counter()
            pool.extend(tiles.candidates(pred.pred_instances, view))
            t4 = time.perf_counter()
            if profile:
                components["forward_ms"] += (t1-t)*1000
                components["d2h_ms"] += (t2-t1)*1000
                components["global_serialization_ms"] += (t3-t2)*1000
                components["candidates_ms"] += (t4-t3)*1000
            if view: del pred
        t = time.perf_counter()
        kept = tiles.merge_candidates(pool)
        merged_at = time.perf_counter()
        if fast:
            merged = direct_rows(kept, identity)
        else:
            prototype.pred_instances = tiles.as_instances(kept)
            tiles.inspect_prediction(prototype.pred_instances, identity)
            merged = previous.serialize(prototype.pred_instances, identity)
        converted_at = time.perf_counter()
        final, changes = refine.refine(captured, merged)
        payload = json.dumps(final, separators=(",", ":"), allow_nan=False)
        torch.cuda.synchronize()
        ended = time.perf_counter()
        timing = dict(total_ms=(ended-begun)*1000, preprocess_ms=(prep_done-begun)*1000,
            five_view_candidates_ms=(t-prep_done)*1000, merge_ms=(merged_at-t)*1000,
            output_conversion_ms=(converted_at-merged_at)*1000,
            refinement_serialization_ms=(ended-converted_at)*1000,
            output_bytes=len(payload.encode()), replaced=len(changes))
        if profile:
            profiles.append(dict(image_id=identity, fast=fast, **components,
                preprocessing_ms=timing["preprocess_ms"], merge_ms=timing["merge_ms"],
                output_conversion_ms=timing["output_conversion_ms"],
                note="Instrumented synchronized stage profiling; excluded from benchmark samples."))
        return (captured, merged, final), timing

    parity, input_hashes = [], []
    with torch.inference_mode():
        for index, identity in enumerate(ids):
            r = requests[identity]
            inputs = dict(image_id=identity, rgb_sha256=sha(r["img_path"]), depth_sha256=sha(r["depth_file"]))
            assert inputs == compact["input_hashes"][str(identity)]
            input_hashes.append(inputs)
            actual, _ = full_call(identity, True, identity in FIXED8)
            case = verify_rows(actual, refs, identity)
            parity.append(case)
            save(out/"parity.json", parity)
            save(out/"input_hashes.json", input_hashes)
            assert all(x["exact"] for x in case["stages"]), "Strict output mismatch; stop, no tolerance relaxation"
            if identity in FIXED8:
                save(out/f"case_{identity}.json", actual[2])
            if index % 16 == 0 or index == len(ids)-1:
                save(out/"progress.json", dict(phase="parity", completed=len(parity), expected=264, **counter, unix=time.time()))
                print(json.dumps(dict(parity_completed=len(parity), **counter)), flush=True)
        # Paired alternating order avoids placing all measurements in different periods.
        for _ in range(5):
            for fast in (False, True):
                actual, _ = full_call(FIXED8[0], fast)
                assert actual[2] == refs[2][FIXED8[0]]
        times = {"reference": [], "passthrough": []}
        peaks = {k: [0., 0.] for k in times}
        for repeat in range(20):
            for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
                torch.cuda.reset_peak_memory_stats()
                actual, timing = full_call(FIXED8[0], fast)
                allocated = torch.cuda.max_memory_allocated()/2**20
                reserved = torch.cuda.max_memory_reserved()/2**20
                check = verify_rows(actual, refs, FIXED8[0])
                assert all(x["exact"] for x in check["stages"]), "Benchmark output mismatch"
                key = "passthrough" if fast else "reference"
                times[key].append(timing)
                peaks[key] = [max(peaks[key][0], allocated), max(peaks[key][1], reserved)]
                save(out/"timing_partial.json", dict(times=times, peaks=peaks))
        timing = {k: summarize_times(times[k], *peaks[k]) for k in times}
        save(out/"timing.json", timing)
        save(out/"profile.json", profiles)
    assert counter == dict(eval_calls=1570, train_calls=0), counter
    assert len(parity) == 264 and len(profiles) == 8
    report = dict(status="passed", image_ids=ids, fixed_ids=FIXED8, strict_parity=True,
        checked_images=len(parity), calls=counter, training_started=False, ap_reevaluated=False,
        active_parameters=44400414, state_tensor_bytes=net, timing=timing,
        latency_change_percent=100*(timing["passthrough"]["mean_ms"]/timing["reference"]["mean_ms"]-1),
        checkpoint_sha256=tiles.SPEC["weight_sha"], source_binding_sha256=sha(out/"binding.json"),
        compact_manifest_sha256=sha(out/"compact_manifest.json"), finished_unix=time.time(),
        artifacts={n:sha(out/n) for n in ("parity.json","input_hashes.json","timing.json","profile.json")})
    save(out/"report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "gpu"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    def timeout(*_):
        raise TimeoutError("Approved phase deadline expired; no extension")
    signal.signal(signal.SIGALRM, timeout)
    limit = CPU_END if args.mode == "cpu" else min(GPU_END, time.time()+1500)
    signal.alarm(max(1, math.ceil(limit-time.time())))
    try:
        (cpu if args.mode == "cpu" else gpu)(out)
    except BaseException as exc:
        save(out/"worker_first_error.json", dict(error=repr(exc), traceback=traceback.format_exc(), unix=time.time()))
        raise

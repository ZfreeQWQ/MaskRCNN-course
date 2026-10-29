"""Bounded eight-image live/cache equivalence; no training or AP evaluation."""
import argparse
import copy
import json
from pathlib import Path
import signal
import statistics
import time
import traceback

START = 1792472086  # 2026-10-20 12:54:46 Beijing; all preparation included.
DEADLINE = START + 1200
IDS = list(range(201960001, 201967002, 1000))
REMAINING = IDS[3:]
PRIOR = "outputs/course_m2f_geometry_gpu_20261020_v3"
OLD = "outputs/course_m2f_tiles_20261019_v1"
GEOM = "outputs/course_m2f_geometry_20261020_v1"
BASE = "outputs/course_m2f_copypaste1536_20261018_v1/evaluate_1536"


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def modules():
    import m2f_tiles as tiles
    import m2f_cached_geometry_refine as refine
    tiles.DEADLINE = refine.DEADLINE = refine.audit.DEADLINE = DEADLINE
    refine.START = refine.audit.START = START
    return tiles, refine


def serialize(pred, identity):
    """CocoMetric native order, including zero scores and empty masks."""
    import numpy as np
    from pycocotools import mask as mu
    rows = []
    for i in range(len(pred)):
        box = pred.bboxes[i].tolist()
        rle = mu.encode(np.asfortranarray(pred.masks[i].numpy().astype(np.uint8)))
        rle["counts"] = rle["counts"].decode()
        rows.append(dict(image_id=identity, bbox=[box[0], box[1], box[2]-box[0], box[3]-box[1]],
                         score=float(pred.scores[i]), category_id=int(pred.labels[i])+1,
                         segmentation=rle))
    return rows


def live_configuration(tiles):
    """One shared adapter for the CPU gate and GPU entry, before model load."""
    cfg, weight, pipeline, images, _manifest = tiles.preflight("/workspace")
    return cfg, weight, pipeline, images


def bind(out):
    tiles, refine = modules()
    root = Path("/workspace")
    bindings = {}
    def check(path, expected):
        actual = sha(root/path)
        assert actual == expected, path
        bindings[path] = actual
    plan = read(root/OLD/"plan.json")
    for name, digest in plan["source_sha256"].items():
        check(OLD+"/source/"+name, digest)
    for name, digest in plan["upstream_sha256"].items():
        check("mmdetection/"+name, digest)
    previous = read(root/GEOM/"report.json")
    check(GEOM+"/report.json", "7dbb3f1d4d624814e43c36c1464ea413095e9d26591d1d30853d0c49819878b6")
    check(GEOM+"/predictions.segm.json", "afce867761b2130bcefa987f8b8078cf3d4cc65353b9634566f3b06f3db96621")
    for name, digest in read(root/GEOM/"plan.json")["source_sha256"].items():
        check(GEOM+"/source/"+name, digest)
    # Bind the exact two parent caches used by the accepted CPU refinement.
    for path, digest in previous["input_sha256"].items():
        if path.endswith("predictions.segm.json") or path.endswith("input_hashes.json"):
            check(path, digest)
    cfg, weight, pipeline, images = live_configuration(tiles)
    save(out/"binding.json", dict(status="passed", files=bindings,
         checkpoint_sha256=tiles.SPEC["weight_sha"], ap_reevaluated=False))
    return cfg, weight, pipeline, images


def cpu(out):
    import torch
    import unittest
    from mmengine.structures import InstanceData
    tiles, refine = modules()
    tests = refine.tests()
    p = InstanceData(bboxes=torch.tensor([[1., 2., 3., 4.], [0., 0., 0., 0.]]),
                     masks=torch.zeros((2, 4, 4), dtype=torch.bool),
                     scores=torch.tensor([.1, 0.]), labels=torch.zeros(2, dtype=torch.long))
    p.masks[0, 2:4, 1:3] = True
    rows = serialize(p, 1)
    assert len(rows) == 2 and rows[0]["bbox"] == [1., 2., 2., 2.]
    assert rows[1]["score"] == 0 and rows[1]["segmentation"]["size"] == [4, 4]
    assert serialize(p, 1) == rows
    tests["tests"] += 3
    suite = unittest.defaultTestLoader.loadTestsFromName("test_m2f_geometry_compact")
    regression = unittest.TextTestRunner(verbosity=2).run(suite)
    assert regression.wasSuccessful() and not regression.skipped and regression.testsRun == 5
    tests["entry_regression_tests"] = regression.testsRun
    cfg, weight, pipeline, images = bind(out)
    root = Path("/workspace")
    assert sha(root/PRIOR/"parity.json") == "8d65beba668f693aa0a23954988ef4061979fd3bc4ebefffb756892c36184b7e"
    previous = read(root/PRIOR/"parity.json")
    assert [x["image_id"] for x in previous] == IDS[:3]
    assert all(s["exact"] for x in previous for s in x["stages"])
    # Compare executable prediction/serialization ASTs, not just result counts.
    import ast
    def algorithm(path):
        tree = ast.parse(Path(path).read_text())
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name in ("serialize", "full_call")}
    assert algorithm(__file__) == algorithm(root/PRIOR/"source/m2f_geometry_gpu_check.py")
    refs = []
    for parent in (BASE, OLD+"/evaluate", GEOM):
        rows = read(root/parent/"predictions.segm.json")
        refs.append(select_rows(rows, IDS))
        del rows
    for identity in IDS[:3]:
        assert read(root/PRIOR/f"{identity}.json") == refs[2][identity]
    hashes = {x["image_id"]: x for x in read(root/BASE/"input_hashes.json")}
    prior_hashes = {x["image_id"]: x for x in read(root/PRIOR/"input_hashes.json")}
    assert all(hashes[i] == prior_hashes[i] for i in IDS[:3])
    requests = {i["id"]: tiles.safe_request(cfg.val_dataloader.dataset.data_root, i)
                for i in images if i["id"] in IDS}
    assert set(requests) == set(IDS)
    cfg.dump(str(out/"live_config.py"))
    save(out/"compact.json", dict(references=refs, requests=requests, pipeline=pipeline,
         weight=str(weight), input_hashes={i: hashes[i] for i in IDS}, prior_parity=previous))
    save(out/"compact_manifest.json", dict(status="passed", files={
        name: sha(out/name) for name in ("live_config.py", "compact.json", "binding.json")},
        reused_parity_sha256=sha(root/PRIOR/"parity.json"),
        unchanged_prediction_ast=True, remaining_ids=REMAINING, prepared_unix=time.time()))
    assert time.time() < START+480, "Preparation gate expired"
    save(out/"cpu_tests.json", tests)


def select_rows(rows, identities):
    selected = {i: [] for i in identities}
    for row in rows:
        if row["image_id"] in selected:
            selected[row["image_id"]].append(row)
    return selected


def typed_pipeline(cfg, compact, tiles):
    """Recover the exact frozen pipeline with tuple types, no full preflight."""
    pipeline = tiles.scaled_pipeline(cfg.val_dataloader.dataset.pipeline, 1536)
    assert json.loads(json.dumps(pipeline)) == compact["pipeline"]
    return pipeline


def repair_check(out):
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    from mmdet.utils import register_all_modules
    tiles, _ = modules()
    manifest = read(out/"compact_manifest.json")
    assert all(sha(out/n) == h for n, h in manifest["files"].items())
    compact = read(out/"compact.json")
    cfg = Config.fromfile(str(out/"live_config.py"))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    pipeline = typed_pipeline(cfg, compact, tiles)
    resize = [x for x in pipeline if x["type"] == "RGBDResize"]
    assert len(resize) == 1 and isinstance(resize[0]["scale"], tuple)
    # Exercise the actual five-view preprocessing that failed; CPU only.
    views = tiles.pack_views(compact["requests"][str(REMAINING[0])],
                             Compose(pipeline[:1]), Compose(pipeline[1:]))
    assert len(views) == 5
    assert all(tuple(v["inputs"].shape) == (5, 1536, 1536) for v in views)
    save(out/"repair_acceptance.json", dict(status="passed", checks=4,
        actual_five_view_preprocessing=True, model_loaded=False, gpu_used=False,
        original_cpu_prepared_unix=manifest["prepared_unix"], unix=time.time()))


def run(out):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    tiles, refine = modules()
    manifest = read(out/"compact_manifest.json")
    assert manifest["status"] == "passed" and manifest["remaining_ids"] == REMAINING
    assert all(sha(out/name) == digest for name, digest in manifest["files"].items())
    compact = read(out/"compact.json")
    cfg = Config.fromfile(str(out/"live_config.py"))
    weight = compact["weight"]
    pipeline = typed_pipeline(cfg, compact, tiles)
    refs = [{int(k): v for k, v in index.items()} for index in compact["references"]]
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    state = torch.load(weight, map_location="cpu")
    trained = Config.fromstring(state["meta"].get("cfg", state["meta"].get("config")), ".py")
    assert trained.model.to_dict() == cfg.model.to_dict()
    assert all(bool(torch.isfinite(t).all()) for t in state["state_dict"].values() if t.is_floating_point())
    model = MODELS.build(cfg.model)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    loader, tail = Compose(pipeline[:1]), Compose(pipeline[1:])
    requests = {int(k): v for k, v in compact["requests"].items()}
    original_hashes = {int(k): v for k, v in compact["input_hashes"].items()}
    input_hashes = []
    for identity in IDS:
        r = requests[identity]
        h = dict(image_id=identity, rgb_sha256=sha(r["img_path"]), depth_sha256=sha(r["depth_file"]))
        assert h == original_hashes[identity]
        input_hashes.append(h)
    save(out/"input_hashes.json", input_hashes)
    counter = dict(eval_calls=0, train_calls=0)
    original_candidates = tiles.candidates

    def full_call(identity):
        captured = []
        def capture(pred, view):
            if view == 0:
                captured.extend(serialize(pred, identity))
            return original_candidates(pred, view)
        torch.cuda.synchronize()
        start = time.perf_counter()
        views = tiles.pack_views(requests[identity], loader, tail)
        batches = [pseudo_collate([v]) for v in views]
        prep = time.perf_counter()
        tiles.candidates = capture
        try:
            prediction, _ = tiles.multiview(model, batches, counter)
        finally:
            tiles.candidates = original_candidates
        merged = serialize(prediction.pred_instances, identity)
        middle = time.perf_counter()
        final, changes = refine.refine(captured, merged)
        payload = json.dumps(final, separators=(",", ":"), allow_nan=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return (captured, merged, final), dict(total_ms=(end-start)*1000,
            preprocess_ms=(prep-start)*1000, five_view_merge_ms=(middle-prep)*1000,
            refinement_serialization_ms=(end-middle)*1000, output_bytes=len(payload.encode()),
            replaced=len(changes))

    comparisons = []
    with torch.inference_mode():
        for identity in REMAINING:
            actual, _ = full_call(identity)
            result = dict(image_id=identity, stages=[])
            for name, observed, reference in zip(("global", "merged", "refined"), actual, refs):
                expected = reference[identity]
                exact = observed == expected
                mismatches = [j for j, (a,b) in enumerate(zip(observed, expected)) if a != b]
                result["stages"].append(dict(name=name, exact=exact, count=len(observed),
                    expected_count=len(expected), mismatch_indices=mismatches))
            comparisons.append(result)
            save(out/"parity.json", comparisons)
            save(out/f"{identity}.json", actual[2])
            assert all(x["exact"] for x in result["stages"]), "Strict live/cache mismatch; no timing or tolerance relaxation"
            print("parity", identity, "passed", flush=True)
        for _ in range(5):
            full_call(IDS[0])
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(20):
            actual, timing = full_call(IDS[0])
            assert actual[2] == refs[2][IDS[0]], "Timed output mismatch"
            times.append(timing)
        timing = dict(warmup=5, repeats=20, image_id=IDS[0], all_calls=times,
            mean_ms=statistics.mean(x["total_ms"] for x in times),
            median_ms=statistics.median(x["total_ms"] for x in times),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            scope="File read/decode, synchronized RGB-D five-view crop/resize/packing, 5 model forwards, D2H/RLE, old merge, geometry refinement, JSON serialization; excludes disk output/transport. Fixed warm-cache image, not cold-storage guarantee.")
        save(out/"timing.json", timing)
    assert counter == dict(eval_calls=150, train_calls=0), counter
    # Frozen MultiScaleRGBDMask2Former has no feature_activity attribute.
    activity = dict(counter)
    assert model.active_parameter_count() == 44400414
    net = sum(x.numel()*x.element_size() for x in model.state_dict().values())
    assert net == 177814584
    assert len(comparisons) == 5 and len(compact["prior_parity"]) == 3
    save(out/"report.json", dict(status="passed", image_ids=IDS, strict_parity=True,
        newly_checked_ids=REMAINING, reused_checked_ids=IDS[:3],
        reused_parity_sha256=manifest["reused_parity_sha256"],
        compact_manifest_sha256=sha(out/"compact_manifest.json"),
        timing=timing, calls=counter, wrapper_activity=activity, active_parameters=44400414,
        state_tensor_bytes=net, training_started=False, ap_reevaluated=False,
        checkpoint_sha256=tiles.SPEC["weight_sha"], source_binding_sha256=sha(out/"binding.json"),
        finished_unix=time.time()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cpu", "repair", "gpu"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    def timeout(*_):
        raise TimeoutError("Original phase/absolute deadline expired")
    signal.signal(signal.SIGALRM, timeout)
    limit = min(DEADLINE-240, START+480 if a.mode == "cpu" else time.time()+480)
    signal.alarm(max(1, int(limit-time.time())))
    try:
        (cpu if a.mode == "cpu" else repair_check if a.mode == "repair" else run)(out)
    except BaseException as exc:
        save(out/"worker_first_error.json", dict(error=repr(exc), traceback=traceback.format_exc(), unix=time.time()))
        raise

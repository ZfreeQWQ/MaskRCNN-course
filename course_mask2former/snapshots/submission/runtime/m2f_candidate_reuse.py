"""Single-call RLE reuse, exact output, bounded single-GPU check; no AP/train."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import m2f_rle_passthrough as base

previous = base.previous
read, save, sha = base.read, base.save, base.sha
START = 1792478551  # 2026-10-20 14:42:31 Beijing, includes all preparation.
CPU_END, GATE_END, GPU_END, DEADLINE = (START+x for x in (1200, 1500, 3000, 3600))
previous.START, previous.DEADLINE = START, DEADLINE
FIXED8 = base.FIXED8
PARENT = "outputs/course_m2f_rle_passthrough_20261020_v1"
PARENT_REPORT_SHA = "6fd44f94a433a9d07a13e16b024063158268f5884f48a52cf196d40631528a98"


class CandidateEncoder:
    """One F-contiguous scratch plane per full call; RLE owns encoded bytes."""
    def __init__(self, tiles):
        import numpy as np
        self.tiles = tiles
        self.buffer = np.zeros((1024, 1024), dtype=np.uint8, order="F")
        self.encoded = self.roundtrips = self.global_shared = 0

    def encode(self, pred, view, identity):
        import numpy as np
        import torch
        from pycocotools import mask as mu
        shape = (1024, 1024) if view == 0 else (640, 640)
        if view not in range(5) or len(pred) > 100 or tuple(pred.masks.shape) != (len(pred), *shape):
            raise ValueError("Invalid prediction/view geometry")
        if pred.masks.dtype != torch.bool or not bool((pred.labels == 0).all()):
            raise ValueError("Mask/label contract")
        if (not bool(torch.isfinite(pred.scores).all()) or
                not bool(torch.isfinite(pred.bboxes).all()) or
                not bool(((pred.scores >= 0) & (pred.scores <= 1)).all())):
            raise ValueError("Invalid scores/boxes")
        rows, pool = [], []
        for index, (mask, score) in enumerate(zip(pred.masks.numpy(), pred.scores.tolist())):
            area = int(np.count_nonzero(mask))
            # Global native output retains empty/zero-score rows. Local filtering
            # remains before mapping, as in the frozen candidate implementation.
            if view and (area == 0 or area > self.tiles.MAX_TILE_AREA):
                continue
            self.buffer.fill(0)
            edge = False
            if view:
                x, y = self.tiles.WINDOWS[view-1]
                edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
                self.buffer[y:y+640, x:x+640] = mask
            else:
                self.buffer[:, :] = mask
            rle = mu.encode(self.buffer)
            self.encoded += 1
            # Retain per-candidate pixel-exact roundtrip; uint8 equality avoids
            # two redundant full-image bool copies, not the correctness check.
            decoded = mu.decode(rle)
            if decoded.dtype != np.uint8 or not np.array_equal(decoded, self.buffer):
                raise ValueError("RLE original-grid roundtrip")
            self.roundtrips += 1
            if view == 0:
                box = pred.bboxes[index].tolist()
                rows.append(dict(image_id=identity,
                    bbox=[box[0], box[1], box[2]-box[0], box[3]-box[1]],
                    score=float(score), category_id=int(pred.labels[index])+1,
                    segmentation=dict(size=list(rle["size"]), counts=rle["counts"].decode())))
            if area:
                pool.append(dict(rle=rle, score=float(score), view=view,
                    index=index, area=area, tile_edge=edge))
                if view == 0:
                    self.global_shared += 1
        assert self.encoded == self.roundtrips
        return rows, pool


def cpu_tests(tiles):
    import numpy as np
    import torch
    from mmengine.structures import InstanceData
    cases = []

    def prediction(masks, shape, scores=None):
        tensor = torch.from_numpy(np.stack(masks).astype(bool)) if masks else torch.zeros((0, *shape), dtype=torch.bool)
        return InstanceData(masks=tensor,
            bboxes=torch.tensor([[.25, .5, 42.75, 63.25]]*len(masks), dtype=torch.float32).reshape(-1, 4),
            scores=torch.tensor(scores if scores is not None else [.123456789]*len(masks), dtype=torch.float32),
            labels=torch.zeros(len(masks), dtype=torch.long))

    encoder = CandidateEncoder(tiles)
    buffer_id = id(encoder.buffer)
    saved = []

    def check(name, view, masks, scores=None):
        shape = (1024, 1024) if view == 0 else (640, 640)
        pred = prediction(masks, shape, scores)
        before = pred.masks.clone()
        rows, pool = encoder.encode(pred, view, 123)
        assert pool == tiles.candidates(pred, view), name
        assert rows == (previous.serialize(pred, 123) if view == 0 else []), name
        assert torch.equal(pred.masks, before) and id(encoder.buffer) == buffer_id
        saved.append((pool, copy.deepcopy(pool), rows, copy.deepcopy(rows)))
        assert all(p == q and a == b for p, q, a, b in saved), "Buffer aliases old RLE"
        cases.append(name)

    full = np.zeros((1024, 1024), dtype=bool)
    one = full.copy(); one[-1, -1] = True
    hole = full.copy(); hole[10:50, 20:60] = True; hole[20:30, 30:40] = False
    check("empty_global_list", 0, [])
    check("empty_zero_score_and_corner", 0, [full, one], [0., 1.])
    check("global_holes_duplicate_native_order", 0, [hole, one, hole], [.9, .2, .9])
    check("global100", 0, [one, full, hole, one]*25, [0., .2, .3, 1.]*25)
    for view in range(1, 5):
        empty = np.zeros((640, 640), dtype=bool)
        corners = empty.copy(); corners[0, 0] = corners[-1, -1] = True
        threshold = empty.copy(); threshold[10:42, 10:42] = True
        oversized = threshold.copy(); oversized[100, 100] = True
        inner = empty.copy(); inner[123:130, 201:208] = True; inner[125, 203] = False
        check(f"window{view}_boundary_1024_1025_empty", view,
            [empty, corners, threshold, oversized, inner], [0., .8, .9, .95, .2])
        check(f"window{view}_empty_list", view, [])
    # Reuse after all windows must not leave pixels in a new global empty mask.
    check("clear_between_views", 0, [full, one], [0., .5])
    for name in ("nonfinite_score", "nonfinite_box", "wrong_label", "wrong_mask_dtype", "wrong_grid"):
        pred = prediction([one], (1024, 1024))
        if name == "nonfinite_score": pred.scores[0] = float("nan")
        elif name == "nonfinite_box": pred.bboxes[0, 0] = float("nan")
        elif name == "wrong_label": pred.labels[0] = 1
        elif name == "wrong_mask_dtype": pred.masks = pred.masks.float()
        else: pred.masks = pred.masks[:, :640, :640]
        try:
            encoder.encode(pred, 0, 123)
        except ValueError:
            cases.append("reject_"+name)
        else:
            raise AssertionError("Invalid input accepted: "+name)
    return dict(status="passed", tests=len(cases), skipped=0, cases=cases,
        one_scratch_buffer=True, encoded=encoder.encoded, roundtrips=encoder.roundtrips,
        global_shared=encoder.global_shared)


def cpu(out):
    import unittest
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.utils import import_modules_from_strings
    from mmdet.utils import register_all_modules
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    parent = Path("/workspace")/PARENT
    assert sha(parent/"report.json") == PARENT_REPORT_SHA
    accepted = read(parent/"terminal_acceptance.json")
    assert accepted["status"] == "passed" and accepted["report_sha256"] == PARENT_REPORT_SHA
    assert read(parent/"status.json")["status"] == "complete"
    plan = read(parent/"plan.json")
    assert all(sha(parent/"source"/n) == h for n, h in plan["source_sha256"].items())
    manifest = read(parent/"compact_manifest.json")
    assert all(sha(parent/n) == h for n, h in manifest["files"].items())
    # Reuse exact accepted compact references; do not parse full val caches again.
    for name in ("compact.json", "live_config.py", "binding.json"):
        shutil.copyfile(parent/name, out/name)
    binding = read(out/"binding.json")
    assert all(sha(Path("/workspace")/n) == h for n, h in binding["files"].items())
    compact = read(out/"compact.json")
    assert sha(compact["weight"]) == binding["checkpoint_sha256"]
    assert len(compact["image_ids"]) == len(set(compact["image_ids"])) == 264
    tiles, _ = previous.modules()
    tests = cpu_tests(tiles)
    suite = unittest.defaultTestLoader.loadTestsFromName("test_m2f_candidate_reuse_contract")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == 7 and not result.skipped
    tests["entry_contract_tests"] = result.testsRun
    cfg = Config.fromfile(str(out/"live_config.py"))
    pipeline = previous.typed_pipeline(cfg, compact, tiles)
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    views = tiles.pack_views(compact["requests"][str(FIXED8[0])],
        Compose(pipeline[:1]), Compose(pipeline[1:]))
    assert len(views) == 5 and all(tuple(v["inputs"].shape) == (5, 1536, 1536) for v in views)
    tests["actual_preprocess_passed"] = True
    save(out/"cpu_tests.json", tests)
    save(out/"parent_binding.json", dict(parent=PARENT, report_sha256=PARENT_REPORT_SHA,
        parent_source_commit=plan["source_commit"], weight_sha256=binding["checkpoint_sha256"]))
    assert time.time() < CPU_END
    save(out/"compact_manifest.json", dict(status="passed", prepared_unix=time.time(),
        files={n:sha(out/n) for n in ("compact.json", "live_config.py", "binding.json",
            "cpu_tests.json", "parent_binding.json")}, no_gpu=True, no_ap=True, no_model=True))


def gpu(out):
    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    manifest = read(out/"compact_manifest.json")
    assert manifest["status"] == "passed" and manifest["prepared_unix"] < CPU_END
    assert all(sha(out/n) == h for n, h in manifest["files"].items())
    compact = read(out/"compact.json")
    ids = compact["image_ids"]
    refs = [{int(k):v for k, v in x.items()} for x in compact["references"]]
    requests = {int(k):v for k, v in compact["requests"].items()}
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
    encoding = dict(encoded=0, roundtrips=0, global_shared=0, fast_calls=0)

    def full_call(identity, fast):
        captured, pool = [], []
        torch.cuda.synchronize()
        begun = time.perf_counter()
        views = tiles.pack_views(requests[identity], loader, tail)
        batches = [pseudo_collate([v]) for v in views]
        encoder = CandidateEncoder(tiles) if fast else None
        prep_done = time.perf_counter()
        for view, batch in enumerate(batches):
            assert time.time() < GPU_END
            pred = model.test_step(batch)[0].cpu()
            counter["eval_calls"] += 1
            if fast:
                rows, candidates = encoder.encode(pred.pred_instances, view, identity)
                if view == 0:
                    captured = rows
            else:
                if view == 0:
                    captured = previous.serialize(pred.pred_instances, identity)
                candidates = tiles.candidates(pred.pred_instances, view)
            pool.extend(candidates)
            del pred
        candidates_done = time.perf_counter()
        kept = tiles.merge_candidates(pool)
        merged_at = time.perf_counter()
        merged = base.direct_rows(kept, identity)
        converted_at = time.perf_counter()
        final, changes = refine.refine(captured, merged)
        payload = json.dumps(final, separators=(",", ":"), allow_nan=False)
        torch.cuda.synchronize()
        ended = time.perf_counter()
        if fast:
            assert encoder.encoded == encoder.roundtrips
            for key in ("encoded", "roundtrips", "global_shared"):
                encoding[key] += getattr(encoder, key)
            encoding["fast_calls"] += 1
        timing = dict(total_ms=(ended-begun)*1000, preprocess_ms=(prep_done-begun)*1000,
            five_view_candidates_ms=(candidates_done-prep_done)*1000,
            merge_ms=(merged_at-candidates_done)*1000,
            output_conversion_ms=(converted_at-merged_at)*1000,
            refinement_serialization_ms=(ended-converted_at)*1000,
            output_bytes=len(payload.encode()), replaced=len(changes))
        return (captured, merged, final), timing

    parity, input_hashes = [], []
    with torch.inference_mode():
        for index, identity in enumerate(ids):
            request = requests[identity]
            inputs = dict(image_id=identity, rgb_sha256=sha(request["img_path"]),
                depth_sha256=sha(request["depth_file"]))
            assert inputs == compact["input_hashes"][str(identity)]
            input_hashes.append(inputs)
            actual, _ = full_call(identity, True)
            case = base.verify_rows(actual, refs, identity)
            parity.append(case)
            save(out/"parity.json", parity)
            save(out/"input_hashes.json", input_hashes)
            assert all(x["exact"] for x in case["stages"]), "Strict mismatch; no tolerance relaxation"
            if identity in FIXED8:
                save(out/f"case_{identity}.json", actual[2])
            if index % 16 == 0 or index == len(ids)-1:
                progress = dict(phase="parity", completed=len(parity), expected=264,
                    **counter, unix=time.time())
                save(out/"progress.json", progress)
                print(json.dumps(progress), flush=True)
        for _ in range(5):
            for fast in (False, True):
                actual, _ = full_call(FIXED8[0], fast)
                assert all(x["exact"] for x in base.verify_rows(actual, refs, FIXED8[0])["stages"])
        times = {"reference": [], "passthrough": []}
        peaks = {k:[0., 0.] for k in times}
        for repeat in range(20):
            for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
                torch.cuda.reset_peak_memory_stats()
                actual, measured = full_call(FIXED8[0], fast)
                allocated = torch.cuda.max_memory_allocated()/2**20
                reserved = torch.cuda.max_memory_reserved()/2**20
                assert all(x["exact"] for x in base.verify_rows(actual, refs, FIXED8[0])["stages"])
                key = "passthrough" if fast else "reference"
                times[key].append(measured)
                peaks[key] = [max(peaks[key][0], allocated), max(peaks[key][1], reserved)]
                save(out/"timing_partial.json", dict(times=times, peaks=peaks))
        timing = {k:base.summarize_times(times[k], *peaks[k]) for k in times}
        save(out/"timing.json", timing)
    assert counter == dict(eval_calls=1570, train_calls=0)
    assert encoding["fast_calls"] == 289 and encoding["encoded"] == encoding["roundtrips"]
    assert len(parity) == len(set(ids)) == 264
    save(out/"encoding.json", encoding)
    save(out/"report.json", dict(status="passed", strict_parity=True, checked_images=264,
        image_ids=ids, fixed_ids=FIXED8, calls=counter, encoding=encoding,
        arm_names=dict(reference="accepted_RLE_passthrough", passthrough="candidate_RLE_reuse"),
        training_started=False, ap_reevaluated=False, active_parameters=44400414,
        state_tensor_bytes=net, timing=timing,
        latency_change_percent=100*(timing["passthrough"]["mean_ms"]/timing["reference"]["mean_ms"]-1),
        checkpoint_sha256=tiles.SPEC["weight_sha"],
        parent_report_sha256=PARENT_REPORT_SHA, finished_unix=time.time(),
        artifacts={n:sha(out/n) for n in ("parity.json", "input_hashes.json", "timing.json", "encoding.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "gpu"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    def timeout(*_):
        raise TimeoutError("Approved phase expired; no extension")
    signal.signal(signal.SIGALRM, timeout)
    limit = CPU_END if args.mode == "cpu" else min(GPU_END, time.time()+1500)
    signal.alarm(max(1, math.ceil(limit-time.time())))
    try:
        (cpu if args.mode == "cpu" else gpu)(out)
    except BaseException as exc:
        save(out/"worker_first_error.json", dict(error=repr(exc),
            traceback=traceback.format_exc(), unix=time.time()))
        raise

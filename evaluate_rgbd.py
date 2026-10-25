"""COCO mask AP/APS on supplied validation only. No test images or labels."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from course.common import atomic_json, configuration
from course.data import RGBDMapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/blender_rgbd_r50.yaml")
    parser.add_argument("--cache", default="data_cache")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--strict-load", action="store_true")
    args = parser.parse_args()
    # All V3 evaluators share a physical-GPU lock. A queued periodic evaluator
    # waits before model allocation, so scale evaluation and training monitors
    # cannot contaminate each other's latency measurements or exhaust memory.
    import fcntl
    import hashlib
    gpu_identity = os.environ.get("CUDA_VISIBLE_DEVICES", "default")
    lock_name = "evaluation_gpu_" + hashlib.sha256(gpu_identity.encode()).hexdigest()[:12] + ".lock"
    gpu_lock = (Path(args.cache).resolve() / lock_name).open("a")
    print("EVAL_GPU_LOCK_WAIT", gpu_identity, flush=True)
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    print("EVAL_GPU_LOCK_ACQUIRED", gpu_identity, flush=True)
    torch.set_num_threads(4)
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    cfg = configuration(args.config)
    model = build_model(cfg).eval()
    if args.strict_load:
        state = torch.load(args.weights, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(state, strict=True)
        del state
    else:
        DetectionCheckpointer(model).load(args.weights)
    from course.experiments import mapper as build_mapper
    mapper = build_mapper(cfg, training=False, cache=args.cache)
    records = json.loads((Path(args.cache) / "index.json").read_text())["val"]
    if args.samples:
        assert args.samples > 0
        indices = np.unique(np.linspace(0, len(records) - 1, min(args.samples, len(records)), dtype=int))
        records = [records[int(i)] for i in indices]
    ids = [record["image_id"] for record in records]
    assert len(ids) == len(set(ids))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(Path(args.cache) / "coco_val.json"))
    detections, timings, preprocess_timings = [], [], []
    profiles = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        for n, record in enumerate(records):
            preprocessing_started = time.perf_counter()
            image = mapper(record)
            assert tuple(image["image"].shape[-2:]) == (cfg.COURSE.TEST_SIZE, cfg.COURSE.TEST_SIZE), "Reported test size must match actual input"
            preprocess_timings.append(time.perf_counter() - preprocessing_started)
            if "preprocess_profile" in image:
                profiles.append(image["preprocess_profile"])
            torch.cuda.synchronize()
            tick = time.perf_counter()
            instances = model([image])[0]["instances"]
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - tick)
            instances = instances.to("cpu")
            assert len(instances) <= 100
            assert instances.pred_masks.shape[-2:] == (1024, 1024)
            for j in range(len(instances)):
                assert int(instances.pred_classes[j]) == 0
                rle = mask_utils.encode(np.asfortranarray(instances.pred_masks[j].numpy().astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                detections.append(dict(image_id=record["image_id"], category_id=1,
                                       score=float(instances.scores[j]), segmentation=rle))
            if (n + 1) % 50 == 0:
                print("EVAL_IMAGE", n + 1, "/", len(records), "PRED", len(detections), flush=True)
    result = dict(architecture="Mask2Former-ResNet50-RGBD", evaluated_subset="val",
                  experiment=cfg.COURSE.NAME, input_size=cfg.COURSE.TEST_SIZE,
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  model_parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                  checkpoint=str(Path(args.weights).resolve()), validation_images=len(records),
                  predictions=len(detections), max_instances_per_image=100,
                  mask_size=[1024, 1024], score_scale="0-1", mask_aps_primary=True,
                  peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                  inference_seconds_per_image=float(np.mean(timings[5:] or timings)),
                  input_preprocess_seconds_per_image=float(np.mean(preprocess_timings[5:] or preprocess_timings)),
                  model_plus_input_seconds_per_image=float(np.mean((np.asarray(timings) + np.asarray(preprocess_timings))[5:]
                                                                  if len(timings) > 5 else
                                                                  np.asarray(timings) + np.asarray(preprocess_timings))),
                  timing_excludes="disk reads, RLE encoding, COCO aggregation; first 5 images warmed up",
                  checkpoint_bytes=os.path.getsize(args.weights))
    if profiles:
        result["input_profile_seconds"] = {k: float(np.mean([p[k] for p in profiles[5:] or profiles])) for k in profiles[0]}
    with contextlib.redirect_stdout(io.StringIO()):
        if detections:
            prediction = coco.loadRes(detections)
            evaluator = COCOeval(coco, prediction, "segm")
            evaluator.params.imgIds = ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.stats
            from course.diagnostics_v3 import coco_small_metrics
            result.update(coco_small_metrics(evaluator))
        else:
            stats = np.zeros(12)
    result.update(mask_ap=float(stats[0]), mask_ap50=float(stats[1]), mask_ap75=float(stats[2]),
                  mask_aps=float(stats[3]), mask_apm=float(stats[4]), mask_apl=float(stats[5]),
                  total_elapsed_seconds=time.monotonic() - started)
    if args.diagnostics:
        from course.diagnostics_v3 import diagnose_small
        result["small_diagnostics"] = diagnose_small(coco, detections, ids)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print("METRICS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

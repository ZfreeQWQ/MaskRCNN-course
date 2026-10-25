"""Paired alternating-order CPU input timing and exact GPU score/mask audit."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from detectron2.modeling import build_model
from course.common import atomic_json, configuration
from course.experiments import mapper


def main():
    torch.set_num_threads(4)
    records = json.loads(Path("data_cache/index.json").read_text())["val"]
    old_cfg = configuration("configs/v3/g01_scale1280.yaml")
    new_cfg = configuration("configs/v4/h00_fast1280.yaml")
    old, new = mapper(old_cfg, False), mapper(new_cfg, False)
    values = {"v3": [], "v4": []}
    profile = {"v3": [], "v4": []}
    # First two cases warm file/index caches, separately recorded, not hidden.
    indices = np.unique(np.linspace(0, len(records)-1, 26, dtype=int))
    warmup = []
    for n, i in enumerate(indices):
        outputs = {}
        for name, obj in (("v3", old), ("v4", new)) if n % 2 == 0 else (("v4", new), ("v3", old)):
            tick = time.perf_counter()
            mapped = obj(records[int(i)])
            elapsed = time.perf_counter()-tick
            outputs[name] = mapped["image"]
            if n >= 2:
                values[name].append(elapsed)
                profile[name].append(mapped["preprocess_profile"])
            else:
                warmup.append(dict(image_id=records[int(i)]["image_id"], mapper=name, seconds=elapsed))
        torch.testing.assert_close(outputs["v3"], outputs["v4"], rtol=0, atol=0)
        print("PAIRED_INPUT", n+1, len(indices), flush=True)
    measurements = {name: dict(mean_seconds=float(np.mean(samples)), median_seconds=float(np.median(samples)),
                              component_mean={k: float(np.mean([p[k] for p in profile[name]])) for k in profile[name][0]})
                    for name, samples in values.items()}
    checks = []
    with torch.inference_mode():
        for config, weights in (("configs/v3/g01_scale1280.yaml", "runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth"),
                                 ("configs/v4/h02_detail1280.yaml", "runs/optimization_v3/g02_geometry_detail/model_final_inference.pth")):
            cfg = configuration(config)
            model = build_model(cfg).eval()
            state = torch.load(Path("/workspace/Mask2Former-course")/weights, map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(state, strict=True)
            for i in (0, len(records)//2):
                a = model([old(records[i])])[0]["instances"]
                b = model([new(records[i])])[0]["instances"]
                torch.testing.assert_close(a.pred_masks, b.pred_masks, rtol=0, atol=0)
                torch.testing.assert_close(a.scores, b.scores, rtol=0, atol=0)
                checks.append(dict(config=config, image_id=records[i]["image_id"], exact_masks=True, exact_scores=True))
    result = dict(status="passed", input_images_checked=len(indices), timing_pairs=len(values["v3"]),
                  protocol="same24 val images, alternating AB/BA, warm-cache input-only; no network/RLE in timings",
                  measurements=measurements, first_two_warmup=warmup, gpu_parity=checks,
                  relative_mean_input_time_reduction=1-measurements["v4"]["mean_seconds"]/measurements["v3"]["mean_seconds"],
                  test_used=False, weights_changed=False)
    atomic_json("/workspace/Mask2Former-course/runs/optimization_v4/input_benchmark.json", result)
    print("INPUT_BENCHMARK", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

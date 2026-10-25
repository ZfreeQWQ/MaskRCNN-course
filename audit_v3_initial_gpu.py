"""Two real val images: zero-detail output must equal saved best V2 exactly."""
import json
from pathlib import Path
import torch
from detectron2.modeling import build_model
from course.common import atomic_json, configuration
from course.experiments import mapper


def main():
    torch.set_num_threads(4)
    baseline_cfg = configuration("configs/v3/g00_fast_control.yaml")
    detail_cfg = configuration("configs/v3/g02_geometry_detail.yaml")
    state = torch.load("/workspace/Mask2Former-course/runs/optimization_v2/d01_geometry_multiscale/model_final_inference.pth",
                       map_location="cpu", weights_only=False)["model"]
    baseline, detail = build_model(baseline_cfg).eval(), build_model(detail_cfg).eval()
    baseline.load_state_dict(state, strict=True)
    incompatible = detail.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys and all(k.startswith("sem_seg_head.refiner.") for k in incompatible.missing_keys)
    records = json.loads(Path("data_cache/index.json").read_text())["val"]
    data = mapper(baseline_cfg, False)
    ids = []
    with torch.inference_mode():
        for record in (records[0], records[len(records)//2]):
            image = data(record)
            old = baseline([image])[0]["instances"]
            new = detail([image])[0]["instances"]
            torch.testing.assert_close(old.pred_masks, new.pred_masks, rtol=0, atol=0)
            torch.testing.assert_close(old.scores, new.scores, rtol=0, atol=0)
            torch.testing.assert_close(old.pred_classes, new.pred_classes, rtol=0, atol=0)
            ids.append(record["image_id"])
    result = dict(status="passed", validation_image_ids=ids, exact_masks=True, exact_scores=True,
                  missing_keys_only_new_refiner=True, new_parameters=sum(p.numel() for p in detail.sem_seg_head.refiner.parameters()),
                  model_parameters=sum(p.numel() for p in detail.parameters()),
                  note="Execution/initialization parity only, not accuracy evaluation")
    destination = Path("/workspace/Mask2Former-course/runs/optimization_v3/initial_gpu_parity.json")
    atomic_json(destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

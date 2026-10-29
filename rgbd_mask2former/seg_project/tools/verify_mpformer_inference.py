#!/usr/bin/env python
"""Real-image initial-weight parity with the frozen selected D inference."""
import json
from pathlib import Path
import train_m2f
import torch
from detectron2.data import DatasetCatalog
from rgbd_seg.data import register_datasets
from rgbd_seg.m2f.config import setup_cfg
from rgbd_seg.m2f.inference import Predictor
from rgbd_seg.m2f.round2_inference import paired_prediction
import rgbd_seg.m2f.mpformer

ROOT = Path(__file__).resolve().parents[1]


def main():
    mp_cfg = setup_cfg(ROOT / "configs/mask2former/round2_mpformer_20261026.yaml")
    base_cfg = setup_cfg(ROOT / "configs/mask2former/round2_d_control_20261026.yaml")
    register_datasets(mp_cfg)
    baseline = Predictor(base_cfg, base_cfg.MODEL.WEIGHTS)
    adapted = Predictor(mp_cfg, mp_cfg.MODEL.WEIGHTS)
    shared = baseline.model.state_dict()
    new = adapted.model.state_dict()
    for name in shared:
        if not torch.equal(shared[name], new[name]):
            raise AssertionError("Pretrained parameter mismatch: " + name)
    additions = sorted(set(new) - set(shared))
    if additions != ["sem_seg_head.predictor.label_enc.weight"]:
        raise AssertionError("Unexpected extra model parameters: " + repr(additions))
    checks = []
    records = sorted(DatasetCatalog.get(mp_cfg.DATASETS.TEST[0]), key=lambda r: r["image_id"])[:3]
    for record in records:
        mapped = baseline.mapper(record)
        old = paired_prediction(baseline.model, mapped)
        current = paired_prediction(adapted.model, mapped)
        if not torch.equal(old["instances"].pred_masks, current["instances"].pred_masks):
            raise AssertionError("MP eval differs from established D masks")
        torch.testing.assert_close(old["instances"].scores, current["instances"].scores, rtol=1e-6, atol=1e-7)
        if adapted.model.sem_seg_head.predictor.last_pilot_stats["pilot_queries"] != 0:
            raise AssertionError("Pilot queries were active at inference")
        checks.append({"image_id": record["image_id"], "instances": len(current["instances"]),
                       "original_geometry": list(current["instances"].image_size), "exact_masks": True,
                       "matching_scores": True, "inference_pilot_queries": 0})
    report = {"valid": True, "shared_pretrained_tensors": len(shared), "new_parameters": additions,
              "checks": checks}
    folder = ROOT / "experiments/mask2former/round2_mpformer_checks_20261026"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "initial_inference_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

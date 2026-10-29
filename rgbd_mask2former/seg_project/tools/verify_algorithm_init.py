#!/usr/bin/env python
"""Real-image and shared-tensor audit before an automatic algorithm job."""
import argparse
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
import rgbd_seg.m2f.frequency_fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    cfg, ref = setup_cfg(args.config), setup_cfg(args.reference_config)
    register_datasets(cfg)
    baseline, adapted = Predictor(ref, ref.MODEL.WEIGHTS), Predictor(cfg, cfg.MODEL.WEIGHTS)
    shared, new = baseline.model.state_dict(), adapted.model.state_dict()
    for name in shared:
        if name not in new or not torch.equal(shared[name], new[name]):
            raise AssertionError("Shared checkpoint tensor differs: " + name)
    additions = sorted(set(new) - set(shared))
    if any(not n.startswith("sem_seg_head.pixel_decoder.frequency_fusion.") for n in additions):
        raise AssertionError("Unexpected newly initialized tensors: " + repr(additions))
    if additions and float(adapted.model.sem_seg_head.pixel_decoder.frequency_fusion.alpha) != 0:
        raise AssertionError("New fusion did not start at zero residual")
    checks = []
    for record in sorted(DatasetCatalog.get(cfg.DATASETS.TEST[0]), key=lambda r: r["image_id"])[:3]:
        mapped = baseline.mapper(record)
        old, current = paired_prediction(baseline.model, mapped), paired_prediction(adapted.model, mapped)
        if not torch.equal(old["instances"].pred_masks, current["instances"].pred_masks):
            raise AssertionError("Initial masks changed")
        torch.testing.assert_close(old["instances"].scores, current["instances"].scores, rtol=1e-6, atol=1e-7)
        stats = getattr(adapted.model.sem_seg_head.predictor, "last_pilot_stats", None)
        if stats and stats.get("pilot_queries", 0):
            raise AssertionError("Pilot queries active at inference")
        checks.append({"image_id": record["image_id"], "exact_masks": True, "matching_scores": True,
                       "instances": len(current["instances"])})
    result = {"valid": True, "shared_tensors": len(shared), "new_tensors": additions, "checks": checks}
    Path(args.report).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

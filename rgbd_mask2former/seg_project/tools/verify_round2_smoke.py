#!/usr/bin/env python
"""Check trained H weights, original inference parity and the real optimizer groups."""
import json
import sys
from pathlib import Path

import train_m2f  # established import paths and model registrations
import torch
from detectron2.data import DatasetCatalog
from rgbd_seg.data import register_datasets
from rgbd_seg.m2f.config import setup_cfg
from rgbd_seg.m2f.experimental_inference import ExperimentalPredictor
from rgbd_seg.m2f.round2_inference import paired_prediction
from rgbd_seg.m2f.round2_trainer import Round2Trainer


def main():
    folder = Path(sys.argv[1])
    cfg = setup_cfg(folder / "config.yaml")
    if cfg.PROJECT.SMALL_OBJECT_POINTS.ENABLED or not cfg.PROJECT.HIRES_MASK.ENABLED:
        raise AssertionError("Expected D+H with S off")
    predictor = ExperimentalPredictor(cfg, str(folder / "model_final.pth"), refine_small=True, chunk_size=4)
    model = predictor.model
    if model.criterion.loss_point_fraction != 0:
        raise AssertionError("S loss point replacement unexpectedly enabled")
    optimizer = Round2Trainer.build_optimizer(cfg, model)
    head_ids = {id(p) for p in model.criterion.hires_head.parameters()}
    head_groups = [g for g in optimizer.param_groups if any(id(p) in head_ids for p in g["params"])]
    if not head_groups or any(abs(g["lr"] - cfg.SOLVER.BASE_LR * 10) > 1e-12 for g in head_groups):
        raise AssertionError("New H parameters do not use the configured learning rate")
    del optimizer
    register_datasets(cfg)
    records = sorted(DatasetCatalog.get(cfg.DATASETS.TEST[0]), key=lambda r: r["image_id"])[:3]
    checks = []
    for record in records:
        variants, previous = predictor(record, check_reference=True)
        result = paired_prediction(model, predictor.mapper(record), chunk_size=4)
        old, new = variants["mask_first"], result["coarse_instances"]
        if not torch.equal(old.pred_masks, new.pred_masks):
            raise AssertionError("New coarse masks differ from established mask-first inference")
        torch.testing.assert_close(old.scores, new.scores, rtol=1e-6, atol=1e-7)
        if len(result["instances"]) > 100 or result["instances"].image_size != (record["height"], record["width"]):
            raise AssertionError("Invalid original-coordinate output geometry or instance count")
        checks.append(dict(result["diagnostics"], coarse_matches_established_mask_first=True,
                           legacy_matches_upstream=previous["legacy_matches_upstream"]))
        del variants, result
    report = {"valid": True, "head_parameter_groups": len(head_groups), "checks": checks}
    (folder / "independent_inference_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

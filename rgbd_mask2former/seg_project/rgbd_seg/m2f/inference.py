"""Mask2Former counterpart of rgbd_seg/inference.py's Predictor (docs/plan.md §7).

Mirrors the Mask R-CNN Predictor exactly (same __call__ contract: one record in, one Instances
out), swapping in the Mask2Former-specific mapper/checkpointer. Reused downstream by
tools/infer_m2f.py through tools/common.py's generic encode_instances, which recomputes bbox
from the predicted mask's RLE rather than trusting Instances.pred_boxes -- important here
specifically, since stock MaskFormer.instance_inference() (third_party/Mask2Former/mask2former/
maskformer_model.py) deliberately returns an all-zero placeholder box per instance (its own
comment: "Uncomment the following to get boxes from masks (this is slow)"), not a real one.
"""
import torch
from detectron2.modeling import build_model

from .config import setup_cfg
from .mapper import RGBDInstanceDatasetMapper
from .checkpoint import M2FCheckpointer
from . import model as _model  # noqa: F401 -- registers RGBDMaskFormer
from . import depth_fusion as _depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone


class Predictor:
    def __init__(self, cfg, weights):
        if cfg.MODEL.DEVICE == "cpu":
            torch.set_num_threads(cfg.PROJECT.CPU_THREADS)
        self.cfg = cfg
        self.mapper = RGBDInstanceDatasetMapper(cfg, is_train=False)
        self.model = build_model(cfg)
        self.model.eval()
        M2FCheckpointer(self.model).load(weights)

    @torch.inference_mode()
    def __call__(self, record):
        return self.model([self.mapper(record)])[0]


def load_predictor(config_file, weights, device=None, opts=()):
    options = list(opts)
    if device:
        options += ["MODEL.DEVICE", device]
    return Predictor(setup_cfg(config_file, options), weights)

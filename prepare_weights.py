"""Adapt the trusted official COCO R50 checkpoint to RGB-D, without randomizing RGB."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import pickle
import numpy as np
import torch

source = Path("weights/mask2former_r50_coco.pkl")
assert source.stat().st_size == 176382166, "Unexpected official checkpoint size"
with source.open("rb") as stream:
    original = pickle.load(stream, encoding="latin1")
weights = OrderedDict((key.replace(".static_query.", ".query_feat."),
                       torch.as_tensor(value).clone() if isinstance(value, np.ndarray) else value.clone())
                      for key, value in original["model"].items())
# PyTorch state_dict metadata is part of the checkpoint format. Dropping it
# triggers MaskFormer-v1 key migration and would silently randomize the
# Mask2Former pixel decoder. Preserve it when adapting the first convolution.
weights._metadata = getattr(original["model"], "_metadata", {}).copy()
weights._metadata.setdefault("sem_seg_head", {})["version"] = 2
weights._metadata.setdefault("sem_seg_head.predictor", {})["version"] = 2
key = "backbone.stem.conv1.weight"
rgb = weights[key]
assert tuple(rgb.shape) == (64, 3, 7, 7)
weights[key] = torch.cat([rgb, torch.zeros_like(rgb[:, :1])], dim=1)
# Adapt 80 classes + no-object to component + no-object. Only the classifier
# is newly initialized; retain pretrained mask/query/pixel-decoder parameters.
excluded = [key for key in weights if ".class_embed." in key]
assert len(excluded) == 2, excluded
for key in excluded:
    del weights[key]
excluded_buffers = ["criterion.empty_weight"]
for key in excluded_buffers:
    assert key in weights
    del weights[key]  # class-dependent CE weights are rebuilt for the new label space
destination = Path("weights/mask2former_r50_coco_rgbd.pth")
torch.save(dict(model=weights), destination)
metadata = dict(source_url="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/"
                           "maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl",
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                adapted_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                retained_tensors=len(weights), excluded_classifier=excluded,
                excluded_class_dependent_buffers=excluded_buffers,
                format_conversion="static_query renamed to query_feat; state_dict version metadata retained",
                rgbd_initialization="RGB channels copied exactly; depth-channel kernels zero",
                depth_reason="Initial RGB features equal COCO; depth kernels are trainable")
destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))

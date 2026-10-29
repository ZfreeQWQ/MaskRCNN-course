"""Trusted local/model-zoo weights and explicit 3-to-4 channel stem adaptation."""
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer


def adapt_rgbd_weights(state_dict, target_channels=4, mode="mean"):
    """Expand only recognized first-layer weights; preserve all three pretrained RGB channels."""
    state_dict = dict(state_dict)
    names = {"conv1_w", "conv1.weight", "stem.conv1.weight", "backbone.bottom_up.stem.conv1.weight"}
    if target_channels != 4:
        return state_dict
    for key in names.intersection(state_dict):
        value = state_dict[key]
        tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        if tensor.ndim == 4 and tensor.shape[1] == 3:
            if mode not in ("mean", "zero"):
                raise ValueError(f"Unknown depth stem initialization: {mode}")
            depth = tensor.mean(dim=1, keepdim=True) if mode == "mean" else torch.zeros_like(tensor[:, :1])
            state_dict[key] = torch.cat([tensor, depth], dim=1)
    return state_dict


class RGBDCheckpointer(DetectionCheckpointer):
    def __init__(self, *args, depth_init="mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_init = depth_init

    def _torch_load(self, path):
        # These are trusted project/model-zoo checkpoints, containing optimizer and trainer state.
        with self.path_manager.open(path, "rb") as stream:
            return torch.load(stream, map_location="cpu", weights_only=False)

    def _load_file(self, filename):
        checkpoint = super()._load_file(filename)
        channels = self.model.backbone.bottom_up.stem.conv1.weight.shape[1]
        checkpoint["model"] = adapt_rgbd_weights(checkpoint["model"], channels, self.depth_init)
        return checkpoint

"""Opt-in early cross-attention relaxation. No upstream edits or new weights."""
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


def relax_mask(forbidden, size, mode, kernel):
    """PyTorch attention uses True=forbidden: dilate ALLOWED pixels, not True."""
    if mode == "off":
        return forbidden
    if mode == "unmasked":
        return torch.zeros_like(forbidden)
    if mode != "dilate" or kernel < 1 or kernel % 2 != 1:
        raise ValueError("Expected off/unmasked/dilate and an odd positive kernel")
    h, w = size
    allowed = (~forbidden).reshape(-1, 1, h, w).float()
    allowed = F.max_pool2d(allowed, kernel, stride=1, padding=kernel // 2).bool()
    return (~allowed).reshape_as(forbidden)


@TRANSFORMER_DECODER_REGISTRY.register()
class EarlyRelaxationDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(self, in_channels, mask_classification=True, *, attention_mode="off",
                 early_layers=2, dilation_kernel=3, **kwargs):
        super().__init__(in_channels, mask_classification, **kwargs)
        if attention_mode not in ("off", "dilate", "unmasked"):
            raise ValueError(attention_mode)
        if not 0 <= early_layers <= self.num_layers:
            raise ValueError("early_layers must refer to existing cross-attention layers")
        if dilation_kernel < 1 or dilation_kernel % 2 != 1:
            raise ValueError("Dilation kernel must be odd and positive")
        self.attention_mode = attention_mode
        self.early_layers = early_layers
        self.dilation_kernel = dilation_kernel
        self._prediction_calls = 0

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        values = super().from_config(cfg, in_channels, mask_classification)
        values.update(attention_mode=cfg.COURSE.SCREEN.ATTENTION_MODE,
                      early_layers=cfg.COURSE.SCREEN.EARLY_LAYERS,
                      dilation_kernel=cfg.COURSE.SCREEN.DILATION_KERNEL)
        return values

    def forward(self, x, mask_features, mask=None):
        self._prediction_calls = 0
        try:
            return super().forward(x, mask_features, mask)
        finally:
            self._prediction_calls = 0

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        logits, masks, forbidden = super().forward_prediction_heads(output, mask_features, attn_mask_target_size)
        # The initial prediction controls layer 0, not layer 1. Later heads and
        # auxiliary supervision remain official, including the all-masked fallback.
        if self._prediction_calls < self.early_layers:
            forbidden = relax_mask(forbidden, attn_mask_target_size, self.attention_mode, self.dilation_kernel)
        self._prediction_calls += 1
        return logits, masks, forbidden

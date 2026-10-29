"""I3: depth-guided 1/2-resolution mask refinement head (docs/plan.md §4 I3).

Mask2Former's coarse mask prediction comes out of the pixel decoder's FPN at stride 4
(`mask_features`, see third_party/Mask2Former/mask2former/modeling/pixel_decoder/msdeformattn.py
's `forward_features`'s `out[-1]`), i.e. a ~42px small target has only ~10x10 coarse-mask
pixels -- too coarse for the fragment/thin-wire boundaries diagnosed in docs/plan.md §2.

Rather than changing the pixel decoder or transformer decoder (which would require editing
third_party/Mask2Former source, or duplicating a lot of it), this module adds a SEPARATE,
lightweight refinement step applied only to the few mask predictions that matter: the matched
(query, target) pairs during training (via rgbd_seg.m2f.criterion.HiResCriterion), and the
top-`test_topk_per_image` selected instances at inference (via `refine_top_k_masks`, called
from tools/infer.py's postprocessing, not from inside MaskFormer.forward()). Mask2Former's
`pred_masks` are resolution-independent as inputs here because point_sample/grid_sample work in
normalized [0,1] coordinates, so this head never needs to match the pixel decoder's exact stride.

Design (see docs/plan.md I3):
  1. Downsample the model's own input tensor (RGB, or RGB+depth-geometry when
     PROJECT.USE_DEPTH, i.e. the same 3- or 9-channel tensor the backbone consumes) by 2x via
     avg-pooling -- a cheap proxy for "stem-resolution" (stride 2) texture, without needing to
     expose any backbone-internal activation.
  2. Two 3x3 conv + GroupNorm + ReLU layers extract local features from that.
  3. The coarse mask logit (for one query) is upsampled 2x (bilinear) and added to a learned
     residual computed from [upsampled coarse logit, step-2 features] -- the residual's final
     conv is zero-initialized, so at init `refined == upsample(coarse)` exactly (same zero-init
     "safe start" philosophy as rgbd_seg.m2f.depth_fusion's gated fusion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiResMaskHead(nn.Module):
    def __init__(self, in_channels=9, hidden=64):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def image_features(self, images):
        """images: (B, C, H, W) model input tensor -> (B, hidden, H/2, W/2)."""
        images = images[:, :self.in_channels]
        down = F.avg_pool2d(images, kernel_size=2, stride=2)
        return self.stem(down)

    def refine(self, coarse_logits, img_feats, batch_idx):
        """coarse_logits: (N, h, w) one per (matched pair / selected instance).
        img_feats: (B, hidden, H/2, W/2) from `image_features`.
        batch_idx: (N,) which batch image each row of coarse_logits belongs to.
        Returns refined logits at (N, H/2, W/2) resolution.
        """
        target_size = img_feats.shape[-2:]
        up = F.interpolate(coarse_logits[:, None], size=target_size, mode="bilinear", align_corners=False)
        feats = img_feats[batch_idx]  # (N, hidden, H/2, W/2)
        delta = self.residual(torch.cat([up, feats], dim=1))
        return (up + delta).squeeze(1)


def refine_top_k_masks(head, pred_masks, images):
    """Inference-time helper: pred_masks (N, h, w) logits for N already-selected instances
    (e.g. after score-based top-k filtering) from a single image; images (1, C, H, W) that
    image's input tensor. Returns refined logits (N, H/2, W/2). Not called from training.
    """
    if pred_masks.shape[0] == 0:
        h2, w2 = images.shape[-2] // 2, images.shape[-1] // 2
        return pred_masks.new_zeros((0, h2, w2))
    img_feats = head.image_features(images)
    batch_idx = torch.zeros(pred_masks.shape[0], dtype=torch.long, device=pred_masks.device)
    return head.refine(pred_masks, img_feats, batch_idx)

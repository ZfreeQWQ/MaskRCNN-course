#!/usr/bin/env python3
"""CPU unit checks for rgbd_seg.m2f.hires_mask.HiResMaskHead (no GPU, no detectron2 needed)."""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rgbd_seg.m2f.hires_mask import HiResMaskHead, refine_top_k_masks


class HiResMaskHeadTests(unittest.TestCase):
    def test_image_features_shape_and_downsample_factor(self):
        head = HiResMaskHead(in_channels=9, hidden=16)
        images = torch.randn(2, 9, 32, 32)
        feats = head.image_features(images)
        self.assertEqual(feats.shape, (2, 16, 16, 16))

    def test_image_features_ignores_extra_channels_beyond_in_channels(self):
        head = HiResMaskHead(in_channels=3, hidden=8)
        images = torch.randn(1, 9, 16, 16)  # 9 channels, head only configured for 3
        feats = head.image_features(images)
        self.assertEqual(feats.shape, (1, 8, 8, 8))

    def test_refine_matches_upsampled_coarse_at_init(self):
        """Zero-initialized residual conv => refined == bilinear-upsampled coarse, exactly."""
        torch.manual_seed(0)
        head = HiResMaskHead(in_channels=3, hidden=8)
        head.eval()
        coarse = torch.randn(4, 10, 10)  # 4 "matched pairs", 10x10 coarse logits
        images = torch.randn(2, 3, 20, 20)  # 2 batch images -> img_feats at 10x10
        img_feats = head.image_features(images)
        batch_idx = torch.tensor([0, 0, 1, 1])
        refined = head.refine(coarse, img_feats, batch_idx)
        expected = F.interpolate(coarse[:, None], size=img_feats.shape[-2:],
                                  mode="bilinear", align_corners=False).squeeze(1)
        torch.testing.assert_close(refined, expected)

    def test_refine_output_shape_matches_image_features_resolution(self):
        head = HiResMaskHead(in_channels=3, hidden=8)
        coarse = torch.randn(3, 7, 7)
        images = torch.randn(1, 3, 32, 32)
        img_feats = head.image_features(images)  # 16x16
        batch_idx = torch.zeros(3, dtype=torch.long)
        refined = head.refine(coarse, img_feats, batch_idx)
        self.assertEqual(refined.shape, (3, 16, 16))

    def test_refine_batch_idx_routes_correct_image_features(self):
        """Swapping which image a pair is routed to should change its refined output when the
        residual branch is non-trivial (after a training-like perturbation from zero-init)."""
        torch.manual_seed(1)
        head = HiResMaskHead(in_channels=3, hidden=8)
        with torch.no_grad():
            for p in head.residual[-1].parameters():
                p.add_(torch.randn_like(p) * 0.1)  # break the zero-init so routing matters
        coarse = torch.randn(1, 8, 8)
        images = torch.stack([torch.zeros(3, 16, 16), torch.ones(3, 16, 16) * 5], dim=0)
        img_feats = head.image_features(images)
        r0 = head.refine(coarse, img_feats, torch.tensor([0]))
        r1 = head.refine(coarse, img_feats, torch.tensor([1]))
        self.assertFalse(torch.allclose(r0, r1))

    def test_all_parameters_require_grad_and_residual_conv_is_zero(self):
        head = HiResMaskHead(in_channels=3, hidden=8)
        self.assertTrue(all(p.requires_grad for p in head.parameters()))
        torch.testing.assert_close(head.residual[-1].weight, torch.zeros_like(head.residual[-1].weight))
        torch.testing.assert_close(head.residual[-1].bias, torch.zeros_like(head.residual[-1].bias))

    def test_gradient_at_init_reaches_coarse_and_final_conv_but_blocks_stem(self):
        """Same zero-init "cold start" as rgbd_seg.m2f.depth_fusion's gated fusion (see that
        module's test_depth_fusion_backbone.py docstring): residual[-1].weight=0 means
        d(refined)/d(stem_output) = residual[-1].weight = 0 at init, so gradient does NOT yet
        reach `stem` or `residual`'s earlier layers -- only `coarse` (via the `up` skip path)
        and `residual[-1]`'s own parameters (which see the pre-activation directly) get a
        gradient on step 0. One optimizer step moves residual[-1].weight off zero and unlocks
        the rest, exactly as test_depth_fusion_training_step.py demonstrates for I1.
        """
        head = HiResMaskHead(in_channels=3, hidden=8)
        coarse = torch.randn(2, 5, 5, requires_grad=True)
        images = torch.randn(1, 3, 10, 10)
        img_feats = head.image_features(images)
        refined = head.refine(coarse, img_feats, torch.zeros(2, dtype=torch.long))
        refined.sum().backward()
        self.assertIsNotNone(coarse.grad)
        self.assertTrue((coarse.grad.abs() > 0).any())
        self.assertTrue(head.residual[-1].weight.grad.abs().sum() > 0)
        self.assertTrue(head.residual[-1].bias.grad.abs().sum() > 0)
        for p in head.stem.parameters():
            self.assertTrue(p.grad is None or p.grad.abs().sum() == 0)

    def test_gradient_reaches_stem_after_breaking_zero_init(self):
        torch.manual_seed(2)
        head = HiResMaskHead(in_channels=3, hidden=8)
        with torch.no_grad():
            head.residual[-1].weight.add_(0.1)  # simulate "after one optimizer step"
        coarse = torch.randn(2, 5, 5)
        images = torch.randn(1, 3, 10, 10)
        img_feats = head.image_features(images)
        refined = head.refine(coarse, img_feats, torch.zeros(2, dtype=torch.long))
        refined.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.stem.parameters()))


class RefineTopKMasksTests(unittest.TestCase):
    def test_empty_instances_returns_correctly_shaped_empty_tensor(self):
        head = HiResMaskHead(in_channels=3, hidden=8)
        pred_masks = torch.zeros(0, 10, 10)
        images = torch.randn(1, 3, 20, 20)
        out = refine_top_k_masks(head, pred_masks, images)
        self.assertEqual(out.shape, (0, 10, 10))

    def test_nonempty_single_image_routes_all_to_batch_index_zero(self):
        head = HiResMaskHead(in_channels=3, hidden=8)
        pred_masks = torch.randn(5, 10, 10)
        images = torch.randn(1, 3, 20, 20)
        out = refine_top_k_masks(head, pred_masks, images)
        self.assertEqual(out.shape, (5, 10, 10))


if __name__ == '__main__':
    unittest.main()

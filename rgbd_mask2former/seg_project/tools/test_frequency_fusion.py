#!/usr/bin/env python
"""Metamorphic checks: DC preservation, reference reassembly, and gate learning."""
import unittest
import train_m2f
import torch
from torch.nn import functional as F
from detectron2.layers import ShapeSpec
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from rgbd_seg.m2f.frequency_fusion import FrequencyResidualFusion, FrequencyPixelDecoder, reassemble


class FrequencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_reassembly_matches_explicit_neighbors_and_gradients(self):
        torch.manual_seed(17)
        x = torch.randn(1, 9, 4, 5, requires_grad=True)
        logits = torch.randn(1, 9, 8, 10, requires_grad=True)
        weights = logits.softmax(1)
        padded = F.pad(x, [1] * 4, mode="replicate")
        expected = sum(F.interpolate(padded[:, :, i:i+4, j:j+5], scale_factor=2, mode="nearest")
                       * weights[:, i*3+j:i*3+j+1] for i in range(3) for j in range(3))
        actual = reassemble(x, weights, 3, up=2, chunk=4)
        torch.testing.assert_close(actual, expected)
        grads = torch.autograd.grad(actual.square().sum(), (x, logits), retain_graph=True)
        reference = torch.autograd.grad(expected.square().sum(), (x, logits))
        for a, b in zip(grads, reference):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)

    def test_constant_features_have_no_frequency_residual(self):
        fusion = FrequencyResidualFusion(8, 4)
        fusion.alpha.data.fill_(0.7)
        high, low = torch.full((2,8,8,10), 2.), torch.full((2,8,4,5), 3.)
        torch.testing.assert_close(fusion(high, low), torch.full_like(high, 5.), atol=1e-6, rtol=1e-6)

    def test_zero_gate_exact_parity_then_encoder_learns(self):
        torch.manual_seed(29)
        fusion = FrequencyResidualFusion(8, 4)
        high, low = torch.randn(2,8,8,10), torch.randn(2,8,4,5)
        baseline = high + F.interpolate(low, scale_factor=2, mode="bilinear", align_corners=False)
        fusion.eval()
        self.assertTrue(torch.equal(fusion(high, low), baseline))
        fusion.train()
        self.assertTrue(torch.equal(fusion(high, low), baseline))
        opt = torch.optim.AdamW(fusion.parameters(), lr=.01)
        for _ in range(3):
            opt.zero_grad()
            fusion(high, low).square().mean().backward()
            opt.step()
        self.assertGreater(float(fusion.alpha.detach().abs()), 0)
        self.assertGreater(float(fusion.low_encoder.weight.grad.norm()), 0)
        self.assertGreater(float(fusion.high_encoder.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.parameters()))

    def test_pixel_decoder_keeps_pretrained_names_shapes_and_output_order(self):
        class Encoder(torch.nn.Module):
            def forward(self, srcs, pos):
                shapes = torch.tensor([x.shape[-2:] for x in srcs])
                sizes = shapes.prod(1)
                starts = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
                return torch.cat([x.flatten(2).transpose(1,2) for x in srcs], 1), shapes, starts
        shapes = {f"res{i}": ShapeSpec(channels=2**(i+2), stride=2**i) for i in range(2,6)}
        kwargs = dict(input_shape=shapes, transformer_dropout=0., transformer_nheads=4,
                      transformer_dim_feedforward=64, transformer_enc_layers=1, conv_dim=32,
                      mask_dim=16, norm="GN", transformer_in_features=["res3","res4","res5"], common_stride=4)
        old, new = MSDeformAttnPixelDecoder(**kwargs), FrequencyPixelDecoder(**kwargs)
        old.transformer, new.transformer = Encoder(), Encoder()
        incompatible = new.load_state_dict(old.state_dict(), strict=False)
        self.assertTrue(all(k.startswith("frequency_fusion.") for k in incompatible.missing_keys))
        self.assertFalse(incompatible.unexpected_keys)
        features = {f"res{i}": torch.randn(2,2**(i+2),2**(6-i),2**(6-i)) for i in range(2,6)}
        old.eval()
        new.eval()
        with torch.no_grad():
            first, second = old.forward_features(features), new.forward_features(features)
        for a,b in zip([first[0],first[1]]+first[2], [second[0],second[1]]+second[2]):
            self.assertTrue(torch.equal(a,b))


if __name__ == "__main__":
    unittest.main()

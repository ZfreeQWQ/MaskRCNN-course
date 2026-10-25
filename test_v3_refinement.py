"""Final-only detail correctness, exact coarse compatibility and gradients."""
import unittest
import torch
from detectron2.modeling import build_model
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from course.common import configuration
from course.refinement_v3 import DetailCriterion, FinalEmbeddingDecoder, GeometryDetailRefiner


def criterion(kind):
    return kind(1, HungarianMatcher(num_points=64), {"loss_mask": 5., "loss_dice": 5.},
                .1, ["masks"], 64, 3., .75)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_decoder_outputs_and_all_aux_layers_exact(self):
        args = dict(in_channels=32, mask_classification=True, num_classes=1, hidden_dim=32,
                    num_queries=5, nheads=8, dim_feedforward=64, dec_layers=2,
                    pre_norm=False, mask_dim=16, enforce_input_project=False)
        torch.manual_seed(4)
        base = MultiScaleMaskedTransformerDecoder(**args)
        torch.manual_seed(4)
        new = FinalEmbeddingDecoder(**args)
        new.load_state_dict(base.state_dict(), strict=True)
        features = [torch.randn(2, 32, s, s) for s in (4, 2, 1)]
        masks = torch.randn(2, 16, 8, 8)
        a, b = base(features, masks), new(features, masks)
        for key in ("pred_logits", "pred_masks"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for old, actual in zip(a["aux_outputs"], b["aux_outputs"]):
            for key in old:
                torch.testing.assert_close(old[key], actual[key], rtol=0, atol=0)
        self.assertEqual(tuple(b["final_mask_embedding"].shape), (2, 5, 16))
        self.assertIsNone(new._final_embedding)

    def test_zero_residual_final_loss_exactly_official(self):
        coarse = torch.randn(1, 2, 8, 8, requires_grad=True)
        detail = torch.zeros(1, 2, 16, 16, requires_grad=True)
        masks = torch.zeros(2, 32, 32)
        masks[0, 4:10, 4:12] = 1
        masks[1, 12:28, 14:30] = 1
        target = [{"masks": masks}]
        indices = [(torch.arange(2), torch.arange(2))]
        torch.manual_seed(8)
        a = criterion(SetCriterion).loss_masks({"pred_masks": coarse}, target, indices, 2)
        torch.manual_seed(8)
        b = criterion(DetailCriterion).loss_masks({"pred_masks": coarse, "detail_residual": detail}, target, indices, 2)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        sum(b.values()).backward()
        self.assertGreater(float(detail.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(detail.grad).all())

    def test_refiner_zero_initialization_and_later_gradients(self):
        refiner = GeometryDetailRefiner(16, 8, 8)
        args = (torch.randn(2, 16, 8, 8), torch.randn(2, 8, 16, 16),
                torch.randn(2, 7, 16, 16), torch.randn(2, 5, 16))
        delta = refiner(*args)
        self.assertEqual(tuple(delta.shape), (2, 5, 16, 16))
        self.assertEqual(float(delta.abs().sum()), 0)
        optimizer = torch.optim.SGD(refiner.parameters(), lr=.1)
        wanted = torch.randn_like(delta)
        for step in range(2):
            optimizer.zero_grad()
            (refiner(*args) - wanted).square().mean().backward()
            self.assertGreater(float(refiner.output.weight.grad.abs().sum()), 0)
            if step:
                self.assertGreater(float(refiner.geometry[0].weight.grad.abs().sum()), 0)
            optimizer.step()

    def test_empty_final_masks_zero_loss_is_differentiable(self):
        masks = torch.randn(1, 2, 8, 8, requires_grad=True)
        detail = torch.randn(1, 2, 16, 16, requires_grad=True)
        result = criterion(DetailCriterion).loss_masks({"pred_masks": masks, "detail_residual": detail},
                     [{"masks": torch.zeros(0, 32, 32)}], [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))], 1)
        loss = sum(result.values())
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertIsNotNone(detail.grad)

    def test_backbone_preserves_features_and_classifier(self):
        torch.manual_seed(42)
        base = build_model(configuration("configs/v3/g00_fast_control.yaml", ["MODEL.DEVICE", "cpu"])).eval()
        torch.manual_seed(42)
        new = build_model(configuration("configs/v3/g02_geometry_detail.yaml", ["MODEL.DEVICE", "cpu"])).eval()
        torch.testing.assert_close(base.sem_seg_head.predictor.class_embed.weight,
                                   new.sem_seg_head.predictor.class_embed.weight, rtol=0, atol=0)
        new.load_state_dict(base.state_dict(), strict=False)
        x = torch.randn(1, 11, 64, 64)
        with torch.no_grad():
            a, b = base.backbone(x), new.backbone(x)
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertEqual(tuple(b["detail_rgb"].shape), (1, 64, 32, 32))
        self.assertEqual(tuple(b["detail_geometry"].shape), (1, 7, 32, 32))
        self.assertIsNone(new.backbone.stem.half_feature)


if __name__ == "__main__":
    unittest.main()

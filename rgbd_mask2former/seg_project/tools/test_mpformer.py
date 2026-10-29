#!/usr/bin/env python
"""Metamorphic tests for GT isolation, unchanged inference and fixed pilot losses."""
import unittest
import train_m2f  # project import paths
import torch
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from mask2former.modeling.matcher import HungarianMatcher
from rgbd_seg.m2f.mpformer import MaskPilotedDecoder, MPCriterion


def make_decoder():
    return MaskPilotedDecoder(in_channels=32, mask_classification=True, num_classes=1,
                             hidden_dim=32, num_queries=6, nheads=4, dim_feedforward=64,
                             dec_layers=3, pre_norm=False, mask_dim=16, enforce_input_project=False)


def make_targets(counts=(2, 1)):
    targets = []
    for count in counts:
        masks = torch.zeros(count, 32, 32)
        for i in range(count):
            masks[i, i + 1:i + 4, i + 2:i + 5] = 1
        targets.append({"labels": torch.zeros(count, dtype=torch.long), "masks": masks})
    return targets


class MPFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.model = make_decoder()
        self.x = [torch.randn(2, 32, h, h) for h in (4, 8, 16)]
        self.features = torch.randn(2, 16, 16, 16)

    def assert_predictions_close(self, first, second):
        for a, b in zip([first] + first["aux_outputs"], [second] + second["aux_outputs"]):
            torch.testing.assert_close(a["pred_logits"], b["pred_logits"], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(a["pred_masks"], b["pred_masks"], rtol=2e-5, atol=2e-6)

    def test_eval_is_exact_original_decoder_and_ignores_gt(self):
        baseline = MultiScaleMaskedTransformerDecoder(in_channels=32, mask_classification=True, num_classes=1,
                        hidden_dim=32, num_queries=6, nheads=4, dim_feedforward=64,
                        dec_layers=3, pre_norm=False, mask_dim=16, enforce_input_project=False)
        baseline.load_state_dict({k: v for k, v in self.model.state_dict().items() if not k.startswith("label_enc.")})
        baseline.eval()
        self.model.eval()
        with torch.no_grad():
            original = baseline(self.x, self.features)
            adapted = self.model(self.x, self.features, targets="must not be read")
        self.assertNotIn("pilot_outputs", adapted)
        self.assert_predictions_close(original, adapted)

    def test_training_ordinary_queries_cannot_read_gt(self):
        self.model.train()
        first = self.model(self.x, self.features, targets=make_targets((2, 1)))
        targets = make_targets((1, 3))
        for target in targets:
            target["masks"] = target["masks"].flip(-1)
        second = self.model(self.x, self.features, targets=targets)
        self.assert_predictions_close(first, second)
        self.assertEqual(first["pred_masks"].shape[1], 6)
        self.assertEqual(first["pilot_outputs"]["pred_masks"].shape[1], 2)
        self.assertEqual(second["pilot_outputs"]["pred_masks"].shape[1], 3)

    def test_losses_finite_and_new_label_embedding_receives_gradient(self):
        self.model.train()
        targets = make_targets((0, 2))
        outputs = self.model(self.x, self.features, targets=targets)
        criterion = MPCriterion(1, HungarianMatcher(1, 1, 1, num_points=32), {}, .1,
                                ["labels", "masks"], 32, 3.0, .75)
        losses = criterion(outputs, targets)
        self.assertEqual(len(losses), 24)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        gradient = self.model.label_enc.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)

    def test_empty_batch_keeps_embedding_in_backward_graph(self):
        self.model.train()
        result = self.model(self.x, self.features, targets=make_targets((0, 0)))
        self.assertNotIn("pilot_outputs", result)
        criterion = MPCriterion(1, HungarianMatcher(1, 1, 1, num_points=32), {}, .1,
                                ["labels", "masks"], 32, 3.0, .75)
        losses = criterion(result, make_targets((0, 0)))
        self.assertEqual(len(losses), 24)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        self.assertIsNotNone(self.model.label_enc.weight.grad)
        self.assertEqual(float(self.model.label_enc.weight.grad.norm()), 0)

    def test_single_pixel_targets_survive_area_pooling(self):
        self.model.train()
        targets = make_targets((1, 1))
        for target in targets:
            target["masks"].zero_()
            target["masks"][0, 3, 7] = 1
        self.model(self.x, self.features, targets=targets)
        self.assertEqual(self.model.last_pilot_stats["vanished_targets_by_scale"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()

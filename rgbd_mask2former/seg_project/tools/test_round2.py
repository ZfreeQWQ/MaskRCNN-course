"""Small deterministic tests of the paired inference protocol, run in project env."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from detectron2.evaluation.evaluator import inference_context
from rgbd_seg.m2f.round2_inference import paired_prediction, PairedInference


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(1, 1, 1))

    def image_features(self, images):
        return images

    def refine(self, masks, features, batch_ids):
        return masks - 3.0


class FakeModel(nn.Module):
    def __init__(self, refine=True):
        super().__init__()
        self.device = torch.device("cpu")
        self.pixel_mean = torch.zeros(3, 1, 1)
        self.pixel_std = torch.ones(3, 1, 1)
        self.size_divisibility = 1
        self.test_topk_per_image = 100
        self.sem_seg_postprocess_before_inference = True
        self.criterion = nn.Module()
        if refine:
            self.criterion.hires_head = FakeHead()
        # 101 queries; one large, one empty, all others small in original space.
        masks = torch.full((1, 101, 40, 40), -2.0)
        masks[0, :, 2:6, 2:6] = 2.0
        masks[0, 0] = 2.0
        masks[0, 1] = -2.0
        self.masks = masks
        self.logits = torch.tensor([4.0, -4.0]).repeat(1, 101, 1)
        self.sem_seg_head = SimpleNamespace(num_classes=1)

    def backbone(self, tensor):
        return tensor

    def forward_head(self, features):
        return {"pred_masks": self.masks, "pred_logits": self.logits}


class CallableHead:
    num_classes = 1

    def __init__(self, model):
        self.model = model

    def __call__(self, features):
        return self.model.forward_head(features)


def fixture(refine=True):
    model = FakeModel(refine)
    model.sem_seg_head = CallableHead(model)
    record = {"image": torch.zeros(3, 40, 40), "height": 40, "width": 40, "image_id": 9}
    return model, record


class PairedInferenceTests(unittest.TestCase):
    def test_control_mask_first_cap_and_original_geometry(self):
        model, record = fixture(False)
        result = paired_prediction(model, record, 3)
        self.assertEqual(len(result["instances"]), 100)
        self.assertEqual(result["instances"].image_size, (40, 40))
        self.assertNotIn("coarse_instances", result)
        self.assertEqual(result["diagnostics"]["refined_queries"], 0)

    def test_prediction_gate_preserves_large_masks_and_changes_small(self):
        model, record = fixture()
        result = paired_prediction(model, record, 4)
        coarse, refined = result["coarse_instances"], result["instances"]
        areas = coarse.pred_masks.flatten(1).sum(1)
        self.assertTrue(torch.equal(coarse.pred_masks[areas > 1024], refined.pred_masks[areas > 1024]))
        self.assertEqual(result["diagnostics"]["refined_queries"], int(((areas > 0) & (areas <= 1024)).sum()))
        self.assertGreater(result["diagnostics"]["changed_pixels"], 0)
        self.assertTrue(torch.all(refined.scores[areas <= 1024] == 0))

    def test_chunk_invariance_and_no_annotation_dependency(self):
        model, record = fixture()
        first = paired_prediction(model, record, 1)
        record["annotations"] = [{"segmentation": "deliberately invalid, must never be read"}]
        second = paired_prediction(model, record, 7)
        for key in ("instances", "coarse_instances"):
            self.assertTrue(torch.equal(first[key].pred_masks, second[key].pred_masks))
            torch.testing.assert_close(first[key].scores, second[key].scores)

    def test_eval_context_restores_training_mode(self):
        model, record = fixture()
        model.train()
        adapter = PairedInference(model)
        with inference_context(adapter):
            self.assertFalse(model.training)
            adapter([record])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()

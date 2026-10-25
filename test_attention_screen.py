import json
from pathlib import Path
import tempfile
import unittest
import torch
from course.attention_screen import EarlyRelaxationDecoder, relax_mask
from course.common import configuration
from course.screen_data import load_panel, make_panels
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class AttentionScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def decoder(self, kind, **extra):
        return kind(in_channels=32, mask_classification=True, num_classes=1, hidden_dim=32,
                    num_queries=5, nheads=8, dim_feedforward=64, dec_layers=3,
                    pre_norm=False, mask_dim=16, enforce_input_project=False, **extra)

    def test_off_exact_official_outputs_weights_rng(self):
        torch.manual_seed(42)
        base = self.decoder(MultiScaleMaskedTransformerDecoder)
        rng_base = torch.get_rng_state()
        torch.manual_seed(42)
        new = self.decoder(EarlyRelaxationDecoder)
        self.assertTrue(torch.equal(rng_base, torch.get_rng_state()))
        new.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(list(base.state_dict()), list(new.state_dict()))
        features = [torch.randn(2, 32, s, s) for s in (4, 2, 1)]
        masks = torch.randn(2, 16, 8, 8)
        a, b = base(features, masks), new(features, masks)
        for old, actual in [(a,b)] + list(zip(a["aux_outputs"], b["aux_outputs"])):
            for key in ("pred_logits", "pred_masks"):
                torch.testing.assert_close(old[key], actual[key], rtol=0, atol=0)
        self.assertEqual(new._prediction_calls, 0)

    def test_dilate_allowed_not_forbidden_and_edges(self):
        forbidden = torch.ones(2, 3, 25, dtype=torch.bool)
        forbidden[:, :, 12] = False
        enlarged = relax_mask(forbidden, (5,5), "dilate", 3)
        self.assertEqual(int((~enlarged).sum()), 2*3*9)
        self.assertTrue(torch.all(enlarged <= forbidden))
        forbidden.fill_(True); forbidden[:,:,0] = False
        self.assertEqual(int((~relax_mask(forbidden, (5,5), "dilate", 3)).sum()), 2*3*4)
        forbidden.fill_(True)
        self.assertTrue(relax_mask(forbidden, (5,5), "dilate", 3).all())
        self.assertFalse(relax_mask(forbidden, (5,5), "unmasked", 3).any())

    def test_only_first_two_heads_change_and_reset_and_gradient(self):
        new = self.decoder(EarlyRelaxationDecoder, attention_mode="dilate", early_layers=2)
        output = torch.randn(5, 2, 32)
        masks = torch.randn(2, 16, 8, 8)
        for call in range(4):
            a = MultiScaleMaskedTransformerDecoder.forward_prediction_heads(new, output, masks, (5,5))
            b = new.forward_prediction_heads(output, masks, (5,5))
            torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
            expected = relax_mask(a[2], (5,5), "dilate", 3) if call < 2 else a[2]
            self.assertTrue(torch.equal(expected, b[2]))
        features = [torch.randn(2,32,s,s) for s in (4,2,1)]
        a = new(features, masks)
        b = new(features, masks)
        torch.testing.assert_close(a["pred_masks"], b["pred_masks"], rtol=0, atol=0)
        loss = a["pred_masks"].square().mean() + a["pred_logits"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(new.query_feat.weight.grad).all())
        self.assertGreater(float(new.query_feat.weight.grad.abs().sum()), 0)

    def test_registered_config_and_matched_budget(self):
        a = configuration("configs/screen/i00_control.yaml")
        b = configuration("configs/screen/i01_early_dilate.yaml")
        self.assertEqual(a.COURSE.SCREEN.ATTENTION_MODE, "off")
        self.assertEqual(b.COURSE.SCREEN.ATTENTION_MODE, "dilate")
        self.assertEqual(a.SOLVER.dump(), b.SOLVER.dump())
        self.assertEqual(a.INPUT.dump(), b.INPUT.dump())
        self.assertEqual(a.SEED, b.SEED)
        self.assertEqual(a.COURSE.TEST_SIZE, 1024)
        args = EarlyRelaxationDecoder.from_config(b, 256, True)
        self.assertEqual(args["early_layers"], 2)

    def test_panels_deterministic_disjoint_scenes_and_val_only(self):
        records, anns = [], []
        for scene in range(40):
            for view in range(4):
                i = scene*4+view
                records.append(dict(image_id=i, file_name=f"/val/{scene:07d}_v{view:03d}.png"))
                anns.append(dict(image_id=i, area=10 if scene%2 else 300, bbox=[0,0,50,50]))
        index, coco = {"val":records}, {"annotations":anns}
        a,b = make_panels(index,coco,scenes_per_panel=10)
        self.assertEqual((a,b), tuple(make_panels(index,coco,scenes_per_panel=10)))
        self.assertFalse(set(a["scenes"]) & set(b["scenes"]))
        self.assertEqual(len(a["image_ids"]),20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"panel.json"
            path.write_text(json.dumps(a))
            self.assertEqual([r["image_id"] for r in load_panel(path, records)], a["image_ids"])
            a["image_ids"].append(-1);path.write_text(json.dumps(a))
            with self.assertRaises(ValueError): load_panel(path,records)
            a["split"]="train";path.write_text(json.dumps(a))
            with self.assertRaises(ValueError): load_panel(path,records)


if __name__ == "__main__":
    unittest.main()

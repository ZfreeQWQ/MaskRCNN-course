import unittest
import io
import torch
from course.common import configuration, cpu_state_dict
from course.query_screen import SpatialQueryInitializer, SeededQueryDecoder
from detectron2.modeling import build_model
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class QueryScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def decoder(self, kind, **extra):
        return kind(in_channels=32, mask_classification=True, num_classes=1, hidden_dim=32,
                    num_queries=5, nheads=8, dim_feedforward=64, dec_layers=3,
                    pre_norm=False, mask_dim=16, enforce_input_project=False, **extra)

    def test_all_modes_zero_residual_exact_official_rng_and_aux(self):
        torch.manual_seed(42)
        base = self.decoder(MultiScaleMaskedTransformerDecoder)
        expected_rng = torch.get_rng_state()
        features = [torch.randn(2,32,s,s) for s in (4,2,1)]
        masks = torch.randn(2,16,8,8)
        geometry = torch.randn(2,7,32,32)
        a = base(features,masks)
        for mode in ("off","p2","p2_geometry"):
            torch.manual_seed(42)
            new = self.decoder(SeededQueryDecoder,query_mode=mode,query_channels=8)
            self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
            missing,unexpected = new.load_state_dict(base.state_dict(),strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(all(k.startswith("query_initializer.") for k in missing))
            b = new(features,masks,query_geometry=geometry)
            for old,actual in [(a,b)] + list(zip(a["aux_outputs"],b["aux_outputs"])):
                for k in ("pred_logits","pred_masks"):
                    torch.testing.assert_close(old[k],actual[k],rtol=0,atol=0)

    def test_query_pooling_zero_then_inner_and_geometry_gradients(self):
        m = SpatialQueryInitializer(16,32,8,geometry=True)
        image,queries,geometry = torch.randn(2,16,8,8),torch.randn(5,32),torch.randn(2,7,16,16)
        self.assertEqual(float(m(image,queries,geometry).abs().sum()),0)
        optimizer = torch.optim.SGD(m.parameters(),lr=.1)
        target = torch.randn(5,2,32)
        for step in range(3):
            optimizer.zero_grad()
            residual = m(image,queries,geometry)
            self.assertLessEqual(float(residual.detach().abs().max()),.100001)
            (residual-target).square().mean().backward()
            self.assertGreater(float(m.output.weight.grad.abs().sum()),0)
            if step:
                for branch in (m.image[0],m.geometry[0],m.key,m.value,m.query):
                    self.assertTrue(torch.isfinite(branch.weight.grad).all())
                    self.assertGreater(float(branch.weight.grad.abs().sum()),0)
            optimizer.step()

    def test_no_cross_image_leakage_and_geometry_is_required(self):
        m = SpatialQueryInitializer(16,32,8,geometry=True)
        torch.nn.init.normal_(m.output.weight,std=.1)
        image,queries,geo = torch.randn(2,16,8,8),torch.randn(5,32),torch.randn(2,7,16,16)
        a=m(image,queries,geo)
        image[1]*=100;geo[1]*=-10
        b=m(image,queries,geo)
        torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
        self.assertFalse(torch.allclose(a[:,1],b[:,1]))
        with self.assertRaises(ValueError): m(image,queries)

    def test_full_model_shared_tensors_exactly_preserved(self):
        torch.manual_seed(42)
        base=build_model(configuration("configs/screen/i00_control.yaml",["MODEL.DEVICE","cpu"]))
        state=base.state_dict()
        for config in ("j01_p2_queries","j02_geometry_queries"):
            torch.manual_seed(42)
            new=build_model(configuration(f"configs/screen/{config}.yaml",["MODEL.DEVICE","cpu"]))
            actual=new.state_dict()
            self.assertTrue(all(torch.equal(v,actual[k]) for k,v in state.items()))
            extra=set(actual)-set(state)
            self.assertTrue(extra)
            self.assertTrue(all(k.startswith("sem_seg_head.predictor.query_initializer.") for k in extra))

    def test_geometry_exposure_reuses_whole_branch_dropout_rng(self):
        torch.manual_seed(7)
        base=build_model(configuration("configs/screen/i00_control.yaml",["MODEL.DEVICE","cpu"])).backbone.train()
        torch.manual_seed(7)
        new=build_model(configuration("configs/screen/j02_geometry_queries.yaml",["MODEL.DEVICE","cpu"])).backbone.train()
        new.load_state_dict(base.state_dict(),strict=True)
        image=torch.randn(2,11,64,64)
        torch.manual_seed(8);a=base(image);rng=torch.get_rng_state()
        torch.manual_seed(8);b=new(image)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        torch.testing.assert_close(b["query_geometry"],image[:,4:]*new.geometry_fusion.last_keep,rtol=0,atol=0)

    def test_lr_only_changes_new_query_parameters(self):
        from train_screen import ScreenTrainer
        cfg=configuration("configs/screen/j02_geometry_queries.yaml",["MODEL.DEVICE","cpu"])
        model=build_model(cfg);optimizer=ScreenTrainer.build_optimizer(cfg,model)
        names={id(p):n for n,p in model.named_parameters()}
        seen=set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                name=names[id(p)]
                if ".query_initializer." in name:
                    expected=cfg.SOLVER.BASE_LR*cfg.COURSE.QUERY.LR_MULTIPLIER;seen.add("query")
                elif name.startswith("backbone.") and not name.startswith("backbone.geometry_fusion."):
                    expected=cfg.SOLVER.BASE_LR*cfg.SOLVER.BACKBONE_MULTIPLIER;seen.add("backbone")
                else: expected=cfg.SOLVER.BASE_LR;seen.add("other")
                self.assertAlmostEqual(group["lr"],expected)
        self.assertEqual(seen,{"query","backbone","other"})

    def test_export_roundtrip_preserves_module_version_metadata(self):
        cfg=configuration("configs/screen/j02_geometry_queries.yaml",["MODEL.DEVICE","cpu"])
        model=build_model(cfg)
        state=cpu_state_dict(model)
        self.assertEqual(state._metadata["sem_seg_head"]["version"],2)
        stream=io.BytesIO();torch.save({"model":state},stream);stream.seek(0)
        loaded=torch.load(stream,map_location="cpu",weights_only=False)["model"]
        clone=build_model(cfg)
        clone.load_state_dict(loaded,strict=True)
        self.assertTrue(all(torch.equal(v,clone.state_dict()[k]) for k,v in state.items()))


if __name__ == "__main__":
    unittest.main()

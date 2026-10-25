import io
import unittest
import torch
from detectron2.modeling import build_model
from course.common import configuration,cpu_state_dict
from course.local_geometry import relative_geometry_features,LocalGeometryResidual


class LocalGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def geometry(self):
        x=torch.zeros(2,7,32,32);x[:,0]=.2;x[:,4]=-1;x[:,6]=1
        return x

    def features(self,x,reliable=False):
        return relative_geometry_features(x,torch.zeros(1,7,1,1),torch.ones(1,7,1,1),(8,8),reliable)

    def test_invalid_depth_has_no_invented_contrast(self):
        x=torch.zeros(2,7,32,32)
        for reliable in (False,True):
            f=self.features(x,reliable)
            self.assertEqual(tuple(f.shape),(2,10,8,8));self.assertTrue(torch.isfinite(f).all())
            self.assertEqual(float(f.abs().sum()),0)
        x=self.geometry();x[:,:,9,9]=0
        self.assertEqual(float(self.features(x)[:,:8].abs().sum()),0)

    def test_single_pixel_depth_edge_survives_pooling(self):
        x=self.geometry();x[:,0,9,9]=.22
        f=self.features(x)
        self.assertGreater(float(f[:,0].max()),1.9)
        # Negative edge is normalized by its own (nearer left) source depth .22.
        self.assertAlmostEqual(float(f[:,1].max()),(.22-.2)/.22/.05,places=5)
        self.assertLess(float(f[:,:4].mean()),.1)

    def test_reliable_mode_excludes_missing_normal_pseudo_edges(self):
        x=self.geometry();x[:,2:5,9,9]=0
        a,b=self.features(x),self.features(x,True)
        self.assertGreater(float(a[:,4:6].max()),.4)
        self.assertEqual(float(b[:,4:6].abs().sum()),0)
        # Real depth discontinuities remain available, even with missing normals.
        x[:,0,9,9]=.22
        self.assertGreater(float(self.features(x,True)[:,:4].max()),1.9)

    def test_zero_residual_then_inner_gradients(self):
        model=LocalGeometryResidual(16,[0]*7,[1]*7,8,True)
        low=torch.randn(2,16,8,8);geometry=self.geometry();geometry[:,0,9,9]=.22
        torch.testing.assert_close(model(low,geometry),low,rtol=0,atol=0)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        for step in range(3):
            optimizer.zero_grad();model(low,geometry).square().mean().backward()
            self.assertGreater(float(model.output.weight.grad.abs().sum()),0)
            if step:
                self.assertTrue(torch.isfinite(model.features[0].weight.grad).all())
                self.assertGreater(float(model.features[0].weight.grad.abs().sum()),0)
            optimizer.step()

    def test_original_initialization_rng_and_shared_tensors_and_export(self):
        torch.manual_seed(42)
        base=build_model(configuration("configs/screen/i00_control.yaml",["MODEL.DEVICE","cpu"]))
        rng=torch.get_rng_state();original=base.state_dict()
        for name in ("k01_local_contrast","k02_reliable_contrast"):
            torch.manual_seed(42)
            cfg=configuration(f"configs/screen/{name}.yaml",["MODEL.DEVICE","cpu"])
            new=build_model(cfg)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            state=new.state_dict();self.assertTrue(all(torch.equal(v,state[k]) for k,v in original.items()))
            self.assertTrue(all(k.startswith("sem_seg_head.local_geometry.") for k in set(state)-set(original)))
            stream=io.BytesIO();torch.save({"model":cpu_state_dict(new)},stream);stream.seek(0)
            loaded=torch.load(stream,map_location="cpu",weights_only=False)["model"]
            new.load_state_dict(loaded,strict=True)

    def test_lr_only_changes_new_local_module(self):
        from train_screen import ScreenTrainer
        cfg=configuration("configs/screen/k02_reliable_contrast.yaml",["MODEL.DEVICE","cpu"])
        model=build_model(cfg);optimizer=ScreenTrainer.build_optimizer(cfg,model)
        names={id(p):n for n,p in model.named_parameters()}
        for group in optimizer.param_groups:
            for p in group["params"]:
                name=names[id(p)]
                expected=cfg.SOLVER.BASE_LR
                if name.startswith("sem_seg_head.local_geometry."):expected*=cfg.COURSE.LOCAL.LR_MULTIPLIER
                elif name.startswith("backbone.") and not name.startswith("backbone.geometry_fusion."):
                    expected*=cfg.SOLVER.BACKBONE_MULTIPLIER
                self.assertAlmostEqual(group["lr"],expected)


if __name__=="__main__":unittest.main()

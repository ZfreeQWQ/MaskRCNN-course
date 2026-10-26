import unittest
import torch
import yaml
from detectron2.modeling import build_model
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from course.common import configuration
from course.modeling import SmallInstanceCriterion
from course.roi_v2 import ROICriterion
from course.screen_protocol import compare_protocol


class WeightScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def criterion(self, kind, weight=2.):
        matcher=HungarianMatcher(cost_class=2,cost_mask=5,cost_dice=5,num_points=64)
        args=(1,matcher,{},.1,["labels","masks"],64,3.,.75)
        if kind=="binary":
            c=SmallInstanceCriterion(*args);c.small_weight=weight;c.foreground_fraction=0.
        elif kind=="area": c=ROICriterion(*args,roi_fraction=0.,area_weight=True)
        else: c=SetCriterion(*args)
        return c

    def data(self):
        masks=torch.zeros(3,16,16);masks[0,1:4,1:4]=1;masks[1,4:10,4:10]=1;masks[2,11:14,11:14]=1
        return dict(pred_masks=torch.randn(1,3,8,8,requires_grad=True),pred_logits=torch.randn(1,3,2,requires_grad=True)),[
            dict(masks=masks,labels=torch.zeros(3,dtype=torch.int64),small=torch.tensor([True,False,True]),
                 original_area=torch.tensor([64.,4096.,256.]))]

    def test_disabled_binary_exact_loss_gradient_and_rng(self):
        outputs,targets=self.data();indices=[(torch.arange(3),torch.arange(3))]
        torch.manual_seed(19);a=self.criterion("official").loss_masks(outputs,targets,indices,3);rng=torch.get_rng_state()
        ga=torch.autograd.grad(sum(a.values()),outputs["pred_masks"],retain_graph=True)[0]
        torch.manual_seed(19);b=self.criterion("binary",1.).loss_masks(outputs,targets,indices,3)
        gb=torch.autograd.grad(sum(b.values()),outputs["pred_masks"])[0]
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)

    def test_weighting_preserves_sampling_rng_and_class_loss(self):
        outputs,targets=self.data()
        for kind in ("binary","area"):
            torch.manual_seed(19);a=self.criterion("official")(outputs,targets);rng=torch.get_rng_state()
            c=self.criterion(kind)
            torch.manual_seed(19);b=c(outputs,targets)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            torch.testing.assert_close(a["loss_ce"],b["loss_ce"],rtol=0,atol=0)
            self.assertFalse(torch.equal(a["loss_dice"],b["loss_dice"]))
            self.assertGreater(c.last_weight_diagnostics["reweighted_targets"],0)
            gradients=torch.autograd.grad(sum(b.values()),outputs["pred_masks"],retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradients).all())

    def test_binary_normalizer_includes_unmatched_targets(self):
        outputs,targets=self.data();outputs["pred_masks"]=torch.zeros(1,2,8,8,requires_grad=True)
        indices=[(torch.tensor([0,1]),torch.tensor([0,1]))]
        c=self.criterion("binary")
        value=c.loss_masks(outputs,targets,indices,3)["loss_mask"]
        torch.testing.assert_close(value,torch.tensor(2.).log()*3/5)
        self.assertAlmostEqual(float(c.last_weight_diagnostics["mean_all_target_weight"]),5/3,places=6)

    def test_auxiliary_and_empty_targets(self):
        outputs,targets=self.data();outputs["aux_outputs"]=[{k:v.clone() for k,v in outputs.items()}]
        for kind in ("binary","area"):
            c=self.criterion(kind);values=c(outputs,targets)
            self.assertEqual(len(values),6)
            self.assertTrue(all(torch.isfinite(v) for v in values.values()))
            empty=[{k:v[:0] for k,v in targets[0].items()}]
            values=c.loss_masks(outputs,empty,[(torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64))],1)
            self.assertTrue(all(v.item()==0 for v in values.values()))

    def test_protocol_requires_explicit_loss_exception_and_keeps_query_settings(self):
        ref=yaml.safe_load(configuration("configs/screen/j01_p2_queries.yaml").dump())
        for name,key in (("p01_p2_small_weight2","COURSE.SMALL_LOSS_WEIGHT"),("p02_p2_area_weight","COURSE.V2.AREA_WEIGHT")):
            actual=yaml.safe_load(configuration(f"configs/screen/{name}.yaml").dump())
            with self.assertRaises(ValueError):compare_protocol(actual,ref)
            changes=compare_protocol(actual,ref,["MODEL.META_ARCHITECTURE",key])
            self.assertEqual(set(changes),{"MODEL.META_ARCHITECTURE",key})
            self.assertEqual(actual["COURSE"]["QUERY"],ref["COURSE"]["QUERY"])
            self.assertEqual(actual["SOLVER"],ref["SOLVER"])

    def test_all_shared_parameters_rng_and_optimizer_groups_unchanged(self):
        from train_screen import ScreenTrainer
        torch.manual_seed(42);base=build_model(configuration("configs/screen/j01_p2_queries.yaml",["MODEL.DEVICE","cpu"]))
        rng=torch.get_rng_state();expected=base.state_dict()
        for name in ("p01_p2_small_weight2","p02_p2_area_weight"):
            cfg=configuration(f"configs/screen/{name}.yaml",["MODEL.DEVICE","cpu"])
            torch.manual_seed(42);m=build_model(cfg)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(set(expected),set(m.state_dict()))
            for k,v in expected.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
            m.load_state_dict(expected,strict=True)
            opt=ScreenTrainer.build_optimizer(cfg,m)
            names={id(p):n for n,p in m.named_parameters()}
            for group in opt.param_groups:
                for p in group["params"]:
                    name=names[id(p)]
                    factor=10 if ".query_initializer." in name else .1 if name.startswith("backbone.") and not name.startswith("backbone.geometry_fusion.") else 1
                    self.assertAlmostEqual(group["lr"],cfg.SOLVER.BASE_LR*factor)


if __name__=="__main__":unittest.main()

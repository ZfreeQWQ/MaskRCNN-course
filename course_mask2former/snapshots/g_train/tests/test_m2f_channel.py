import copy,unittest
from unittest.mock import patch
import torch
from torch import nn
from mmdet.models import Mask2Former
from rgbd.multiscale import MultiScaleRGBDMask2Former,SpatialFusion
from rgbd.course_channel import ChannelRGBDMask2Former,ChannelSpatialFusion
from test_course_feature_models import tiny_init
import m2f_channel_spec as s

class ChannelTests(unittest.TestCase):
    def model(self,enabled=True,parent=False):
        cls=MultiScaleRGBDMask2Former if parent else ChannelRGBDMask2Former
        kw={} if parent else dict(course_channel=dict(enabled=enabled))
        with patch.object(Mask2Former,'__init__',tiny_init):
            return cls(depth_fusion=dict(enabled=False),multiscale_depth=dict(enabled=True,reliability_gate=True,source='depth'),**kw)
    def data(self):
        x=torch.randn(2,5,64,64);x[:,3].abs_();x[:,4]=1;return x
    def test_constructor_preserves_rng_and_common_tensors(self):
        torch.manual_seed(20261013);a=self.model();rng=torch.get_rng_state()
        torch.manual_seed(20261013);b=self.model(parent=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,a.state_dict()[k]),k)
        self.assertEqual(sum(p.numel() for p in a.spatial_fusion.channel_mlps.parameters()),s.ADDED)
    def test_disabled_exact(self):
        torch.manual_seed(3);a=self.model(False);torch.manual_seed(3);b=self.model(parent=True)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        x=self.data();self.assertTrue(all(torch.equal(v,w) for v,w in zip(a.extract_feat(x),b.extract_feat(x))))
    def test_identity_on_nonzero_depth_residual(self):
        a=self.model();b=self.model(parent=True)
        for p in b.spatial_fusion.projections.parameters():nn.init.normal_(p,std=.01)
        for p in b.spatial_fusion.gates.parameters():nn.init.normal_(p,std=.01)
        result=a.load_state_dict(b.state_dict(),strict=False)
        self.assertTrue(all('channel_mlps' in k for k in result.missing_keys));self.assertFalse(result.unexpected_keys)
        for train in (False,True):
            a.train(train);b.train(train);x=self.data();rng=torch.get_rng_state()
            self.assertTrue(all(torch.equal(v,w) for v,w in zip(a.extract_feat(x),b.extract_feat(x))))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_effect_and_all_four_levels_gradient(self):
        a=self.model();x=self.data()
        for p in a.spatial_fusion.projections.parameters():nn.init.normal_(p,std=.01)
        for mlp in a.spatial_fusion.channel_mlps:nn.init.normal_(mlp[-1].weight,std=.01)
        before=copy.deepcopy(a.state_dict());opt=torch.optim.AdamW(a.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad();ys=a.extract_feat(x);loss=sum(y.square().mean() for y in ys);loss.backward()
            for p in a.spatial_fusion.channel_mlps.parameters():self.assertTrue(torch.isfinite(p.grad).all())
            opt.step()
        for k,v in a.state_dict().items():
            if 'channel_mlps' in k:self.assertFalse(torch.equal(v,before[k]),k)
        self.assertEqual(a.spatial_fusion.channel_calls['train_samples'],6)
    def test_only_depth_path_scaled(self):
        a=self.model();x=self.data()
        for mlp in a.spatial_fusion.channel_mlps:nn.init.constant_(mlp[-1].bias,2.)
        # A zero depth projection cannot change RGB even with non-unit channels.
        rgb=a.backbone(x[:,:3]);out=a.extract_feat(x)
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(rgb,out)))
    def test_bad_options(self):
        with patch.object(Mask2Former,'__init__',tiny_init):
            for opts in ({'enabled':1},{'unknown':True}):
                with self.assertRaises(ValueError):ChannelRGBDMask2Former(course_channel=opts)
    def test_budget(self):
        self.assertEqual(s.DEADLINE-s.START,43200);self.assertEqual(s.GPU_MAX,37800)
        hard,work=s.lease_end(s.START,0,s.SMOKE_END,900)
        self.assertEqual(hard-s.START,900);self.assertEqual(hard-work,20)
        with self.assertRaises(TimeoutError):s.lease_end(s.DEADLINE,0,s.EVAL_END,2700)
    def test_only_one_fresh_arm(self):
        self.assertEqual(s.job()['flags'],dict(multiscale_depth=True,reliability_gate=True,course_copy_paste=True,functional_smoke=False))
    def test_model_recipe_delta(self):
        import m2f_channel_worker as w
        import train_course_directions as t
        args=w.args_for();cats=[dict(id=1,name='component',supercategory='component')]
        a=w.build(args,cats).to_dict();b=t.build_config(args,cats).to_dict()
        self.assertEqual(a['model'].pop('course_channel'),dict(enabled=True))
        a['model']['type']=b['model']['type'];a['custom_imports']['imports'].remove('rgbd.course_channel')
        self.assertEqual(a['custom_hooks'].pop(),dict(type='ChannelAuditHook'))
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()

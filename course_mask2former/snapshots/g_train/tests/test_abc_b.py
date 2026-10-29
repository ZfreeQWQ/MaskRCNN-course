import unittest,torch
from mmcv.ops import point_sample
from mmdet.models.dense_heads import Mask2FormerHead
from rgbd.course_tiny_matching import tiny_pixels,replace_tail,TinyMatchedHead
class MatchingContracts(unittest.TestCase):
    def fixture(self):
        masks=torch.zeros(4,128,128);masks[0,1,2]=1;masks[1,9:17,9:17]=1;masks[2,30:39,30:39]=1
        return masks
    def test_exact_quota_prefix_and_targets(self):
        m=self.fixture();saved=m.clone();pixels,shape=tiny_pixels(m);self.assertEqual([i for i,p in pixels],[0,1])
        torch.manual_seed(7);c=torch.rand(1,12544,2);state=torch.get_rng_state().clone()
        out,n=replace_tail(c,pixels,shape,torch.Generator().manual_seed(13))
        self.assertEqual(n,1024);self.assertTrue(torch.equal(c[:,:-1024],out[:,:-1024]));self.assertTrue(torch.equal(m,saved));self.assertTrue(torch.equal(state,torch.get_rng_state()))
        values=point_sample(m[:,None],out.repeat(4,1,1)).squeeze(1)
        self.assertTrue((values[:2].max(1).values>=.999).all())
    def test_empty_identity(self):
        c=torch.rand(1,12544,2);out,n=replace_tail(c,[],(128,128),torch.Generator());self.assertEqual(n,0);self.assertTrue(torch.equal(c,out))
    def test_independent_reproducible_rng(self):
        m=self.fixture();p,shape=tiny_pixels(m);c=torch.rand(1,12544,2)
        a,_=replace_tail(c,p,shape,torch.Generator().manual_seed(9));b,_=replace_tail(c,p,shape,torch.Generator().manual_seed(9));self.assertTrue(torch.equal(a,b))
    def test_inference_and_loss_sampler_unchanged(self):
        self.assertIs(TinyMatchedHead.predict,Mask2FormerHead.predict);self.assertIs(TinyMatchedHead.forward,Mask2FormerHead.forward)
        self.assertIs(TinyMatchedHead._loss_by_feat_single,Mask2FormerHead._loss_by_feat_single)
    def test_no_unknown_quota(self):
        with self.assertRaises(AssertionError):replace_tail(torch.rand(1,12544,2),[],(128,128),torch.Generator(),512)

def actual_matching_check(head,path):
    from mmengine.structures import InstanceData
    from launch_nyu_sweep import atomic_json
    head.train();head.tiny_matching_enabled=False
    masks=torch.zeros(2,128,128);masks[0,5,8]=1;masks[1,20:40,20:40]=1
    gt=InstanceData(labels=torch.zeros(2,dtype=torch.long),masks=masks)
    cls=torch.randn(100,2);pred=torch.randn(100,32,32);meta=dict(img_shape=(128,128),batch_input_shape=(128,128))
    torch.manual_seed(117);original=Mask2FormerHead._get_targets_single(head,cls,pred,gt,meta);rng=torch.get_rng_state().clone()
    torch.manual_seed(117);disabled=head._get_targets_single(cls,pred,gt,meta)
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(original[:6],disabled[:6]))
    head.tiny_matching_enabled=True;torch.manual_seed(117);enabled=head._get_targets_single(cls,pred,gt,meta)
    assert torch.equal(rng,torch.get_rng_state()) and head.match_activity['replaced_points']==1024
    assert torch.equal(gt.masks,masks) and head.match_activity['tiny_targets']==1
    atomic_json(path,dict(status='passed',actual_pretrained_head=True,disabled_targets_exact=True,global_rng_exact=True,clean_targets_unchanged=True,enabled_quota1024=True,no_new_parameters=True))

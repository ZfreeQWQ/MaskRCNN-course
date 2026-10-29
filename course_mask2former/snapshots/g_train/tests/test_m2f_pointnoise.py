import copy,unittest
from unittest.mock import patch
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import Mask2FormerHead
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from rgbd.course_pilot import PilotedHead
from rgbd.course_pointnoise import NoisyPilotedHead,noisy_occupancy,decoder,guided_loss
from m2f_local_decoder import decoder as old_decoder,guided_loss as old_loss
import m2f_pointnoise_spec as spec

class PointNoiseTests(unittest.TestCase):
    def generator(self,seed=17):return torch.Generator().manual_seed(seed)
    def test_one_and_four_cells_unchanged(self):
        x=torch.zeros(2,16,16,dtype=torch.bool);x[0,0,0]=True;x[1,:2,:2]=True
        y,n,a=noisy_occupancy(x,self.generator());self.assertTrue(torch.equal(x,y));self.assertEqual(n.tolist(),[0,0])
    def test_budget_nonempty_clean_preserved(self):
        x=torch.rand(32,16,16)>.5;old=x.clone()
        y,n,a=noisy_occupancy(x,self.generator())
        self.assertTrue(torch.equal(x,old));self.assertTrue(y.flatten(1).any(1).all())
        self.assertTrue(torch.equal((x^y).flatten(1).sum(1),n));self.assertTrue((n<=torch.floor(.2*a.double())).all())
        self.assertGreater(int(n.sum()),0)
    def test_zero_ratio_exact(self):
        x=torch.ones(3,16,16,dtype=torch.bool);y,n,a=noisy_occupancy(x,self.generator(),0)
        self.assertTrue(torch.equal(x,y));self.assertEqual(int(n.sum()),0)
    def test_global_rng_isolated(self):
        x=torch.ones(3,16,16,dtype=torch.bool);torch.manual_seed(7);state=torch.get_rng_state().clone()
        noisy_occupancy(x,self.generator());self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_reproducible_independent_layers(self):
        x=torch.ones(8,16,16,dtype=torch.bool);a=noisy_occupancy(x,self.generator())[0]
        self.assertTrue(torch.equal(a,noisy_occupancy(x,self.generator())[0]))
        gen=self.generator();a=noisy_occupancy(x,gen)[0];b=noisy_occupancy(x,gen)[0]
        self.assertFalse(torch.equal(a,b))
    def test_empty_is_error(self):
        with self.assertRaises(AssertionError):noisy_occupancy(torch.zeros(1,3,3,dtype=torch.bool),self.generator())
    def test_ratio_fixed_limit(self):
        for ratio in (-.1,.21,1):
            with self.assertRaises(ValueError):noisy_occupancy(torch.ones(1,2,2,dtype=torch.bool),self.generator(),ratio)
    def test_inference_unchanged(self):
        self.assertIs(NoisyPilotedHead.forward,Mask2FormerHead.forward)
        self.assertIs(NoisyPilotedHead.predict,Mask2FormerHead.predict)
    def test_budget_single_run(self):
        self.assertEqual(spec.DEADLINE-spec.START,18*3600);self.assertEqual(spec.GPU_MAX,15*3600)
        s=spec.arm_spec();self.assertEqual(s.job()['max_job_seconds'],14*3600)
        hard,work=s.lease_end(100,53900,999999,900);self.assertEqual((hard,work),(200,180))
        with self.assertRaises(TimeoutError):s.lease_end(100,53950,999999,900)
    def test_targets_used_clean_in_loss(self):
        import ast,inspect
        # All point-target/loss operations are exactly the accepted clean-loss AST.
        a=ast.parse(inspect.getsource(guided_loss)).body[0].body
        b=ast.parse(inspect.getsource(old_loss)).body[0].body
        self.assertEqual(ast.dump(ast.Module(body=a[3:],type_ignores=[])),ast.dump(ast.Module(body=b[3:],type_ignores=[])))

def actual_noise_check(head,path):
    """Real pretrained head, no optimizer, no GPU, no AP; not a mock loss test."""
    assert type(head) is NoisyPilotedHead
    before={k:v.clone() for k,v in head.state_dict().items()}
    head.train();pixels=torch.randn(2,256,32,32,requires_grad=True)
    memories=[torch.randn(2,256,k,k,requires_grad=True) for k in (4,8,16)]
    samples=[]
    for i in range(2):
        m=np.zeros((2,128,128),dtype=np.uint8);m[0,17+i,23]=1;m[1,20:110,20:110]=1
        ds=DetDataSample(metainfo=dict(batch_input_shape=(128,128),img_shape=(128,128),ori_shape=(128,128)))
        ds.gt_instances=InstanceData(labels=torch.zeros(2,dtype=torch.long),masks=BitmapMasks(m,128,128));samples.append(ds)
    target=head.preprocess_gt([samples[0].gt_instances],[None])[0];clean=target.masks.clone()
    head.zero_grad(set_to_none=True)
    generator=torch.Generator().manual_seed(5);activity=torch.zeros(5,dtype=torch.long)
    a=old_decoder(head,pixels[:1],[z[:1] for z in memories],target.labels,target.masks)
    b=decoder(head,pixels[:1],[z[:1] for z in memories],target.labels,target.masks,generator,0,activity)
    assert all(torch.equal(x,y) for ax,bx in zip(a,b) for x,y in zip(ax,bx))
    torch.manual_seed(79);a=old_loss(head,pixels[:1],[z[:1] for z in memories],target);rng=torch.get_rng_state().clone()
    torch.manual_seed(79);b=guided_loss(head,pixels[:1],[z[:1] for z in memories],target,generator,0,activity)
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    with patch.object(head.pixel_decoder,'forward',return_value=(pixels,memories)) as forward:
        torch.manual_seed(71);ordinary=Mask2FormerHead.loss(head,(pixels,),samples);rng=torch.get_rng_state().clone()
        torch.manual_seed(71);actual=head.loss((pixels,),samples)
        assert forward.call_count==2 and torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,actual[k]) for k,v in ordinary.items())
        assert torch.isfinite(actual['loss_mask_pilot']) and actual['loss_mask_pilot']>0
        actual['loss_mask_pilot'].backward()
        for prefix in ('transformer_decoder.','mask_embed.','cls_embed.'):
            grads=[p.grad for n,p in head.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
        assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in memories)
        assert head.noise_activity['flips']>0 and head.noise_activity['tiny_unchanged_guides']>0
        head.eval();activity_before=copy.deepcopy(head.noise_activity);calls=head.pilot_calls
        blank=[DetDataSample(metainfo=d.metainfo) for d in samples]
        with torch.no_grad():a=head((pixels,),blank);b=Mask2FormerHead.forward(head,(pixels,),blank)
        assert all(torch.equal(x,y) for ax,bx in zip(a,b) for x,y in zip(ax,bx))
        assert calls==head.pilot_calls and activity_before==head.noise_activity
    assert torch.equal(clean,target.masks)
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    from launch_nyu_sweep import atomic_json
    atomic_json(path,dict(status='passed',real_pretrained_head=True,zero_noise_decoder_exact=True,
        zero_noise_loss_exact=True,ordinary_losses_exact=True,ordinary_and_auxiliary_RNG_preserved=True,
        shared_pixel_forward_once=True,clean_target_unchanged=True,auxiliary_gradient_finite=True,
        tiny_guides_nonempty=True,actual_noise=activity_before,inference_exact=True,no_GT_inference=True,
        parameters_unchanged=True))
    head.zero_grad(set_to_none=True)

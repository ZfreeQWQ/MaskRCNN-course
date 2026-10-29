import copy,json,unittest
from unittest.mock import patch
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.models.dense_heads import Mask2FormerHead
from rgbd.course_pilot import PilotedHead,selected_indices
from m2f_local_decoder import decoder
import m2f_final_spec as spec

def actual_pilot_check(head,path):
    assert type(head) is PilotedHead
    before={k:v.clone() for k,v in head.state_dict().items()}
    head.train();pixels=torch.randn(2,256,32,32,requires_grad=True)
    memories=[torch.randn(2,256,k,k,requires_grad=True) for k in (4,8,16)]
    samples=[]
    for i in range(2):
        m=np.zeros((2,128,128),dtype=np.uint8);m[0,17+i,23]=1;m[1,40:65,50:80]=1
        ds=DetDataSample(metainfo=dict(batch_input_shape=(128,128),img_shape=(128,128),ori_shape=(128,128)))
        ds.gt_instances=InstanceData(labels=torch.zeros(2,dtype=torch.long),masks=BitmapMasks(m,128,128))
        samples.append(ds)
    head.zero_grad(set_to_none=True)
    with patch.object(head.pixel_decoder,'forward',return_value=(pixels,memories)) as forward:
        torch.manual_seed(71);ordinary=Mask2FormerHead.loss(head,(pixels,),samples);rng=torch.get_rng_state().clone()
        torch.manual_seed(71);actual=head.loss((pixels,),samples)
        assert forward.call_count==2
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,actual[k]) for k,v in ordinary.items())
        assert torch.isfinite(actual['loss_mask_pilot']) and actual['loss_mask_pilot'].item()>0
        actual['loss_mask_pilot'].backward()
        assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum()>0
        assert all(m.grad is not None and torch.isfinite(m.grad).all() for m in memories)
        for prefix in ('transformer_decoder.','mask_embed.','cls_embed.'):
            grads=[p.grad for n,p in head.named_parameters() if n.startswith(prefix) and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
        # A single-pixel guide must remain nonempty at each attention resolution.
        target=head.preprocess_gt([samples[0].gt_instances],[None])[0]
        c,m=decoder(head,pixels[:1],[z[:1] for z in memories],target.labels,target.masks)
        assert all(torch.isfinite(z).all() for z in (*c,*m))
        head.eval();before_calls=head.pilot_calls
        blank=[DetDataSample(metainfo=d.metainfo) for d in samples]
        with torch.no_grad():
            a=head((pixels,),blank);b=Mask2FormerHead.forward(head,(pixels,),blank)
        assert all(torch.equal(x,y) for ax,bx in zip(a,b) for x,y in zip(ax,bx))
        assert head.pilot_calls==before_calls
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
    result=dict(status='passed',ordinary_losses_exact=True,RNG_preserved=True,
        shared_pixel_forward_once=True,auxiliary_gradient_finite=True,one_pixel_guide=True,
        inference_exact=True,no_inference_GT=True,parameters_unchanged=True)
    from launch_nyu_sweep import atomic_json
    atomic_json(path,result)
    head.zero_grad(set_to_none=True)

class FinalTests(unittest.TestCase):
    def test_finite_scope(self):
        self.assertEqual(spec.ARMS,('channel','pilot'))
        self.assertEqual(spec.DEADLINE-spec.START,36*3600)
        self.assertEqual(spec.GPU_MAX,30*3600)
    def test_selection_zero_and_all(self):
        self.assertEqual(len(selected_indices(0,0,0,'cpu')),0)
        for n in (1,7,16):
            for step in range(5):self.assertEqual(set(selected_indices(n,step,0,'cpu').tolist()),set(range(n)))
    def test_selection_cap_unique_in_range(self):
        for n in (17,32,51,100):
            ids=selected_indices(n,7,1,'cpu').tolist()
            self.assertEqual(len(ids),16);self.assertEqual(len(set(ids)),16)
            self.assertTrue(all(0<=i<n for i in ids))
    def test_selection_no_rng(self):
        torch.manual_seed(3);before=torch.get_rng_state().clone()
        selected_indices(45,9,1,'cpu');self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_selection_covers_all_over_rotation(self):
        for n in range(1,101):
            union=set()
            for step in range(n):union.update(selected_indices(n,step,0,'cpu').tolist())
            self.assertEqual(union,set(range(n)))
    def test_inference_method_unchanged(self):
        self.assertIs(PilotedHead.forward,Mask2FormerHead.forward)
        self.assertIs(PilotedHead.predict,Mask2FormerHead.predict)
    def test_pilot_only_overrides_training(self):
        import inspect
        self.assertEqual({k for k,v in PilotedHead.__dict__.items() if inspect.isfunction(v)},{'loss'})

def preflight(output):
    import os,time
    from pathlib import Path
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from launch_nyu_sweep import atomic_json,sha256
    import m2f_channel_worker as old
    import train_course_directions as t
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not Path('/dev/nvidiactl').exists()
    assert time.time()<spec.PREP_END
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_m2f_final','test_m2f_channel','test_course_directions_training',
        'test_course_copy_paste','test_course_feature_models','test_course_resolution']))
    assert result.wasSuccessful() and not result.skipped
    register_all_modules(init_default_scope=True)
    args=old.args_for();cfg=t.build_config(args,[dict(id=1,name='component',supercategory='component')])
    cfg.custom_imports.imports.append('rgbd.course_pilot');import_modules_from_strings(**cfg.custom_imports)
    cfg.model.type='PilotedRGBDMask2Former';cfg.model.course_pilot=dict(enabled=True,max_queries=16,weight=.5,no_noise=True)
    model=MODELS.build(cfg.model);model.init_weights()
    assert sha256(args.checkpoint)==t.COCO_SHA
    state=t.base.component_state(torch.load(args.checkpoint,map_location='cpu')['state_dict'])
    t.initial_state(model,state)
    actual_pilot_check(model.panoptic_head,Path(output)/'pilot_actual_cpu.json')
    atomic_json(Path(output)/'report.json',dict(status='passed',tests=result.testsRun,zero_skip=True,
        actual_pretrained_head=True,COCO_sha256=t.COCO_SHA,unix=time.time(),no_GPU=True))

if __name__=='__main__':
    import sys
    preflight(sys.argv[2]) if len(sys.argv)>1 and sys.argv[1]=='--actual-preflight' else unittest.main()

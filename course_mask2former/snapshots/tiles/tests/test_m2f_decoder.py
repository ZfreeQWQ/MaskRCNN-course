import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.models.dense_heads.maskformer_head import MaskFormerHead
from mmdet.models.seg_heads.panoptic_fusion_heads.maskformer_fusion_head import MaskFormerFusionHead
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
import m2f_decoder_probe as p
import run_m2f_decoder as launch


class FakeHead(nn.Module):
    predict=MaskFormerHead.predict
    def __init__(self):
        super().__init__()
        g=torch.Generator().manual_seed(7)
        self.cls=torch.randn(1,100,2,generator=g)
        self.masks=torch.randn(1,100,4,5,generator=g)
    def forward(self,x,samples):return [self.cls]*10,[self.masks]*10


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):register_all_modules(init_default_scope=True)
    def test_original_budget(self):self.assertEqual(launch.DEADLINE-launch.PREPARATION_UNIX,7200)
    def test_no_new_training(self):self.assertEqual(set(p.ARMS),{'copy_paste','cp_detail'})
    def test_same_accepted_weights(self):
        self.assertEqual(p.ARMS['copy_paste']['weight_sha'],'bdea9c0f87864941e9ba34abceb561328fc4a950051ea4e0ad59cf81219ccd1a')
        self.assertEqual(p.ARMS['cp_detail']['parameters'],44402718)
    def test_cap(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):p.budget(d,time.time()+20,p.CAP_BYTES+1)
    def test_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):p.budget(d,1)
    def test_no_replay(self):
        with self.assertRaises(ValueError):launch.start(launch.ROOT/'outputs/wrong','a'*40)
    def test_no_missing_commit(self):
        with self.assertRaises(ValueError):launch.start(launch.ROOT/'outputs/wrong',None)
    def test_container_scope(self):
        name,c=launch.docker(launch.ROOT/'outputs/example',dict(owner_token='123',image_id='image'),'capture',['python','x.py'],True)
        self.assertIn('device=0',c);self.assertIn('--read-only',c);self.assertIn('none',c)
        self.assertIn('ai-course-directions-owner=123',c);self.assertEqual(c.count('--volume'),3)
    def test_cpu_no_gpu(self):
        _,c=launch.docker(launch.ROOT/'outputs/example',dict(owner_token='123',image_id='image'),'cpu',['python','x.py'])
        self.assertNotIn('--gpus',c);self.assertIn('CUDA_VISIBLE_DEVICES=',c)
    def test_recorder_no_mutation(self):
        h=FakeHead();r=p.Recorder();handle=h.register_forward_hook(r)
        out=h(None,None);saved=r.take();handle.remove()
        self.assertIs(out,saved);self.assertEqual(r.calls,1)
    def test_recorder_missing(self):
        with self.assertRaises(ValueError):p.Recorder().take()
    def test_recorder_overwrite(self):
        r=p.Recorder();r(None,None,([],[]))
        with self.assertRaises(ValueError):r(None,None,([],[]))
    def test_real_production_head_and_fusion(self):
        h=FakeHead();meta=dict(batch_input_shape=(20,24),img_shape=(18,21),ori_shape=(13,15))
        samples=[DetDataSample(metainfo=meta)]
        fusion=MaskFormerFusionHead(num_things_classes=1,num_stuff_classes=0,
            test_cfg=dict(panoptic_on=False,instance_on=True,semantic_on=False,max_per_image=100))
        cls,masks=h.predict(None,samples)
        off=fusion.predict(cls,masks,samples,rescale=True)[0]['ins_results']
        r=p.Recorder();handle=h.register_forward_hook(r)
        cls,masks=h.predict(None,samples);on=fusion.predict(cls,masks,samples,rescale=True)[0]['ins_results']
        handle.remove();all_cls,all_masks=r.take()
        p.assert_same_prediction(off,on)
        binary,prob,quality,scores=p.stage_values(all_cls[-1],all_masks[-1],meta)
        p.assert_final(binary,prob,scores,on)
        rows=p.encode_layer(binary,prob,quality,scores)
        self.assertEqual(len(rows),100);self.assertEqual(rows[0]['segmentation']['size'],[13,15])
    def test_nonfinite_rejected(self):
        h=FakeHead();h.cls[0,0,0]=float('nan')
        with self.assertRaises(ValueError):p.stage_values(h.cls,h.masks,dict())
    def test_query_shape(self):
        with self.assertRaises(ValueError):p.stage_values(torch.zeros(1,99,2),torch.zeros(1,99,4,4),dict())
    def test_real_two_resize_not_direct(self):
        h=FakeHead();meta=dict(batch_input_shape=(20,24),img_shape=(18,21),ori_shape=(13,15))
        x=p.map_logits(h.masks,meta)
        direct=torch.nn.functional.interpolate(h.masks,size=(13,15),mode='bilinear',align_corners=False)[0]
        self.assertFalse(torch.allclose(x,direct))
    def test_max_cardinality_not_greedy(self):
        m=p.match_iou(np.array([[.9,.6],[.8,.1]]))
        self.assertEqual(m,{1:0,0:1})
    def test_no_double_match(self):self.assertEqual(len(p.match_iou(np.array([[.7,.8]]))),1)
    def test_empty_matching(self):self.assertEqual(p.match_iou(np.zeros((100,0))),{})
    def test_no_below_threshold_match(self):self.assertEqual(p.match_iou(np.array([[.49]])),{})
    def layers(self):
        a=np.zeros((8,8),np.uint8);a[1:3,1:3]=1;b=np.zeros_like(a);b[5:7,5:7]=1
        enc=lambda x:mu.encode(np.asfortranarray(x))
        gt=[enc(a)];anns=[dict(id=1,area=4)]
        layers=[]
        for i,mask in enumerate([a,b]):
            layers.append(dict(stage=i,queries=[dict(query=0,score=.9,class_probability=.95,mask_quality=.9/.95,segmentation=enc(mask))]))
        return layers,anns,gt
    def test_early_loss_and_query_tracking(self):
        layers,anns,gt=self.layers();r=p.analyze_one(layers,anns,gt)[0]
        self.assertTrue(r['early_covered_final_lost']);self.assertEqual(r['coverage_loss_transitions'],[1])
        self.assertEqual(r['early_covering_query_ids'],[0]);self.assertEqual(r['early_queries_final_iou'],{'0':0.})
    def test_final_covered_low_score_not_geometry(self):
        layers,anns,gt=self.layers();layers[1]=copy.deepcopy(layers[0]);layers[1]['stage']=1;layers[1]['queries'][0]['score']=.01
        r=p.analyze_one(layers,anns,gt)[0]
        self.assertTrue(r['final_covered']);self.assertFalse(r['final_display_matched']);self.assertFalse(r['early_covered_final_lost'])
    def test_predeclared_route_thresholds(self):
        layers,anns,gt=self.layers();r=p.analyze_one(layers,anns,gt)[0]
        records=[copy.deepcopy(r) for _ in range(100)]
        self.assertTrue(p.summarize(records,2)['early_layer_hypothesis_supported'])
        self.assertFalse(p.summarize(records[:99],2)['early_layer_hypothesis_supported'])
    def test_tiny_split(self):
        layers,anns,gt=self.layers();a=p.analyze_one(layers,anns,gt)[0];b=copy.deepcopy(a);b['area']=257
        x=p.summarize([a,b],2);self.assertEqual(x['tiny']['gt'],1);self.assertEqual(x['small_257_1024']['gt'],1)
    def test_source_no_optimizer_or_test_request(self):
        self.assertNotIn('build_optim_wrapper',Path(p.__file__).read_text())
        self.assertNotIn('instances_test.json',Path(p.__file__).read_text())


if __name__=='__main__':unittest.main()

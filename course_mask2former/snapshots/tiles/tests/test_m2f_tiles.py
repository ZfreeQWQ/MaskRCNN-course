import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmengine.dataset import Compose
from mmdet.utils import register_all_modules
from pycocotools import mask as mu
import m2f_tiles as t
import run_m2f_tiles as launch


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        import rgbd.course_data, rgbd.data

    def raw(self):
        y,x=np.mgrid[:1024,:1024];depth=(y*1024+x+1).astype(np.float32)/10000
        return dict(img=np.stack([x,y,x+y],axis=-1).astype(np.float32),rgbd_depth=depth,
            rgbd_raw_valid=(x%2==0),rgbd_label_valid=np.ones((1024,1024),bool),
            ori_shape=(1024,1024),img_shape=(1024,1024),img_id=1,img_path='val.png')

    def pred(self,mask,scores=None):
        if mask.ndim==2:mask=mask[None]
        n=len(mask)
        return InstanceData(masks=torch.as_tensor(mask,dtype=torch.bool),scores=torch.tensor(scores or [.8]*n),
            labels=torch.zeros(n,dtype=torch.long),bboxes=torch.zeros(n,4))

    def test_absolute_budget(self):self.assertEqual(t.DEADLINE-t.PREPARATION_UNIX,14400)
    def test_deadline_rejected(self):
        with patch('m2f_tiles.time.time',return_value=t.DEADLINE):
            with self.assertRaises(TimeoutError):t.check_time()
    def test_deadline_valid(self):
        with patch('m2f_tiles.time.time',return_value=t.PREPARATION_UNIX):t.check_time()
    def test_windows_cover_original(self):
        cover=np.zeros((1024,1024),int)
        for x,y in t.WINDOWS:cover[y:y+640,x:x+640]+=1
        self.assertTrue((cover>=1).all());self.assertEqual(int(cover.max()),4)
    def test_original_unmutated(self):
        r=self.raw();v=t.crop_loaded(r,(384,384));v['img'][0,0]=0
        self.assertEqual(r['img'][384,384,0],384);self.assertEqual(r['ori_shape'],(1024,1024))
    def test_all_planes_same_crop(self):
        r=self.raw()
        for x,y in t.WINDOWS:
            v=t.crop_loaded(r,(x,y))
            for key in ('img','rgbd_depth','rgbd_raw_valid','rgbd_label_valid'):
                np.testing.assert_array_equal(v[key],r[key][y:y+640,x:x+640])
            self.assertEqual(v['ori_shape'],(640,640))
    def test_global_identity(self):
        r=self.raw();v=t.crop_loaded(r,None)
        for k in ('img','rgbd_depth','rgbd_raw_valid','rgbd_label_valid'):np.testing.assert_array_equal(r[k],v[k])
    def test_reject_unplanned_window(self):
        with self.assertRaises(ValueError):t.crop_loaded(self.raw(),(1,1))
    def test_reject_gt(self):
        r=self.raw();r['gt_masks']=[]
        with self.assertRaises(ValueError):t.crop_loaded(r,None)
    def test_reject_misalignment(self):
        r=self.raw();r['rgbd_depth']=r['rgbd_depth'][:640]
        with self.assertRaises(ValueError):t.crop_loaded(r,None)
    def test_real_depth_aware_resize_and_pack(self):
        r=self.raw();r['rgbd_depth'][384:390,384:390]=0;r['rgbd_raw_valid'][384:390,384:390]=False
        tail=Compose([dict(type='RGBDResize',scale=(2559,1536),keep_ratio=True),
            dict(type='PackRGBDDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])
        v=tail(t.crop_loaded(r,(384,384)))
        self.assertEqual(tuple(v['inputs'].shape),(5,1536,1536));self.assertEqual(tuple(v['data_samples'].ori_shape),(640,640))
        self.assertEqual(float(v['inputs'][3,0,0]),0.);self.assertEqual(float(v['inputs'][4,0,0]),0.)
        self.assertTrue(torch.isfinite(v['inputs']).all())
    def test_tile_restoration(self):
        a=np.zeros((640,640),bool);a[2:6,3:8]=1;c=t.candidates(self.pred(a),4)[0]
        m=mu.decode(c['rle']);self.assertEqual(int(m.sum()),20);self.assertEqual(int(m[386:390,387:392].sum()),20)
    def test_five_real_packed_views(self):
        raw=self.raw();tail=Compose([dict(type='RGBDResize',scale=(2559,1536),keep_ratio=True),
            dict(type='PackRGBDDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])
        values=t.pack_views({},lambda request:copy.deepcopy(raw),tail)
        self.assertEqual(len(values),5)
        self.assertEqual(values[0]['data_samples'].ori_shape,(1024,1024))
        self.assertTrue(all(v['data_samples'].ori_shape==(640,640) for v in values[1:]))
        self.assertTrue(all(tuple(v['inputs'].shape)==(5,1536,1536) for v in values))
        self.assertTrue(torch.equal(values[0]['inputs'],tail(copy.deepcopy(raw))['inputs']))
        self.assertTrue(all('gt_instances' not in v['data_samples'] for v in values))
    def test_nonfinite_bbox(self):
        p=self.pred(np.ones((640,640),bool));p.bboxes[0,0]=float('nan')
        with self.assertRaises(ValueError):t.candidates(p,1)
    def test_tile_area_filter(self):
        a=np.zeros((2,640,640),bool);a[0,:32,:32]=1;a[1,:33,:32]=1
        self.assertEqual(len(t.candidates(self.pred(a),1)),1)
    def test_global_large_retained(self):
        a=np.ones((1024,1024),bool);self.assertEqual(len(t.candidates(self.pred(a),0)),1)
    def test_empty_ignored(self):self.assertEqual(t.candidates(self.pred(np.zeros((640,640),bool)),1),[])
    def test_wrong_view_geometry(self):
        with self.assertRaises(ValueError):t.candidates(self.pred(np.zeros((640,640),bool)),0)
    def test_nonfinite_scores(self):
        with self.assertRaises(ValueError):t.candidates(self.pred(np.ones((640,640),bool),[float('nan')]),1)
    def test_wrong_label(self):
        p=self.pred(np.ones((640,640),bool));p.labels[0]=1
        with self.assertRaises(ValueError):t.candidates(p,1)
    def test_edges_recorded_not_silently_removed(self):
        a=np.zeros((640,640),bool);a[0:2,10:12]=1
        self.assertTrue(t.candidates(self.pred(a),1)[0]['tile_edge'])
    def pair(self):
        a=np.zeros((1024,1024),bool);a[390:394,390:394]=1
        b=np.zeros((640,640),bool);b[6:10,6:10]=1
        return t.candidates(self.pred(a),0)+t.candidates(self.pred(b),4)
    def test_cross_view_duplicate_removed(self):self.assertEqual(len(t.merge_candidates(self.pair())),1)
    def test_equal_score_global_first(self):self.assertEqual(t.merge_candidates(self.pair())[0]['view'],0)
    def test_higher_score_tile_wins(self):
        p=self.pair();p[1]['score']=.9;self.assertEqual(t.merge_candidates(p)[0]['view'],4)
    def test_no_score_threshold(self):
        p=self.pair()[:1];p[0]['score']=.001;self.assertEqual(len(t.merge_candidates(p)),1)
    def test_max100(self):
        pool=[]
        for i in range(105):
            a=np.zeros((1024,1024),np.uint8);a[3*i:3*i+2,0:2]=1
            pool.append(dict(rle=mu.encode(np.asfortranarray(a)),score=.8,view=0,index=i,area=4,tile_edge=False))
        self.assertEqual(len(t.merge_candidates(pool)),100)
    def test_instances_and_box_coordinates(self):
        p=t.as_instances(t.merge_candidates(self.pair()));self.assertEqual(tuple(p.masks.shape),(1,1024,1024))
        self.assertEqual(p.bboxes[0].tolist(),[390,390,394,394]);self.assertEqual(int(p.masks.sum()),16)
    def test_empty_instances(self):self.assertEqual(tuple(t.as_instances([]).masks.shape),(0,1024,1024))
    def test_composite_five_calls(self):
        class Model:
            def test_step(inner,batch):
                shape=(1024,1024) if batch==0 else (640,640)
                a=np.zeros(shape,bool);a[100:102,100:102]=1
                s=DetDataSample(metainfo=dict(img_id=1));s.pred_instances=self.pred(a);return [s]
        counts=dict(eval_calls=0,train_calls=0)
        with patch('m2f_tiles.check_time'):
            p,info=t.multiview(Model(),list(range(5)),counts)
        self.assertEqual(counts,dict(eval_calls=5,train_calls=0));self.assertEqual(len(info['candidate_counts']),5)
    def test_runtime_container_scope(self):
        name,c=launch.docker(launch.ROOT/'outputs/example',dict(owner_token='123',image_id='image'),'evaluate',['python','x.py'],True)
        self.assertIn('device=0',c);self.assertIn('--read-only',c);self.assertIn('none',c);self.assertIn('ai-course-directions-owner=123',c)
        self.assertTrue(name.startswith('ai-m2f-tiles-'));self.assertEqual(c.count('--volume'),3)
    def test_cpu_no_gpu(self):
        _,c=launch.docker(launch.ROOT/'outputs/example',dict(owner_token='123',image_id='image'),'cpu',['python','x.py'])
        self.assertNotIn('--gpus',c);self.assertIn('CUDA_VISIBLE_DEVICES=',c)
    def test_no_wrong_output(self):
        with self.assertRaises(ValueError):launch.start(launch.ROOT/'outputs/wrong','a'*40)
    def test_no_unknown_commit(self):
        with self.assertRaises(ValueError):launch.start(launch.OUTPUT,None)
    def test_fixed_reference(self):
        self.assertEqual(t.SPEC['weight_sha'],'bdea9c0f87864941e9ba34abceb561328fc4a950051ea4e0ad59cf81219ccd1a')
        self.assertEqual(t.SPEC['parameters'],44400414)
    def test_no_training_code(self):
        code=Path(t.__file__).read_text();self.assertNotIn('build_optim_wrapper',code);self.assertNotIn('instances_test.json',code)


if __name__=='__main__':unittest.main()

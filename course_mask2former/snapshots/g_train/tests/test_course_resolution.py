import copy
from pathlib import Path
import unittest
import numpy as np

from course_resolution_probe import (parser, scaled_pipeline, safe_request,
                                     check_coverage, check_baseline)


class ResolutionTests(unittest.TestCase):
    def pipeline(self):
        return [dict(type='LoadCourseRGBDImage'),
                dict(type='RGBDResize', scale=(1706,1024), keep_ratio=True),
                dict(type='LoadAnnotations', with_mask=True, with_bbox=True),
                dict(type='PackRGBDDetInputs')]

    def test_opt_in_default(self):
        self.assertEqual(parser().parse_args(['--mode','smoke','--output','x']).course_eval_scale,1024)

    def test_only_scale_changes_and_original_immutable(self):
        original=self.pipeline();saved=copy.deepcopy(original)
        a=scaled_pipeline(original);b=scaled_pipeline(original,1536)
        self.assertEqual(original,saved)
        self.assertEqual(b[1]['scale'],(2559,1536))
        b[1]['scale']=a[1]['scale'];self.assertEqual(a,b)
        self.assertNotIn('LoadAnnotations',[t['type'] for t in a])

    def test_unknown_scale_or_pipeline_rejected(self):
        for scale in (True,1280,2048):
            with self.assertRaises(ValueError):scaled_pipeline(self.pipeline(),scale)
        p=self.pipeline();p[1]['scale']=(1024,1024)
        with self.assertRaises(ValueError):scaled_pipeline(p)

    def test_scope_and_geometry(self):
        image=dict(id=1,file_name='rgb/val/x.png',depth_file_name='depth/val/x.png',
                   width=1024,height=1024,depth_scale_m=.001,depth_invalid_code=0,scene_id='s')
        self.assertEqual(safe_request('/tmp/data',image)['img_id'],1)
        for field,value in [('file_name','rgb/test/x.png'),('depth_file_name','../x.png'),('width',1536)]:
            im=dict(image);im[field]=value
            with self.assertRaises(ValueError):safe_request('/tmp/data',im)

    def test_coverage_empty_duplicates_order(self):
        check_coverage([1,2],[1,2])
        for ids in ([],[1],[2,1],[1,1]):
            with self.assertRaises(ValueError):check_coverage(ids,[1,2])

    def test_baseline_stop_gate(self):
        r={'course/segm_mAP'+s:v for s,v in zip(('', '_50','_75','_s','_m','_l'),(.676,.888,.749,.284,.696,.941))}
        check_baseline(r,r)
        bad=dict(r);bad['course/segm_mAP_s']=.285
        with self.assertRaises(ValueError):check_baseline(bad,r)
        bad['course/segm_mAP_s']=float('nan')
        with self.assertRaises(ValueError):check_baseline(bad,r)

    def test_paired_resize_validity_and_depth(self):
        from rgbd.data import RGBDResize,PackRGBDDetInputs
        depth=np.ones((4,4),dtype=np.float32)*3;depth[0,0]=0
        valid=depth>0
        sample=dict(img=np.ones((4,4,3),dtype=np.float32),ori_shape=(4,4),img_shape=(4,4),
                    img_id=1,img_path='synthetic',rgbd_depth=depth,
                    rgbd_raw_valid=valid.copy(),rgbd_label_valid=valid.copy())
        result=RGBDResize(scale=(8,8),keep_ratio=True)(sample)
        self.assertEqual(result['img'].shape,(8,8,3))
        self.assertEqual(result['ori_shape'],(4,4))
        self.assertTrue(np.allclose(result['rgbd_depth'][result['rgbd_depth']>0],3))
        self.assertTrue((result['rgbd_depth'][:2,:2]==0).all())
        self.assertEqual(result['rgbd_raw_valid'].dtype,np.bool_)
        self.assertFalse(result['rgbd_raw_valid'][:2,:2].any())
        packed=PackRGBDDetInputs(meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))(result)
        self.assertEqual(tuple(packed['inputs'].shape),(5,8,8))
        self.assertEqual(tuple(packed['data_samples'].label_valid.data.shape),(1,8,8))
        self.assertEqual(len(packed['data_samples'].gt_instances),0)


if __name__=='__main__':unittest.main(verbosity=2)

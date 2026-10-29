#!/usr/bin/env python
"""Mask identity/merging, cut-edge, coverage, and actual RGB-D mapper regressions."""
import unittest
import json
from pathlib import Path
import train_m2f
import numpy as np
from pycocotools import mask as mu
from rgbd_seg.m2f.multiview_inference import windows, touches_internal_edge, merge_views, resize_view, tile_rows


def row(mask, score=.8, view='full'):
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return dict(image_id=1, category_id=1, score=score, segmentation=rle, view=view)


class Tests(unittest.TestCase):
    def test_windows_cover_all_and_overlap(self):
        coverage = np.zeros((1024,1024), dtype=int)
        ws = windows(1024,1024)
        self.assertEqual(len(ws),4)
        for x,y,w,h in ws:
            coverage[y:y+h,x:x+w] += 1
        self.assertTrue((coverage > 0).all())
        self.assertEqual(int(coverage.max()),4)

    def test_only_internal_cut_edges_rejected(self):
        mask=np.zeros((640,640),dtype=bool); mask[:4,:4]=True
        self.assertFalse(touches_internal_edge(mask,(0,0,640,640),(1024,1024)))
        self.assertTrue(touches_internal_edge(mask,(384,384,640,640),(1024,1024)))

    def test_duplicate_uses_stronger_identity_without_union(self):
        a=np.zeros((30,30),dtype=bool);a[5:10,5:10]=True;a[15:17,15:17]=True
        out=merge_views([row(a,.8),row(a,.95,'0')],tile_weight=.9)
        self.assertEqual(len(out),1)
        self.assertTrue(np.array_equal(mu.decode(out[0]['segmentation']),a))
        self.assertAlmostEqual(out[0]['score'],.855)

    def test_disjoint_fragments_with_overlapping_bboxes_stay_distinct(self):
        a=np.eye(30,dtype=bool);b=np.fliplr(a).copy()
        out=merge_views([row(a),row(b,.9,'0')])
        self.assertEqual(len(out),2)

    def test_within_view_original_predictions_not_suppressed(self):
        a=np.eye(20,dtype=bool)
        self.assertEqual(len(merge_views([row(a),row(a,.7)])),2)

    def test_contained_cross_view_fragment_is_not_added_twice(self):
        a=np.zeros((30,30),dtype=bool);a[5:15,5:15]=True;b=np.zeros_like(a);b[5:8,5:8]=True
        self.assertEqual(len(merge_views([row(a,.95),row(b,.8,'0')])),1)

    def test_empty_and_max100(self):
        self.assertEqual(merge_views([]),[])
        a=np.eye(20,dtype=bool)
        self.assertEqual(len(merge_views([row(a) for _ in range(110)])),100)

    def test_tile_mask_native_offset_and_rle(self):
        import torch
        from detectron2.structures import Instances
        inst=Instances((10,10));inst.pred_masks=torch.zeros((1,10,10),dtype=torch.bool)
        inst.pred_masks[0,4:6,4:6]=True;inst.scores=torch.tensor([.9])
        rows=tile_rows(inst,123,(8,8,10,10),(20,20),'0')
        self.assertEqual(len(rows),1)
        mask=mu.decode(rows[0]['segmentation'])
        self.assertEqual(mask.shape,(20,20));self.assertEqual(int(mask.sum()),4)
        self.assertTrue(mask[12:14,12:14].all())
        self.assertEqual(rows[0]['bbox'],[12.,12.,2.,2.])

    def test_pilot_and_confirmation_are_disjoint_scenes(self):
        from eval_multiview import pilot_ids
        rows=[dict(id=4*s+v,scene_id=str(s)) for s in range(70) for v in range(4)]
        selected=pilot_ids({'images':rows})
        self.assertEqual(len(selected),256)
        self.assertEqual(selected,pilot_ids({'images':list(reversed(rows))}))
        design={r['scene_id'] for r in rows if r['id'] in selected}
        heldout={r['scene_id'] for r in rows if r['id'] not in selected}
        self.assertFalse(design & heldout)

    def test_real_mapper_full_rgb_and_depth_exact(self):
        import torch
        from detectron2.data import transforms as T
        from rgbd_seg.m2f.generalization import setup_cfg
        from rgbd_seg.m2f.mapper import RGBDInstanceDatasetMapper
        from common import image_record
        root=Path(train_m2f.PROJECT)
        cfg=setup_cfg(root/'experiments/mask2former/generalization_20261027/configs/crop.yaml')
        info=json.loads((root/'datasets/blender/annotations/instances_val.json').read_text())['images'][0]
        record=image_record(info,root.parent/'blender')
        normal=RGBDInstanceDatasetMapper(cfg,is_train=False)
        native=RGBDInstanceDatasetMapper(cfg,is_train=False);native.augs=T.AugmentationList([])
        mapped=native(record);view=resize_view(mapped,1536)
        self.assertTrue(torch.equal(view['image'],normal(record)['image']))
        self.assertNotIn('annotations',view)
        crop=resize_view(mapped,1536,(384,384,640,640))
        self.assertEqual(tuple(crop['image'].shape),(9,1536,1536))
        self.assertEqual((crop['height'],crop['width']),(640,640))


if __name__=='__main__':
    unittest.main()

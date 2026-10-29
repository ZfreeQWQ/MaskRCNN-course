import copy
import unittest
from unittest.mock import patch
import numpy as np
from pycocotools import mask as mu
import m2f_multi_choice as m


def row(x=10,w=10,score=.9,category=1):
    a=np.zeros((1024,1024),dtype=np.uint8,order="F");a[10:20,x:x+w]=1
    r=mu.encode(a);r["counts"]=r["counts"].decode()
    return dict(image_id=1,category_id=category,score=score,segmentation=r)


class Contracts(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(m.choose([],[],{}),([],[]))

    def test_original_choice(self):
        base=[row()]; merged=base+[row(x=11,score=.8),row(x=12,score=.7)]
        _,old=m.geometry.refine(base,merged)
        self.assertEqual(m.candidates(base,merged)[0][0],old[0])

    def test_excludes_global_category_and_large(self):
        base=[row()]
        merged=base+[row(category=2),row(w=150)]
        self.assertEqual(m.candidates(base,merged),{})

    def test_threshold(self):
        base=[row(w=12)]
        self.assertTrue(m.candidates(base,[row(x=14,w=12)]))
        self.assertFalse(m.candidates(base,[row(x=15,w=12)]))

    def test_select_second_preserve(self):
        base=[row()]; merged=base+[row(x=11,score=.8),row(x=12,score=.7)]
        before=copy.deepcopy(base)
        with patch.object(m.quality,"predict",return_value=np.array([.1,.2])):
            out,d=m.choose(base,merged,{})
        self.assertEqual(base,before)
        self.assertEqual(out[0]["segmentation"],merged[2]["segmentation"])
        self.assertEqual(out[0]["score"],base[0]["score"])
        self.assertTrue(d[0]["different_from_max_overlap"])

    def test_negative_and_tie(self):
        base=[row()];merged=base+[row(x=11,score=.8),row(x=12,score=.7)]
        with patch.object(m.quality,"predict",return_value=np.array([0.,-.1])):
            out,_=m.choose(base,merged,{})
        self.assertEqual(out,base)
        with patch.object(m.quality,"predict",return_value=np.array([.2,.2])):
            _,d=m.choose(base,merged,{})
        self.assertEqual(d[0]["local_index"],1)

    def test_same_gt(self):
        base=[row()];merged=base+[row(x=11,score=.8),row(x=12,score=.7)]
        gt=row(x=12)
        class Coco:
            def annToRLE(self,a):return a["segmentation"]
        anns=[dict(id=4,area=100,category_id=1,segmentation=gt["segmentation"])]
        pairs,records=m.supervised(base,merged,anns,Coco())
        self.assertEqual([p["gt_id"] for p in pairs],[4,4])
        self.assertEqual(pairs[1]["local_iou"],1.)
        self.assertEqual(records[0]["multiple"],1.)

    def test_gate_and_dedup(self):
        import json
        records=[dict(gt_id=1,image_id=1,gt_area=100,single=.6,multiple=.8)]*2
        gate=m.oracle_summary(records)
        json.dumps(gate,allow_nan=False)
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(gate["groups"]["tiny"]["count"],1)
        self.assertFalse(m.oracle_summary([dict(gt_id=1,image_id=1,gt_area=100,single=.8,multiple=.8)])["gate_passed"])
        self.assertFalse(m.oracle_summary([dict(gt_id=1,image_id=1,gt_area=500,single=.6,multiple=.8)])["gate_passed"])

    def test_unmatched_and_crowd(self):
        class Coco:
            def annToRLE(self,a):return a["segmentation"]
        base=[row()];merged=[row(x=11)]
        pairs,_=m.supervised(base,merged,[dict(id=1,area=100,category_id=1,
            iscrowd=1,segmentation=base[0]["segmentation"])],Coco())
        self.assertIsNone(pairs[0]["gt_id"]);self.assertEqual(pairs[0]["target_delta"],0)

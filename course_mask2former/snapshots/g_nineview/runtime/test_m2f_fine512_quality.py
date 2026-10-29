import copy
import inspect
import unittest
import numpy as np
import m2f_quality_selector as q
import m2f_fine512_quality as f
from test_m2f_fine512_geometry import row

def head(delta):
    h=q.fit(np.zeros((2,13)),np.array([delta,delta])); return h

class Coco:
    def annToRLE(self,a):return a['segmentation']

def ann(r,ident=2,area=64):
    return dict(id=ident,area=area,category_id=r['category_id'],segmentation=r['segmentation'])

class QualityContracts(unittest.TestCase):
    def test_no_gt_in_application(self):
        self.assertEqual(list(inspect.signature(f.apply).parameters),['current','new_raw','before','kept','selector'])
    def test_positive_only_changes_geometry(self):
        old,new=row(),row(width=7,score=.1); r,a=f.apply([old],[new],[old],[dict(origin='original',index=0)],head(.1))
        self.assertEqual(r[0]['segmentation'],new['segmentation']);self.assertEqual(r[0]['score'],old['score']);self.assertEqual(a['accepted'],1)
    def test_zero_negative_identity(self):
        old,new=row(),row(width=7)
        for v in (0.,-.1):self.assertEqual(f.apply([old],[new],[old],[dict(origin='original',index=0)],head(v))[0],[old])
    def test_tiny_additions_immutable(self):
        old,new=row(),row(width=7);r,a=f.apply([old],[new],[old],[dict(origin='new',index=77)],head(.1))
        self.assertEqual(r,[old]);self.assertEqual(a['accepted'],0);self.assertEqual(a['protected_tiny_outputs'],1)
    def test_pair_universe_not_final_only(self):
        a,b=row(score=.8),row(width=7,score=.7);n=row(width=7)
        # b wins reverse nearest neighbour, so a cannot acquire it after b is dropped.
        r,d=f.apply([a,b],[n],[a],[dict(origin='original',index=0)],head(.1));self.assertEqual(r,[a]);self.assertEqual(d['accepted'],0)
    def test_mapping_mismatch_rejected(self):
        a=row();b=row(score=.7)
        with self.assertRaises(AssertionError):f.apply([a],[],[b],[dict(origin='original',index=0)],head(.1))
    def test_order_and_inputs_preserved(self):
        a,b=row(),row(x=400);n=row(width=7);c=copy.deepcopy([a,b]);r,d=f.apply(c,[n],[b,a],[dict(origin='original',index=1),dict(origin='original',index=0)],head(.1))
        self.assertEqual(c,[a,b]);self.assertEqual(r[0],b);self.assertEqual([x['score'] for x in r],[b['score'],a['score']])
    def test_same_gt_not_best_new(self):
        a,n=row(width=8),row(width=7);p=f.supervised([a],[n],[ann(a,1),ann(n,2,56)],Coco())[0]
        self.assertEqual(p['gt_id'],1);self.assertEqual(p['old_iou'],1.);self.assertAlmostEqual(p['new_iou'],.875);self.assertAlmostEqual(p['target_delta'],-.125)
    def test_large_identity_target(self):
        a,n=row(),row(width=7);p=f.supervised([a],[n],[ann(row(width=40,height=40),2,1600)],Coco())[0]
        self.assertEqual(p['target_delta'],0.);self.assertEqual(p['target_kind'],'identity_zero');self.assertNotEqual(p['actual_delta'],0.)
    def test_no_gt_identity_target(self):
        p=f.supervised([row()],[row(width=7)],[],Coco())[0];self.assertEqual(p['target_delta'],0.);self.assertIsNone(p['gt_id'])
    def test_ignore_crowd(self):
        a=row();z=ann(a);z['iscrowd']=1
        self.assertIsNone(f.supervised([a],[row(width=7)],[z],Coco())[0]['gt_id'])
    def test_feature_count_and_finite_fit(self):
        x=q.features(row(),row(width=7));self.assertEqual(len(x),13)
        h=q.fit([x,x],[.2,-.1]);self.assertEqual(h['trainable_parameters'],105);self.assertTrue(np.isfinite(q.predict([x],h)).all());self.assertEqual(h['lambda_l2'],1.)
    def test_no_pair_identity(self):
        a=row();self.assertEqual(f.apply([a],[],[a],[dict(origin='original',index=0)],head(.1))[0],[a])
    def test_duplicate_identity_rejected(self):
        a=row()
        with self.assertRaises(AssertionError):f.apply([a],[],[a,a],[dict(origin='original',index=0)]*2,head(.1))

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import torch
import m2f_boundary_refiner as ref
import m2f_tiny_verifier as v


def row(x=100, y=100, side=8, score=.9, category=1):
    mask=np.zeros((1024,1024),np.uint8);mask[y:y+side,x:x+side]=1
    return dict(image_id=1,category_id=category,score=score,segmentation=ref.encode(mask))


class Contracts(unittest.TestCase):
    def test_label_partition(self):
        self.assertEqual([v.target(a,b) for a,b in ((.5,256),(.49,10),(.09,10),(.99,257),(0,None))],
                         [1,-1,0,0,0])
    def test_duplicate_excluded(self):
        r=row();self.assertEqual(v.eligible([r],[r]),[])
    def test_large_excluded(self):
        self.assertEqual(v.eligible([],[row(side=17)]),[])
    def test_independent(self):
        self.assertEqual(len(v.eligible([row()],[row(200)])),1)
    def test_class_specific(self):
        self.assertEqual(len(v.eligible([row()],[row(category=2)])),1)
    def test_probability_threshold(self):
        a,b=row(),row(200)
        result,detail=v.merge([a],[b],v.eligible([a],[b]),[.499])
        self.assertEqual(result,[a]);self.assertEqual(detail["proposed"],0)
    def test_score_and_unchanged(self):
        a,b=row(),row(200)
        result,_=v.merge([a],[b],v.eligible([a],[b]),[.5])
        self.assertEqual(result[0],a);self.assertAlmostEqual(result[1]["score"],.45)
    def test_five_cap(self):
        merged=[row(20+i*50) for i in range(7)]
        result,detail=v.merge([],merged,v.eligible([],merged),[1.]*7)
        self.assertEqual(len(result),5);self.assertEqual(detail["accepted"],7)
    def test_top100_displacement_record(self):
        current=[row(10+(i%10)*80,10+(i//10)*80,score=.1) for i in range(100)]
        merged=[row(850,850)]
        final,detail=v.merge(current,merged,v.eligible(current,merged),[1.])
        self.assertEqual(len(final),100)
        self.assertEqual(detail["top100_dropped"][0]["origin"],"original")
    def test_zero_new_preserves_original_duplicates(self):
        a=row();final,detail=v.merge([a,a],[],[],[])
        self.assertEqual(final,[a,a]);self.assertEqual(detail["suppressed"],[])
    def test_real_network_backward(self):
        torch.manual_seed(20261013);model=v.Verifier()
        x=torch.randn(2,6,128,128);x[:,5]=0;x[:,5,64,64]=1
        logits=model(x);self.assertEqual(tuple(logits.shape),(2,))
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([1.,0.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__=="__main__":unittest.main()

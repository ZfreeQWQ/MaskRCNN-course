import copy
import unittest
import numpy as np
from pycocotools import mask as mu
import m2f_absolute_quality as q


def row(x=4, y=4, w=8, h=8, score=.9, category=1):
    mask = np.zeros((1024,1024), order="F", dtype=np.uint8)
    mask[y:y+h,x:x+w] = 1
    rle = mu.encode(mask); rle["counts"] = rle["counts"].decode()
    return dict(image_id=1, category_id=category, score=score, segmentation=rle)


class Contracts(unittest.TestCase):
    def test_global_local(self):
        a, b = row(), row(x=5)
        f = q.features([a], [a,b])
        self.assertEqual(f.shape, (2,13))
        self.assertEqual(f[0,1], 0); self.assertEqual(f[1,1], 1)
        self.assertEqual(f[0,7], 1)

    def test_containment(self):
        small, big = row(), row(x=0,y=0,w=64,h=64)
        f = q.features([big], [small,big])
        self.assertAlmostEqual(f[0,9], 1)
        self.assertEqual(f[1,9], 0)

    def test_category(self):
        f = q.features([row(category=2)], [row()])
        self.assertEqual(f[0,7], 0); self.assertEqual(f[0,9], 0)

    def test_empty(self):
        self.assertEqual(q.features([],[]).shape, (0,13))
        self.assertEqual(q.rescore([],[],{}), ([],[]))

    def test_labels(self):
        a, b = row(), row(x=30)
        class Coco:
            def annToRLE(self, ann): return ann["segmentation"]
        anns = [dict(id=1,category_id=1,area=64,segmentation=a["segmentation"]),
                dict(id=2,category_id=1,area=64,segmentation=b["segmentation"],iscrowd=1)]
        y = q.labels([a,b],anns,Coco())
        self.assertEqual([v["target_iou"] for v in y],[1,0])
        self.assertIsNone(y[1]["gt_id"])

    def test_fit_and_no_mutation(self):
        rows = [row(score=.3),row(x=30,score=.8)]
        before = copy.deepcopy(rows)
        model = q.fit(q.features(rows,rows), [0.,1.])
        out, order = q.rescore(rows,rows,model)
        self.assertEqual(rows,before)
        self.assertEqual(order,[1,0])
        self.assertEqual(len(model["coefficients"]),104)
        self.assertTrue(all(0 <= r["score"] <= 1 for r in out))

    def test_ties(self):
        rows = [row(),row(x=30)]
        model = q.fit(np.zeros((3,13)), [1,1,1])
        out, order = q.rescore([],rows,model)
        self.assertEqual(order,[0,1])

    def test_nonfinite_rejected(self):
        with self.assertRaises(AssertionError): q.features([], [row(score=float("nan"))])

    def test_no_ground_truth_in_inference(self):
        import inspect
        self.assertEqual(list(inspect.signature(q.features).parameters), ["base","rows"])
        self.assertEqual(list(inspect.signature(q.rescore).parameters), ["base","rows","model"])

import copy
import inspect
import unittest
import numpy as np
from pycocotools import mask as mu
from m2f_fine512_geometry import refine


def row(x=100, y=100, width=8, height=8, score=.9, category=1):
    mask = np.zeros((1024, 1024), np.uint8)
    mask[y:y+height, x:x+width] = 1
    r = mu.encode(np.asfortranarray(mask)); r['counts'] = r['counts'].decode()
    return dict(image_id=1, category_id=category, score=score, segmentation=r,
                bbox=[float(v) for v in mu.toBbox(r)])


class Contracts(unittest.TestCase):
    def test_no_gt_interface(self):
        self.assertEqual(list(inspect.signature(refine).parameters), ['before', 'new_raw'])
    def test_no_source_identity(self):
        a = [row()]; self.assertEqual(refine(a, [])[0], a)
    def test_same_geometry_identity(self):
        a = [row()]; b, d = refine(a, a); self.assertEqual(a, b); self.assertEqual(d['changed'], 0)
    def test_geometry_only(self):
        a, n = row(), row(width=7, score=.1)
        before, source = copy.deepcopy([a]), copy.deepcopy([n]); r, d = refine(before, source)
        self.assertEqual(r[0]['segmentation'], n['segmentation']); self.assertEqual(r[0]['bbox'], n['bbox'])
        self.assertEqual(r[0]['score'], a['score']); self.assertEqual(before, [a]); self.assertEqual(source, [n])
        self.assertEqual(d['changed'], 1)
    def test_probability_not_consulted(self):
        a, n = row(), row(width=7)
        n['probability'] = 0
        self.assertEqual(refine([a], [n])[1]['changed'], 1)
    def test_class_specific(self):
        a = [row()]; self.assertEqual(refine(a, [row(width=7, category=2)])[0], a)
    def test_overlap_threshold(self):
        a = [row()]; self.assertEqual(refine(a, [row(width=3)])[0], a)
        self.assertEqual(refine(a, [row(width=4)])[1]['changed'], 1)
    def test_large_excluded_both_sides(self):
        a = [row(width=40, height=40)]
        self.assertEqual(refine(a, [row(width=30, height=30)])[0], a)
        a = [row(width=32, height=32)]
        self.assertEqual(refine(a, [row(width=33, height=33)])[0], a)
    def test_reciprocal_prevents_reuse(self):
        a = [row(score=.8), row(score=.7)]; n = [row(width=7)]
        r, d = refine(a, n); self.assertEqual([p['index'] for p in d['pairs']], [0]); self.assertEqual(r[1], a[1])
    def test_ties_by_source_score(self):
        a = [row()]; n = [row(width=7, score=.2), row(height=7, score=.9)]
        self.assertEqual(refine(a, n)[1]['pairs'][0]['new_index'], 1)
    def test_tie_by_index(self):
        a = [row()]; n = [row(width=7), row(height=7)]
        self.assertEqual(refine(a, n)[1]['pairs'][0]['new_index'], 0)
    def test_count_order_no_new_instance(self):
        a = [row(x=10+i*9, score=1-i/101) for i in range(100)]
        r, _ = refine(a, [row(x=900, y=900)])
        self.assertEqual(r, a); self.assertEqual(len(r), 100)
    def test_full100_geometry_change_preserves_slots(self):
        a = [row(x=10+i*9, score=1-i/101) for i in range(100)]
        r, d = refine(a, [row(x=900)])
        self.assertEqual(d['changed'], 1); self.assertEqual(len(r), 100)
        self.assertEqual([x['score'] for x in r], [x['score'] for x in a])
        self.assertEqual(r[:99], a[:99])
    def test_empty_mask_ignored(self):
        a = [row()]; self.assertEqual(refine(a, [row(width=0)])[0], a)
    def test_reject_invalid(self):
        a = row(); a['score'] = float('nan')
        with self.assertRaises(AssertionError):refine([a], [])


if __name__ == '__main__':unittest.main()

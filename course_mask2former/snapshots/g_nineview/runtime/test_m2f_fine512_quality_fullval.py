import ast
import inspect
import unittest
from pathlib import Path
import run_m2f_fine512_quality_fullval as x

class FullvalContracts(unittest.TestCase):
    def test_no_refit_call(self):
        tree=ast.parse(inspect.getsource(x.main))
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('fit',calls);self.assertEqual(calls.count('metrics'),1)
    def test_forbidden_fit(self):
        with self.assertRaises(AssertionError):x.forbidden_fit([],[])
    def test_deadline(self):
        self.assertEqual(x.DEADLINE-x.START,1800);self.assertEqual(x.AP_END-x.START,1200)
    def test_area_strata(self):
        class C:anns={1:dict(area=16),2:dict(area=17),3:dict(area=256),4:dict(area=257)}
        b=dict(tiny=[1,2],small=[1,2,4],false_positives=[])
        a=dict(tiny=[2,3],small=[2,3],false_positives=[1])
        d=x.transfers(C,b,a);self.assertEqual(d['tiny_gained'],[3]);self.assertEqual(d['tiny_lost'],[1])
        self.assertEqual(d['area_bins']['1-16']['lost'],[1]);self.assertEqual(d['area_bins']['65-256']['gained'],[3])
        self.assertEqual(d['area_bins']['257-1024']['lost'],[4])
    def test_gt_not_in_apply_interface(self):
        self.assertEqual(list(inspect.signature(x.f.apply).parameters),['current','new_raw','before','kept','selector'])

if __name__=='__main__':unittest.main()

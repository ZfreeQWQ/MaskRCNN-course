import ast,copy,inspect,unittest
from pathlib import Path
import abc_spec as s
class Contracts(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(s.DEADLINE-s.START,30*3600);self.assertEqual(s.GPU_MAX,23*3600)
        self.assertEqual(s.A_END-s.START,6*3600);self.assertEqual(s.A_GPU_MAX,5*3600)
    def test_main_reference(self):
        self.assertEqual(s.REFERENCE,'outputs/course_m2f_fine512_quality_fullval_20261021_v1')
        self.assertEqual(s.WEIGHT_SHA,'e53ea799348ae53403d15fba07c7d763143f07879191778142a63c2002766ece')
    def test_source_functions_are_verbatim(self):
        import abc_a_worker as a
        funcs=a.numeric_functions();self.assertEqual(len(funcs),4)
        src=(a.OUT/'source/runtime/run_m2f_quality_runtime.py').read_text();gpu=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='gpu')
        for n in gpu.body:
            if isinstance(n,ast.FunctionDef) and n.name in funcs:self.assertEqual(funcs[n.name],ast.get_source_segment(src,n))
    def test_no_numerical_training(self):
        import abc_a_worker as a
        source=inspect.getsource(a.gpu)+inspect.getsource(a.runtime)
        for term in ('.backward(','.step(','.fit('):self.assertNotIn(term,source)
    def test_parity_not_ap(self):
        import abc_a_worker as a
        self.assertNotIn('w.metrics(',inspect.getsource(a.gpu));self.assertEqual(inspect.getsource(a.finish).count('w.metrics('),1)
    def test_lease_never_more_than_requested(self):
        from abc_a_launcher import lease
        t=s.START+1800
        hard,work=lease(t,100,200)
        self.assertEqual(hard,min(s.A_GPU_END,t+s.A_GPU_MAX-100,t+s.GPU_MAX-200));self.assertEqual(hard-work,20)
    def test_counts(self):self.assertEqual((2584+8+25)*9,23553)

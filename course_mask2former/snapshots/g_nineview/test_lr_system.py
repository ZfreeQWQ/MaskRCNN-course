import inspect,json,unittest
from unittest.mock import patch
import lr_system_spec as p
class Contracts(unittest.TestCase):
    def test_original_deadlines(self):
        self.assertEqual(p.END-p.START,21600);self.assertEqual(p.CPU_END-p.START,3600);self.assertEqual(p.GPU_END-p.START,18000);self.assertEqual(p.GPU_MAX,14400)
    def test_single_target(self):
        import lr_system_launcher as l
        src=l.transformed();self.assertIn('for card in (0,):',src);self.assertNotIn('for card in (0,1):',src);self.assertIn('verify_uuid()',src);self.assertIn('LockBundle()',src)
    def test_deadline_lease(self):
        import abc_a_launcher as l
        now=p.START+800;hard,work=l.lease(now,300,300);self.assertEqual(hard,min(p.GPU_END,now+14400-300));self.assertEqual(hard-work,20)
    def test_owner_stop_only(self):
        import lr_system_launcher as l
        src=l.transformed();self.assertIn("d['Config']['Labels']['ai-course-directions-owner']==s.A_OWNER",src);self.assertIn('external GPU compute conflict',src)
    def test_no_updates(self):
        import lr_system_worker as w
        import abc_a_worker as a
        for src in (inspect.getsource(w.gpu),inspect.getsource(w.single_runtime),inspect.getsource(a.runtime),w.finish_source()):
            for forbidden in ('.backward(','.step(','.fit('):self.assertNotIn(forbidden,src)
    def test_one_new_AP(self):
        import lr_system_worker as w
        self.assertEqual(w.finish_source().count('w.metrics('),1);self.assertNotIn('w.metrics(',inspect.getsource(w.gpu))
    def test_numeric_functions_bound(self):
        import abc_a_worker as a
        self.assertEqual(set(a.numeric_functions()),{'m2f_flags','head_flags','new_views','actual_full'})
    def test_separate_primary(self):
        self.assertNotEqual(p.WEIGHT,p.OLD_WEIGHT);self.assertIn('lr3e5_long',p.WEIGHT);self.assertIn('B_into_A',p.PRIOR)
    def test_exact_global_parity(self):
        import lr_system_worker as w
        self.assertIn('case_',inspect.getsource(w.compare_global));self.assertIn('compare_global(actual',inspect.getsource(w.gpu))
    def test_standalone_uses_accepted_M2F_flags(self):
        import lr_system_worker as w
        import abc_a_worker as a
        import torch
        a.w.base.configure_torch();self.assertFalse(torch.backends.cudnn.allow_tf32)
        w.frozen_m2f_flags();self.assertTrue(torch.backends.cudnn.allow_tf32);self.assertFalse(torch.backends.cudnn.deterministic)
        self.assertIn('frozen_m2f_flags()',inspect.getsource(w.single_runtime))
    def test_actual_reference_CPU_backend(self):
        import ctypes,glob,pathlib,numpy as np,os
        lib=ctypes.CDLL(glob.glob(str(pathlib.Path(np.__file__).parent.parent/'numpy.libs/*openblas*.so'))[0])
        core=lib.openblas_get_corename64_;core.restype=ctypes.c_char_p
        self.assertEqual(core().decode(),'Prescott');self.assertEqual(os.environ['OPENBLAS_CORETYPE'],'Prescott')
    def test_budget_counts(self):
        self.assertEqual((2584+8+25)*9,23553);self.assertEqual(23553+8*9+8+25,23658)
    def test_primary_single_timing(self):
        import lr_system_worker as w
        src=inspect.getsource(w.gpu);self.assertIn("states['LR_single1536']['calls']==dict(eval=33,train=0)",src);self.assertIn('warmup=5,repeats=20',src)
if __name__=='__main__':unittest.main()

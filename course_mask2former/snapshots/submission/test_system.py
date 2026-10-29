import ast,inspect,unittest
import system_spec as p
class SystemContracts(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(p.DEADLINE-p.START,43200);self.assertEqual(p.GPU_MAX,28800);self.assertEqual(p.B_GPU_MAX,12600);self.assertLessEqual(p.B_END,p.DEADLINE)
    def test_no_training(self):
        import abc_a_worker as a
        from system_worker import transformed
        for src in (inspect.getsource(a.runtime),transformed('gpu'),transformed('finish')):
            for forbidden in ('.backward(','.step(','.fit('):self.assertNotIn(forbidden,src)
    def test_reference_and_candidate_separate(self):
        self.assertNotEqual(p.BASE_WEIGHT,p.WEIGHT);self.assertNotEqual(p.BASE_WEIGHT_SHA,p.WEIGHT_SHA);self.assertIn('arms/B/',p.WEIGHT);self.assertIn('A_recovery1',p.PRIOR)
    def test_same_runtime_functions(self):
        import abc_a_worker as a
        self.assertEqual(set(a.numeric_functions()),{'m2f_flags','head_flags','new_views','actual_full'})
    def test_one_new_AP_zero_GPU_AP(self):
        from system_worker import transformed
        self.assertEqual(transformed('finish').count('w.metrics('),1);self.assertNotIn('w.metrics(',transformed('gpu'))
    def test_baseline_uses_pointnoise_actual_cache(self):
        from system_worker import transformed
        src=transformed('gpu');self.assertIn("('baseline',ROOT/p.BASE_WEIGHT)",src);self.assertIn('assert_baseline(actual,ref)',src);self.assertNotIn('r.assert_parity(',src)
    def test_B_global_standalone_equivalence(self):
        from system_worker import transformed
        self.assertIn("assert_new_global(actual,b['image_id'])",transformed('gpu'))
    def test_transformed_source_has_no_spec_shadow(self):
        from system_worker import transformed
        tree=ast.parse(transformed('gpu'))
        self.assertFalse(any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='p' for n in ast.walk(tree)))
    def test_call_budget(self):self.assertEqual((2584+8+25)*9,23553);self.assertEqual((8+25)*9,297)
    def test_same_card(self):
        from system_launcher import transformed
        src=transformed();self.assertIn('for card in (0,):',src);self.assertNotIn('for card in (0,1):',src);self.assertIn('LOCK_NB',src)
    def test_more_restrictive_lease(self):
        import abc_a_launcher as l
        t=p.START+1000;hard,work=l.lease(t,100,200);self.assertEqual(hard,min(p.B_GPU_END,t+p.B_GPU_MAX-100,t+p.GPU_MAX-200));self.assertEqual(hard-work,20)
if __name__=='__main__':unittest.main()

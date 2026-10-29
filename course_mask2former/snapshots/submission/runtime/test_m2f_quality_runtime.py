import ast,copy,inspect,unittest
import run_m2f_quality_runtime as r
import launch_m2f_quality_runtime as launcher

class RuntimeContracts(unittest.TestCase):
    def test_budget_and_calls(self):
        self.assertEqual(r.DEADLINE-r.START,1800);self.assertEqual(9*(8+5+20),297)
        ns=launcher.build();self.assertEqual(ns['GPU_END']-r.START,1200)
    def test_parity_rejects_mask_score_order_activity(self):
        base={k:[1,2] for k in ('global_rows','current','merged','jobs','probabilities','before','before_activity')};base['image_id']=1
        ref=dict(base=base,**{k:[1,2] for k in ('new_raw','new_origins','rows','activity')})
        actual={k:v for k,v in base.items() if k!='image_id'};actual.update({k:ref[k] for k in ('new_raw','new_origins','rows','activity')})
        r.assert_parity(actual,ref)
        for key in actual:
            bad=copy.deepcopy(actual);bad[key]=[2,1]
            with self.assertRaises(AssertionError):r.assert_parity(bad,ref)
    def test_no_cache_predictions_to_actual_path(self):
        node=ast.parse(inspect.getsource(r.gpu)).body[0]
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='actual_full')
        self.assertEqual([a.arg for a in fn.args.args],['request','i'])
        text=ast.unparse(fn);self.assertNotIn('ref[',text);self.assertNotIn('read(',text)
        self.assertIn('f.apply',text);self.assertIn('tiny.merge',text)
    def test_no_AP_fit_training_or_new512_head(self):
        text=inspect.getsource(r.gpu)+inspect.getsource(r.finish)
        for value in ('w.metrics(','.backward(','.step(','.fit(',"verifier_512_2000.pth"):
            self.assertNotIn(value,text)
    def test_exact_final_mask_contract(self):
        self.assertIn("actual[key]==ref[key]",inspect.getsource(r.assert_parity))
    def test_timing_includes_quality_and_json(self):
        text=inspect.getsource(r.gpu)
        self.assertLess(text.index('final,activity=f.apply'),text.index('json.dumps(final'))
        self.assertLess(text.index('json.dumps(final'),text.index('ms=(time.perf_counter()-t)'))

if __name__=='__main__':unittest.main()

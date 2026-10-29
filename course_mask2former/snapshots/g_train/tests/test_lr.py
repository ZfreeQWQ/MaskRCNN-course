import copy,unittest
from lr_contract import KEYS,check_delta
from lr_spec import lease_end,START,DEADLINE,GPU_MAX
class LRContracts(unittest.TestCase):
    def pair(self):
        a={k:{} for k in KEYS};a.update(optim_wrapper={'optimizer':{'lr':1e-5},'clip_grad':.01},custom_hooks=[{'type':'existing'}])
        b=copy.deepcopy(a);b['optim_wrapper']['optimizer']['lr']=3e-5;b['custom_hooks'].append({'type':'LearningRateAuditHook'});return a,b
    def test_single_delta_and_inputs_unchanged(self):
        a,b=self.pair();saved=copy.deepcopy((a,b));self.assertTrue(check_delta(a,b)['learning_rate_only']);self.assertEqual((a,b),saved)
    def test_reject_other_recipe_changes(self):
        for key in KEYS:
            a,b=self.pair()
            if key=='custom_hooks':b[key].append({'type':'unapproved'})
            else:b[key]['unapproved']=True
            with self.assertRaises(AssertionError):check_delta(a,b)
    def test_reject_wrong_lr(self):
        for lr in (1e-5,1e-4,0):
            a,b=self.pair();b['optim_wrapper']['optimizer']['lr']=lr
            with self.assertRaises(AssertionError):check_delta(a,b)
    def test_budget_never_extended(self):
        hard,work=lease_end(START+100,0,START+900,900);self.assertEqual(hard,START+900);self.assertEqual(hard-work,20)
        hard,work=lease_end(START+100,GPU_MAX-100,DEADLINE,50400);self.assertEqual(hard,START+200)
        with self.assertRaises(TimeoutError):lease_end(START+100,GPU_MAX,DEADLINE,50400)

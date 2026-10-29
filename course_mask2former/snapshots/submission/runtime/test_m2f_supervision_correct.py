import json
import unittest
import numpy as np
import run_m2f_supervision_correct as r


class CorrectionContracts(unittest.TestCase):
    def test_only_large_changes(self):
        t=np.ones((5,3,3),bool);c=np.zeros_like(t)
        records=[dict(gt_area=a) for a in (None,17,256,1024,1025)]
        result=r.corrected_target(t,c,records)
        self.assertTrue(result[:4].all());self.assertFalse(result[4].any())
        self.assertTrue(t.all())

    def test_gate_negative_small(self):
        self.assertFalse(r.learned_gate(dict(small=dict(net_crossings=0),tiny=dict(net_crossings=1))))

    def test_gate_tiny_loss(self):
        self.assertFalse(r.learned_gate(dict(small=dict(net_crossings=100),tiny=dict(net_crossings=-1))))

    def test_gate_small_gain_allowed(self):
        self.assertTrue(r.learned_gate(dict(small=dict(net_crossings=1),tiny=dict(net_crossings=0))))

    def test_signed_net_not_only_gains(self):
        p=[dict(before=.9,after=.4,area=5),dict(before=.49,after=.51,area=500)]
        s=r.threshold_summary(p,"after")
        self.assertLess(s["small"]["net_crossings"],0)
        self.assertFalse(r.learned_gate(s))

    def test_native_json(self):
        self.assertEqual(json.loads(json.dumps(r.native(dict(a=np.int64(3),b=np.bool_(True))))),dict(a=3,b=True))
        with self.assertRaises(TypeError):r.native(float("nan"))

    def test_budget(self):
        self.assertEqual(r.DEADLINE-r.START,3600)
        self.assertEqual(r.TRAIN_END-r.START,1800)
        self.assertEqual(r.GPU_END-r.START,2700)
        self.assertEqual(r.EVAL_END-r.START,3000)

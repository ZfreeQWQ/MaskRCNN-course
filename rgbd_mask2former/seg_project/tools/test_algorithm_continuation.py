#!/usr/bin/env python
"""Check that AdamW moments survive and restored updates match uninterrupted ones."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import train_m2f
import torch
from rgbd_seg.m2f.algorithm_continuation import audit_optimizer, ContinuationWarmup


class ContinuationTests(unittest.TestCase):
    def test_restored_adamw_produces_identical_next_update(self):
        torch.manual_seed(29)
        model=torch.nn.Linear(4,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(3,4)
        for _ in range(5):
            optimizer.zero_grad()
            model(x).square().sum().backward()
            optimizer.step()
        other=copy.deepcopy(model)
        restored=torch.optim.AdamW(other.parameters(),lr=.01)
        saved=copy.deepcopy(optimizer.state_dict())
        restored.load_state_dict(saved)
        self.assertTrue(audit_optimizer(saved,restored.state_dict())['valid'])
        for m,o in ((model,optimizer),(other,restored)):
            o.zero_grad()
            m(x).square().sum().backward()
            o.step()
        for a,b in zip(model.parameters(),other.parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_empty_or_corrupt_moments_are_rejected(self):
        p=torch.nn.Parameter(torch.ones(2))
        opt=torch.optim.AdamW([p],lr=.01)
        p.sum().backward()
        opt.step()
        saved=copy.deepcopy(opt.state_dict())
        corrupt=copy.deepcopy(saved)
        corrupt['state'][0]['exp_avg'].zero_()
        with self.assertRaises(AssertionError):
            audit_optimizer(saved,corrupt)
        with self.assertRaises(AssertionError):
            audit_optimizer(saved,torch.optim.AdamW([p],lr=.01).state_dict())

    def test_full_checkpoint_staging_preserves_iteration_and_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'parent'
            source.mkdir()
            p=torch.nn.Parameter(torch.ones(2))
            opt=torch.optim.AdamW([p],lr=.01)
            p.sum().backward()
            opt.step()
            state={'iteration':7,'model':{'p':p.detach()},'trainer':{'iteration':7,
                   '_trainer':{'optimizer':opt.state_dict()},'hooks':{'LRScheduler':{'last_epoch':8}}}}
            torch.save(state,source/'model_final.pth')
            script=Path(__file__).with_name('prepare_algorithm_resume.py')
            target=Path(tmp)/'continued'
            result=subprocess.run([sys.executable,str(script),'--source-output',str(source),'--output',str(target)],
                                  capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((target/'resume_seed.json').read_text())
            self.assertEqual(report['completed_iterations'],8)
            restored=torch.load(target/'resume_seed.pth',map_location='cpu',weights_only=False)
            self.assertTrue(audit_optimizer(state['trainer']['_trainer']['optimizer'],restored['trainer']['_trainer']['optimizer'])['valid'])


if __name__=='__main__':
    unittest.main()

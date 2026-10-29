import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

source=Path(__file__).resolve().parents[1]/'rgbd_seg/m2f/resume_eval.py'
spec=importlib.util.spec_from_file_location('resume_eval',source)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ResumeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.output=Path(self.temp.name)
        self.cfg=NS(OUTPUT_DIR=str(self.output),TEST=NS(EVAL_PERIOD=1804),
                    PROJECT=NS(EVAL_LIMIT=0,SKIP_FINAL_EVAL=False))
    def tearDown(self): self.temp.cleanup()
    def test_normal_resume_unchanged(self):
        self.assertFalse(module.needs_expansion_evaluation(self.cfg,1804))
    def test_non_evaluation_checkpoint_does_not_add_evaluation(self):
        (self.output/'gpu_expansion.json').write_text('{}')
        self.assertFalse(module.needs_expansion_evaluation(self.cfg,902))
    def test_interrupted_boundary_is_evaluated(self):
        (self.output/'gpu_expansion.json').write_text('{}')
        self.assertTrue(module.needs_expansion_evaluation(self.cfg,1804))
    def test_completed_boundary_is_not_repeated(self):
        (self.output/'gpu_expansion.json').write_text('{}')
        (self.output/'latest_eval_metrics.json').write_text(json.dumps({'iteration':1803}))
        self.assertFalse(module.needs_expansion_evaluation(self.cfg,1804))
    def test_smoke_does_not_trigger_full_evaluation(self):
        (self.output/'gpu_expansion.json').write_text('{}')
        self.cfg.PROJECT.EVAL_LIMIT=12
        self.assertFalse(module.needs_expansion_evaluation(self.cfg,1804))


if __name__=='__main__': unittest.main()

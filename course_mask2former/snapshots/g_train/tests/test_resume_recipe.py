"""A checkpoint must not silently resume with different optimization switches."""
from pathlib import Path
import sys
import unittest

from mmengine.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_coco_experiment import verify_resume_recipe


class ResumeRecipeTests(unittest.TestCase):
    def config(self):
        return Config(dict(model=dict(panoptic_head=dict(type='WeightedBoundaryMask2FormerHead',
                                                         small_object_loss=dict(enabled=True))),
                           train_dataloader=dict(batch_size=2), val_dataloader=dict(batch_size=1),
                           optim_wrapper=dict(optimizer=dict(lr=1e-5)), param_scheduler=[],
                           randomness=dict(seed=20260922),
                           experiment_provenance=dict(dataset_manifest_sha256='data',
                                                      implementation_sha256={'head': 'code'}),
                           train_cfg=dict(max_iters=100), work_dir='first'))

    def test_total_iteration_and_output_may_change(self):
        previous = self.config()
        current = self.config()
        current.train_cfg.max_iters = 120
        current.work_dir = 'resumed'
        verify_resume_recipe(previous.pretty_text, current)

    def test_switch_data_and_code_changes_are_rejected(self):
        previous = self.config()
        for key in ('model', 'train_dataloader', 'randomness', 'experiment_provenance'):
            with self.subTest(key=key):
                current = self.config()
                current[key] = dict(changed=True)
                with self.assertRaisesRegex(ValueError, 'Resume recipe differs'):
                    verify_resume_recipe(previous.pretty_text, current)

    def test_missing_original_recipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lacks its saved config'):
            verify_resume_recipe(None, self.config())


if __name__ == '__main__':
    unittest.main()

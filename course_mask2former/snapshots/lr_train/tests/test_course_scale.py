"""Course scale/crop configuration invariants in the pinned MMDetection env."""
import copy
import unittest
import numpy as np

from run_course_experiment import parser, build_config
from run_course_workflow import variants
from analyze_course_baseline import boundary_f1


class CourseScaleTests(unittest.TestCase):
    def config(self, scale=False, crop=False):
        args = parser().parse_args(['train', '--output', '/tmp/course_config_only',
                                   '--multiscale-depth', '--reliability-gate'])
        args.course_large_scale_jitter = scale
        args.small_object_crop = crop
        return build_config(args, [dict(id=1, name='component', supercategory='component')])

    def test_defaults_and_scale_only_changes_training_resize(self):
        base = self.config().to_dict()
        enabled = self.config(True).to_dict()
        self.assertEqual(base['train_dataloader']['dataset']['pipeline'][3]['ratio_range'], (0.8, 1.25))
        self.assertEqual(enabled['train_dataloader']['dataset']['pipeline'][3]['ratio_range'], (0.8, 2.0))
        enabled['train_dataloader']['dataset']['pipeline'][3]['ratio_range'] = (0.8, 1.25)
        self.assertEqual(enabled, base)

    def test_crop_and_scale_independent_and_no_validation_change(self):
        base = self.config()
        for scale in (False, True):
            for crop in (False, True):
                cfg = self.config(scale, crop)
                for key in ('model','val_dataloader','test_dataloader','val_evaluator',
                            'test_evaluator','optim_wrapper','param_scheduler','randomness','train_cfg'):
                    self.assertEqual(cfg[key], base[key], key)
                transforms = cfg.train_dataloader.dataset.pipeline
                self.assertEqual(transforms[4]['type'], 'RGBDSmallObjectAwareCrop' if crop else 'RGBDRandomCrop')
                self.assertEqual(transforms[3]['ratio_range'], (0.8, 2.0) if scale else (0.8, 1.25))
                self.assertEqual(transforms[4]['crop_size'], (1024, 1024))
                self.assertFalse(cfg.model.panoptic_head.ignore_void)

    def test_two_arms_are_incremental_not_loss_combinations(self):
        arms = variants('scale_crop')
        self.assertEqual(len(arms), 2)
        a,b = [copy.deepcopy(flags) for _,flags in arms]
        self.assertTrue(b.pop('small_object_crop'))
        self.assertEqual(a,b)
        self.assertNotIn('small_object_loss', a)
        self.assertNotIn('course_foreground_points', a)

    def test_boundary_diagnostic(self):
        a = np.zeros((32,32), dtype=bool)
        a[8:20,8:20] = True
        self.assertEqual(boundary_f1(a,a), 1.)
        self.assertEqual(boundary_f1(a,np.zeros_like(a)), 0.)
        self.assertLess(boundary_f1(a,np.roll(a,8,axis=1)), 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)

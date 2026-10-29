"""Config isolation, original coordinates and fail-closed training acceptance."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run_course_experiment as old
from train_course_directions import (parser, build_config, finite_tree, initial_state,
                                     CourseDirectionsAuditHook, SIX_METRICS, verify_reference_recipe)


class DirectionsTrainingTests(unittest.TestCase):
    def args(self, flags=()):
        return parser().parse_args(['train', '--output', '/tmp/directions_config_only',
            '--multiscale-depth', '--reliability-gate', *flags])

    def config(self, flags=()):
        return build_config(self.args(flags), [dict(id=1,name='component',supercategory='component')]).to_dict()

    def test_all_off_preserves_existing_recipe_exactly_except_audit_hook(self):
        a = self.args()
        actual = build_config(a, [dict(id=1,name='component',supercategory='component')]).to_dict()
        expected = old.build_config(a, [dict(id=1,name='component',supercategory='component')]).to_dict()
        self.assertEqual(actual['custom_hooks'].pop(), dict(type='CourseDirectionsAuditHook'))
        self.assertEqual(actual, expected)

    def test_default_switches_off_and_fixed_schedule(self):
        a = self.args()
        for name in ('course_detail','course_depth_reliability','course_copy_paste','course_high_resolution'):
            self.assertFalse(getattr(a,name))
        self.assertEqual(self.config()['param_scheduler'][0]['milestones'], [18000,24000])

    def test_accepted_frozen_30k_shared_config_has_no_drift(self):
        args=self.args()
        categories=[dict(id=1,name='component',supercategory='component')]
        result=verify_reference_recipe(args,categories,build_config(args,categories))
        self.assertTrue(result['shared_base_matches'])
        self.assertEqual(result['differences'],[])

    def test_detail_changes_only_model_import_optimizer_key(self):
        base, treatment = self.config(), self.config(['--course-detail'])
        self.assertEqual(treatment['model'].pop('course_detail'), {'enabled':True})
        self.assertEqual(treatment['model'].pop('course_depth_reliability'), {'enabled':False})
        self.assertEqual(treatment['model']['type'], 'CourseFeatureRGBDMask2Former')
        treatment['model']['type'] = base['model']['type']
        treatment['custom_imports']['imports'].remove('rgbd.course_feature_models')
        self.assertEqual(treatment['optim_wrapper']['paramwise_cfg']['custom_keys'].pop('course_detail_adapter'),
                         dict(lr_mult=10.,decay_mult=1.))
        self.assertEqual(treatment,base)

    def test_reliability_parameter_free_config_only_model(self):
        base,treatment=self.config(),self.config(['--course-depth-reliability'])
        self.assertEqual(treatment['model'].pop('course_detail'), {'enabled':False})
        self.assertEqual(treatment['model'].pop('course_depth_reliability'), {'enabled':True})
        treatment['model']['type']=base['model']['type']
        treatment['custom_imports']['imports'].remove('rgbd.course_feature_models')
        self.assertEqual(treatment,base)

    def test_copy_paste_train_only_before_geometry(self):
        base,treatment=self.config(),self.config(['--course-copy-paste'])
        ds=treatment['train_dataloader']['dataset']
        transform=ds['pipeline'].pop(2)
        self.assertEqual(transform['type'],'CourseRGBDSmallCopyPaste')
        self.assertEqual(transform['ann_file'],ds['ann_file'])
        self.assertEqual(transform['data_root'],ds['data_root'])
        self.assertIn('course_copy_paste_activity',ds['pipeline'][-1].pop('meta_keys'))
        treatment['custom_imports']['imports'].remove('rgbd.course_copy_paste')
        self.assertEqual(treatment,base)

    def test_1536_changes_both_training_geometry_and_validation_not_scheduler(self):
        base,treatment=self.config(),self.config(['--course-high-resolution'])
        self.assertEqual(treatment['model']['data_preprocessor']['batch_augments'][0]['size'],(1536,1536))
        self.assertEqual(treatment['train_dataloader']['dataset']['pipeline'][3]['scale'],(1536,1536))
        self.assertEqual(treatment['train_dataloader']['dataset']['pipeline'][4]['crop_size'],(1536,1536))
        self.assertEqual(treatment['val_dataloader']['dataset']['pipeline'][1]['scale'],(2559,1536))
        self.assertEqual(treatment['val_evaluator'],base['val_evaluator'])
        self.assertEqual(treatment['param_scheduler'],base['param_scheduler'])
        self.assertEqual(treatment['train_dataloader']['batch_size'],2)
        self.assertFalse(treatment['resume'])

    def test_bad_legacy_flag_recipe_and_smoke_rejected(self):
        for flags in (['--small-object-loss'],['--image-size','1536'],['--iters','10000'],
                      ['--functional-smoke'],['--seed','7']):
            with self.assertRaises(ValueError): self.config(flags)
        self.config(['--functional-smoke','--iters','20','--val-interval','20'])

    def test_independent_switch_combinations_are_supported_not_auto_scheduled(self):
        cfg=self.config(['--course-detail','--course-depth-reliability','--course-copy-paste','--course-high-resolution'])
        self.assertTrue(cfg['model']['course_detail']['enabled'])
        self.assertTrue(cfg['model']['course_depth_reliability']['enabled'])
        self.assertEqual(cfg['train_dataloader']['dataset']['pipeline'][2]['type'],'CourseRGBDSmallCopyPaste')

    def test_finite_tree_checks_nested_optimizer(self):
        self.assertTrue(finite_tree({'a':[torch.ones(1),2,None]}))
        self.assertFalse(finite_tree({'state':{1:{'exp_avg':torch.tensor(float('nan'))}}}))
        self.assertFalse(finite_tree({'a':float('inf')}))

    def test_initialization_rejects_extra_or_missing_keys(self):
        model=torch.nn.Module()
        model.spatial_fusion=torch.nn.Linear(1,1,bias=False)
        model.course_detail_adapter=torch.nn.Conv2d(1,1,1,bias=False)
        torch.nn.init.zeros_(model.course_detail_adapter.weight)
        with self.assertRaises(ValueError): initial_state(model,{},False)
        initial=initial_state(model,{},True)
        self.assertIn('course_detail_adapter.weight',initial)
        with self.assertRaises(ValueError): initial_state(model,{'invented':torch.ones(1)},True)
        torch.nn.init.ones_(model.course_detail_adapter.weight)
        with self.assertRaises(ValueError): initial_state(model,{},True)

    def test_coverage_rejects_missing_duplicates_and_wrong_order(self):
        hook=CourseDirectionsAuditHook();hook.expected_ids=[1,2]
        runner=SimpleNamespace(iter=5000,work_dir='/tmp/no_write')
        metrics={k:.5 for k in SIX_METRICS}
        for ids in ([1],[2,1],[1,1]):
            hook.val_ids=ids
            with self.assertRaises(RuntimeError): hook.after_val_epoch(runner,metrics)
        hook.val_ids=[1,2]
        with patch('train_course_directions.atomic_json'):
            hook.after_val_epoch(runner,metrics)
        self.assertEqual(hook.coverage,[dict(iteration=5000,completed_images=2,unique_images=2)])
        metrics[SIX_METRICS[0]]=float('nan')
        with self.assertRaises(RuntimeError): hook.after_val_epoch(runner,metrics)


if __name__=='__main__':
    unittest.main(verbosity=2)

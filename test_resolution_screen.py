import copy
from collections import OrderedDict
import unittest
import torch
import yaml
from detectron2.modeling import build_model
from course.common import configuration
from course.experiments import mapper
from course.resolution_screen import validate_resolution_pair,assert_same_initial_state,paired_scores


class ResolutionScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def config(self,name):return yaml.safe_load(configuration(f'configs/screen/{name}.yaml').dump())

    def test_only_training_size_and_shared_eval_size_change(self):
        ref,old=self.config('q00_p2_eval1280'),self.config('j01_p2_queries')
        candidate=self.config('q01_p2_train1280');d=validate_resolution_pair(candidate,ref,old)
        self.assertEqual(set(d['training_difference']),{'INPUT.IMAGE_SIZE'})
        self.assertEqual(set(d['common_inference_difference_from_old_report']),{'COURSE.TEST_SIZE'})
        for key in ('DATASETS','SEED','SOLVER','DATALOADER'):self.assertEqual(candidate[key],old[key])

    def test_reject_loss_query_or_mixed_eval_changes(self):
        ref,old=self.config('q00_p2_eval1280'),self.config('j01_p2_queries')
        for section,key,value in [('COURSE','SMALL_LOSS_WEIGHT',2.),('COURSE','TEST_SIZE',1024),('SOLVER','IMS_PER_BATCH',1)]:
            candidate=self.config('q01_p2_train1280');candidate[section][key]=value
            with self.assertRaises((ValueError,AssertionError)):validate_resolution_pair(candidate,ref,old)
        candidate=self.config('q01_p2_train1280');candidate['COURSE']['QUERY']['SPATIAL_SIGMA']=.05
        with self.assertRaises(AssertionError):validate_resolution_pair(candidate,ref,old)

    def test_initial_state_includes_rng_preserved_p2_and_metadata(self):
        torch.manual_seed(42);a=build_model(configuration('configs/screen/q00_p2_eval1280.yaml',['MODEL.DEVICE','cpu']));rng=torch.get_rng_state()
        torch.manual_seed(42);b=build_model(configuration('configs/screen/q01_p2_train1280.yaml',['MODEL.DEVICE','cpu']))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertGreater(assert_same_initial_state(a.state_dict(),b.state_dict()),0)
        plain=OrderedDict(b.state_dict())
        with self.assertRaises(AssertionError):assert_same_initial_state(a.state_dict(),plain)
        with torch.no_grad():next(b.parameters()).add_(1)
        with self.assertRaises(AssertionError):assert_same_initial_state(a.state_dict(),b.state_dict())

    def test_pairing_cannot_credit_inference_upscale(self):
        a=[dict(additional_training_step=s,input_size=1280,panel_sha256='fixed',validation_images=200,mask_aps=.5 if s==0 else .52) for s in (0,1000,2000,3000)]
        b=copy.deepcopy(a)
        for row in b[1:]:row['mask_aps']=.51
        self.assertAlmostEqual(paired_scores(a,b)['3000'],.01)
        b[-1]['input_size']=1024
        with self.assertRaises(AssertionError):paired_scores(a,b)

    def test_mapper_receives_actual_requested_train_scale(self):
        cfg=configuration('configs/screen/q01_p2_train1280.yaml')
        m=mapper(cfg,True)
        self.assertEqual(m.cfg.INPUT.IMAGE_SIZE,1280)
        self.assertEqual(cfg.COURSE.TEST_SIZE,1280)
        self.assertEqual(cfg.COURSE.SMALL_LOSS_WEIGHT,1.)
        self.assertFalse(cfg.COURSE.V2.AREA_WEIGHT)
        self.assertTrue(all('1280' in str(a) for a in m.augmentations.augs if a.__class__.__name__ in ('ResizeScale','FixedSizeCrop')))


if __name__=='__main__':unittest.main()

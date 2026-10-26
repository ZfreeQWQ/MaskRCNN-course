import copy
import unittest
from unittest.mock import patch

import torch
import yaml
from detectron2.modeling import build_model
from detectron2.projects.point_rend.point_features import point_sample
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.modeling.criterion import SetCriterion
from course.common import configuration
from course.screen_protocol import compare_protocol
from course.coverage_screen import (positive_anchors, shared_match_points, matched_loss_points,
                                    CoverageMatcher, CoverageCriterion)


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def targets(self):
        masks=torch.zeros(3,32,32)
        masks[0,3,7]=1
        masks[1,9:12,12:15]=1
        masks[2,1:31,1:31]=1
        small=torch.tensor([True,True,False])
        anchors,valid=positive_anchors(masks,small,4)
        return dict(masks=masks,labels=torch.zeros(3,dtype=torch.int64),small=small,
                    coverage_points=anchors,coverage_valid=valid)

    def outputs(self):
        return dict(pred_masks=torch.randn(1,5,8,8,requires_grad=True),pred_logits=torch.randn(1,5,2))

    def test_anchor_hits_one_pixel_and_never_consumes_rng(self):
        torch.manual_seed(4);rng=torch.get_rng_state()
        t=self.targets()
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        labels=point_sample(t['masks'][:,None],t['coverage_points'],align_corners=False).squeeze(1)
        self.assertTrue(torch.equal(labels[:2],torch.ones_like(labels[:2])))
        self.assertEqual(t['coverage_valid'].tolist(),[True,True,False])
        anchors,valid=positive_anchors(torch.zeros(0,32,32),torch.zeros(0,dtype=torch.bool),4)
        self.assertEqual(anchors.shape,(0,4,2));self.assertEqual(len(valid),0)

    def test_shared_cap_round_robin_and_unchanged_uniform_prefix(self):
        t=self.targets();coords=torch.rand(1,32,2)
        actual,k=shared_match_points(coords,t['coverage_points'],t['coverage_valid'],3)
        self.assertEqual(k,3)
        torch.testing.assert_close(actual[:,:-3],coords[:,:-3],rtol=0,atol=0)
        torch.testing.assert_close(actual[0,-3:],torch.stack([t['coverage_points'][0,0],
            t['coverage_points'][1,0],t['coverage_points'][0,1]]),rtol=0,atol=0)
        empty,zero=shared_match_points(coords,t['coverage_points'],torch.zeros(3,dtype=torch.bool),256)
        self.assertIs(empty,coords);self.assertEqual(zero,0)

    def test_loss_only_random_tail_small_rows_and_rng(self):
        t=self.targets();coords=torch.rand(3,32,2);rng=torch.get_rng_state()
        actual,k=matched_loss_points(coords,t['coverage_points'],t['coverage_valid'])
        self.assertEqual(k,8);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(actual[:,:-4],coords[:,:-4],rtol=0,atol=0)
        torch.testing.assert_close(actual[2],coords[2],rtol=0,atol=0)
        labels=point_sample(t['masks'][:,None],actual,align_corners=False).squeeze(1)
        self.assertTrue(torch.equal(labels[:2,-4:],torch.ones(2,4)))

    def test_matcher_off_exact_original_assignment_and_rng(self):
        output=self.outputs();target=[self.targets()]
        base=HungarianMatcher(2,5,5,32)
        changed=CoverageMatcher(2,5,5,32,match_points=0)
        torch.manual_seed(9);a=base(output,target);rng=torch.get_rng_state()
        torch.manual_seed(9);b=changed(output,target)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for x,y in zip(a[0],b[0]):self.assertTrue(torch.equal(x,y))

    def test_matcher_uses_same_coordinate_set_for_all_queries_and_gts(self):
        output=self.outputs();t=self.targets();seen=[]
        def audited(masks,coords,**kw):
            seen.append(coords.detach().clone())
            return point_sample(masks,coords,**kw)
        model=CoverageMatcher(2,5,5,32,match_points=4)
        torch.manual_seed(10);HungarianMatcher(2,5,5,32)(output,[t]);rng=torch.get_rng_state()
        torch.manual_seed(10)
        with patch('course.coverage_screen.point_sample',side_effect=audited):
            assignment=model(output,[t])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(len(assignment[0][0]),3);self.assertEqual(len(seen),2)
        for coords in seen:
            for row in coords:torch.testing.assert_close(row,seen[0][0],rtol=0,atol=0)
        self.assertEqual(model.last_diagnostics['replaced_points_across_images'],8)
        empty={k:v[:0] for k,v in t.items()}
        self.assertEqual(len(model(output,[empty])[0][0]),0)

    def criterion(self,kind,points=0):
        kwargs=dict(num_classes=1,matcher=HungarianMatcher(2,5,5,32),weight_dict={},eos_coef=.1,
                    losses=['labels','masks'],num_points=32,oversample_ratio=3,importance_sample_ratio=.75)
        return kind(**kwargs,**(dict(loss_points=points) if kind is CoverageCriterion else {}))

    def test_loss_off_exact_original_losses_gradients_and_rng(self):
        out=self.outputs();target=[self.targets()];indices=[(torch.tensor([0,1,2]),torch.tensor([0,1,2]))]
        gradients=[]
        for kind in (SetCriterion,CoverageCriterion):
            src=out['pred_masks'].detach().clone().requires_grad_();o=dict(pred_masks=src)
            torch.manual_seed(15);loss=self.criterion(kind).loss_masks(o,target,indices,3)
            rng=torch.get_rng_state();sum(loss.values()).backward()
            gradients.append((loss,src.grad,rng))
        for key in gradients[0][0]:torch.testing.assert_close(gradients[0][0][key],gradients[1][0][key],rtol=0,atol=0)
        torch.testing.assert_close(gradients[0][1],gradients[1][1],rtol=0,atol=0)
        self.assertTrue(torch.equal(gradients[0][2],gradients[1][2]))

    def test_loss_floor_finite_gradients_same_rng_and_empty_targets(self):
        t=self.targets();ids=[(torch.tensor([0,1,2]),torch.tensor([0,1,2]))]
        out=self.outputs()
        torch.manual_seed(16);self.criterion(SetCriterion).loss_masks(out,[t],ids,3);rng=torch.get_rng_state()
        torch.manual_seed(16);criterion=self.criterion(CoverageCriterion,4)
        loss=criterion.loss_masks(out,[t],ids,3)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));sum(loss.values()).backward()
        self.assertTrue(torch.isfinite(out['pred_masks'].grad).all())
        self.assertGreater(float(out['pred_masks'].grad.abs().sum()),0)
        self.assertEqual(criterion.last_diagnostics['replaced_points_across_matches'],8)
        empty={k:v[:0] for k,v in t.items()}
        losses=criterion.loss_masks(out,[empty],[(torch.zeros(0,dtype=torch.int64),torch.zeros(0,dtype=torch.int64))],1)
        self.assertEqual(float(sum(losses.values()).detach()),0)

    def test_models_strict_shared_init_rng_and_narrow_protocol_changes(self):
        torch.manual_seed(42)
        ref=build_model(configuration('configs/screen/j01_p2_queries.yaml',['MODEL.DEVICE','cpu']))
        rng=torch.get_rng_state();state=ref.state_dict()
        reference=yaml.safe_load(configuration('configs/screen/j01_p2_queries.yaml').dump())
        del reference['COURSE']['COVERAGE']  # old immutable protocol semantics
        for name,key in [('n01_p2_match_floor4','MATCH_POINTS'),('n02_p2_loss_floor4','LOSS_POINTS')]:
            cfg=configuration(f'configs/screen/{name}.yaml',['MODEL.DEVICE','cpu'])
            torch.manual_seed(42);model=build_model(cfg)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            model.load_state_dict(state,strict=True)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
            actual=yaml.safe_load(configuration(f'configs/screen/{name}.yaml').dump())
            allowed=['MODEL.META_ARCHITECTURE','COURSE.COVERAGE.'+key]
            with self.assertRaises(ValueError):compare_protocol(actual,reference)
            self.assertEqual(set(compare_protocol(actual,reference,allowed)),set(allowed))
            invalid=copy.deepcopy(actual);invalid['COURSE']['COVERAGE']['MAX_MATCH_REPLACEMENTS']=512
            with self.assertRaises(ValueError):compare_protocol(invalid,reference,allowed)


if __name__=='__main__':unittest.main()

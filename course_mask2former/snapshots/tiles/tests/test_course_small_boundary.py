"""Real assignment/loss regression for the bounded small-boundary treatment."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
import test_course_head as base
from rgbd.small_boundary_head import (CourseSmallBoundaryMask2FormerHead,
                                      checked_small_boundary, empty_counters)
from run_course_experiment import parser, build_config, CourseCurveHook
from run_course_workflow import variants, formal_protocol, validate_gpus
from launch_nyu_sweep import docker_command


class CourseSmallBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.CourseHeadTests.setUpClass()
        cls.course = base.CourseHeadTests()
        cls.fixture = cls.course.fixture

    def head(self, enabled=True):
        head = self.course.head()
        head.__class__ = CourseSmallBoundaryMask2FormerHead
        head.small_boundary_loss = checked_small_boundary(dict(enabled=enabled, area_threshold=64.))
        head.small_boundary_counters = empty_counters()
        return head

    def test_disabled_exact_loss_and_rng(self):
        for counts in ((2,1),(0,0),(0,2)):
            inputs = self.fixture.inputs(counts=counts)
            reference = self.fixture.losses(self.course.head(), inputs)
            rng = torch.random.get_rng_state()
            actual = self.fixture.losses(self.head(False), inputs)
            self.assertEqual(reference.keys(), actual.keys())
            self.fixture.assert_base_equal(actual, reference)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_enabled_preserves_original_losses_rng_and_one_assignment_per_layer(self):
        inputs = self.fixture.inputs(layers=3)
        reference = self.fixture.losses(self.course.head(), inputs)
        rng = torch.random.get_rng_state()
        head = self.head()
        with patch.object(head, 'get_targets', wraps=head.get_targets) as targets, \
             patch.object(head, '_boundary_region_loss', wraps=head._boundary_region_loss) as auxiliary:
            actual = self.fixture.losses(head, inputs)
        self.assertEqual(set(actual)-set(reference), {'loss_small_boundary'})
        self.fixture.assert_base_equal(actual, reference)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(targets.call_count, 3)
        self.assertEqual(auxiliary.call_count, 1)
        self.assertEqual(head.small_boundary_counters['calls'], 1)
        self.assertEqual(head.small_boundary_counters['area_eligible'], 2)
        self.assertGreater(actual['loss_small_boundary'].item(), 0)
        self.fixture.assert_finite_backward(actual, inputs)

    def test_selection_skip_divisor_and_gradient(self):
        head = self.head()
        target = torch.zeros(4,32,32, device=self.fixture.device)
        target[0,4:8,4:8] = 1  # eligible and resolved
        target[1,7,7] = 1       # eligible but unresolved
        target[2,12:28,12:28] = 1  # too large
        logits = torch.zeros(4,8,8, device=self.fixture.device, requires_grad=True)
        actual = head._boundary_region_loss(logits, target, 4)
        reference = self.head()._boundary_region_loss(logits[:1], target[:1], 1)
        self.assertTrue(torch.equal(actual, reference))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0,1,1].item(),0)
        self.assertGreater(logits.grad[0,0,1].item(),0)
        self.assertEqual(logits.grad[1:].abs().sum().item(),0)
        self.assertEqual(head.small_boundary_counters, dict(calls=0,area_eligible=2,
                         resolvable=1,skipped_subpixel=1,with_boundary=1,nonzero_calls=1))

    def test_empty_and_mixed_finite_backward(self):
        for counts in ((0,0),(0,2)):
            inputs = self.fixture.inputs(counts=counts)
            head = self.head()
            losses = self.fixture.losses(head, inputs)
            self.fixture.assert_finite_backward(losses, inputs)
            self.assertEqual(head.small_boundary_counters['calls'], 1)
            if counts == (0,0):
                self.assertEqual(losses['loss_small_boundary'].item(),0)

    def test_no_selected_and_subpixel_connected_zero(self):
        for mode in ('large','tiny','empty'):
            target = torch.zeros(1,32,32,device=self.fixture.device)
            if mode == 'large': target[0,4:28,4:28] = 1
            if mode == 'tiny': target[0,7,7] = 1
            logits = torch.zeros(1,8,8,device=self.fixture.device,requires_grad=True)
            loss = self.head()._boundary_region_loss(logits,target,1)
            self.assertEqual(loss.item(),0)
            loss.backward()
            self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_exact_threshold_and_extreme_finite(self):
        target = torch.zeros(2,32,32,device=self.fixture.device)
        target[0,4:12,4:12] = 1  # exactly64
        target[1] = target[0]; target[1,20,20] = 1  #65
        logits = torch.full((2,8,8),1000.,device=self.fixture.device,requires_grad=True)
        head = self.head()
        loss = head._boundary_region_loss(logits,target,2)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0].abs().sum().item(),0)
        self.assertEqual(logits.grad[1].abs().sum().item(),0)
        self.assertEqual(head.small_boundary_counters['area_eligible'],1)

    def test_defaults_validation(self):
        self.assertFalse(checked_small_boundary()['enabled'])
        for options in (dict(enabled=1),dict(area_threshold=0),dict(loss_weight=float('nan')),
                        dict(radius=True),dict(min_target_peak=1.1),dict(unrecognized=True)):
            with self.assertRaises(ValueError): checked_small_boundary(options)

    def config(self, flags=()):
        args = parser().parse_args(['train','--output','/tmp/config_only',
            '--multiscale-depth','--reliability-gate','--iters','30000',
            '--val-interval','5000','--keep-checkpoints','6',*flags])
        return build_config(args,[dict(id=1,name='component',supercategory='component')])

    def test_only_config_difference_is_training_auxiliary(self):
        control = self.config().to_dict()
        treatment = self.config(['--course-small-boundary']).to_dict()
        self.assertEqual(treatment['model']['panoptic_head'].pop('small_boundary_loss'),
                         checked_small_boundary(dict(enabled=True)))
        self.assertEqual(treatment['model']['panoptic_head']['type'],'CourseSmallBoundaryMask2FormerHead')
        treatment['model']['panoptic_head']['type'] = control['model']['panoptic_head']['type']
        treatment['custom_imports']['imports'].remove('rgbd.small_boundary_head')
        self.assertEqual(control,treatment)
        with self.assertRaises(ValueError): self.config(['--course-small-boundary','--boundary-loss'])

    def test_single_treatment_same_budget_and_command(self):
        jobs=variants('small_boundary')
        self.assertEqual(jobs,[('gated_small_boundary_long',dict(multiscale_depth=True,
                              reliability_gate=True,course_small_boundary=True))])
        protocol=formal_protocol('small_boundary')
        control=formal_protocol('budget_probe')
        self.assertEqual({k:v for k,v in protocol.items() if k!='stage'},
                         {k:v for k,v in control.items() if k!='stage'})
        with self.assertRaises(ValueError): validate_gpus('small_boundary',[0,1],{})
        plan=dict(workspace='/mnt/nfs/fenghaibin/AI_Class',container_output_root='/workspace/outputs/example',
                  container_name_prefix='test',image='test-image',runner_script='run_course_experiment.py',
                  recipe=dict(iters=30000,val_interval=5000,keep_checkpoints=6,schedule='decay'))
        name,flags=jobs[0]
        _,cmd=docker_command(plan,dict(id=name,seed=20261013,output_rel='runs/'+name,flags=flags),0)
        self.assertIn('--course-small-boundary',cmd)
        self.assertNotIn('--boundary-loss',cmd)
        self.assertEqual(cmd[cmd.index('--iters')+1],'30000')

    def test_activity_hook_only_selected_steps_and_no_control_writes(self):
        runner=SimpleNamespace(model=SimpleNamespace(panoptic_head=SimpleNamespace()),
                               iter=99,max_iters=30000,work_dir='/tmp/not_written')
        hook=CourseCurveHook()
        with patch('run_course_experiment.atomic_json') as write:
            hook.after_train_iter(runner,99)
            write.assert_not_called()
            runner.model.panoptic_head.small_boundary_counters=empty_counters()
            hook.after_train_iter(runner,99)
            self.assertEqual(write.call_args.args[1]['iteration'],100)
            write.reset_mock();runner.iter=100
            hook.after_train_iter(runner,100)
            write.assert_not_called()
            runner.iter=19;runner.max_iters=20
            hook.after_train_iter(runner,19)
            self.assertEqual(write.call_args.args[1]['iteration'],20)


if __name__ == '__main__':
    unittest.main(verbosity=2)

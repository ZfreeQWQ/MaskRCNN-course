"""Synthetic fixtures only: malformed evidence must never enter real results."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_nyu_sweep_summary as old
import summarize_nyu_v2 as subject
from nyu_v2_plan import experiment_jobs
from launch_nyu_sweep import docker_command


class V2SummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = old.SweepSummaryTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan = self.fixture.plan
        self.plan.update(schema_version=2, jobs=experiment_jobs(self.plan['seeds']))
        self.plan['recipe'].update(val_interval=1000, branch_lr_mult=10., foreground_quota=64)
        self.fixture.save_plan()

    def report(self, index=0):
        r = self.fixture.report(index)
        args = r['args']
        r.update(initial_spatial_sha256='f' * 64, active_parameters=44004873,
                 spatial_parameters=400000, spatial_projection_changed=args['multiscale_depth'],
                 spatial_encoder_changed=args['multiscale_depth'], spatial_gate_changed=args['reliability_gate'],
                 learning_curve=[dict(iteration=1000, metrics={k: r['metrics']['nyuv2_void/segm_' + k]
                                      for k in ('mAP', 'mAP_s')})])
        return r

    def write(self, index=0, report=None):
        return self.fixture.write_report(index, self.report(index) if report is None else report)

    def test_complete_and_paired_contrasts(self):
        for index in range(24):
            self.write(index)
        result = subject.summarize(self.root)
        self.assertEqual(result['counts']['passed'], 24)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(all(r['delta_AP']['n'] == 3 for r in result['contrasts']))
        self.assertIn('学习曲线', subject.markdown(result))

    def test_empty_partial_missing_baseline_not_zero_gain(self):
        self.assertEqual(subject.summarize(self.root)['counts']['missing'], 24)
        self.write(3)
        result = subject.summarize(self.root)
        self.assertIsNone(result['combinations'][3]['delta_AP']['mean'])
        self.assertEqual(result['counts']['passed'], 1)

    def test_changed_switch_budget_curve_and_updates_are_invalid(self):
        mutations = [lambda r: r['args'].update(foreground_points=True),
                     lambda r: r['args'].update(branch_lr_mult=20),
                     lambda r: r.update(spatial_gate_changed=True),
                     lambda r: r.update(learning_curve=[]),
                     lambda r: r['learning_curve'][0].update(iteration=999),
                     lambda r: r['learning_curve'][0]['metrics'].update(mAP_s=.99)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                report = self.report()
                mutate(report)
                self.write(report=report)
                result = subject.summarize(self.root)
                self.assertEqual(result['invalid_count'], 1)
                self.assertEqual(result['validated_runs'], [])

    def test_paired_initialization_and_shared_args_are_checked(self):
        self.write(0)
        for key in ('initial_spatial_sha256', 'dataset_manifest_sha256'):
            report = self.report(1)
            report[key] = '9' * 64
            self.write(1, report)
            self.assertEqual(subject.summarize(self.root)['invalid_count'], 2)

    def test_failed_running_malformed_excluded(self):
        self.write(0, dict(status='failed', error='test fixture'))
        self.write(1, dict(status='running'))
        self.write(2).write_text('{')
        self.assertEqual(subject.summarize(self.root)['counts'],
                         dict(passed=0, missing=21, running=1, failed=1, invalid=1))

    def test_plan_must_match_predeclared_design(self):
        for change in (lambda p: p['jobs'].pop(), lambda p: p['jobs'][0].update(output_rel='../escape'),
                       lambda p: p['jobs'][0]['flags'].update(foreground_points=True)):
            plan = copy.deepcopy(self.plan)
            change(plan)
            with self.assertRaises(ValueError):
                subject.validate_plan(plan)

    def test_command_selects_v2_and_all_recipe_flags(self):
        self.plan.update(runner_script='run_nyu_v2.py', image='pinned-image', container_name_prefix='test-v2')
        _, cmd = docker_command(self.plan, self.plan['jobs'][7], 1)
        self.assertIn('/workspace/outputs/sweep/source/run_nyu_v2.py', cmd)
        self.assertIn('--rgb-control', cmd)
        self.assertIn('--foreground-points', cmd)
        self.assertIn('--val-interval', cmd)
        self.assertNotIn('--depth-fusion', cmd)


if __name__ == '__main__':
    unittest.main()

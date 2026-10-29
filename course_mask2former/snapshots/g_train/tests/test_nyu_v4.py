"""Standard-library plan/command/evidence tests; no synthetic scores are results."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_nyu_v2_summary as old
import summarize_nyu_v4 as subject
from launch_nyu_sweep import docker_command
from nyu_v4_plan import experiment_jobs, scheduler_config


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = old.V2SummaryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan = self.fixture.plan
        self.plan.update(schema_version=4, seeds=[20261013], jobs=experiment_jobs([20261013]))
        self.plan['recipe'].update(schedule='constant')
        self.fixture.fixture.save_plan()

    def report(self, index):
        r = self.fixture.report(index)
        r['args'].update(self.plan['jobs'][index]['recipe_overrides'])
        a = r['args']
        factor = 1. if a['schedule'] == 'constant' else .01
        r.update(declared_scheduler=scheduler_config(a['schedule'], a['iters']),
                 final_learning_rates=sorted(set(a['lr'] * x * factor for x in (.1, 1., a['branch_lr_mult']))),
                 training_policy_diagnostics=[dict(iteration=1000,
                    first_val_image_residual_rgb_l2_ratios=[.1] * 4 if a['multiscale_depth'] else None)])
        return r

    def write(self, index, report=None):
        self.fixture.write(index, self.report(index) if report is None else report)

    def test_complete_six_cells_and_matched_contrasts(self):
        for i in range(6):
            self.write(i)
        result = subject.summarize(self.root)
        self.assertEqual(result['counts']['passed'], 6)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(all(c['delta_AP']['n'] == 1 for c in result['contrasts']))
        self.assertIn(('depth_decay_1', 'rgb_decay'), [(r['on'], r['off']) for r in result['contrasts']])

    def test_overrides_not_duplicate_cli_options(self):
        self.plan.update(image='test', runner_script='run_nyu_v4.py', container_name_prefix='test')
        for job in self.plan['jobs']:
            _, cmd = docker_command(self.plan, job, 0)
            for key, value in job['recipe_overrides'].items():
                opt = '--' + key.replace('_', '-')
                self.assertEqual(cmd.count(opt), 1)
                self.assertEqual(cmd[cmd.index(opt) + 1], str(value))
            self.assertNotIn('--foreground-points', cmd)

    def test_plan_tampering_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['jobs'][1]['recipe_overrides']['schedule'] = 'constant'
        with self.assertRaises(ValueError):
            subject.validate_plan(plan)

    def test_bad_report_policy_and_diagnostics_excluded(self):
        for mutate in (lambda r: r['args'].update(schedule='constant'),
                       lambda r: r['args'].update(branch_lr_mult=20.),
                       lambda r: r.update(declared_scheduler=[]),
                       lambda r: r.update(final_learning_rates=[1e-5]),
                       lambda r: r.update(training_policy_diagnostics=[])):
            report = self.report(3)
            mutate(report)
            self.write(3, report)
            self.assertEqual(subject.summarize(self.root)['counts']['invalid'], 1)

    def test_unknown_shared_difference_is_still_rejected(self):
        self.write(0)
        r = self.report(1)
        r['args']['small_crop_prob'] = .8
        self.write(1, r)
        self.assertEqual(subject.summarize(self.root)['counts']['invalid'], 2)

    def test_empty_not_zero_and_scheduler(self):
        self.assertEqual(subject.summarize(self.root)['counts']['missing'], 6)
        self.assertEqual(scheduler_config('constant', 5000), [])
        self.assertEqual(scheduler_config('decay', 5000)[0]['milestones'], [3000, 4000])
        for mode, steps in [('bad', 5000), ('decay', 1)]:
            with self.assertRaises(ValueError):
                scheduler_config(mode, steps)


if __name__ == '__main__':
    unittest.main()

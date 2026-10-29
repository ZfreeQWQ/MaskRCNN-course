"""Stdlib synthetic evidence + scheduler tests; never dispatch real containers."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_nyu_sweep_summary as old
import summarize_nyu_v3 as summary
import run_nyu_v3_all as workflow
from launch_nyu_v3 import stage_plan, launch
from nyu_v3_plan import experiment_jobs


class V3SummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = old.SweepSummaryTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.plan = self.fixture.plan
        self.plan.update(stage_plan('screen', self.plan['seeds']))
        self.fixture.save_plan()

    def report(self, index):
        r = self.fixture.report(index)
        args = r['args']
        count = args['iters'] * args['batch_size']
        matching = 5 if args['small_matching'] else 0
        r.update(initial_spatial_sha256='f' * 64, active_parameters=44004873, spatial_parameters=397340,
            spatial_projection_changed=args['multiscale_depth'], spatial_encoder_changed=args['multiscale_depth'],
            spatial_gate_changed=args['reliability_gate'],
            advanced_depth_counters=dict(relative_training_samples=count if args['relative_depth'] else 0,
                dropout_training_samples=count if args['modality_dropout'] else 0,
                dropped_samples=count // 5 if args['modality_dropout'] else 0),
            small_matching_counters=dict(calls=matching, augmented_calls=matching,
                foreground_points=matching * int(12544 * args['matching_fraction'])),
            learning_curve=[dict(iteration=1000, metrics={key: r['metrics']['nyuv2_void/segm_' + key]
                for key in ('mAP', 'mAP_s')})])
        return r

    def write(self, index, report=None):
        return self.fixture.write_report(index, self.report(index) if report is None else report)

    def test_complete_factorial_shared_flags_and_pairs(self):
        for index in range(10):
            self.write(index)
        r = summary.summarize(self.root)
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['counts']['passed'], 10)
        self.assertTrue(all(len(x['pairs']) == 4 for x in r['switch_context_effects']))
        self.assertIn('不是独立重复', summary.markdown(r))

    def test_bad_counters_flags_and_source_are_invalid(self):
        for mutate in (lambda r: r['args'].update(relative_depth=False),
                       lambda r: r['advanced_depth_counters'].update(relative_training_samples=0),
                       lambda r: r['advanced_depth_counters'].update(dropped_samples=0),
                       lambda r: r['small_matching_counters'].update(augmented_calls=0),
                       lambda r: r['small_matching_counters'].update(foreground_points=1)):
            report = self.report(8)  # all three new switches
            mutate(report)
            self.write(8, report)
            r = summary.summarize(self.root)
            self.assertEqual(r['invalid_count'], 1)
            self.assertEqual(r['validated_runs'], [])

    def test_disabled_counters_must_be_zero(self):
        report = self.report(1)
        report['advanced_depth_counters']['dropped_samples'] = 1
        self.write(1, report)
        self.assertEqual(summary.summarize(self.root)['invalid_count'], 1)

    def test_matching_pairs_require_both_sides(self):
        self.write(8)
        r = summary.summarize(self.root)
        self.assertTrue(all(not x['pairs'] for x in r['switch_context_effects']))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Path(temp.name)
        self.root = self.workspace / 'outputs/workflow'

    def worker_files(self):
        self.root.mkdir(parents=True)
        plan = workflow.workflow_plan([1, 2, 3])
        plan.update(workspace=str(self.workspace), output=str(self.root), seeds=[1, 2, 3],
                    gpus=[0, 1], snapshot_sha256={})
        (self.root / 'workflow_plan.json').write_text(json.dumps(plan))

    def fake_launch(self, workspace, output, gpus, stage, seeds):
        path = Path(output)
        path.mkdir()
        (path / 'status.json').write_text(json.dumps(dict(status='complete', counts=dict(passed=10))))

    def test_plan_all_combinations_seeds_and_fixed_stages(self):
        plan = workflow.workflow_plan([1, 2, 3])
        self.assertEqual([len(s['jobs']) for s in plan['stages']], [10, 30])
        self.assertEqual(plan['total_jobs'], 40)
        self.assertEqual([s['recipe']['iters'] for s in plan['stages']], [20, 5000])
        self.assertEqual(len({j['id'] for j in experiment_jobs([1, 2, 3])}), 30)

    def test_busy_gpu_does_not_create_output_or_dispatch(self):
        args = argparse.Namespace(workspace=str(self.workspace), output=str(self.root), seeds=[1], gpus=[0])
        with patch.object(workflow, 'gpu_snapshot', return_value={0: dict(memory_used_mib=5000, utilization=80)}), \
                patch.object(workflow.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                workflow.start(args)
            popen.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_success_dispatches_smoke_then_confirm(self):
        self.worker_files()
        with patch.object(workflow, 'launch', side_effect=self.fake_launch) as launch_mock, \
                patch.object(workflow, 'validate_stage', return_value={'implementation': 'same'}) as verify, \
                patch.object(workflow.signal, 'signal'):
            workflow.worker(self.root)
        self.assertEqual([call.args[3] for call in launch_mock.call_args_list], ['smoke', 'confirm'])
        self.assertEqual(verify.call_args_list[1].args[3], {'implementation': 'same'})
        state = json.loads((self.root / 'workflow_status.json').read_text())
        self.assertEqual(state['status'], 'complete')

    def test_failed_smoke_never_launches_confirmation(self):
        self.worker_files()
        with patch.object(workflow, 'launch', side_effect=self.fake_launch) as launch_mock, \
                patch.object(workflow, 'validate_stage', side_effect=ValueError('invalid smoke')), \
                patch.object(workflow.signal, 'signal'):
            with self.assertRaisesRegex(ValueError, 'invalid smoke'):
                workflow.worker(self.root)
        self.assertEqual(launch_mock.call_count, 1)
        self.assertTrue((self.root / 'smoke/STOP').exists())
        self.assertEqual(json.loads((self.root / 'workflow_status.json').read_text())['status'], 'failed')

    def test_stop_before_start_never_dispatches(self):
        self.worker_files()
        (self.root / 'STOP').touch()
        with patch.object(workflow, 'launch') as launch_mock, patch.object(workflow.signal, 'signal'):
            workflow.worker(self.root)
        launch_mock.assert_not_called()
        self.assertEqual(json.loads((self.root / 'workflow_status.json').read_text())['status'], 'stopped')

    def test_replay_never_overwrites_prior_state(self):
        self.worker_files()
        path = self.root / 'workflow_status.json'
        path.write_text('{"status":"complete"}')
        with self.assertRaisesRegex(RuntimeError, 'replay'):
            workflow.worker(self.root)
        self.assertEqual(path.read_text(), '{"status":"complete"}')

    def test_launcher_snapshot_dependencies_and_entrypoint(self):
        with patch('launch_nyu_v3.prepare') as prepare:
            launch('/workspace', '/workspace/outputs/test', [0, 1], 'confirm', [1, 2, 3])
        kwargs = prepare.call_args.kwargs
        self.assertIn('run_nyu_v3.py', kwargs['extra_train_files'])
        self.assertIn('summarize_nyu_v2.py', kwargs['extra_source_files'])
        self.assertEqual(kwargs['plan_extensions']['runner_script'], 'run_nyu_v3.py')
        self.assertEqual(len(kwargs['jobs']), 30)

    def test_bad_stage_seeds_and_paths_rejected(self):
        for seeds in ([], [1, 1], [-1], [True]):
            with self.assertRaises(ValueError):
                workflow.workflow_plan(seeds)
        with self.assertRaises(ValueError):
            stage_plan('typo', [1])
        args = argparse.Namespace(workspace=str(self.workspace), output=str(self.workspace), seeds=[1], gpus=[0])
        with self.assertRaises(ValueError):
            workflow.start(args)


if __name__ == '__main__':
    unittest.main()

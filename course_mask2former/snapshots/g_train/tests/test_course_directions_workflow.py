"""Host-only fail-fast queue tests: no Torch, Docker, SSH, data, or GPU required."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import run_course_directions as queue


class DirectionsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.root = self.workspace / 'outputs' / 'new'
        self.root.mkdir(parents=True)
        (self.root / 'source').mkdir()
        (self.root / 'logs').mkdir()
        self.metrics = {key: .5 for key in queue.METRICS}
        self.provenance = dict(source_checkpoint_sha256=queue.INITIAL_SHA,
            instances_train_annotations_sha256='train', instances_val_annotations_sha256='val',
            smoke_train_annotations_sha256='smoke_train', smoke_val_annotations_sha256='smoke_val',
            instances_train_images=14432, instances_val_images=2584,
            smoke_train_images=8, smoke_val_images=8,
            smoke_missing_area_metrics=['course/segm_mAP_l'])
        source = self.root / 'source/train_course_directions.py'
        source.write_text('# test fixture\n')
        self.plan = dict(workspace=str(self.workspace), output=str(self.root),
            image_id='sha256:test', container_name_prefix='test-owned-queue',
            owner_token='owned-nonce',
            expected_provenance=self.provenance, source_sha256={source.name: queue.sha256(source)},
            cpu_tests=['test_course_directions_workflow'], max_seconds=queue.TOTAL_SECONDS,
            deadline_unix=time.time() + queue.TOTAL_SECONDS, max_gpu_wait_seconds=queue.GPU_WAIT_SECONDS,
            reference_baseline=dict(report='baseline.json', metrics=self.metrics,
                high_resolution=dict(report='resolution1536.json', metrics={k: .6 for k in queue.METRICS})))

    def fixture_report(self, identity='detail_long', smoke=False):
        phase = 'smoke' if smoke else 'formal'
        job = next(j for j in queue.stage_jobs(smoke) if j['id'] == identity)
        recipe = queue.stage_recipe(smoke)
        run = self.root / phase / 'runs' / identity
        run.mkdir(parents=True)
        endpoints = list(range(recipe['val_interval'], recipe['iters'] + 1, recipe['val_interval']))
        for iteration in endpoints:
            (run / f'iter_{iteration}.pth').write_bytes(b'owned-test-checkpoint')
        curve = [dict(iteration=i, metrics=self.metrics) for i in endpoints]
        queue.atomic_json(run / 'learning_curve.json', curve)
        prefix = 'smoke' if smoke else 'instances'
        size = 1536 if job['flags'].get('course_high_resolution') else 1024
        report = dict(status='passed', final_iteration=recipe['iters'],
            final_optimizer_steps=[recipe['iters']], test_inputs_used=False, test_annotations_used=False,
            args=dict(recipe, seed=job['seed'], **job['flags'],
                      data_root='/workspace/' + queue.DATA, checkpoint='/workspace/' + queue.CHECKPOINT),
            train_annotations_sha256=self.provenance[prefix + '_train_annotations_sha256'],
            val_annotations_sha256=self.provenance[prefix + '_val_annotations_sha256'],
            training_images=self.provenance[prefix + '_train_images'],
            validation_images=self.provenance[prefix + '_val_images'],
            source_checkpoint_sha256=queue.INITIAL_SHA, implementation_sha256=self.plan['source_sha256'],
            finite_checks=dict(model=True, optimizer=True, loss=True, grad=True),
            finite_train_iterations=recipe['iters'],
            effective_train_size=size, effective_eval_size=size,
            classifier_changed=True, depth_adapter_changed=False,
            spatial_projection_changed=bool(job['flags'].get('multiscale_depth')),
            spatial_encoder_changed=bool(job['flags'].get('multiscale_depth')),
            spatial_gate_changed=bool(job['flags'].get('reliability_gate')),
            detail_adapter_changed=bool(job['flags'].get('course_detail')),
            metrics=self.metrics, learning_curve=curve,
            validation_coverage=[dict(iteration=i, completed_images=self.provenance[prefix + '_val_images'],
                                      unique_images=self.provenance[prefix + '_val_images']) for i in endpoints],
            checkpoint=queue.container_path(run / f"iter_{recipe['iters']}.pth", self.workspace),
            checkpoint_sha256=queue.sha256(run / f"iter_{recipe['iters']}.pth"), active_parameters=44400414)
        queue.atomic_json(run / 'report.json', report)
        return phase, job, run, report

    def test_exact_six_arms_and_no_all_on(self):
        jobs = queue.jobs()
        self.assertEqual([j['id'] for j in jobs], ['rgb_long', 'depth_ungated_long', 'detail_long',
                         'reliable_depth_long', 'copy_paste_long', 'high_resolution_long'])
        self.assertEqual(jobs[0]['flags'], {})
        self.assertEqual(jobs[1]['flags'], {'multiscale_depth': True})
        self.assertEqual([sum(j['flags'].get(f, False) for f in queue.NEW_FLAGS) for j in jobs], [0, 0, 1, 1, 1, 1])
        self.assertEqual([j['max_job_seconds'] for j in jobs], [43200] * 5 + [86400])
        self.assertTrue(all(j['seed'] == 20261013 for j in jobs))

    def test_fixed_budget_and_smoke_recipe(self):
        formal, smoke = queue.stage_recipe(False), queue.stage_recipe(True)
        self.assertEqual((formal['iters'], formal['val_interval'], formal['keep_checkpoints']), (30000, 5000, 6))
        self.assertEqual((smoke['iters'], smoke['val_interval'], smoke['keep_checkpoints']), (20, 20, 1))
        for key in formal.keys() - {'iters', 'val_interval', 'keep_checkpoints'}:
            self.assertEqual(formal[key], smoke[key])
        self.assertEqual(queue.TOTAL_SECONDS, 96 * 3600)

    def test_training_command_gpu0_fresh_course_data_and_independent_switches(self):
        stage = queue.stage_plan(self.root, self.plan, 'formal')
        for job in stage['jobs']:
            _, command = queue.training_command(stage, job)
            for key, expected in [('--gpus', 'device=0'), ('--image-size', '1024'),
                                  ('--iters', '30000'), ('--checkpoint', '/workspace/' + queue.CHECKPOINT),
                                  ('--data-root', '/workspace/' + queue.DATA)]:
                self.assertEqual(command[command.index(key) + 1], expected)
            self.assertFalse(any('test' == item or '/test/' in item for item in command))
            self.assertNotIn('--resume', command)
            for flag in queue.NEW_FLAGS:
                self.assertEqual('--' + flag.replace('_', '-') in command, bool(job['flags'].get(flag)))

    def test_cpu_container_cannot_allocate_gpu(self):
        _, command = queue.cpu_command(self.root, self.plan)
        self.assertNotIn('--gpus', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=', command)
        self.assertIn('AI_CLASS_TEST_DEVICE=cpu', command)
        self.assertEqual(command[-4:], ['-m', 'unittest', '-v', 'test_course_directions_workflow'])

    def test_diagnostics_command_uses_fresh_gpu0_container_without_training_flags(self):
        job = queue.stage_jobs(False)[-1]
        name, command = queue.diagnostics_command(self.root, self.plan, job)
        self.assertIn('diagnostics-high_resolution_long', name)
        self.assertEqual(command[command.index('--gpus') + 1], 'device=0')
        self.assertNotIn('--iters', command)
        self.assertNotIn('--checkpoint', command)
        self.assertTrue(command[-4].endswith('--run'))
        self.assertTrue(command[-3].endswith('formal/runs/high_resolution_long'))

    def test_diagnostics_acceptance_requires_fixed_inputs_and_full_timing(self):
        job = queue.stage_jobs(False)[0]
        output = self.root / 'diagnostics' / job['id']
        output.mkdir(parents=True)
        hashes = [dict(image_id=i, rgb_sha256='rgb', depth_sha256='depth') for i in queue.FIXED_IDS]
        queue.atomic_json(output / 'input_hashes.json', hashes)
        queue.atomic_json(output / 'cases.json', [dict(image_id=i) for i in queue.FIXED_IDS])
        for identity in queue.FIXED_IDS:
            queue.atomic_json(output / f'case_{identity}_predictions.json', [])
        digest = queue.sha256(output / 'input_hashes.json')
        trained = dict(checkpoint_sha256='checkpoint', report_sha256='trained', parameters=10)
        report = dict(status='passed', completed_images=8, fixed_case_ids=list(queue.FIXED_IDS),
            checkpoint_sha256='checkpoint', training_report_sha256='trained', input_hashes_sha256=digest,
            active_parameters=10, state_tensor_bytes=40, input_scale=1024, original_grid_verified=True,
            training_started=False, test_inputs_used=False, test_annotations_used=False,
            gt_sent_to_model=False, ap_reevaluated=False,
            timing=dict(warmup=5, repeats=20, image_id=queue.FIXED_IDS[0], all_seconds=[.1] * 20,
                        mean_ms=100., median_ms=100., peak_allocated_mib=100.))
        queue.atomic_json(output / 'report.json', report)
        with patch.object(queue, 'FIXED_INPUTS_SHA', digest):
            self.assertEqual(queue.validate_diagnostics(self.root, self.plan, job, trained)['status'], 'passed')
            report['timing']['mean_ms'] = 50
            queue.atomic_json(output / 'report.json', report)
            with self.assertRaisesRegex(ValueError, 'Timing summary'):
                queue.validate_diagnostics(self.root, self.plan, job, trained)

    def test_all_six_smokes_before_any_formal(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        phases = []
        def fake_run(phase, name, command, timeout):
            phases.append(phase)
            supervisor.state['steps'].append(dict(phase=phase, exit_code=0))
        def fake_accept(root, plan, phase, job):
            return dict(status='passed', metrics=self.metrics)
        with patch.object(queue, 'verify_inputs', return_value=(self.provenance, self.plan['reference_baseline'])), \
                patch.object(supervisor, 'wait_idle'), patch.object(supervisor, 'run_command', side_effect=fake_run), \
                patch.object(queue, 'validate_report', side_effect=fake_accept), \
                patch.object(queue, 'validate_diagnostics', return_value=dict(status='passed')):
            supervisor.execute()
        self.assertEqual(phases, ['cpu'] + ['smoke/' + j['id'] for j in queue.jobs()] +
                         [phase + '/' + j['id'] for j in queue.jobs() for phase in ('formal', 'diagnostics')])
        self.assertEqual(supervisor.state['status'], 'complete')
        summary = json.loads((self.root / 'summary.json').read_text())
        high = summary['results']['high_resolution_long']
        self.assertEqual(high['primary_reference_report'], 'resolution1536.json')
        self.assertAlmostEqual(high['delta_vs_primary_reference_points']['course/segm_mAP_s'], -10)

    def test_first_smoke_failure_stops_queue_and_preserves_original(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        phases = []
        def fake_run(phase, *unused):
            phases.append(phase)
            if phase == 'smoke/depth_ungated_long':
                raise RuntimeError('first actual CUDA error')
            supervisor.state['steps'].append(dict(phase=phase))
        with patch.object(queue, 'verify_inputs', return_value=(self.provenance, self.plan['reference_baseline'])), \
                patch.object(supervisor, 'wait_idle'), patch.object(supervisor, 'run_command', side_effect=fake_run), \
                patch.object(queue, 'validate_report', return_value=dict(status='passed')):
            with self.assertRaisesRegex(RuntimeError, 'first actual CUDA error'):
                supervisor.execute()
        self.assertEqual(phases, ['cpu', 'smoke/rgb_long', 'smoke/depth_ungated_long'])
        self.assertFalse((self.root / 'formal').exists())
        error = json.loads((self.root / 'first_error.json').read_text())
        self.assertEqual(error['message'], 'first actual CUDA error')
        self.assertEqual(supervisor.state['status'], 'failed')

    def test_busy_gpu_wait_is_bounded_and_never_stops_any_container(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        busy = {0: dict(memory_used_mib=10000, utilization=90)}
        with patch.object(queue, 'gpu_snapshot', return_value=busy), \
                patch.object(supervisor, 'remaining', return_value=1000), \
                patch.object(queue.time, 'monotonic', side_effect=[0, queue.GPU_WAIT_SECONDS]), \
                patch.object(queue.subprocess, 'run') as run:
            with self.assertRaisesRegex(TimeoutError, 'remained occupied'):
                supervisor.wait_idle()
            run.assert_not_called()

    def test_total_deadline_includes_gpu_waits(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        with patch.object(supervisor, 'remaining', return_value=-1), patch.object(queue, 'gpu_snapshot') as snapshot:
            with self.assertRaisesRegex(TimeoutError, 'total workflow'):
                supervisor.wait_idle()
            snapshot.assert_not_called()

    def test_stop_marker_prevents_dispatch(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        (self.root / 'STOP').touch()
        with patch.object(queue.subprocess, 'Popen') as popen:
            with self.assertRaises(queue.StopRequested):
                supervisor.run_command('cpu', 'own', ['unused'], 10)
            popen.assert_not_called()

    def test_cleanup_targets_exact_owned_container_only(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        child = Mock()
        supervisor.active = child, 'test-owned-exact-name'
        with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='owned-nonce')) as run:
            supervisor.stop_owned()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'stop', '--time', '20', 'test-owned-exact-name'])
        child.wait.assert_called_once_with(timeout=25)
        self.assertIsNone(supervisor.active)

    def test_cleanup_refuses_container_with_other_owner_label(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        supervisor.active = Mock(), 'same-name-different-owner'
        with patch.object(queue.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='someone-else')) as run:
            supervisor.stop_owned()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:2], ['docker', 'inspect'])
        self.assertIn('refused to stop', supervisor.state['cleanup_errors'][0])

    def test_job_timeout_stops_only_owned_container_then_raises_original(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        child = Mock()
        child.poll.return_value = None
        child.pid = 123
        with patch.object(queue.subprocess, 'Popen', return_value=child), \
                patch.object(supervisor, 'check_stop'), \
                patch.object(queue.time, 'monotonic', side_effect=[0, 11]):
            with self.assertRaisesRegex(TimeoutError, 'Per-job deadline'):
                supervisor.run_command('formal/rgb_long', 'only-our-container', ['unused'], 10)
        self.assertEqual(supervisor.active, (child, 'only-our-container'))

    def test_worker_replay_never_changes_status_or_launches(self):
        queue.atomic_json(self.root / 'plan.json', self.plan)
        queue.atomic_json(self.root / 'status.json', {'status': 'failed', 'error': 'original'})
        before = (self.root / 'status.json').read_bytes()
        with patch.object(queue.Supervisor, 'execute') as execute:
            with self.assertRaisesRegex(RuntimeError, 'already started'):
                queue.worker(self.root)
            execute.assert_not_called()
        self.assertEqual((self.root / 'status.json').read_bytes(), before)

    def test_existing_output_rejected_before_any_mutation(self):
        args = queue.parser().parse_args(['start', '--workspace', str(self.workspace), '--output', str(self.root)])
        with patch.object(queue, 'verify_inputs') as verify:
            with self.assertRaisesRegex(ValueError, 'NEW directory'):
                queue.start(args)
            verify.assert_not_called()

    def test_formal_report_accepted_only_with_six_full_validations(self):
        phase, job, run, report = self.fixture_report()
        accepted = queue.validate_report(self.root, self.plan, phase, job)
        self.assertEqual(accepted['final_iteration'], 30000)
        report['validation_coverage'][-1]['unique_images'] = 2583
        queue.atomic_json(run / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'cover the whole'):
            queue.validate_report(self.root, self.plan, phase, job)

    def test_smoke_is_separate_and_not_claimed_full_dataset(self):
        phase, job, _, _ = self.fixture_report(smoke=True)
        accepted = queue.validate_report(self.root, self.plan, phase, job)
        self.assertTrue(accepted['functional_smoke'])
        self.assertEqual(accepted['final_iteration'], 20)

    def test_smoke_allows_minus_one_only_for_verified_absent_size_strata(self):
        self.metrics['course/segm_mAP_l'] = -1
        phase, job, run, report = self.fixture_report(smoke=True)
        accepted = queue.validate_report(self.root, self.plan, phase, job)
        self.assertEqual(accepted['metrics']['course/segm_mAP_l'], -1)
        report['metrics']['course/segm_mAP_s'] = -1
        queue.atomic_json(run / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'out-of-range'):
            queue.validate_report(self.root, self.plan, phase, job)

    def test_coco_area_boundaries_and_ignored_gt_detect_missing_strata(self):
        self.assertEqual(queue.missing_area_metrics(dict(annotations=[dict(area=100), dict(area=2000),
                         dict(area=10000, iscrowd=True)])), ['course/segm_mAP_l'])
        self.assertEqual(queue.missing_area_metrics(dict(annotations=[dict(area=1024), dict(area=9216)])), [])
        with self.assertRaises(ValueError):
            queue.missing_area_metrics(dict(annotations=[]))

    def test_nonzero_docker_client_exit_keeps_container_for_owned_cleanup(self):
        supervisor = queue.Supervisor(self.root, self.plan)
        child = Mock(pid=123)
        child.poll.return_value = 125
        with patch.object(queue.subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'exited 125'):
                supervisor.run_command('smoke/rgb_long', 'only-our-container', ['unused'], 1800)
        self.assertEqual(supervisor.active, (child, 'only-our-container'))

    def test_rejects_early_step_nonfinite_metrics_test_access_flags_and_hashes(self):
        phase, job, run, original = self.fixture_report()
        changes = [dict(final_optimizer_steps=[29999]), dict(test_inputs_used=True),
                   dict(source_checkpoint_sha256='wrong'), dict(checkpoint_sha256='wrong'),
                   dict(effective_train_size=1536), dict(finite_checks={}),
                   dict(metrics={key: -1 for key in queue.METRICS}),
                   dict(implementation_sha256={})]
        for update in changes:
            with self.subTest(update=update):
                report = copy.deepcopy(original)
                report.update(update)
                queue.atomic_json(run / 'report.json', report)
                with self.assertRaises(ValueError):
                    queue.validate_report(self.root, self.plan, phase, job)
        report = copy.deepcopy(original)
        report['args']['course_small_boundary'] = True
        queue.atomic_json(run / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'switch'):
            queue.validate_report(self.root, self.plan, phase, job)

    def test_final_endpoint_cannot_be_replaced_by_intermediate_best(self):
        phase, job, run, report = self.fixture_report()
        report['metrics'] = dict(self.metrics, **{'course/segm_mAP_s': .9})
        queue.atomic_json(run / 'report.json', report)
        with self.assertRaisesRegex(ValueError, 'substituted'):
            queue.validate_report(self.root, self.plan, phase, job)

    def test_missing_retained_checkpoint_and_changed_source_fail(self):
        phase, job, run, _ = self.fixture_report()
        (run / 'iter_5000.pth').unlink()
        with self.assertRaisesRegex(ValueError, 'retention'):
            queue.validate_report(self.root, self.plan, phase, job)
        (self.root / 'source/train_course_directions.py').write_text('# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'Frozen source changed'):
            queue.assert_intact(self.root, self.plan)


if __name__ == '__main__':
    unittest.main(verbosity=2)

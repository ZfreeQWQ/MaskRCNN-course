"""Host-side queue safety checks; no CUDA, Docker, SSH, or training is invoked."""
from argparse import Namespace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_nyu_sweep as launcher


PINNED_CHECKPOINT_SHA = '41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879'


class FactorialAndCommandTests(unittest.TestCase):
    def test_three_seeds_cover_every_cell_once_with_identical_switch_encoding(self):
        seeds = [20260923, 20260924, 20260925]
        jobs = launcher.experiment_jobs(seeds)
        self.assertEqual(len(jobs), 48)
        self.assertEqual(len({j['id'] for j in jobs}), 48)
        self.assertEqual(len({j['output_rel'] for j in jobs}), 48)
        for seed in seeds:
            cells = [j for j in jobs if j['seed'] == seed]
            self.assertEqual({j['mask'] for j in cells}, set(range(16)))
            self.assertEqual([j['mask'] for j in cells[:5]], [0, 1, 2, 4, 8])
            self.assertEqual(cells[-1]['mask'], 15)
            for cell in cells:
                self.assertEqual(cell['flags'], {
                    flag: bool(cell['mask'] & 1 << index)
                    for index, flag in enumerate(launcher.FLAGS)})
        self.assertEqual(jobs, launcher.experiment_jobs(seeds))

    def test_empty_duplicate_negative_and_noninteger_seeds_are_rejected(self):
        for seeds in ([], [1, 1], [-1], [1.5], ['1']):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                launcher.experiment_jobs(seeds)

    def plan(self):
        return dict(workspace='/mnt/nfs/fenghaibin/AI_Class',
                    container_output_root='/workspace/outputs/test-sweep',
                    container_name_prefix='ai-nyu-1234', image=launcher.IMAGE,
                    recipe=dict(iters=1000, batch_size=2, image_size=512,
                                workers=2, lr=1e-5))

    def test_docker_flags_budget_and_nfs_bindings_are_explicit(self):
        plan = self.plan()
        jobs = launcher.experiment_jobs([123])
        for job in jobs:
            with self.subTest(mask=job['mask']):
                name, command = launcher.docker_command(plan, job, 1)
                self.assertIsInstance(command, list)
                self.assertEqual(command[:3], ['docker', 'run', '--rm'])
                for key, value in {'--pull': 'never', '--gpus': 'device=1',
                                   '--workdir': '/workspace',
                                   '--workspace': '/workspace', '--iters': '1000',
                                   '--batch-size': '2', '--image-size': '512',
                                   '--workers': '2', '--lr': '1e-05', '--seed': '123',
                                   '--data-root': '/workspace/' + launcher.DATA,
                                   '--checkpoint': '/workspace/' + launcher.CHECKPOINT,
                                   '--output': '/workspace/outputs/test-sweep/' + job['output_rel']}.items():
                    self.assertEqual(command[command.index(key) + 1], value)
                self.assertEqual(command[command.index('--name') + 1], name)
                self.assertIn('--shm-size=32g', command)
                self.assertIn('/mnt/nfs/fenghaibin/AI_Class:/workspace', command)
                self.assertIn('/mnt/nfs/fenghaibin/AI_Class:/mnt/nfs/fenghaibin/AI_Class', command)
                self.assertIn('HOME=/workspace/.home', command)
                self.assertIn('TMPDIR=/workspace/.tmp', command)
                self.assertIn('PYTHONPATH=/workspace/outputs/test-sweep/source:/workspace/mmdetection', command)
                self.assertIn('/workspace/outputs/test-sweep/source/run_nyu_experiment.py', command)
                self.assertNotIn('--resume', command)
                for flag, enabled in job['flags'].items():
                    self.assertEqual('--' + flag.replace('_', '-') in command, enabled)

    def test_gpu_and_run_identity_make_container_names_unique(self):
        names = {launcher.docker_command(self.plan(), job, gpu)[0]
                 for job in launcher.experiment_jobs([1, 2, 3]) for gpu in (0, 1)}
        self.assertEqual(len(names), 96)

    def test_container_path_rejects_outside_workspace(self):
        self.assertEqual(launcher.container_path('/tmp/root/outputs/one', '/tmp/root'),
                         '/workspace/outputs/one')
        with self.assertRaises(ValueError):
            launcher.container_path('/tmp/root-other/outputs/one', '/tmp/root')

    def test_log_progress_uses_latest_iteration_and_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.log'
            self.assertEqual(launcher.log_progress(path), {})
            path.write_text('x' * 20000 + '\nIter(train) [ 10/1000] lr: 1e-5 time: 0.502\n'
                            'Iter(train) [ 20/1000] lr: 1e-5 time: 0.498\n')
            self.assertEqual(launcher.log_progress(path), dict(
                iteration=20, total_iterations=1000, seconds_per_iteration=0.498))


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name).resolve()
        self.workspace = base / 'workspace'
        self.origin = base / 'code'
        self.output = self.workspace / 'outputs' / 'new-sweep'
        self.origin.mkdir()
        for name in launcher.SOURCE_FILES:
            (self.origin / name).write_text('# frozen fixture ' + name)
        for package in ('optimizations', 'rgbd'):
            (self.origin / package).mkdir()
            (self.origin / package / '__init__.py').write_text('# package')
            (self.origin / package / 'implementation.py').write_text('# implementation')
        self.manifest = self.workspace / launcher.DATA / 'manifest.json'
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"fixed_development_split": true}')
        self.completion = self.manifest.with_name('export_complete.json')
        self.completion.write_text(json.dumps(dict(manifest_sha256=launcher.sha256(self.manifest))))
        self.checkpoint = self.workspace / launcher.CHECKPOINT
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b'fixture checkpoint')
        executable = self.workspace / 'envs/mmdet/bin/python'
        executable.parent.mkdir(parents=True)
        executable.write_text('fixture python')
        self.args = Namespace(workspace=str(self.workspace), output=str(self.output),
                              seeds=[1, 2, 3], gpus=[0, 1], iters=1000,
                              batch_size=2, image_size=512, workers=2, lr=1e-5)
        self.real_sha = launcher.sha256
        self.popen = Mock(return_value=types.SimpleNamespace(pid=12345))
        self.run = Mock(return_value=subprocess.CompletedProcess([], 0, 'sha256:fixture-image\n', ''))
        patches = [patch.object(launcher, '__file__', str(self.origin / 'launch_nyu_sweep.py')),
                   patch.object(launcher, 'gpu_snapshot', return_value={
                       0: dict(memory_used_mib=0, utilization=0),
                       1: dict(memory_used_mib=0, utilization=0)}),
                   patch.object(launcher.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=100 * 2**30)),
                   patch.object(launcher, 'sha256', side_effect=lambda path:
                                PINNED_CHECKPOINT_SHA if Path(path) == self.checkpoint else self.real_sha(path)),
                   patch.object(launcher.subprocess, 'run', self.run),
                   patch.object(launcher.subprocess, 'Popen', self.popen),
                   patch('sys.stdout', new_callable=io.StringIO)]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def test_prepare_freezes_exact_source_and_starts_detached_supervisor(self):
        launcher.prepare(self.args)
        plan = json.loads((self.output / 'plan.json').read_text())
        self.assertEqual(len(plan['jobs']), 48)
        self.assertEqual(plan['expected_provenance']['dataset_manifest_sha256'], self.real_sha(self.manifest))
        self.assertEqual(plan['expected_provenance']['source_checkpoint_sha256'], PINNED_CHECKPOINT_SHA)
        self.assertEqual(plan['image_id'], 'sha256:fixture-image')
        self.assertEqual(plan['container_output_root'], '/workspace/outputs/new-sweep')
        for relative, checksum in plan['snapshot_sha256'].items():
            self.assertEqual(self.real_sha(self.output / 'source' / relative), checksum)
            self.assertEqual((self.origin / relative).read_bytes(),
                             (self.output / 'source' / relative).read_bytes())
        self.assertIn('rgbd/implementation.py', plan['expected_provenance']['implementation_sha256'])
        self.assertIn('optimizations/implementation.py', plan['expected_provenance']['implementation_sha256'])
        self.assertEqual(self.popen.call_count, 1)
        command = self.popen.call_args.args[0]
        options = self.popen.call_args.kwargs
        self.assertEqual(command[2], str(self.output / 'source/launch_nyu_sweep.py'))
        self.assertEqual(command[3:], ['worker', '--output', str(self.output)])
        self.assertIs(options['start_new_session'], True)
        self.assertIs(options['close_fds'], True)
        self.assertEqual(options['stdin'], subprocess.DEVNULL)
        self.assertEqual(options['stderr'], subprocess.STDOUT)
        self.assertEqual(json.loads((self.output / 'launch.json').read_text())['pid'], 12345)
        (self.origin / 'run_nyu_experiment.py').write_text('# changed after launch')
        self.assertEqual(self.real_sha(self.output / 'source/run_nyu_experiment.py'),
                         plan['snapshot_sha256']['run_nyu_experiment.py'])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('keep historical result')
        with self.assertRaises(FileExistsError):
            launcher.prepare(self.args)
        self.assertEqual(sentinel.read_text(), 'keep historical result')
        self.popen.assert_not_called()
        self.run.assert_not_called()

    def test_output_must_be_below_workspace_outputs(self):
        for output in (self.workspace, self.workspace / 'elsewhere', self.workspace.parent / 'outputs/run'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                launcher.prepare(Namespace(**{**vars(self.args), 'output': str(output)}))
        self.popen.assert_not_called()

    def test_bad_manifest_hash_is_rejected_before_output_creation(self):
        self.completion.write_text('{"manifest_sha256": "wrong"}')
        with self.assertRaisesRegex(ValueError, 'completion marker'):
            launcher.prepare(self.args)
        self.assertFalse(self.output.exists())
        self.popen.assert_not_called()

    def test_changed_checkpoint_is_rejected_before_output_creation(self):
        with patch.object(launcher, 'sha256', side_effect=self.real_sha):
            with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
                launcher.prepare(self.args)
        self.assertFalse(self.output.exists())
        self.popen.assert_not_called()

    def test_busy_gpu_is_rejected_without_starting_anything(self):
        with patch.object(launcher, 'gpu_snapshot', return_value={
                0: dict(memory_used_mib=4096, utilization=50),
                1: dict(memory_used_mib=0, utilization=0)}):
            with self.assertRaisesRegex(RuntimeError, 'not idle'):
                launcher.prepare(self.args)
        self.assertFalse(self.output.exists())
        self.popen.assert_not_called()

    def test_insufficient_disk_and_missing_source_are_rejected(self):
        with patch.object(launcher.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=2**30)):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient NFS'):
                launcher.prepare(self.args)
        (self.origin / 'run_nyu_experiment.py').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'source'):
            launcher.prepare(self.args)
        self.assertFalse(self.output.exists())
        self.popen.assert_not_called()


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name).resolve() / 'outputs/sweep'
        for directory in ('logs', 'runs', 'source'):
            (self.output / directory).mkdir(parents=True, exist_ok=True)
        (self.output / 'source/frozen.py').write_text('# unchanged source')
        self.plan = dict(workspace=str(self.output.parents[1]),
                         container_output_root='/workspace/outputs/sweep',
                         container_name_prefix='unit-test-only', image=launcher.IMAGE,
                         image_id='sha256:fixture-image', gpus=[0],
                         recipe=dict(iters=1000, batch_size=2, image_size=512, workers=2, lr=1e-5),
                         jobs=launcher.experiment_jobs([1])[:5], max_job_seconds=3600,
                         snapshot_sha256={'frozen.py': launcher.sha256(self.output / 'source/frozen.py')})
        self.outcomes = []
        self.started = []
        self.summary_invalid = 0
        self.summary_return_code = 0
        self.docker_image_id = 'sha256:fixture-image'
        self.popen = Mock(side_effect=self.fake_popen)
        self.run = Mock(side_effect=self.fake_run)
        patches = [patch.dict(sys.modules, {'summarize_nyu_sweep': types.ModuleType('summarize_nyu_sweep')}),
                   patch.object(launcher.signal, 'signal'),
                   patch.object(launcher.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=100 * 2**30)),
                   patch.object(launcher.subprocess, 'run', self.run),
                   patch.object(launcher.subprocess, 'Popen', self.popen),
                   patch.object(launcher.time, 'sleep'), patch('sys.stdout', new_callable=io.StringIO)]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def fake_run(self, command, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect']:
            return subprocess.CompletedProcess(command, 0, self.docker_image_id, '')
        if command[:2] == ['docker', 'stop']:
            return subprocess.CompletedProcess(command, 0, '', '')
        if 'summarize_nyu_sweep.py' in command[1]:
            (self.output / 'summary.json').write_text(json.dumps(dict(invalid_count=self.summary_invalid)))
            return subprocess.CompletedProcess(command, self.summary_return_code, '', 'fixture summary error')
        raise AssertionError(f'Unexpected external command in test: {command}')

    def fake_popen(self, command, **kwargs):
        self.assertEqual(command[:2], ['docker', 'run'])
        self.started.append(command)
        job_id = Path(command[command.index('--output') + 1]).name
        destination = self.output / 'runs' / job_id
        destination.mkdir()
        outcome = self.outcomes.pop(0) if self.outcomes else 'passed'
        if outcome != 'missing':
            (destination / 'report.json').write_text(json.dumps(dict(status=outcome)))
        return types.SimpleNamespace(pid=1000 + len(self.started), poll=lambda: 0)

    def run_worker(self):
        (self.output / 'plan.json').write_text(json.dumps(self.plan))
        launcher.run_worker(self.output)
        return json.loads((self.output / 'status.json').read_text())

    def test_complete_queue_records_all_successes_and_summary(self):
        self.plan['gpus'] = [0, 1]
        state = self.run_worker()
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['counts'], dict(pending=0, running=0, passed=5, failed=0))
        self.assertEqual(len(self.started), 5)
        self.assertTrue(all(j['exit_code'] == 0 for j in state['jobs'].values()))
        names = [c[c.index('--name') + 1] for c in self.started]
        self.assertEqual(len(set(names)), 5)
        self.assertTrue((self.output / 'summary.json').is_file())

    def test_three_consecutive_missing_or_failed_reports_stop_remaining_queue(self):
        self.outcomes = ['failed', 'missing', 'failed']
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertEqual(state['counts'], dict(pending=2, running=0, passed=0, failed=3))
        self.assertIn('Three consecutive', state['stop_reason'])
        self.assertEqual(len(self.started), 3)

    def test_isolated_failure_does_not_hide_failure_or_block_all_other_jobs(self):
        self.outcomes = ['failed', 'passed', 'passed', 'passed', 'passed']
        state = self.run_worker()
        self.assertEqual(state['status'], 'completed_with_failures')
        self.assertEqual(state['counts'], dict(pending=0, running=0, passed=4, failed=1))

    def test_stop_file_prevents_new_work(self):
        (self.output / 'STOP').touch()
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertEqual(state['counts'], dict(pending=5, running=0, passed=0, failed=0))
        self.popen.assert_not_called()

    def test_changed_snapshot_prevents_dispatch(self):
        (self.output / 'source/frozen.py').write_text('# changed')
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertIn('Snapshot changed', state['stop_reason'])
        self.popen.assert_not_called()

    def test_deleted_snapshot_prevents_dispatch_without_crashing(self):
        (self.output / 'source/frozen.py').unlink()
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertIn('Snapshot changed', state['stop_reason'])
        self.popen.assert_not_called()

    def test_changed_cached_docker_image_prevents_dispatch(self):
        self.docker_image_id = 'sha256:unexpected'
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertIn('Docker image changed', state['stop_reason'])
        self.popen.assert_not_called()

    def test_invalid_experiment_evidence_stops_remaining_dispatch(self):
        self.summary_invalid = 1
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertEqual(len(self.started), 1)
        self.assertIn('invalid experiment evidence', state['stop_reason'])

    def test_initial_summary_failure_prevents_unvalidated_dispatch(self):
        self.summary_return_code = 1
        state = self.run_worker()
        self.assertEqual(state['status'], 'stopped')
        self.assertIn('summary validation failed', state['stop_reason'])
        self.assertIn('fixture summary error', state['summary_error'])
        self.popen.assert_not_called()

    def test_second_supervisor_does_not_overwrite_live_status(self):
        (self.output / 'plan.json').write_text(json.dumps(self.plan))
        original = {'status': 'running', 'pid': 56789, 'sentinel': 'preserve'}
        status = self.output / 'status.json'
        status.write_text(json.dumps(original))
        with patch.object(launcher.fcntl, 'flock', side_effect=BlockingIOError('already locked')):
            with patch.object(sys, 'argv', ['launch_nyu_sweep.py', 'worker', '--output', str(self.output)]):
                with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
                    launcher.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(json.loads(status.read_text()), original)
        self.popen.assert_not_called()

    def test_restarting_finished_worker_does_not_overwrite_historical_status(self):
        (self.output / 'plan.json').write_text(json.dumps(self.plan))
        original = {'status': 'complete', 'sentinel': 'keep historical audit'}
        status = self.output / 'status.json'
        status.write_text(json.dumps(original))
        with patch.object(sys, 'argv', ['launch_nyu_sweep.py', 'worker', '--output', str(self.output)]):
            with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as raised:
                launcher.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(json.loads(status.read_text()), original)
        self.popen.assert_not_called()

    def test_error_cleanup_stops_only_recorded_owned_running_containers(self):
        (self.output / 'plan.json').write_text(json.dumps(self.plan))
        own_name = launcher.docker_command(self.plan, self.plan['jobs'][0], 0)[0]
        completed_name = launcher.docker_command(self.plan, self.plan['jobs'][1], 0)[0]
        state = dict(jobs={
            'own': dict(status='running', container_name=own_name),
            'other': dict(status='running', container_name='unrelated-user-container'),
            'finished': dict(status='passed', container_name=completed_name)})
        launcher.stop_owned_after_error(self.output, state)
        self.assertEqual(self.run.call_count, 1)
        self.assertEqual(self.run.call_args.args[0], ['docker', 'stop', '--time', '30', own_name])
        self.assertEqual(self.run.call_args.kwargs['timeout'], 45)
        self.assertEqual(state['jobs']['own']['status'], 'interrupted')
        self.assertEqual(state['jobs']['other']['status'], 'running')
        self.assertEqual(state['jobs']['finished']['status'], 'passed')

    def test_second_gpu_launch_exception_cleans_up_first_owned_container(self):
        self.plan['gpus'] = [0, 1]
        (self.output / 'plan.json').write_text(json.dumps(self.plan))
        first = types.SimpleNamespace(pid=1234, poll=lambda: None)
        self.popen.side_effect = [first, OSError('fixture second Docker launch failed')]
        with patch.object(sys, 'argv', ['launch_nyu_sweep.py', 'worker', '--output', str(self.output)]):
            with self.assertRaisesRegex(OSError, 'second Docker launch failed'):
                launcher.main()
        state = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(state['status'], 'supervisor_failed')
        self.assertIn('second Docker launch failed', state['error'])
        self.assertEqual(state['jobs'][self.plan['jobs'][0]['id']]['status'], 'interrupted')
        stops = [call.args[0] for call in self.run.call_args_list if call.args[0][:2] == ['docker', 'stop']]
        self.assertEqual(stops, [['docker', 'stop', '--time', '30',
                                 launcher.docker_command(self.plan, self.plan['jobs'][0], 0)[0]]])


if __name__ == '__main__':
    unittest.main()

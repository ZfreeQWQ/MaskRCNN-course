"""Standard-library-only evidence validation; no torch, GPU or dataset needed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    'nyu_summary', Path(__file__).resolve().parents[1] / 'summarize_nyu_experiment.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def report(name):
    smoke, depth = name == 'all_on_smoke', name != 'rgb_100'
    count = 2 if smoke else 100
    args = dict(mode='train', trained_checkpoint=False, output=f'/output/{name}',
                workspace='/workspace', data_root='/workspace/data/nyuv2/development_indoor8_v1',
                checkpoint='/workspace/pretrained.pth', iters=count, batch_size=2,
                image_size=256 if smoke else 512, workers=2, lr=1e-5, seed=20260923,
                depth_fusion=depth, small_object_crop=smoke, small_object_loss=smoke,
                boundary_loss=smoke, small_area_threshold=1024.0, small_crop_prob=0.5,
                small_loss_max_weight=2.0, boundary_loss_weight=0.1)
    categories = [{'id': i + 1, 'name': name, 'coco80_class_index': index}
                  for i, (name, index) in enumerate(zip(M.EXPECTED_NAMES, M.EXPECTED_COCO))]
    return dict(status='passed', protocol=M.PROTOCOL, args=args,
                dataset_manifest_sha256='a' * 64, source_checkpoint_sha256='b' * 64,
                checkpoint_sha256=('d' if depth else 'c') * 64,
                checkpoint=f'/output/{name}/iter_{count}.pth',
                implementation_sha256={'run_nyu_experiment.py': 'e' * 64, 'rgbd/model.py': 'f' * 64},
                categories=categories, parameters=43999999, mmdetection_commit='c' * 40,
                evaluation_policy='mask prediction intersect label_valid; discard empty predictions; COCO segm',
                training_images=640, final_iteration=count, final_optimizer_steps=[count],
                classifier_changed=True, depth_adapter_changed=depth,
                depth_adapter_l2=0.001 if depth else 0.0,
                first_loss=20.0, last_loss=15.0,
                seconds_including_evaluation=100, whole_run_peak_allocated_mib=3000,
                metrics={f'nyuv2_void/segm_{metric}': (0.126 if depth else 0.123)
                         for metric in M.METRICS})


def benchmark(name):
    depth = name == 'depth_100'
    return dict(status='complete', protocol='fixed-first-validation-frame-cpu-to-prediction-batch1-v1',
                checkpoint_sha256=('d' if depth else 'c') * 64, checkpoint_iteration=100,
                config_sha256=('f' if depth else 'a') * 64, script_sha256='b' * 64,
                seed=20260924, batch_size=1, warmup=5, repeats=2,
                input={'cpu_tensor_sha256': 'c' * 64, 'cpu_input': {'shape': [5, 512, 683]}},
                environment={'gpu_name': 'fixture GPU', 'autocast': False},
                parameters=43999999, parameter_storage_bytes=175999996,
                parameter_dtypes=['torch.float32'],
                model_flags=dict(depth_fusion={'enabled': depth},
                                 small_object_loss={'enabled': False},
                                 boundary_loss={'enabled': False}, ignore_void=True),
                latency_ms=dict(samples=[10.0, 12.0], mean=11, median=11, p90=11.8,
                                p95=11.9, minimum=10, maximum=12, population_std=1),
                images_per_second_reciprocal_mean_latency=1000 / 11,
                cuda_memory_mib={'peak_allocated': 1000, 'peak_reserved': 1200},
                timing_scope='fixture cpu to prediction', memory_scope='fixture allocator')


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nyu-summary-test-')
        self.root = Path(self.temporary.name) / 'runs'
        self.reports = {name: report(name) for name in M.NAMES}
        self.flush()

    def tearDown(self):
        self.temporary.cleanup()

    def flush(self):
        for name, value in self.reports.items():
            destination = self.root / name / 'report.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(value))

    def benches(self, values=None):
        for name, value in (values or {n: benchmark(n) for n in ('rgb_100', 'depth_100')}).items():
            (self.root / name / M.BENCHMARK_NAME).write_text(json.dumps(value))

    def test_valid_no_benchmark_and_percentage_point_conversion(self):
        result = M.summarize(self.root)
        self.assertEqual(result['benchmark']['status'], 'not_measured')
        self.assertEqual(result['comparison'][0]['mask_APs_points'], 12.3)
        self.assertEqual(result['comparison'][1]['mask_APs_points'], 12.6)
        self.assertEqual(result['comparison'][1]['delta_APs_vs_RGB_percentage_points'], 0.3)
        self.assertFalse(result['integration_smoke']['comparable_for_AP_ranking'])
        text = M.markdown(result)
        self.assertIn('未测', text)
        self.assertIn('百分点', text)

    def test_requires_all_three_passed_reports(self):
        for name in M.NAMES:
            values = copy.deepcopy(self.reports)
            values[name]['status'] = 'running'
            with self.assertRaisesRegex(ValueError, 'status is not passed'):
                M.validate_reports(values)

    def test_rejects_any_shared_provenance_mismatch(self):
        for key in ('dataset_manifest_sha256', 'source_checkpoint_sha256', 'implementation_sha256',
                    'parameters', 'categories', 'mmdetection_commit', 'evaluation_policy', 'training_images'):
            values = copy.deepcopy(self.reports)
            if key.endswith('sha256') and key != 'implementation_sha256':
                values['depth_100'][key] = '0' * 64
            elif key == 'implementation_sha256':
                values['depth_100'][key]['rgbd/model.py'] = '0' * 64
            elif key == 'categories':
                values['depth_100'][key][0]['name'] = 'wrong'
            elif isinstance(values['depth_100'][key], int):
                values['depth_100'][key] += 1
            else:
                values['depth_100'][key] += 'wrong'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'differs from RGB'):
                M.validate_reports(values)

    def test_rejects_budget_difference_and_extra_unknown_arg(self):
        values = copy.deepcopy(self.reports)
        values['depth_100']['args']['lr'] = 2e-5
        with self.assertRaisesRegex(ValueError, 'args differ'):
            M.validate_reports(values)
        values = copy.deepcopy(self.reports)
        values['depth_100']['args']['new_option'] = True
        with self.assertRaisesRegex(ValueError, 'args differ'):
            M.validate_reports(values)

    def test_rejects_wrong_flags_or_parameter_updates(self):
        for name, key, wrong in [('rgb_100', 'classifier_changed', False),
                                 ('rgb_100', 'depth_adapter_changed', True),
                                 ('depth_100', 'depth_adapter_changed', False),
                                 ('depth_100', 'depth_adapter_l2', 0.0),
                                 ('rgb_100', 'final_iteration', 99),
                                 ('depth_100', 'final_optimizer_steps', [99])]:
            values = copy.deepcopy(self.reports)
            values[name][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_reports(values)
        values = copy.deepcopy(self.reports)
        values['depth_100']['args']['boundary_loss'] = True
        with self.assertRaisesRegex(ValueError, 'switches do not match'):
            M.validate_reports(values)

    def test_absent_area_range_is_na_not_negative_accuracy(self):
        for value in self.reports.values():
            value['metrics']['nyuv2_void/segm_mAP_s'] = -1
        self.flush()
        result = M.summarize(self.root)
        self.assertIsNone(result['comparison'][0]['mask_APs_points'])
        self.assertIsNone(result['comparison'][1]['delta_APs_vs_RGB_percentage_points'])
        self.assertIn('不可用', M.markdown(result))

    def test_nonfinite_or_invalid_aggregate_metric_rejected(self):
        for value in (float('nan'), float('inf'), None, -0.5, 1.1):
            values = copy.deepcopy(self.reports)
            values['depth_100']['metrics']['nyuv2_void/segm_mAP_s'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.validate_reports(values)

    def test_paired_benchmark_validates_and_renders(self):
        self.benches()
        result = M.summarize(self.root)
        self.assertEqual(result['benchmark']['status'], 'complete')
        self.assertEqual(result['benchmark']['rows']['depth_100']['mean_ms'], 11)
        self.assertIn('固定一张', M.markdown(result))

    def test_one_sided_benchmark_rejected(self):
        self.benches({'rgb_100': benchmark('rgb_100')})
        with self.assertRaisesRegex(ValueError, 'Only one paired benchmark'):
            M.summarize(self.root)

    def test_benchmark_shared_condition_mismatch_rejected(self):
        for key in ('protocol', 'script_sha256', 'input', 'environment', 'parameters',
                    'warmup', 'repeats', 'timing_scope'):
            values = {n: benchmark(n) for n in ('rgb_100', 'depth_100')}
            if key in ('input', 'environment'):
                values['depth_100'][key]['unexpected'] = True
            elif isinstance(values['depth_100'][key], int):
                values['depth_100'][key] += 1
            else:
                values['depth_100'][key] += 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_benchmarks(values, self.reports)

    def test_benchmark_failed_or_wrong_checkpoint_rejected(self):
        for key, wrong in [('status', 'running'), ('checkpoint_sha256', '0' * 64),
                           ('checkpoint_iteration', 99)]:
            values = {n: benchmark(n) for n in ('rgb_100', 'depth_100')}
            values['depth_100'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                M.validate_benchmarks(values, self.reports)

    def test_benchmark_inconsistent_timings_rejected(self):
        values = {n: benchmark(n) for n in ('rgb_100', 'depth_100')}
        values['depth_100']['latency_ms']['mean'] = 999
        with self.assertRaisesRegex(ValueError, 'mean does not match'):
            M.validate_benchmarks(values, self.reports)

    def test_output_files_created_once_and_never_overwritten(self):
        result = M.summarize(self.root)
        output = self.root / 'summary'
        M.write_summary(result, output)
        self.assertEqual(json.loads((output / 'summary.json').read_text()), result)
        self.assertIn('同预算对照', (output / 'summary.md').read_text())
        before = (output / 'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            M.write_summary(result, output)
        self.assertEqual((output / 'summary.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

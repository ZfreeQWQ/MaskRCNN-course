"""Pure-standard-library coverage for strict partial factorial summaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_nyu_sweep as sweep


class SweepSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = dict(schema_version=1, created_at_utc='2026-09-22T00:00:00+00:00',
                         workspace='/mnt/nfs/fenghaibin/AI_Class',
                         container_output_root='/workspace/outputs/sweep',
                         recipe=dict(iters=1000, batch_size=2, image_size=512, workers=2, lr=1e-5),
                         seeds=[20260923, 20260924, 20260925], jobs=[])
        for seed in self.plan['seeds']:
            for mask in range(16):
                job_id = f's{seed}_m{mask:02d}'
                self.plan['jobs'].append(dict(id=job_id, seed=seed, mask=mask,
                                              flags=sweep.flags_for(mask), output_rel=f'runs/{job_id}'))
        self.save_plan()

    def save_plan(self):
        (self.root / 'plan.json').write_text(json.dumps(self.plan))

    def report(self, index=0):
        job = self.plan['jobs'][index]
        output = self.plan['container_output_root'] + '/' + job['output_rel']
        args = dict(mode='train', trained_checkpoint=False, output=output, seed=job['seed'],
                    workspace='/workspace', data_root='/workspace/data/nyuv2/development_indoor8_v1',
                    checkpoint='/workspace/checkpoints/source.pth', small_area_threshold=1024.,
                    small_crop_prob=.5, small_loss_max_weight=2., boundary_loss_weight=.1,
                    **self.plan['recipe'], **job['flags'])
        return dict(status='passed', args=args, protocol=sweep.PROTOCOL,
                    evaluation_policy=sweep.EVALUATION,
                    dataset_manifest_sha256='a' * 64, source_checkpoint_sha256='b' * 64,
                    checkpoint_sha256='c' * 64,
                    implementation_sha256={key: 'd' * 64 for key in sweep.REQUIRED_IMPLEMENTATION},
                    mmdetection_commit='e' * 40,
                    categories=[dict(id=i + 1, name=name, coco80_class_index=coco, nyuv2_source_id=nyu)
                                for i, (name, coco, nyu) in enumerate(zip(sweep.NAMES, sweep.COCO, sweep.NYU))],
                    parameters=44004879, training_images=640, final_iteration=1000,
                    final_optimizer_steps=[1000], classifier_changed=True,
                    depth_adapter_changed=job['flags']['depth_fusion'],
                    depth_adapter_l2=.1 if job['flags']['depth_fusion'] else 0., loss_records=1000,
                    first_loss=40., last_loss=10., seconds_including_evaluation=600.,
                    whole_run_peak_allocated_mib=4000., checkpoint=output + '/iter_1000.pth',
                    metrics={f'nyuv2_void/segm_{key}': .3 + job['mask'] * .001
                             for key in ('mAP', 'mAP_s', 'mAP_50', 'mAP_75', 'mAP_m', 'mAP_l')})

    def write_report(self, index=0, report=None):
        path = self.root / self.plan['jobs'][index]['output_rel'] / 'report.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(index) if report is None else report))
        return path

    def test_empty_is_incomplete_not_zero_gains(self):
        result = sweep.summarize(self.root)
        self.assertEqual(result['counts']['missing'], 48)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['combinations'][1]['delta_AP']['mean'])

    def test_report_without_baseline_has_no_delta(self):
        self.write_report(1)
        result = sweep.summarize(self.root)
        self.assertEqual(result['combinations'][1]['AP']['mean'], 30.1)
        self.assertEqual(result['combinations'][1]['delta_AP']['n'], 0)

    def test_pairs_same_seed_only_and_percentage_points(self):
        for index in (0, 1, 17):
            self.write_report(index)
        result = sweep.summarize(self.root)
        self.assertEqual(result['combinations'][1]['AP']['n'], 2)
        self.assertEqual(result['combinations'][1]['delta_AP']['n'], 1)
        self.assertEqual(result['combinations'][1]['delta_AP']['mean'], .1)
        self.assertIsNone(result['combinations'][1]['delta_AP']['sample_std'])

    def test_complete_full_factorial_and_context_effects(self):
        for index in range(48):
            self.write_report(index)
        result = sweep.summarize(self.root)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['counts']['passed'], 48)
        self.assertEqual(result['combinations'][15]['delta_AP']['mean'], 1.5)
        self.assertEqual(result['combinations'][15]['AP']['sample_std'], 0.)
        for bit, effect in enumerate(result['switch_context_paired_effects']):
            self.assertEqual(effect['delta_AP']['n'], 24)
            self.assertEqual(effect['delta_AP']['mean'], .1 * (1 << bit))
        text = sweep.markdown(result)
        self.assertIn('全部 16 种组合', text)
        self.assertIn('不是官方 NYUv2', text)
        self.assertIn('48/48', text)

    def test_failed_running_and_truncated_are_excluded(self):
        self.write_report(0, dict(status='failed', error='OOM'))
        self.write_report(1, dict(status='running'))
        path = self.write_report(2)
        path.write_text('{')
        result = sweep.summarize(self.root)
        self.assertEqual(result['counts'], dict(passed=0, missing=45, running=1, failed=1, invalid=1))
        self.assertEqual(result['invalid_count'], 1)
        self.assertIsNone(result['combinations'][2]['AP']['mean'])

    def test_invalid_evidence_never_enters_means(self):
        edits = [
            lambda r: r['args'].update(iters=999),
            lambda r: r['args'].update(seed=0),
            lambda r: r['args'].update(depth_fusion=True),
            lambda r: r.update(final_optimizer_steps=[999, 1000]),
            lambda r: r.update(final_optimizer_steps=[True]),
            lambda r: r.update(classifier_changed=False),
            lambda r: r.update(depth_adapter_changed=True),
            lambda r: r.update(depth_adapter_l2=.1),
            lambda r: r.update(loss_records=999),
            lambda r: r.update(checkpoint='/other/iter_1000.pth'),
            lambda r: r.update(checkpoint_sha256='broken'),
            lambda r: r.update(implementation_sha256={}),
            lambda r: r['categories'][0].update(nyuv2_source_id=100),
            lambda r: r['metrics'].update({'nyuv2_void/segm_mAP': float('nan')}),
            lambda r: r['metrics'].update({'nyuv2_void/segm_mAP_s': -1}),
            lambda r: r['metrics'].update({'nyuv2_void/segm_mAP': 30}),
            lambda r: r['args'].update(output='/another/run'),
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                report = self.report()
                edit(report)
                self.write_report(report=report)
                result = sweep.summarize(self.root)
                self.assertEqual(result['counts']['invalid'], 1)
                self.assertEqual(result['counts']['passed'], 0)

    def test_conflicting_provenance_quarantines_entire_comparison(self):
        self.write_report(0)
        report = self.report(1)
        report['dataset_manifest_sha256'] = 'f' * 64
        self.write_report(1, report)
        result = sweep.summarize(self.root)
        self.assertEqual(result['counts']['invalid'], 2)
        self.assertEqual(result['validated_runs'], [])
        self.assertIsNone(result['shared_training_conditions'])

    def test_unplanned_hyperparameter_difference_rejected(self):
        self.write_report(0)
        report = self.report(1)
        report['args']['small_crop_prob'] = .6
        self.write_report(1, report)
        self.assertEqual(sweep.summarize(self.root)['counts']['invalid'], 2)

    def test_pinned_provenance_rejects_bad_first_report(self):
        self.plan['expected_provenance'] = dict(dataset_manifest_sha256='f' * 64)
        self.save_plan()
        self.write_report()
        self.assertEqual(sweep.summarize(self.root)['counts']['invalid'], 1)

    def test_bad_plan_rejected(self):
        edits = [lambda p: p['jobs'].pop(),
                 lambda p: p['jobs'][0].update(output_rel='../escape'),
                 lambda p: p['jobs'][0].update(flags=sweep.flags_for(1)),
                 lambda p: p['jobs'][1].update(id=p['jobs'][0]['id']),
                 lambda p: p['seeds'].append(p['seeds'][0])]
        for edit in edits:
            with self.subTest(edit=edit):
                plan = copy.deepcopy(self.plan)
                edit(plan)
                with self.assertRaises(ValueError):
                    sweep.validate_plan(plan)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            path = self.root / self.plan['jobs'][0]['output_rel']
            path.parent.mkdir(parents=True)
            path.symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError):
                sweep.summarize(self.root)

    def test_sample_standard_deviation(self):
        result = sweep.stats([1., 2., 3.])
        self.assertEqual(result, dict(n=3, mean=2., sample_std=1.))

    def test_atomic_replace_and_no_leftover(self):
        path = self.root / 'summary.json'
        sweep.atomic_write(path, 'old')
        sweep.atomic_write(path, 'new')
        self.assertEqual(path.read_text(), 'new')
        self.assertEqual(list(self.root.glob('.summary.json.*')), [])


if __name__ == '__main__':
    unittest.main()

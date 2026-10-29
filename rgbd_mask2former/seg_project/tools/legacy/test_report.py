"""Small CPU report fixtures. Synthetic scores never represent trained-model results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from common import PROJECT, write_json
from build_report import build_report, load_experiment


class ReportEvidence(unittest.TestCase):
    def test_final_checkpoint_does_not_prove_final_validation_finished(self):
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix='report_fixture_') as temporary:
            run = Path(temporary) / 'pilot_b0'
            run.mkdir()
            (run / 'config.yaml').write_text(yaml.safe_dump({
                'SOLVER': {'MAX_ITER': 3000, 'IMS_PER_BATCH': 1},
                'PROJECT': {'EXPERIMENT': 'B0', 'EVAL_LIMIT': 0}}))
            (run / 'model_final.pth').write_bytes(b'synthetic checkpoint placeholder')
            item = load_experiment('B0', run)
            self.assertEqual(item['completed_iterations'], 3000)
            self.assertTrue(item['optimizer_updates_complete'])
            self.assertFalse(item['complete'])
            # The scheduler writes this only after train.py has exited successfully.
            write_json(run / 'training_complete.json', {'stage': 'pilot'})
            self.assertTrue(load_experiment('B0', run)['complete'])

    def test_missing_artifacts_stay_pending(self):
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix='report_fixture_') as temporary:
            project = Path(temporary)
            output = project / 'reports/missing.md'
            experiments, comparisons = build_report(['B0=' + str(project / 'outputs/missing/report_artifacts')],
                                                    output, 'Synthetic missing-artifact fixture', project)
            self.assertFalse(experiments[0]['complete'])
            self.assertEqual(comparisons, [])
            text = output.read_text()
            self.assertIn('缺实验', text)
            self.assertIn('最终选择文件尚未生成', text)
            self.assertIn('数据验证产物缺失', text)
            self.assertNotIn('完整val 2,584图 |', text)

    def test_deltas_budgets_selection_and_candidates(self):
        actual_cfg = yaml.safe_load((PROJECT / 'outputs/smoke_m1/config.yaml').read_text())
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix='report_fixture_') as temporary:
            project = Path(temporary)
            specifications = []
            for role, aps, ap in [('B0', 10, 20), ('B1', 12, 21), ('M1', 15, 20), ('A1', 11, 19)]:
                run = project / 'outputs' / ('1x_' + role.lower())
                artifact = run / 'report_artifacts'
                artifact.mkdir(parents=True)
                cfg = copy.deepcopy(actual_cfg)
                cfg['SOLVER'].update(MAX_ITER=144320, IMS_PER_BATCH=1)
                cfg['PROJECT']['EXPERIMENT'] = role
                (run / 'config.yaml').write_text(yaml.safe_dump(cfg))
                write_json(run / 'training_complete.json', dict(stage='1x'))
                write_json(artifact / 'metrics.json', dict(units='percent (0-100)', num_images=2584, smoke=False,
                    metrics=dict(segm=dict(APs=aps, AP=ap, AP50=30, AP75=10, APm=25, APl=None), bbox=dict(AP=23)),
                    selection=dict(iteration=99999)))
                write_json(artifact / 'benchmark.json', dict(smoke=False, warmup=20, iterations=200,
                    input_shape=[1, 4, 1024, 1024], device_name='Synthetic fixture GPU', torch_version='fixture',
                    score_threshold=.05, tta=False, tiling=False, latency_model_mean_ms=100 + aps,
                    fps_model=1000/(100+aps), peak_allocated_MiB=1000+aps,
                    parameters=1000000, model_state_MiB=4, checkpoint_MiB=8))
                specifications.append(f'{role}={artifact}')
            write_json(project / 'outputs/final_selection.json', dict(
                selected_directory=str(project / 'outputs/1x_m1'), selected_score_APs_AP=[15, 20],
                criterion='Synthetic fixture comparison', candidates={'fixture_m1': [15, 20]},
                test_labels_used=False, inference_score_threshold=.05))
            write_json(project / 'reports/final_error_analysis/error_candidates.json', dict(
                score_threshold=.5, small_area_definition='fixture', counts={'small_missed': 2},
                cases={'small_missed': [dict(image_id=123, annotation_id=456, semantic_type='synthetic',
                                           gt_area=42, best_iou=.1)]}))
            output = project / 'reports/synthetic.md'
            experiments, comparisons = build_report(specifications, output, 'Synthetic report fixture only', project)
            self.assertEqual(experiments[0]['epochs'], 10)
            self.assertEqual(experiments[0]['best_completed_iterations'], 100000)
            self.assertEqual(comparisons[0]['delta_APs'], 2)
            self.assertEqual(comparisons[1]['delta_AP'], -1)
            self.assertEqual(comparisons[1]['delta_latency_ms'], 3)
            self.assertEqual(comparisons[1]['delta_peak_MiB'], 3)
            self.assertTrue(comparisons[0]['comparable_accuracy'])
            self.assertTrue(comparisons[0]['comparable_speed'])
            text = output.read_text()
            self.assertIn('小目标分数增加同时整体AP下降', text)
            self.assertIn('0.208epoch', text)
            self.assertIn('1x为10epoch', text)
            self.assertIn('DDP 2x为20epoch', text)
            self.assertIn('未定义', text)
            self.assertIn('Synthetic fixture comparison', text)
            self.assertIn('| 123 | 456 | synthetic |', text)
            self.assertIn('未内嵌预测文件/模型来源', text)
            # Make unequal budget explicit; a delta alone must not become a fair ablation.
            run_b1 = project / 'outputs/1x_b1'
            cfg = yaml.safe_load((run_b1 / 'config.yaml').read_text())
            cfg['SOLVER']['MAX_ITER'] = 288640
            (run_b1 / 'config.yaml').write_text(yaml.safe_dump(cfg))
            _, unequal = build_report(specifications, output, 'Synthetic unequal-budget fixture', project)
            self.assertFalse(unequal[0]['comparable_accuracy'])
            self.assertIn('SOLVER.MAX_ITER', unequal[0]['accuracy_notes'][0])
            component = project / 'outputs/1x_m1_no_small_anchors'
            artifact = component / 'report_artifacts'
            artifact.mkdir(parents=True)
            cfg['PROJECT']['EXPERIMENT'] = 'M1_NO_SMALL_ANCHORS'
            cfg['SOLVER']['MAX_ITER'] = 144320
            (component / 'config.yaml').write_text(yaml.safe_dump(cfg))
            write_json(component / 'training_complete.json', dict(stage='1x'))
            write_json(artifact / 'metrics.json', dict(units='percent (0-100)', num_images=2584, smoke=False,
                       metrics=dict(segm=dict(APs=14, AP=19))))
            specifications.append(f'M1_NO_SMALL_ANCHORS={artifact}')
            _, components = build_report(specifications, output, 'Synthetic component fixture', project)
            self.assertEqual(components[-1]['left'], 'M1_NO_SMALL_ANCHORS')
            self.assertEqual(components[-1]['delta_APs'], 1)

    def test_real_smoke_input_is_not_formal(self):
        artifact = PROJECT / 'outputs/smoke_m1/report_artifacts'
        if not (artifact / 'metrics_smoke.json').is_file():
            self.skipTest('real smoke artifacts not available')
        with tempfile.TemporaryDirectory(dir=PROJECT, prefix='report_fixture_') as temporary:
            output = Path(temporary) / 'report.md'
            experiments, _ = build_report([f'M1={artifact}'], output, 'Real smoke contract only')
            item = experiments[0]
            self.assertFalse(item['full_val'])
            self.assertFalse(item['standard_speed'])
            self.assertEqual(item['metrics']['num_images'], 4)
            self.assertAlmostEqual(item['metrics']['metrics']['segm']['APs'], 3.029869129952497)
            self.assertIn('smoke/子集，仅链路验证', output.read_text())
            self.assertIn('smoke/标准条件未满足', output.read_text())
            self.assertIn('p1=0.0550m', output.read_text())


if __name__ == '__main__':
    unittest.main()

"""Guard against premature evaluation, invalid selection, and silent control drift."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import finalize_continue_results as workflow
from common import write_json


def result(name, aps=.5, ap=.8, images=2584):
    return dict(name=name, num_images=images, metrics=dict(APs=aps, AP=ap, AP50=.9, AP75=.85))


class ResultWorkflowTests(unittest.TestCase):
    def test_training_must_finish_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'manager_state.json', dict(status='running', manager_pid=123))
            with patch.object(workflow, 'alive', return_value=True):
                self.assertFalse(workflow.training_complete(root))
            with patch.object(workflow, 'alive', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'manager died'):
                    workflow.training_complete(root)

    def test_failed_training_never_proceeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'manager_state.json', dict(status='failed', exit_code=1))
            (root / 'manager.exit_code').write_text('1')
            with self.assertRaises(RuntimeError):
                workflow.training_complete(root)

    def test_success_requires_final_checkpoint_and_dead_launcher(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / 'manager_state.json', dict(status='completed', exit_code=0, training_pid=123))
            (root / 'manager.exit_code').write_text('0')
            (root / 'last_checkpoint').write_text('model_final.pth')
            with patch.object(workflow, 'alive', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'Missing final'):
                    workflow.training_complete(root)
                (root / 'model_final.pth').write_bytes(b'checkpoint')
                self.assertTrue(workflow.training_complete(root))
            with patch.object(workflow, 'alive', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'still alive'):
                    workflow.training_complete(root)

    def test_lower_score_and_exact_ties_preserve_baseline(self):
        baseline = result('baseline')
        self.assertIs(workflow.choose([baseline, result('worse', .49), result('tie')]), baseline)
        self.assertEqual(workflow.choose([baseline, result('better', .51)])['name'], 'better')
        self.assertEqual(workflow.choose([baseline, result('ap_tie_break', .5, .81)])['name'], 'ap_tie_break')

    def test_selection_rejects_smoke_percentages_and_nonfinite_metrics(self):
        for invalid in (result('smoke', images=12), result('percent', aps=51), result('nan', aps=float('nan'))):
            with self.assertRaises(ValueError):
                workflow.choose([result('baseline'), invalid])

    def test_legacy_control_drift_blocks_publication(self):
        best = dict(metrics=dict(segm=dict(APs=50, AP=80, AP50=90, AP75=85)))
        self.assertAlmostEqual(workflow.check_reproduction(result('same'), best)['APs'], 0)
        with self.assertRaisesRegex(ValueError, 'drift'):
            workflow.check_reproduction(result('different', aps=.51), best)

    def scenario(self, improves, export_fails=False):
        # Exercise publication and baseline fallback with isolated artifacts and
        # external GPU/evaluation boundaries replaced, without starting training.
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            training, baseline, output = (root / name for name in ('training', 'baseline', 'output'))
            for directory in (training, baseline, output):
                directory.mkdir()
            annotations, metadata = root / 'val.json', root / 'test.json'
            annotations.write_text('{"images": []}')
            metadata.write_text('{"images": []}')
            (training / 'best_segm_APs.pth').write_bytes(b'frozen model')
            (training / 'config.yaml').write_text('INPUT: {TEST_IMAGE_SIZE: 1536}')
            aps = .51 if improves else .49
            best = dict(metrics=dict(segm=dict(APs=aps*100, AP=80, AP50=90, AP75=85)))
            write_json(training / 'best_metrics.json', best)
            write_json(training / 'manager_state.json', dict(status='completed'))
            write_json(baseline / 'val_results.json', dict(selected='baseline', results=[result('baseline')]))
            write_json(baseline / 'execution_audit.json', dict(annotations_sha256=workflow.digest(annotations)))
            (baseline / 'submission.csv').write_bytes(b'preserved baseline artifact')
            validation = dict(valid=True, submission_ready=True, num_images=2112, num_instances=123)
            write_json(baseline / 'test_complete.json', dict(selected='baseline', validation=validation,
                       sha256=workflow.digest(baseline / 'submission.csv')))
            for name, value in dict(TRAIN=training, BASELINE=baseline, ROOT=output,
                                    ANNOTATIONS=str(annotations), TEST_METADATA=str(metadata), SOURCE_NAMES=[]).items():
                stack.enter_context(patch.object(workflow, name, value))
            stack.enter_context(patch.object(workflow, 'training_complete', return_value=True))
            stack.enter_context(patch.object(workflow, 'state'))
            infer = stack.enter_context(patch.object(workflow, 'infer'))
            stack.enter_context(patch.object(workflow, 'merge', return_value=dict(num_images=2584)))
            stack.enter_context(patch.object(workflow, 'evaluate', side_effect=lambda job: result(job[0], aps)))
            def export(command, **kwargs):
                self.assertEqual(command[command.index('--metadata') + 1], str(metadata))
                if export_fails:
                    raise RuntimeError('export validation failed')
                (output / 'submission.csv').write_bytes(b'validated new submission')
                write_json(output / 'submission.csv.validation.json', validation)
            stack.enter_context(patch.object(workflow.subprocess, 'run', side_effect=export))
            if export_fails:
                with self.assertRaisesRegex(RuntimeError, 'export validation failed'):
                    workflow.finalize()
                self.assertFalse((output / 'test_complete.json').exists())
            else:
                workflow.finalize()
                completed = workflow.read_json(output / 'test_complete.json')
                self.assertEqual(completed['sha256'], workflow.digest(output / 'submission.csv'))
                self.assertEqual(completed['selected'], 'continued1536_legacy' if improves else 'baseline')
                self.assertEqual([call.args[0] for call in infer.call_args_list], ['val', 'test'] if improves else ['val'])
                self.assertEqual((baseline / 'submission.csv').read_bytes(), b'preserved baseline artifact')

    def test_workflow_preserves_baseline_without_test_inference(self):
        self.scenario(False)

    def test_workflow_exports_improved_model_using_only_test_metadata(self):
        self.scenario(True)

    def test_workflow_does_not_publish_failed_export(self):
        self.scenario(True, export_fails=True)


if __name__ == '__main__':
    unittest.main()

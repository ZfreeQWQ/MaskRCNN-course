"""CPU-only contracts for fixed-eight post-acceptance course diagnostics."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import course_directions_diagnostics as diag
from run_course_directions import (BASE_RECIPE, INITIAL_SHA, NEW_FLAGS, OLD_FLAGS,
                                   TRAIN_SHA, VAL_SHA, jobs)


def fixture(name='detail_long'):
    job = next(job for job in jobs() if job['id'] == name)
    flags = dict.fromkeys((*OLD_FLAGS, *NEW_FLAGS, 'multiscale_depth', 'reliability_gate',
                           'functional_smoke', 'trained_checkpoint'), False)
    flags.update(job['flags'])
    metrics = {'course/segm_mAP' + suffix: value for suffix, value in zip(
        ('', '_50', '_75', '_s', '_m', '_l'), (.676, .888, .749, .284, .696, .941))}
    size = 1536 if flags['course_high_resolution'] else 1024
    steps = list(range(5000, 30001, 5000))
    return dict(status='passed', args=dict(BASE_RECIPE, seed=20261013, **flags),
                final_iteration=30000, final_optimizer_steps=[30000],
                training_images=14432, validation_images=2584,
                protocol='component-official-train-val-standard-coco-mask-maxdets100',
                test_inputs_used=False, test_annotations_used=False,
                finite_checks=dict(model=True, optimizer=True, loss=True, grad=True),
                train_annotations_sha256=TRAIN_SHA, val_annotations_sha256=VAL_SHA,
                source_checkpoint_sha256=INITIAL_SHA,
                effective_train_size=size, effective_eval_size=size,
                metrics=metrics, learning_curve=[dict(iteration=s, metrics=dict(metrics)) for s in steps],
                validation_coverage=[dict(iteration=s, completed_images=2584, unique_images=2584) for s in steps],
                implementation_sha256={'train_course_directions.py': 'training-source'})


def pipeline(size=1024):
    return [dict(type='LoadCourseRGBDImage'),
            dict(type='RGBDResize', scale=(round(size*1333/800), size), keep_ratio=True),
            dict(type='LoadAnnotations', with_mask=True, with_bbox=True),
            dict(type='PackRGBDDetInputs')]


def val_doc():
    identifiers = [*diag.FIXED_IDS, *range(1, 2577)]
    return dict(images=[dict(id=i, file_name=f'rgb/val/{i}.png',
                            depth_file_name=f'depth/val/{i}.png', width=1024, height=1024,
                            depth_scale_m=.001, depth_invalid_code=0, scene_id=str(i))
                       for i in identifiers])


class DirectionsDiagnosticsTests(unittest.TestCase):
    def test_exact_six_accepted_arm_identities(self):
        for job in jobs():
            name = job['id']
            self.assertEqual(diag.accepted_identity(fixture(name), name),
                             1536 if name == 'high_resolution_long' else 1024)

    def test_unknown_arm_is_not_silently_diagnosed(self):
        with self.assertRaisesRegex(ValueError, 'six'):
            diag.accepted_identity(fixture(), 'another_algorithm')

    def test_rejects_smoke_or_nonaccepted_run(self):
        for changes in (dict(status='running'), dict(final_iteration=10000),
                        dict(final_optimizer_steps=[25000]), dict(training_images=64),
                        dict(validation_images=8), dict(test_inputs_used=True),
                        dict(test_annotations_used=True)):
            report = fixture(); report.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                diag.accepted_identity(report, 'detail_long')
        report = fixture(); report['args']['functional_smoke'] = True
        with self.assertRaises(ValueError):
            diag.accepted_identity(report, 'detail_long')

    def test_nonfinite_acceptance_and_hash_changes_fail(self):
        for field in ('train_annotations_sha256', 'val_annotations_sha256', 'source_checkpoint_sha256'):
            report = fixture(); report[field] = 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                diag.accepted_identity(report, 'detail_long')
        report = fixture(); report['finite_checks']['grad'] = False
        with self.assertRaises(ValueError):
            diag.accepted_identity(report, 'detail_long')

    def test_budget_seed_and_learning_rate_are_not_interchangeable(self):
        for key, value in (('iters', 10000), ('seed', 20261014), ('lr', 1e-4),
                           ('batch_size', 1), ('schedule', 'constant')):
            report = fixture(); report['args'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'recipe'):
                diag.accepted_identity(report, 'detail_long')

    def test_switches_match_exact_single_treatment(self):
        for key in ('course_copy_paste', 'course_high_resolution', 'course_small_boundary', 'rgb_control'):
            report = fixture(); report['args'][key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'flags'):
                diag.accepted_identity(report, 'detail_long')
        report = fixture(); report['args']['course_detail'] = False
        with self.assertRaisesRegex(ValueError, 'flags'):
            diag.accepted_identity(report, 'detail_long')

    def test_bool_switch_cannot_be_string_or_number(self):
        for value in ('true', 1, None):
            report = fixture(); report['args']['course_detail'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                diag.accepted_identity(report, 'detail_long')

    def test_scale_cannot_be_changed_after_training(self):
        report = fixture(); report['effective_eval_size'] = 1536
        with self.assertRaisesRegex(ValueError, 'scale'):
            diag.accepted_identity(report, 'detail_long')

    def test_all_six_curves_and_coverages_are_required(self):
        for key in ('learning_curve', 'validation_coverage'):
            report = fixture(); report[key].pop(0)
            with self.subTest(key=key), self.assertRaises(ValueError):
                diag.accepted_identity(report, 'detail_long')
        report = fixture(); report['validation_coverage'][3]['unique_images'] = 2583
        with self.assertRaisesRegex(ValueError, 'coverages'):
            diag.accepted_identity(report, 'detail_long')

    def test_missing_nonfinite_and_substituted_metrics_fail(self):
        for value in (float('nan'), float('inf'), -1., 1.1, True):
            report = fixture(); report['metrics']['course/segm_mAP_s'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                diag.accepted_identity(report, 'detail_long')
        report = fixture(); report['metrics']['course/segm_mAP_s'] = .3
        with self.assertRaisesRegex(ValueError, '30k endpoint'):
            diag.accepted_identity(report, 'detail_long')

    def test_annotation_free_pipeline_preserves_every_other_setting(self):
        for size in (1024, 1536):
            original = pipeline(size); before = copy.deepcopy(original)
            result = diag.annotation_free_pipeline(original, size)
            self.assertEqual(original, before)
            self.assertEqual(result, [original[0], original[1], original[3]])
            result[1]['scale'] = (1, 1)
            self.assertEqual(original, before)

    def test_unknown_geometry_or_augmentation_fails(self):
        for size in (True, 1280, 2048):
            with self.assertRaises(ValueError):
                diag.annotation_free_pipeline(pipeline(), size)
        for mutation in ('resize', 'type', 'duplicate_annotations', 'flip'):
            value = pipeline()
            if mutation == 'resize': value[1]['scale'] = (1024, 1024)
            if mutation == 'type': value[0]['type'] = 'LoadImageFromFile'
            if mutation == 'duplicate_annotations': value.append(value[2])
            if mutation == 'flip': value.insert(2, dict(type='RGBDRandomFlip'))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                diag.annotation_free_pipeline(value, 1024)

    def test_fixed_ids_and_order_never_depend_on_metric(self):
        doc = val_doc()
        doc['images'].reverse()
        requests = diag.select_fixed(doc, '/tmp/course/prepared')
        self.assertEqual([r['img_id'] for r in requests], diag.FIXED_IDS)
        self.assertEqual(len(requests), 8)
        self.assertTrue(all('/rgb/val/' in r['img_path'] for r in requests))

    def test_missing_duplicate_or_wrong_val_count_fails(self):
        doc = val_doc(); doc['images'].pop()
        with self.assertRaises(ValueError): diag.select_fixed(doc, '/tmp/course/prepared')
        doc = val_doc(); doc['images'][0] = doc['images'][1]
        with self.assertRaises(ValueError): diag.select_fixed(doc, '/tmp/course/prepared')
        doc = val_doc(); doc['images'][0]['id'] = 5000
        with self.assertRaises(ValueError): diag.select_fixed(doc, '/tmp/course/prepared')

    def test_fixed_input_cannot_point_to_train_test_or_other_grid(self):
        for key, value in (('file_name', 'rgb/train/a.png'), ('depth_file_name', 'depth/test/a.png'),
                           ('file_name', '../escape.png'), ('width', 1536)):
            doc = val_doc(); doc['images'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                diag.select_fixed(doc, '/tmp/course/prepared')

    def test_input_hash_schema_matches_existing_fixed8_diagnostic(self):
        requests = [dict(img_id=1, img_path='r', depth_file='d')]
        with patch.object(diag, 'sha256', side_effect=lambda path: str(path) + '-hash'):
            self.assertEqual(diag.input_hashes(requests),
                             [dict(image_id=1, rgb_sha256='r-hash', depth_sha256='d-hash')])

    def test_parser_only_accepts_run_and_output_not_scale_or_ap_mode(self):
        args = diag.parser().parse_args(['--run', '/workspace/outputs/x', '--output', '/workspace/outputs/y'])
        self.assertEqual(set(vars(args)), {'run', 'output'})

    def test_nonworkspace_output_rejected_without_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'must-not-create'
            args = diag.parser().parse_args(['--run', directory, '--output', str(output)])
            with self.assertRaises(ValueError): diag.run(args)
            self.assertFalse(output.exists())

    def test_verify_requires_formal_path_final_checkpoint_and_matching_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root/'formal/runs/detail_long'
            run.mkdir(parents=True)
            source = root/'source'
            report = fixture()
            (root/'plan.json').write_text(json.dumps(dict(source_sha256=report['implementation_sha256'])))
            with patch.object(diag, 'verify_run', return_value=(report, run/'iter_30000.pth', source)):
                self.assertEqual(diag.verify_direction(run)[3], 1024)
                with self.assertRaisesRegex(ValueError, 'formal'):
                    diag.verify_direction(root/'smoke/runs/detail_long')
            with patch.object(diag, 'verify_run', return_value=(report, run/'iter_15000.pth', source)):
                with self.assertRaisesRegex(ValueError, 'final 30k'):
                    diag.verify_direction(run)
            (root/'plan.json').write_text(json.dumps(dict(source_sha256={})))
            with patch.object(diag, 'verify_run', return_value=(report, run/'iter_30000.pth', source)):
                with self.assertRaisesRegex(ValueError, 'implementation'):
                    diag.verify_direction(run)


if __name__ == '__main__':
    unittest.main(verbosity=2)

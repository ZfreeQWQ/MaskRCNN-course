"""CPU-only contract/regression tests; all images and scores are synthetic fixtures."""
import csv
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_util
from common import load_images, read_json, require_depth_metadata, write_csv, write_json
from evaluate import evaluate
from validate_submission import validate
from error_analysis import analyze


class ToolContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='seg_tools_', dir='/private/rgzndzy/seg_project')
        self.directory = Path(self.temporary.name)
        # Sparse mask has SMALL area but MEDIUM bbox; detects incorrect bbox-area segm APs.
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[1:5, 1:5] = 1
        mask[90:94, 90:94] = 1
        rle = mask_util.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('ascii')
        self.pred = dict(image_id=1, category_id=1, score=.9,
                         bbox=mask_util.toBbox(rle).tolist(), segmentation=rle)
        self.images = [dict(id=i, file_name=f'rgb/test/{i}.png', height=128, width=128, split='test') for i in [1, 2]]
        self.metadata = self.directory / 'metadata.json'
        self.json_path = self.directory / 'predictions.json'
        self.csv_path = self.directory / 'predictions.csv'
        write_json(self.metadata, {'images': self.images})
        write_json(self.json_path, [self.pred])
        write_csv(self.csv_path, self.images, [self.pred])
        self.gt = dict(images=[dict(row, split='val', file_name=f"rgb/val/{row['id']}.png") for row in self.images],
                       categories=[dict(id=1, name='component')],
                       annotations=[dict(self.pred, id=1, area=32, iscrowd=0, semantic_type='fixture')])
        self.gt['annotations'][0].pop('score')
        self.gt_path = self.directory / 'val.json'
        write_json(self.gt_path, self.gt)

    def tearDown(self):
        self.temporary.cleanup()

    def test_submission_json_empty_and_rle(self):
        result = validate(self.metadata, self.json_path, self.csv_path, expected_images=2)
        self.assertEqual(result['empty_images'], 1)
        with self.csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        value = json.loads(rows[0]['PredictionString'])
        self.assertEqual(value, [dict(category_id=1, score=.9, segmentation=self.pred['segmentation'])])
        np.testing.assert_array_equal(mask_util.decode(value[0]['segmentation']), mask_util.decode(self.pred['segmentation']))
        self.assertEqual(rows[1]['PredictionString'], '[]')

    def test_reject_blank_null_nonfinite_or_wrong_mask_csv(self):
        for value in ['', 'null', '[null]', '[NaN]', '[Infinity]',
                      json.dumps([dict(category_id=1, score=.9, segmentation=None)]),
                      json.dumps([dict(category_id=1, score=.8, segmentation=self.pred['segmentation'])])]:
            with self.subTest(value=value):
                with self.csv_path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['image_id', 'PredictionString'])
                    writer.writerow([1, value])
                    writer.writerow([2, '[]'])
                with self.assertRaises(ValueError):
                    validate(self.metadata, self.json_path, self.csv_path, 2)

    def test_legacy_boxes_require_explicit_format(self):
        write_csv(self.csv_path, self.images, [self.pred], csv_format='legacy-boxes')
        with self.assertRaisesRegex(ValueError, 'JSON'):
            validate(self.metadata, self.json_path, self.csv_path, 2)
        result = validate(self.metadata, self.json_path, self.csv_path, 2, csv_format='legacy-boxes')
        self.assertFalse(result['submission_ready'])

    def test_reject_excess_instances_and_duplicate_csv_ids(self):
        with self.assertRaisesRegex(ValueError, '100'):
            write_csv(self.csv_path, self.images, [self.pred] * 101)
        write_csv(self.csv_path, [self.images[0], self.images[0]], [self.pred])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate(self.metadata, self.json_path, self.csv_path, 2)

    def test_reject_missing_csv_image(self):
        write_csv(self.csv_path, self.images[:1], [self.pred])
        with self.assertRaisesRegex(ValueError, 'missing'):
            validate(self.metadata, self.json_path, self.csv_path, 2)

    def test_reject_corrupt_mask_bbox(self):
        broken = dict(self.pred, bbox=[1, 1, 90, 90])
        write_json(self.json_path, [broken])
        with self.assertRaisesRegex(ValueError, 'mask bounds'):
            validate(self.metadata, self.json_path, self.csv_path, 2)

    def test_test_annotation_guard(self):
        with self.assertRaisesRegex(ValueError, 'must not contain annotations'):
            load_images(self.gt_path, 'test')

    def test_null_depth_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing/invalid depth_scale_m'):
            require_depth_metadata([dict(id=1, depth_scale_m=None, depth_invalid_code=None)])
        require_depth_metadata([dict(id=1, depth_scale_m=.0001, depth_invalid_code=0,
                                     depth_file_name='depth/fixture.png')])

    def test_true_small_mask_coco_ap(self):
        result = evaluate(str(self.gt_path), [self.pred])
        self.assertAlmostEqual(result['metrics']['segm']['APs'], 100)
        self.assertAlmostEqual(result['metrics']['segm']['AP'], 100)
        empty = evaluate(str(self.gt_path), [])
        self.assertEqual(empty['metrics']['segm']['APs'], 0)

    def test_small_missed_and_false_positive_candidates(self):
        result = analyze(self.gt, [])
        self.assertEqual(result['counts']['small_missed'], 1)
        result = analyze(self.gt, [self.pred])
        self.assertFalse(result['counts'].get('small_missed', 0))

    def test_benchmark_smoke_and_standard_guards_on_cpu(self):
        import torch
        from benchmark import benchmark
        torch.set_num_threads(1)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, records):
                return []

        class Mapper:
            use_depth = True

            def __call__(self, record):
                return dict(image=torch.zeros((4, 1024, 1024)))

        class Predictor:
            model = Model()
            mapper = Mapper()

        images = [dict(id=1, height=1024, width=1024, file_name='fixture.png')]
        result = benchmark(Predictor(), images, '.', self.json_path, 1, 2, smoke=True)
        self.assertEqual(result['parameters'], 1)
        self.assertEqual(result['input_shape'], [1, 4, 1024, 1024])
        self.assertIsNone(result['peak_allocated_MiB'])
        self.assertTrue(result['smoke'])
        with self.assertRaisesRegex(ValueError, 'Standard benchmark requires warmup'):
            benchmark(Predictor(), images, '.', self.json_path, 1, 2)
        with self.assertRaisesRegex(ValueError, 'distinct source'):
            benchmark(Predictor(), images, '.', self.json_path, 20, 200)

    def test_merge_preserves_empty_image_coverage_and_rejects_overlap(self):
        shards = []
        for index, image_id in enumerate([1, 2]):
            path = self.directory / f'shard{index}.json'
            write_json(path, [self.pred] if image_id == 1 else [])
            write_json(str(path) + '.manifest.json', dict(
                split='test', smoke=False, image_ids=[image_id], weights='fixture', config='fixture',
                score_threshold=.05, tta=False, tiling=False, wall_seconds=1))
            shards.append(str(path))
        output = self.directory / 'merged.json'
        command = [sys.executable, str(Path(__file__).with_name('merge_predictions.py')),
                   '--metadata', str(self.metadata), '--split', 'test', '--output', str(output),
                   '--csv', str(self.csv_path), '--shards']
        subprocess.run(command + shards, check=True, capture_output=True)
        result = validate(self.metadata, output, self.csv_path, 2, str(output) + '.manifest.json')
        self.assertEqual(result['empty_images'], 1)
        failed = subprocess.run(command + [shards[0], shards[0]], capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('overlapping', failed.stderr)


if __name__ == '__main__':
    unittest.main()

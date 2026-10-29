"""Offline NYUv2 development-export regressions; never download a dataset."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    'nyu_development', Path(__file__).resolve().parents[1] / 'export_nyu_development.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
HAS_COCO = importlib.util.find_spec('pycocotools') is not None


def fixture_taxonomy():
    return {'description': 'Synthetic development fixture, not real taxonomy',
            'categories': [{'id': 1, 'name': 'book', 'coco80_class_index': 73, 'coco_name': 'book'},
                           {'id': 2, 'name': 'bottle', 'coco80_class_index': 39, 'coco_name': 'bottle'}]}


class GuardedPixels:
    def __init__(self, values, allowed, touched):
        self.values, self.allowed, self.touched = values, allowed, touched

    def __getitem__(self, index):
        if index + 1 not in self.allowed:
            raise AssertionError('Official test pixels were read')
        self.touched.append(index + 1)
        return self.values[index]


def fake_hdf5():
    scenes = ['a', 'a', 'b', 'b', 'c', 'c', 'heldout']
    names = ['book', 'bottle', 'chair']
    data, touched = {}, []
    for key, strings in [('names', names), ('scenes', scenes)]:
        references = []
        for i, value in enumerate(strings):
            ref = f'{key}_ref_{i}'
            references.append(ref)
            data[ref] = np.frombuffer(value.encode('utf-16-le'), dtype='<u2')
        data[key] = np.array([references], dtype=object)
    rgb = np.arange(7 * 3 * 4 * 3, dtype=np.uint8).reshape(7, 3, 4, 3)
    labels = np.array([[1, 1, 0, 0], [1, 2, 2, 3], [1, 2, 2, 3]], dtype=np.uint16)
    instances = np.array([[0, 1, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1]], dtype=np.uint16)
    depths = np.array([[1, 2, np.nan, 4], [2, 2, 3, 4], [0, 2, 3, 4]], dtype=np.float32)
    raw = np.array([[0, 2, np.nan, 4], [0, 2, 3, 4], [0, 2, 3, 4]], dtype=np.float32)
    maps = {'images': rgb.transpose(0, 3, 2, 1),
            'depths': np.repeat(depths.T[None], 7, axis=0),
            'rawDepths': np.repeat(raw.T[None], 7, axis=0),
            'labels': np.repeat(labels.T[None], 7, axis=0),
            'instances': np.repeat(instances.T[None], 7, axis=0)}
    # Keep one frame with no selected target: it must NOT be filtered out.
    maps['labels'][4] = 3
    for key, value in maps.items():
        data[key] = GuardedPixels(value, set(range(1, 7)), touched)
    official = {'train': [1, 2, 3, 4, 5, 6], 'test': [7]}
    return data, official, touched, rgb, labels


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.scenes = ['a', 'a', 'b', 'b', 'c', 'c', 'test']
        self.split = {'train': [1, 2, 3, 4, 5, 6], 'test': [7]}

    def test_deterministic_scene_disjoint_complete(self):
        actual = M.split_scenes(self.split, self.scenes)
        reordered = {k: list(reversed(v)) for k, v in self.split.items()}
        self.assertEqual(actual, M.split_scenes(reordered, self.scenes))
        train, val = map(set, actual['frame_ids'].values())
        self.assertEqual(train | val, set(self.split['train']))
        self.assertFalse(train & val)
        self.assertFalse(set(actual['scene_ids']['train']) & set(actual['scene_ids']['val']))
        self.assertEqual(actual['official_test_frame_ids_excluded'], [7])
        self.assertEqual(len(actual['scene_ids']['val']), 1)

    def test_rejects_official_frame_overlap(self):
        with self.assertRaisesRegex(ValueError, 'frame IDs overlap'):
            M.split_scenes({'train': [1, 2], 'test': [2]}, self.scenes)

    def test_rejects_official_scene_overlap(self):
        with self.assertRaisesRegex(ValueError, 'scenes overlap'):
            M.split_scenes({'train': [1, 3], 'test': [2]}, self.scenes)

    def test_rejects_invalid_split_and_fraction(self):
        for fraction in (0, 1, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                M.split_scenes(self.split, self.scenes, val_fraction=fraction)
        for ids in ([1, 1], [0, 3], [1.0, 3], [True, 3], [8, 3], []):
            with self.assertRaises(ValueError):
                M.split_scenes({'train': ids, 'test': [7]}, self.scenes)

    def test_degenerate_one_scene_rejected(self):
        with self.assertRaisesRegex(ValueError, 'At least two'):
            M.split_scenes({'train': [1, 2], 'test': [7]}, self.scenes)

    def test_true_dataset_metadata_split(self):
        root = Path(__file__).resolve().parents[1] / 'evidence/public_20260922/data/nyuv2'
        if not (root / 'inspection.json').exists():
            self.skipTest('Local downloaded-data inspection evidence unavailable')
        inspection = json.loads((root / 'inspection.json').read_text())
        frame_scenes = {row['frame_id']: row['scene'] for row in inspection['frames']}
        scenes = [frame_scenes[i] for i in range(1, 1450)]
        official = json.loads((root / 'official_split.json').read_text())
        result = M.split_scenes(official, scenes)
        self.assertEqual(sum(len(ids) for ids in result['frame_ids'].values()), 795)
        self.assertEqual(len(result['official_test_frame_ids_excluded']), 654)
        self.assertTrue(result['no_scene_leakage'])


class TaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nyu-taxonomy-test-')
        self.path = Path(self.temporary.name) / 'things.json'

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, value):
        M.NYU.write_json(self.path, value)
        return M.load_taxonomy(self.path, ['book', 'bottle', 'chair'])

    def test_fixed_order_is_classifier_order(self):
        value = fixture_taxonomy()
        value['categories'].reverse()
        _, categories = self.load(value)
        self.assertEqual([x['id'] for x in categories], [1, 2])
        self.assertEqual([x['coco80_class_index'] for x in categories], [39, 73])

    def test_invalid_or_duplicate_mapping_rejected(self):
        for key, wrong in [('id', 0), ('id', True), ('name', 'cup'),
                           ('coco80_class_index', 80), ('coco80_class_index', 74),
                           ('coco80_class_index', True), ('coco_name', 'chair')]:
            value = fixture_taxonomy()
            value['categories'][0][key] = wrong
            with self.assertRaises(ValueError):
                self.load(value)
        value = fixture_taxonomy()
        value['categories'].append(value['categories'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.load(value)

    def test_actual_fixed_eight_class_taxonomy(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / 'evidence/public_20260922/data/nyuv2/categories.json'
        if not evidence.exists():
            self.skipTest('Local category metadata unavailable')
        rows = json.loads(evidence.read_text())
        _, categories = M.load_taxonomy(root / 'nyuv2_things_v1.json', [r['name'] for r in rows])
        self.assertEqual([c['name'] for c in categories],
                         ['book', 'bottle', 'chair', 'bowl', 'cup', 'vase', 'sofa', 'bed'])
        self.assertEqual([c['coco80_class_index'] for c in categories], [73, 39, 56, 45, 41, 75, 57, 59])


@unittest.skipUnless(HAS_COCO, 'pycocotools required for real RLE export tests')
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nyu-export-test-')
        self.root = Path(self.temporary.name)
        self.taxonomy = self.root / 'things.json'
        M.NYU.write_json(self.taxonomy, fixture_taxonomy())
        self.output = self.root / 'export'
        self.data, self.official, self.touched, self.rgb, self.labels = fake_hdf5()
        self.source = {'mat': {'sha256': 'synthetic-no-real-data'}, 'split': {'sha256': 'fixture'}}

    def tearDown(self):
        self.temporary.cleanup()

    def export(self):
        with redirect_stdout(io.StringIO()):
            return M.export_development(self.data, self.official, self.taxonomy, self.output, self.source)

    def test_end_to_end_and_no_holdout_pixels(self):
        from PIL import Image
        from pycocotools import mask as mask_api

        manifest = self.export()
        self.assertEqual(set(self.touched), set(self.official['train']))
        self.assertEqual(manifest['exported_images'], 6)
        self.assertEqual(sum(s['images_with_no_selected_instances'] for s in manifest['statistics'].values()), 1)
        self.assertEqual(sum(s['instances']['count'] for s in manifest['statistics'].values()), 15)
        self.assertEqual(len(manifest['files']), 36)
        for split in ('train', 'val'):
            coco = json.loads((self.output / f'annotations/instances_{split}.json').read_text())
            for image in coco['images']:
                self.assertTrue(np.array_equal(np.array(Image.open(self.output / image['file_name'])),
                                               self.rgb[image['id'] - 1]))
                depth = np.load(self.output / image['depth_file'], allow_pickle=False)
                self.assertEqual(depth.dtype, np.float32)
                self.assertTrue(np.isfinite(depth).all())
                self.assertEqual(depth[0, 2], 0)
                label_valid = np.load(self.output / image['label_valid_file'], allow_pickle=False)
                if image['id'] != 5:
                    self.assertTrue(np.array_equal(label_valid, self.labels > 0))
                self.assertEqual(label_valid.dtype, np.uint8)
            for annotation in coco['annotations']:
                binary = mask_api.decode(annotation['segmentation'])
                self.assertEqual(int(binary.sum()), annotation['area'])
                self.assertIn(annotation['nyuv2_instance_id'], (0, 1))
        self.assertEqual(M.verify_export(self.output), manifest)

    def test_idempotent_reuse_does_not_read_pixels_or_rewrite(self):
        first = self.export()
        before = {str(p): p.stat().st_mtime_ns for p in self.output.rglob('*') if p.is_file()}
        with patch.object(M.NYU, 'read_frame', side_effect=AssertionError('Must not reread pixels')):
            self.assertEqual(first, self.export())
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in self.output.rglob('*') if p.is_file()})

    def test_tamper_detected(self):
        manifest = self.export()
        target = self.output / next(e['path'] for e in manifest['files'] if e['path'].endswith('.npy'))
        with target.open('ab') as stream:
            stream.write(b'x')
        with self.assertRaisesRegex(ValueError, 'bytes/SHA256 mismatch'):
            M.verify_export(self.output)

    def test_manifest_tamper_detected(self):
        self.export()
        manifest_path = self.output / 'manifest.json'
        manifest_path.write_text(manifest_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest SHA256 mismatch'):
            M.verify_export(self.output)

    def test_unknown_extra_files_rejected(self):
        self.export()
        (self.output / 'another-protocol.txt').touch()
        with self.assertRaisesRegex(ValueError, 'Unexpected or missing'):
            M.verify_export(self.output)

    def test_changed_taxonomy_rejected_without_rewrite(self):
        self.export()
        value = fixture_taxonomy()
        value['description'] += ' changed'
        M.NYU.write_json(self.taxonomy, value)
        with self.assertRaisesRegex(ValueError, 'Different completed'):
            self.export()

    def test_unowned_output_rejected(self):
        self.output.mkdir()
        (self.output / 'user.txt').touch()
        with self.assertRaisesRegex(ValueError, 'Non-empty output'):
            self.export()

    def test_interrupted_export_has_no_completion_marker_and_can_resume(self):
        real_read = M.NYU.read_frame
        calls = []

        def fail_once(data, frame_id):
            calls.append(frame_id)
            if len(calls) == 2:
                raise RuntimeError('simulated interruption')
            return real_read(data, frame_id)

        with patch.object(M.NYU, 'read_frame', side_effect=fail_once):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                self.export()
        self.assertTrue((self.output / 'export_spec.json').exists())
        self.assertFalse((self.output / 'export_complete.json').exists())
        manifest = self.export()
        self.assertEqual(manifest['exported_images'], 6)
        self.assertEqual(M.verify_export(self.output), manifest)


if __name__ == '__main__':
    unittest.main()

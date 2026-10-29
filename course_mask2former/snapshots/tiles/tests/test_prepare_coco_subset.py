import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    'coco_subset', Path(__file__).resolve().parents[1] / 'prepare_coco_subset.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def source(split, start, count):
    images = [{'id': i, 'file_name': f'{i:012}.jpg', 'width': 16, 'height': 12,
               'coco_url': f'http://images.cocodataset.org/{split}/{i:012}.jpg'}
              for i in range(start, start + count)]
    anns = [{'id': i['id'], 'image_id': i['id'], 'category_id': 1,
             'segmentation': [[0, 0, 8, 0, 8, 8, 0, 8]], 'bbox': [0, 0, 8, 8],
             'area': 64, 'iscrowd': 0} for i in images if i['id'] % 3]
    return {'images': images, 'annotations': anns,
            'categories': [{'id': 1, 'name': 'fixture', 'supercategory': 'fixture'}]}


class Response(io.BytesIO):
    def __init__(self, payload, url, status=200, headers=None):
        super().__init__(payload)
        self.status, self.url = status, url
        self.headers = headers or {'Content-Length': str(len(payload))}

    def geturl(self):
        return self.url


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='coco-subset-unit-')
        self.root = Path(self.temp.name)
        self.train = source('train2017', 1, 20)
        self.val = source('val2017', 100, 8)
        buffer = io.BytesIO()
        Image.new('RGB', (16, 12), (42, 80, 90)).save(buffer, format='JPEG')
        self.jpeg = buffer.getvalue()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as output:
            for split, data in [('train2017', self.train), ('val2017', self.val)]:
                output.writestr(f'annotations/instances_{split}.json', json.dumps(data))
            output.writestr('../never-extract-me', 'unsafe archive entry')
        self.archive = archive.getvalue()
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def open(self, request, timeout):
        url = request.full_url
        self.calls.append(url)
        return Response(self.archive if url == M.ANNOTATIONS_URL else self.jpeg, url)

    def args(self):
        return argparse.Namespace(data_root=str(self.root / 'subset'), train_count=6,
                                  val_count=3, seed=20260922, workers=4)

    def run_prepare(self, args=None):
        with patch.object(M, 'urlopen', self.open), redirect_stdout(io.StringIO()):
            return M.prepare(args or self.args())

    def test_end_to_end_and_idempotent_no_network_rerun(self):
        first = self.run_prepare()
        self.assertEqual(first['statistics']['train2017']['images'], 6)
        self.assertEqual(len(first['images']), 9)
        self.assertEqual(first['leakage_checks']['train_val_image_id_overlap'], 0)
        self.assertEqual(first['leakage_checks']['train_val_file_name_overlap'], 0)
        self.assertEqual(len(self.calls), 10)
        self.assertFalse((self.root / 'never-extract-me').exists())
        self.calls.clear()
        second = self.run_prepare()
        self.assertEqual(first, second)
        self.assertEqual(self.calls, [])

    def test_verify_only_portable_subset_without_cache_or_network(self):
        expected = self.run_prepare()
        portable = self.root / 'transferred' / 'subset'
        shutil.copytree(self.root / 'subset', portable)
        self.assertFalse((portable.parent / '_coco2017_cache').exists())
        before = {str(path.relative_to(portable)): M.sha256(path)
                  for path in portable.rglob('*') if path.is_file()}
        with patch.object(M, 'urlopen', side_effect=AssertionError('must stay offline')), \
                patch.object(M, 'prepare_sources', side_effect=AssertionError('no source cache')), \
                redirect_stdout(io.StringIO()):
            actual = M.verify_subset(portable)
        self.assertEqual(expected, actual)
        after = {str(path.relative_to(portable)): M.sha256(path)
                 for path in portable.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        image = portable / expected['images'][0]['path']
        Image.new('RGB', (16, 12), (150, 1, 1)).save(image, format='JPEG')
        with self.assertRaisesRegex(RuntimeError, 'Image size/SHA256 mismatch'):
            M.verify_subset(portable)

    def test_deterministic_unfiltered_sampling(self):
        a = M.sample_split(self.train, 'train2017', 6, 20260922)
        b = M.sample_split(self.train, 'train2017', 6, 20260922)
        c = M.sample_split(self.train, 'train2017', 6, 123)
        self.assertEqual(a, b)
        self.assertNotEqual([i['id'] for i in a['images']], [i['id'] for i in c['images']])
        all_images = M.sample_split(self.train, 'train2017', 20, 20260922)
        self.assertEqual(M.statistics(all_images)['images_without_annotations'], 6)

    def test_conflicting_counts_refused_before_network(self):
        self.run_prepare()
        args = self.args()
        args.train_count = 7
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'seed/count conflict'):
            self.run_prepare(args)
        self.assertEqual(self.calls, [])

    def test_modified_subset_annotation_refused(self):
        self.run_prepare()
        annotation = self.root / 'subset/annotations/instances_train2017.json'
        # Write through the operating-system fixture, not application code.
        with annotation.open('ab') as stream:
            stream.write(b' ')
        with self.assertRaisesRegex(RuntimeError, 'overwrite conflicting file'):
            self.run_prepare()

    def test_modified_decodable_image_refused(self):
        manifest = self.run_prepare()
        image_path = self.root / 'subset' / manifest['images'][0]['path']
        Image.new('RGB', (16, 12), (100, 1, 1)).save(image_path, format='JPEG')
        with self.assertRaisesRegex(RuntimeError, 'differs from completed manifest'):
            self.run_prepare()

    def test_malformed_segmentation_rejected(self):
        self.train['annotations'][0]['segmentation'] = []
        with self.assertRaisesRegex(ValueError, 'Missing or malformed polygons'):
            M.sample_split(self.train, 'train2017', 20, 20260922)

    def test_unsafe_filename_rejected(self):
        self.train['images'][0]['file_name'] = '../outside.jpg'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            M.sample_split(self.train, 'train2017', 20, 20260922)

    def test_image_decode_and_dimensions(self):
        path = self.root / 'wrong.jpg'
        Image.new('RGB', (15, 12)).save(path, format='JPEG')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            M.validate_image(path, self.train['images'][0])

    def test_download_resume_range(self):
        destination = self.root / 'download'
        with destination.with_name('download.partial').open('wb') as stream:
            stream.write(b'ab')

        def open_range(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=2-')
            return Response(b'cdef', request.full_url, 206,
                            {'Content-Range': 'bytes 2-5/6', 'Content-Length': '4'})

        def validate(path):
            self.assertEqual(path.read_bytes(), b'abcdef')

        with patch.object(M, 'urlopen', open_range):
            M.download('https://fixture.invalid/download', destination, validate)
        self.assertEqual(destination.read_bytes(), b'abcdef')

    def test_download_server_ignores_range(self):
        destination = self.root / 'download'
        with destination.with_name('download.partial').open('wb') as stream:
            stream.write(b'junk')
        with patch.object(M, 'urlopen', lambda request, timeout:
                          Response(b'abcdef', request.full_url)):
            M.download('https://fixture.invalid/download', destination,
                       lambda path: self.assertEqual(path.read_bytes(), b'abcdef'))
        self.assertEqual(destination.read_bytes(), b'abcdef')

    def test_download_invalid_existing_preserved(self):
        destination = self.root / 'download'
        with destination.open('wb') as stream:
            stream.write(b'bad')
        with patch.object(M, 'urlopen', side_effect=AssertionError('must not download')):
            with self.assertRaises(AssertionError):
                M.download('https://fixture.invalid/download', destination,
                           lambda path: self.assertEqual(path.read_bytes(), b'good'))
        self.assertEqual(destination.read_bytes(), b'bad')

    def test_cocoeval_small_area_upper_boundary(self):
        self.train['annotations'][0]['area'] = 1024
        self.train['annotations'][1]['area'] = 1025
        self.assertEqual(M.statistics(self.train)['small_instances_including_crowd'],
                         len(self.train['annotations']) - 1)

    def test_s3_transport_and_official_source_are_both_recorded(self):
        manifest = self.run_prepare()
        self.assertEqual(manifest['sources']['url'],
                         'https://s3.amazonaws.com/images.cocodataset.org/'
                         'annotations/annotations_trainval2017.zip')
        self.assertEqual(manifest['sources']['official_source_url'],
                         'http://images.cocodataset.org/annotations/'
                         'annotations_trainval2017.zip')
        for image in manifest['images']:
            self.assertEqual(image['url'], M.COCO_TRANSPORT_BASE + image['path'])
            self.assertEqual(image['official_source_url'],
                             'http://images.cocodataset.org/' + image['path'])
        self.assertTrue(all(url.startswith(M.COCO_TRANSPORT_BASE) for url in self.calls))

    def test_http_download_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unencrypted'):
            M.download('http://fixture.invalid/image', self.root / 'image', lambda path: None)

    def test_cancelled_download_does_not_start_network(self):
        stop = threading.Event()
        stop.set()
        with patch.object(M, 'urlopen', side_effect=AssertionError('must not download')):
            with self.assertRaises(M.CancelledError):
                M.download('https://fixture.invalid/image', self.root / 'image',
                           lambda path: None, stop_event=stop)

    def test_first_failure_cancels_pending_image_queue(self):
        started = []

        def fail_or_wait(url, destination, validator, retries=4, stop_event=None):
            started.append(url)
            if len(started) == 1:
                raise OSError('fixture failure')
            self.assertTrue(stop_event.wait(2), 'failure did not signal running downloads')
            raise M.CancelledError('fixture cancellation')

        with patch.object(M, 'download', fail_or_wait):
            with self.assertRaisesRegex(OSError, 'fixture failure'):
                M.prepare_images(self.root, {'train2017': self.train, 'val2017': self.val},
                                 None, workers=1)
        # At most one racing in-flight task may start before the main thread
        # handles the first failed future; the other 26+ tasks must not run.
        self.assertLessEqual(len(started), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

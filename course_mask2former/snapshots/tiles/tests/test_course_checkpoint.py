import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from course_checkpoint import resolve_final_checkpoint


class FinalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name).resolve()
        self.report = dict(status='passed', final_iteration=30000,
                           args={'iters': 30000}, final_optimizer_steps=[30000],
                           checkpoint='/workspace/run/iter_30000.pth')
        (self.run/'iter_30000.pth').write_bytes(b'test checkpoint')

    def test_30k(self):
        self.assertEqual(resolve_final_checkpoint(self.run, self.report), self.run/'iter_30000.pth')

    def test_legacy_10k(self):
        (self.run/'iter_10000.pth').write_bytes(b'legacy')
        report = dict(status='passed', final_iteration=10000, args={'iters': 10000},
                      final_optimizer_steps=[10000], checkpoint='/host/run/iter_10000.pth')
        self.assertEqual(resolve_final_checkpoint(self.run, report), self.run/'iter_10000.pth')

    def test_reject_inconsistent_reports(self):
        mutations = [('status', 'running'), ('status', 'failed'),
                     ('final_iteration', 0), ('final_iteration', True),
                     ('final_iteration', '30000'), ('args', {'iters': 10000}),
                     ('final_optimizer_steps', [10000]),
                     ('checkpoint', '/workspace/run/iter_25000.pth'),
                     ('checkpoint', '')]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                report = copy.deepcopy(self.report)
                report[key] = value
                with self.assertRaises(ValueError):
                    resolve_final_checkpoint(self.run, report)

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            resolve_final_checkpoint(self.run/'missing', self.report)


if __name__ == '__main__':
    unittest.main()

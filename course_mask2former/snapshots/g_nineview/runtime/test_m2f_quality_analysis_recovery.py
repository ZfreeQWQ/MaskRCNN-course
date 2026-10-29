"""Regression: GT-filtered records cannot define the diagnostic image set."""
import unittest
from run_m2f_quality_select import analysis_manifest_ids


class ManifestTests(unittest.TestCase):
    def test_keeps_images_without_small_gt(self):
        manifest = {"image_ids": list(range(256))}
        pairs = [{"image_id": i, "annotation_id": i + 1000} for i in range(142)]
        self.assertEqual(analysis_manifest_ids(manifest, pairs), list(range(256)))

    def test_empty_pairs_still_keeps_manifest(self):
        self.assertEqual(len(analysis_manifest_ids({"image_ids": list(range(256))}, [])), 256)

    def test_duplicate_image_rejected(self):
        with self.assertRaises(AssertionError):
            analysis_manifest_ids({"image_ids": [0] * 256}, [])

    def test_outside_pair_rejected(self):
        with self.assertRaises(AssertionError):
            analysis_manifest_ids({"image_ids": list(range(256))},
                                  [{"image_id": 999, "annotation_id": 1}])

    def test_duplicate_gt_rejected(self):
        with self.assertRaises(AssertionError):
            analysis_manifest_ids({"image_ids": list(range(256))},
                                  [{"image_id": 1, "annotation_id": 1}] * 2)


if __name__ == "__main__":
    unittest.main()

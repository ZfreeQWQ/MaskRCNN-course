import copy
import unittest
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from course.instance_audit import audit_instances


def rle(mask):
    result = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    result["counts"] = result["counts"].decode("ascii")
    return result


class InstanceAuditTests(unittest.TestCase):
    def data(self):
        a = np.zeros((16, 16)); a[1:5, 1:5] = 1
        b = np.zeros_like(a); b[8:12, 8:12] = 1
        annotations = [dict(id=i+1, image_id=1, category_id=1, segmentation=rle(m),
                            area=16, bbox=mask_utils.toBbox(rle(m)).tolist(), iscrowd=0)
                       for i, m in enumerate((a, b))]
        coco = COCO()
        coco.dataset = dict(info={}, images=[dict(id=1, height=16, width=16)],
                            categories=[dict(id=1, name="part")], annotations=annotations)
        coco.createIndex()
        return coco, a, b

    def test_perfect_duplicate_and_merge_heuristics(self):
        coco, a, b = self.data()
        pred = [dict(image_id=1, category_id=1, score=s, segmentation=rle(m))
                for m, s in ((a, .9), (a, .8), (b, .7), (a+b, .6))]
        original = copy.deepcopy(pred)
        result = audit_instances(coco, pred, [1])
        self.assertEqual(pred, original)
        self.assertEqual(result["per_image"][0]["duplicate_extras_score_ge01"], 2)
        # a+b has IoU exactly .5, so it is not counted by the strict <.5 merge heuristic.
        self.assertEqual(result["per_image"][0]["merge_like_score_ge01"], 0)
        self.assertEqual(result["groups"]["all_small"]["counts_best_iou"]["0.95"], 2)
        self.assertEqual(result["coco_one_to_one_small_matches"]["0.75"]["matched_gt"], 2)
        self.assertEqual(result["density"]["instances_le20"]["images"], 1)

    def test_empty_predictions_no_fake_recall(self):
        coco, _, _ = self.data()
        result = audit_instances(coco, [], [1])
        self.assertEqual(result["baseline"]["mask_aps"], 0.)
        self.assertEqual(result["groups"]["all_small"]["counts_best_iou"]["0.5"], 0)
        self.assertEqual(result["groups"]["all_small"]["instances"], 2)

    def test_oracle_overlap_must_not_be_reported_as_one_to_one_recall(self):
        coco, a, b = self.data()
        pred = [dict(image_id=1, category_id=1, score=.9, segmentation=rle(a+b))]
        result = audit_instances(coco, pred, [1])
        self.assertEqual(result["groups"]["all_small"]["counts_best_iou"]["0.5"], 2)
        self.assertEqual(result["coco_one_to_one_small_matches"]["0.5"]["matched_gt"], 1)

    def test_merge_like_requires_multi_gt_coverage_and_low_iou(self):
        coco, a, b = self.data()
        merged = a+b
        merged[13:16, :] = 1
        pred = [dict(image_id=1, category_id=1, score=.9, segmentation=rle(merged))]
        result = audit_instances(coco, pred, [1])
        self.assertEqual(result["per_image"][0]["merge_like_score_ge01"], 1)
        self.assertEqual(result["groups"]["all_small"]["counts_best_iou"]["0.5"], 0)

    def test_reject_nonpanel_predictions(self):
        coco, a, _ = self.data()
        with self.assertRaises(AssertionError):
            audit_instances(coco, [dict(image_id=2, category_id=1, score=1, segmentation=rle(a))], [1])

    def test_records_previously_mutated_by_coco_loadres(self):
        coco, a, b = self.data()
        pred = [dict(image_id=1, category_id=1, score=.9, segmentation=rle(m)) for m in (a, b)]
        coco.loadRes(pred)
        self.assertIsInstance(pred[0]["bbox"], np.ndarray)
        result = audit_instances(coco, pred, [1])
        self.assertAlmostEqual(result["baseline"]["mask_aps"], 1.)
        self.assertEqual(result["coco_one_to_one_small_matches"]["0.75"]["matched_gt"], 2)


if __name__ == "__main__":
    unittest.main()

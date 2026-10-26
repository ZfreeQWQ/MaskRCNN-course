import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools import mask as masks
from course.submission import (validate_prediction, merge_shards, attach_constant_test_intrinsics,
                               remove_empty_instances, repair_empty_masks)


class SubmissionTest(unittest.TestCase):
    def example(self):
        m = np.zeros((1024,1024), dtype=np.uint8, order="F")
        m[0,0] = 1
        r = masks.encode(m)
        r["counts"] = r["counts"].decode("ascii")
        return [dict(category_id=1, score=.5, segmentation=r)]

    def test_masks_and_empty(self):
        validate_prediction(self.example(), decode=True)
        validate_prediction([])

    def test_bad_contract(self):
        for value in (float("nan"), -1, 2, True):
            p = self.example(); p[0]["score"] = value
            with self.assertRaises(ValueError): validate_prediction(p)
        with self.assertRaises(ValueError): validate_prediction(self.example()*101)
        p=self.example();p[0]["segmentation"]["size"]=[512,512]
        with self.assertRaises(ValueError): validate_prediction(p)

    def empty_instance(self):
        r = masks.encode(np.zeros((1024,1024), dtype=np.uint8, order="F"))
        r["counts"] = r["counts"].decode("ascii")
        return dict(category_id=1, score=0., segmentation=r)

    def test_empty_instance_rejected_but_empty_image_valid(self):
        with self.assertRaisesRegex(ValueError, "Empty instance masks"):
            validate_prediction([self.empty_instance()])
        validate_prediction([])

    def test_filter_preserves_one_pixel_and_scores(self):
        good = self.example()
        good[0]["score"] = 0.  # zero score is not the same as an empty mask
        kept, n = remove_empty_instances([self.empty_instance(), *good, self.empty_instance()])
        self.assertEqual(n, 2)
        self.assertEqual(kept, good)
        self.assertEqual(remove_empty_instances([self.empty_instance()]), ([], 1))

    def test_repair_empty_row_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);meta=r/"meta.json";src=r/"source.csv"
            meta.write_text(json.dumps(dict(images=[dict(id=2),dict(id=1)])))
            with src.open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["image_id","PredictionString"])
                w.writerow([1,json.dumps([self.empty_instance(),*self.example()])])
                w.writerow([2,json.dumps([self.empty_instance()])])
            result=repair_empty_masks(src,meta,r/"fixed.csv")
            self.assertEqual(result["removed_empty_masks"],2)
            self.assertEqual(result["empty_prediction_rows"],1)
            with (r/"fixed.csv").open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([row["image_id"] for row in rows],["2","1"])
            self.assertEqual(rows[0]["PredictionString"],"[]")
            self.assertEqual(json.loads(rows[1]["PredictionString"]),self.example())
            with self.assertRaises(FileExistsError): repair_empty_masks(src,meta,r/"fixed.csv")
            meta.write_text(json.dumps(dict(images=[dict(id=3)])))
            with self.assertRaises(ValueError): repair_empty_masks(src,meta,r/"bad.csv")

    def test_merge_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);meta=r/"meta.json"
            meta.write_text(json.dumps(dict(images=[dict(id=2),dict(id=1)])))
            for name, key in (("a",1),("b",2)):
                with (r/name).open("w",newline="") as f:
                    w=csv.writer(f);w.writerow(["image_id","PredictionString"])
                    w.writerow([key,json.dumps(self.example())])
            result=merge_shards([r/"a",r/"b"],meta,r/"out.csv")
            self.assertEqual(result["rows"],2)
            with (r/"out.csv").open() as f:
                self.assertEqual([x["image_id"] for x in csv.DictReader(f)],["2","1"])
            with self.assertRaises(ValueError): merge_shards([r/"a",r/"a"],meta,r/"bad.csv")
            with self.assertRaises(ValueError): merge_shards([r/"a"],meta,r/"missing.csv")
            with self.assertRaises(FileExistsError): merge_shards([r/"a",r/"b"],meta,r/"out.csv")

    def test_explicit_camera_assumption(self):
        class Mapper: pass
        m=Mapper();K=[[2,0,1],[0,2,1],[0,0,1]]
        m.cameras=dict(cameras={"1":dict(K=K,split="train")})
        attach_constant_test_intrinsics(m,[dict(image_id=3)])
        self.assertEqual(m.cameras["cameras"]["3"]["split"],"test")
        m.cameras["cameras"]["2"]=dict(K=[[3,0,1],[0,3,1],[0,0,1]],split="val")
        with self.assertRaises(ValueError): attach_constant_test_intrinsics(m,[dict(image_id=4)])


if __name__ == "__main__": unittest.main()

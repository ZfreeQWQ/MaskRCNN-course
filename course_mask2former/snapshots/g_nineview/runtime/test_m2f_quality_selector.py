import copy
import unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
import m2f_quality_selector as q


def row(x=10, y=10, w=10, h=10, score=.8):
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    rle = mu.encode(np.asfortranarray(mask)); rle["counts"] = rle["counts"].decode()
    return dict(image_id=1, category_id=1, score=score, segmentation=rle,
                bbox=[x, y, w, h])


class Coco:
    def annToRLE(self, a):
        return a["segmentation"]


class Tests(unittest.TestCase):
    def test_features(self):
        a = row()
        v = q.features(a, a)
        self.assertEqual(len(v), 13)
        self.assertEqual(v[5:8], [1., 1., 1.])
        self.assertEqual(v[-1], 0.)

    def test_features_no_extra_gt(self):
        a, b = row(), row(x=11)
        expected = q.features(a, b)
        a.update(ground_truth="forbidden_unused", scene_id="unused", image_id=999)
        self.assertEqual(q.features(a, b), expected)

    def test_features_nonfinite(self):
        with self.assertRaises(AssertionError):
            q.features(row(score=float("nan")), row())

    def test_fit_deterministic_finite(self):
        rng = np.random.default_rng(1); x = rng.normal(size=(140, 13))
        x[:, 0] = 1.
        y = np.tanh(x[:, 1]) * .2
        a, b = q.fit(x, y), q.fit(x, y)
        self.assertEqual(a, b)
        self.assertTrue(np.isfinite(q.predict(x, a)).all())
        self.assertEqual(a["trainable_parameters"], 105)

    def test_fit_constant_target(self):
        model = q.fit(np.zeros((10, 13)), np.zeros(10))
        self.assertEqual(q.predict([[0]*13], model).tolist(), [0.])

    def test_fit_validation_not_used(self):
        x = np.arange(260, dtype=float).reshape(20, 13) / 100
        model = q.fit(x, np.ones(20)*.1); saved = copy.deepcopy(model)
        q.predict(np.ones((40, 13))*1000, model)
        self.assertEqual(model, saved)

    def test_fit_nan(self):
        with self.assertRaises(AssertionError):
            q.fit(np.zeros((2, 13)), [0., float("nan")])

    def test_accept_equals_original(self):
        import m2f_cached_geometry_refine as old
        a, b = row(), row(x=11, score=.99)
        model = q.fit(np.zeros((2, 13)), [.1, .1])
        actual, changes = q.choose([a], [b], model)
        self.assertEqual(actual, old.refine([a], [b])[0])
        self.assertTrue(changes[0]["accepted"])
        self.assertEqual(actual[0]["score"], .8)

    def test_reject_keeps_original(self):
        a, b = row(), row(x=11, score=.99)
        model = q.fit(np.zeros((2, 13)), [-.1, -.1])
        actual, changes = q.choose([a], [b], model)
        self.assertEqual(actual, [a])
        self.assertFalse(changes[0]["accepted"])

    def test_zero_is_reject(self):
        model = q.fit(np.zeros((2, 13)), [0., 0.])
        self.assertEqual(q.choose([row()], [row(x=11)], model)[0], [row()])

    def test_empty_and_no_candidate(self):
        model = q.fit(np.zeros((2, 13)), [.1, .1])
        self.assertEqual(q.choose([], [], model), ([], []))
        self.assertEqual(q.choose([row()], [row(x=60)], model), ([row()], []))

    def test_same_gt_not_independent(self):
        a, b = row(), row(x=12, score=.99)
        anns = [dict(id=1, area=100, category_id=1, segmentation=a["segmentation"]),
                dict(id=2, area=100, category_id=1, segmentation=b["segmentation"])]
        p = q.supervised_pairs([a], [b], anns, Coco())[0]
        self.assertEqual(p["gt_id"], 1)
        self.assertLess(p["target_delta"], 0)

    def test_gt_tie_by_id(self):
        a, b = row(), row(x=11)
        anns = [dict(id=i, area=100, category_id=1, segmentation=a["segmentation"]) for i in (8, 3)]
        self.assertEqual(q.supervised_pairs([a], [b], anns, Coco())[0]["gt_id"], 3)

    def test_ignore_crowd_wrong_class(self):
        a, b = row(), row(x=11)
        anns = [dict(id=1, area=100, category_id=1, iscrowd=1, segmentation=a["segmentation"]),
                dict(id=2, area=100, category_id=2, segmentation=a["segmentation"])]
        p = q.supervised_pairs([a], [b], anns, Coco())[0]
        self.assertIsNone(p["gt_id"]); self.assertEqual(p["target_delta"], 0.)

    def test_request_split_guard(self):
        image = dict(id=1, file_name="rgb/train/a.png", depth_file_name="depth/train/a.png",
                     height=1024, width=1024, depth_scale_m=.0001, depth_invalid_code=0, scene_id="a")
        request = q.train_request("/tmp/quality_data", image)
        self.assertNotIn("gt_masks", request)
        for bad in ("rgb/val/a.png", "rgb/test/a.png", "../train/a.png"):
            with self.assertRaises(AssertionError):
                q.train_request("/tmp/quality_data", dict(image, file_name=bad))

    def test_manifest_deterministic_disjoint(self):
        images = [dict(id=i, scene_id=str(i//4)) for i in range(4000)]
        a = q.select_manifest(images)
        self.assertEqual(a, q.select_manifest(list(reversed(images))))
        self.assertEqual([len(a[k]) for k in ("fit", "check")], [384, 128])
        self.assertFalse({r["scene_id"] for r in a["fit"]} & {r["scene_id"] for r in a["check"]})

    def test_insufficient_manifest(self):
        with self.assertRaises(AssertionError):
            q.select_manifest([dict(id=1, scene_id="a")])

    def test_controller_deadlines(self):
        import run_m2f_quality_select as worker
        self.assertEqual(worker.START, 1792482558)
        self.assertEqual([worker.CPU_END-worker.START, worker.GPU_END-worker.START,
                          worker.EVAL_END-worker.START, worker.DEADLINE-worker.START],
                         [1800, 4500, 5700, 7200])

    def test_controller_gpu_command(self):
        import launch_m2f_quality_select as c
        name, args = c.command("gpu")
        self.assertEqual(args[args.index("--gpus")+1], "device=0")
        self.assertEqual(args[args.index("--memory")+1], "32g")
        self.assertEqual(args[args.index("--network")+1], "none")
        self.assertIn("--read-only", args)
        self.assertIn(c.OWNER[:10], name)
        self.assertIn("run_m2f_quality_select.py", args)

    def test_controller_cpu_no_gpu(self):
        import launch_m2f_quality_select as c
        for stage in ("cpu", "evaluate", "analyze"):
            _, args = c.command(stage)
            self.assertNotIn("--gpus", args)
            self.assertEqual(args[args.index("--runtime")+1], "runc")
            self.assertEqual(args[args.index("--memory")+1], "8g")
            self.assertIn("CUDA_VISIBLE_DEVICES=", args)
            self.assertIn("NVIDIA_VISIBLE_DEVICES=void", args)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Stdlib entry/immutability contracts; real mask fixtures run in CPU worker."""
import inspect
import unittest
import launch_m2f_candidate_reuse as launcher
import m2f_candidate_reuse as worker


class Contracts(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(worker.START, 1792478551)
        self.assertEqual(worker.DEADLINE-worker.START, 3600)
        self.assertEqual(worker.CPU_END-worker.START, 1200)
        self.assertEqual(worker.GATE_END-worker.START, 1500)
        self.assertEqual(worker.GPU_END-worker.START, 3000)

    def test_single_card_and_entry(self):
        _, gpu = launcher.command("gpu")
        _, cpu = launcher.command("cpu")
        self.assertEqual(gpu[gpu.index("--gpus")+1], "device=0")
        self.assertNotIn("--gpus", cpu)
        self.assertIn("CUDA_VISIBLE_DEVICES=", cpu)
        self.assertIn("m2f_candidate_reuse.py", gpu)
        self.assertNotIn("m2f_rle_passthrough.py", gpu)
        self.assertIn("--read-only", gpu)

    def test_scratch_and_roundtrip(self):
        source = inspect.getsource(worker.CandidateEncoder)
        self.assertEqual(source.count("mu.encode("), 1)
        self.assertIn("mu.decode(", source)
        self.assertIn("np.array_equal(decoded, self.buffer)", source)
        self.assertIn("self.buffer.fill(0)", source)
        self.assertNotIn(".astype(bool)", source)

    def test_old_reference_is_passthrough(self):
        source = inspect.getsource(worker.gpu)
        self.assertIn("previous.serialize(pred.pred_instances, identity)", source)
        self.assertIn("tiles.candidates(pred.pred_instances, view)", source)
        self.assertIn("merged = base.direct_rows(kept, identity)", source)
        self.assertNotIn("tiles.as_instances(", source)
        self.assertIn("tiles.merge_candidates(pool)", source)
        self.assertIn("refine.refine(captured, merged)", source)

    def test_cpu_reuses_compact(self):
        source = inspect.getsource(worker.cpu)
        self.assertNotIn("predictions.segm.json", source)
        self.assertNotIn("previous.bind(", source)
        self.assertIn("cpu_tests(tiles)", source)
        self.assertIn("tiles.pack_views", source)

    def test_calls_and_no_full_ap(self):
        self.assertEqual(5*(264+2*5+2*20), 1570)
        source = inspect.getsource(worker.gpu)
        self.assertNotIn("COCOeval", source)
        self.assertNotIn("feature_activity", source)
        self.assertNotIn("backward(", source)

    def test_immutable_parent(self):
        self.assertEqual(worker.PARENT_REPORT_SHA,
            "6fd44f94a433a9d07a13e16b024063158268f5884f48a52cf196d40631528a98")
        self.assertNotEqual(launcher.OUT.name, worker.PARENT.split("/")[-1])


if __name__ == "__main__":
    unittest.main()

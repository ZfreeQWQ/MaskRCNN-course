#!/usr/bin/env python
"""Queue safety checks without allocating a GPU or starting training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import run_algorithm_queue as module


class QueueTests(unittest.TestCase):
    def test_long_mp_replaces_short_parent_for_continuation_selection(self):
        candidates={"mp":{"APs":53.,"AP":82.},"mp_long":{"APs":53.,"AP":82.},"frequency":{"APs":52.,"AP":81.}}
        eligible=module.continuation_candidates(candidates)
        self.assertNotIn("mp",eligible)
        self.assertEqual(max(eligible,key=lambda k:(eligible[k]["APs"],eligible[k]["AP"])),"mp_long")

    def test_six_epoch_generated_recipe_does_not_decay_at_two_epochs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module,"OUT",Path(tmp)):
            queue=module.Queue.__new__(module.Queue)
            config=queue.config('six',Path('base.yaml'),Path('weights.pth'),Path(tmp)/'out',10824)
            recipe=module.yaml.safe_load(config.read_text())
            self.assertEqual(recipe['SOLVER']['MAX_ITER'],10824)
            self.assertEqual(recipe['SOLVER']['STEPS'],[8659,10066])
            self.assertEqual(recipe['TEST']['EVAL_PERIOD'],1804)

    def test_worker_records_success_and_failure_durably(self):
        worker = Path(__file__).with_name("algorithm_queue_worker.py")
        with tempfile.TemporaryDirectory() as tmp:
            for code in (0, 7):
                folder = Path(tmp) / str(code)
                folder.mkdir()
                (folder/"command.json").write_text(json.dumps({"command": [sys.executable, "-c", f"raise SystemExit({code})"],
                                                              "cwd": tmp, "poll_seconds": .05}))
                result = subprocess.run([sys.executable, str(worker), str(folder)], capture_output=True, timeout=10)
                record = json.loads((folder/"result.json").read_text())
                self.assertEqual(record["exit_code"], code)
                self.assertEqual(record["status"], "completed" if code == 0 else "failed")
                self.assertEqual(result.returncode, 0 if code == 0 else 1)

    def test_full_validation_rejects_subset_and_partial_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"inference").mkdir()
            (folder/"model_final.pth").touch()
            (folder/"best_segm_APs.pth").write_bytes(b"x"*1000001)
            (folder/"metrics.json").write_text(json.dumps({"iteration": 3607})+"\n")
            (folder/"best_metrics.json").write_text(json.dumps({"iteration":3607, "metrics":{"segm":{"APs":53., "AP":82.}}}))
            diagnostic = folder/"inference/round2_diagnostics.json"
            diagnostic.write_text(json.dumps({"images":16, "ranking":"mask-first"}))
            with self.assertRaises(RuntimeError):
                module.Queue.validate_full(folder, 3608, Path("config.yaml"))
            diagnostic.write_text(json.dumps({"images":2584, "ranking":"mask-first"}))
            self.assertEqual(module.Queue.validate_full(folder, 3608, Path("config.yaml"))["APs"], 53.)
            with self.assertRaises(RuntimeError):
                module.Queue.validate_full(folder, 7216, Path("config.yaml"))

    def test_failed_smoke_never_launches_formal_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "OUT", Path(tmp)), patch.object(module, "EXP", Path(tmp)):
            class FakeQueue(module.Queue):
                def __init__(self):
                    self.state = {"candidates": {}, "decisions": []}
                    self.calls = []
                def save(self):
                    pass
                def run_job(self, name, command, *args, **kwargs):
                    self.calls.append(name)
                    if name.endswith("_init_audit"):
                        Path(command[command.index("--report")+1]).write_text('{"valid": true}')
                        return True
                    return False
            queue = FakeQueue()
            queue.algorithm("frequency", Path("base.yaml"), Path("weights.pth"), 2, True, False)
            self.assertEqual(queue.calls, ["frequency_init_audit", "frequency_smoke"])
            self.assertEqual(queue.state["candidates"], {})
            self.assertEqual(queue.state["decisions"][-1]["status"], "failed_or_skipped")

    def test_process_guard_recognizes_real_trainer_only(self):
        # This test runner itself must not be mistaken for an active training job.
        self.assertNotIn(__import__("os").getpid(), [p["pid"] for p in module.trainers()])


if __name__ == "__main__":
    unittest.main()

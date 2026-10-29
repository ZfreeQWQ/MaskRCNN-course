"""Synthetic, standard-library CPU contract checks; no real training results used."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import audit_final_selection as audit_tool

PROJECT = Path(__file__).resolve().parents[1]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fixture_run(root, label, ap, aps, final=False):
    directory = root / (f"final_ddp_2x_{label}" if final else f"1x_{label}")
    directory.mkdir()
    batch, count = (4, 72160) if final else (1, 144320)
    channels = 4 if label in ("b1", "m1") else 3
    write(directory / "training_complete.json", {"stage": "2x" if final else "1x", "num_gpus": batch})
    config = f'''PROJECT:
  EXPERIMENT: {label.upper()}
  USE_DEPTH: {str(channels == 4).lower()}
  EVAL_LIMIT: 0
  SKIP_FINAL_EVAL: false
SOLVER:
  IMS_PER_BATCH: {batch}
  MAX_ITER: {count}
DATASETS:
  TRAIN:
  - blender_train
  TEST:
  - blender_val
TEST:
  DETECTIONS_PER_IMAGE: 100
  AUG:
    ENABLED: false
INPUT:
  MIN_SIZE_TEST: 1024
  MAX_SIZE_TEST: 1024
MODEL:
  ROI_HEADS:
    SCORE_THRESH_TEST: 0.05
    NMS_THRESH_TEST: 0.5
SEED: 20261016
OUTPUT_DIR: {directory}
'''
    (directory / "config.yaml").write_text(config)
    metrics = {kind: {"AP": ap, "APs": aps, "AP50": 60, "AP75": 40, "APm": 50, "APl": 60}
               for kind in ("bbox", "segm")}
    best = {"iteration": count - 1, "completed_iterations": count, "selection_metric": "segm/APs",
            "tie_breaker": "segm/AP", "metrics": metrics, "checkpoint": "best_segm_APs.pth"}
    write(directory / "best_metrics.json", best)
    write(directory / "latest_eval_metrics.json", {"iteration": count - 1, "metrics": metrics})
    (directory / "best_segm_APs.pth").write_bytes(b"synthetic-not-a-real-checkpoint")
    (directory / "model_final.pth").write_bytes(b"synthetic-not-a-real-checkpoint")
    write(directory / "best_val_predictions.json", [{"image_id": 0}])
    evaluation = {f"{kind}/{key}": value for kind, values in metrics.items() for key, value in values.items()}
    evaluation["iteration"] = count
    (directory / "metrics.json").write_text(json.dumps({"iteration": count - 1, "total_loss": 1}) + "\n" + json.dumps(evaluation) + "\n")
    write(directory / "report_artifacts/metrics.json", {
        "smoke": False, "num_images": 2584, "evaluation_scope": audit_tool.SCOPE,
        "units": "percent (0-100)", "primary_metric": "segm/APs", "metrics": metrics,
        "selection": best, "source": str(directory / "best_metrics.json"),
        "predictions": str(directory / "best_val_predictions.json")})
    size = (directory / "best_segm_APs.pth").stat().st_size
    write(directory / "report_artifacts/benchmark.json", {
        "iterations": 200, "warmup": 20, "smoke": False, "unique_source_images": 200,
        "input_shape": [1, channels, 1024, 1024], "tta": False, "tiling": False,
        "score_threshold": 0.05, "config": str(directory / "config.yaml"),
        "weights": str(directory / "best_segm_APs.pth"), "checkpoint_bytes": size,
        "checkpoint_MiB": size / 2**20, "model_ms_samples": [20] * 200,
        "read_preprocess_ms_samples": [5] * 200, "latency_model_mean_ms": 20,
        "read_preprocess_mean_ms": 5, "fps_model": 50, "fps_read_plus_model": 40,
        "device": "cuda:0", "device_name": "SYNTHETIC", "torch_version": "SYNTHETIC",
        "peak_allocated_MiB": 1000, "peak_reserved_MiB": 1200, "parameters": 100,
        "trainable_parameters": 100, "model_state_bytes": 400, "model_state_MiB": 400 / 2**20,
        "timing_scope": (audit_tool.MODEL_TIMING_SCOPE + "; read/preprocess includes "
                         + ("RGB+depth" if channels == 4 else "RGB")
                         + " decode and mapper; excludes JSON RLE encoding")})
    return directory


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_selection_", dir=PROJECT / "reports")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root / "instances_val.json"
        write(self.metadata, {"images": [{"id": i, "width": 1024, "height": 1024} for i in range(2584)]})
        dirs = {name: fixture_run(self.root, name, ap, aps) for name, ap, aps in (
            ("b0", 49, 19), ("b1", 50, 20), ("m1", 49.999, 99), ("a1", 55, 90))}
        self.final = fixture_run(self.root, "m1", 51, 21, final=True)
        self.args = argparse.Namespace(**{key: str(path) for key, path in dirs.items()},
            final=[str(self.final)], val_metadata=str(self.metadata),
            scheduler_selection=str(self.root / "scheduler.json"), output=None)

    def test_ap_drop_disqualifies_high_aps_and_rgb_control(self):
        result = audit_tool.audit(self.args)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["recommended_directory"], str(self.final))
        m1 = next(row for row in result["runs"] if row["label"] == "m1")
        self.assertFalse(m1["ap_gate_pass"])
        self.assertFalse(m1["eligible_rgbd_candidate"])
        self.assertEqual(result["scheduler_selection"]["status"], "pending")

    def test_missing_formal_completion_is_incomplete(self):
        (self.final / "training_complete.json").unlink()
        result = audit_tool.audit(self.args)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("recommended_directory", result)

    def test_b1_fallback_and_deterministic_exact_ties(self):
        rows = [
            {"label": "b1", "directory": "1x_b1", "metrics": {"segm": {"AP": 50, "APs": 20}}},
            {"label": "m1", "directory": "1x_m1", "metrics": {"segm": {"AP": 49, "APs": 99}}},
            {"label": "final", "directory": "final_ddp_2x_m1", "metrics": {"segm": {"AP": 48, "APs": 100}}}]
        self.assertEqual(audit_tool.choose(rows, 50), ("1x_b1", ["1x_b1"]))
        rows[1]["metrics"]["segm"] = {"AP": 50, "APs": 20}
        self.assertEqual(audit_tool.choose(rows, 50), ("1x_b1", ["1x_b1", "1x_m1"]))

    def test_wrong_raw_benchmark_samples_are_incomplete(self):
        path = self.final / "report_artifacts/benchmark.json"
        benchmark = audit_tool.read_json(path)
        benchmark["model_ms_samples"][0] = 200
        write(path, benchmark)
        result = audit_tool.audit(self.args)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("aggregate inconsistent" in issue for issue in result["issues"]))

    def test_realistic_rgb_rgbd_timing_scope_diff_allowed(self):
        result = audit_tool.audit(self.args)
        self.assertEqual(result["status"], "complete")
        scopes = [row["benchmark"]["timing_scope"] for row in result["runs"]]
        self.assertNotEqual(scopes[0], scopes[1])
        path = self.final / "report_artifacts/benchmark.json"
        benchmark = audit_tool.read_json(path)
        benchmark["timing_scope"] = benchmark["timing_scope"].replace("excludes JSON RLE", "includes JSON RLE")
        write(path, benchmark)
        self.assertEqual(audit_tool.audit(self.args)["status"], "incomplete")

    def test_changed_seed_or_nms_is_incomplete(self):
        path = self.final / "config.yaml"
        original = path.read_text()
        for before, after in (("SEED: 20261016", "SEED: 99"), ("NMS_THRESH_TEST: 0.5", "NMS_THRESH_TEST: 0.6")):
            with self.subTest(field=before):
                path.write_text(original.replace(before, after))
                self.assertEqual(audit_tool.audit(self.args)["status"], "incomplete")

    def test_partial_val_or_unfinished_logs_are_incomplete(self):
        path = self.final / "config.yaml"
        path.write_text(path.read_text().replace("EVAL_LIMIT: 0", "EVAL_LIMIT: 16"))
        self.assertEqual(audit_tool.audit(self.args)["status"], "incomplete")
        path.write_text(path.read_text().replace("EVAL_LIMIT: 16", "EVAL_LIMIT: 0"))
        (self.final / "metrics.json").write_text('{}\n')
        self.assertEqual(audit_tool.audit(self.args)["status"], "incomplete")

    def test_original_lexicographic_winner_fails_policy(self):
        candidates = {str(Path(self.args.b1)): [20, 50], str(Path(self.args.m1)): [99, 49.999], str(self.final): [21, 51]}
        write(Path(self.args.scheduler_selection), {"selected_directory": self.args.m1,
              "selected_score_APs_AP": [99, 49.999], "candidates": candidates, "test_labels_used": False})
        result = audit_tool.audit(self.args)
        self.assertEqual(result["status"], "complete")
        original = result["scheduler_selection"]
        self.assertTrue(original["candidate_scores_match"])
        self.assertFalse(original["ap_gate_pass"])
        self.assertFalse(original["conforms"])

    def test_cli_exit_and_output_guard(self):
        argv = [sys.executable, str(PROJECT / "tools/audit_final_selection.py")]
        for key in ("b0", "b1", "m1", "a1", "val_metadata", "scheduler_selection"):
            argv += ["--" + key.replace("_", "-"), getattr(self.args, key)]
        argv += ["--final", str(self.final)]
        guarded = subprocess.run(argv + ["--output", str(self.root / "outputs/final_selection.json")], capture_output=True, text=True)
        self.assertEqual(guarded.returncode, 2)
        self.assertFalse((self.root / "outputs").exists())
        (self.final / "training_complete.json").unlink()
        completed = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(json.loads(completed.stdout)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main(verbosity=2)

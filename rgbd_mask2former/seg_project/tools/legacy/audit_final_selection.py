"""Read-only CPU audit of explicit formal-run evidence; never selects test inputs."""
import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

POLICY = "formal-v1-2026-10-16-AP-nondecrease-vs-1x-B1"
AP_KEYS = ("AP", "AP50", "AP75", "APs", "APm", "APl")
SCOPE = "complete fixed blender_val, official COCOEvaluator, maxDets=100"
MODEL_TIMING_SCOPE = "model includes device transfer, forward, D2 postprocessing"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, minimum=0):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def resolved_scalars(path):
    """Only the scalar/block-list subset emitted by Detectron2 cfg.dump().

    This is deliberately not a general YAML loader. Unknown scalar text is kept
    verbatim; all policy-relevant keys are checked explicitly by the caller.
    """
    values, stack = {}, []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        if content.startswith("- "):
            while stack and stack[-1][0] > indent:
                stack.pop()
            key = ".".join(item[1] for item in stack)
            require(key not in values or isinstance(values[key], list), "unsupported config list")
            values.setdefault(key, []).append(content[2:].strip("\"'"))
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        require(":" in content, "unsupported resolved config syntax")
        key, raw = content.split(":", 1)
        name = ".".join([item[1] for item in stack] + [key])
        raw = raw.strip()
        if not raw:
            stack.append((indent, key))
            continue
        if raw in ("true", "false", "null"):
            value = {"true": True, "false": False, "null": None}[raw]
        else:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                value = raw
        values[name] = value
    return values


def named_path(value, expected):
    return Path(value).resolve() == expected.resolve()


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_run(directory, label, val_ids, final=False):
    directory = directory.resolve()
    result = {"directory": str(directory), "label": label, "verified": False}
    try:
        # A marker is the first gate: do not inspect active/incomplete runs.
        marker = read_json(directory / "training_complete.json")
        architecture = label if not final else directory.name.rsplit("_", 1)[-1]
        stage = marker["stage"]
        require(stage in (("1x", "2x") if final else ("1x",)), "formal stage required; pilot/smoke excluded")
        if final:
            require(architecture in ("b1", "m1") and directory.name == f"final_ddp_{stage}_{architecture}",
                    "final directory must identify final_ddp formal stage and RGB-D architecture")
        else:
            require(directory.name == f"1x_{label}", "four formal control directories must be named 1x_b0/b1/m1/a1")
        require(marker["num_gpus"] == (4 if final else 1), "unexpected training world size")
        cfg = resolved_scalars(directory / "config.yaml")
        batch, max_iter = cfg["SOLVER.IMS_PER_BATCH"], cfg["SOLVER.MAX_ITER"]
        exposures = 14432 * (10 if stage == "1x" else 20)
        require(type(batch) is int and type(max_iter) is int
                and batch == (4 if final else 1) and max_iter * batch == exposures,
                "actual config does not match formal image-exposure budget")
        require(cfg["PROJECT.EXPERIMENT"] == architecture.upper(), "architecture/config mismatch")
        require(cfg["PROJECT.USE_DEPTH"] is (architecture in ("b1", "m1")), "depth modality mismatch")
        require(cfg["PROJECT.EVAL_LIMIT"] == 0 and cfg["PROJECT.SKIP_FINAL_EVAL"] is False,
                "full final evaluation is required")
        require(cfg["DATASETS.TRAIN"] == ["blender_train"] and cfg["DATASETS.TEST"] == ["blender_val"],
                "fixed train/val datasets required")
        require(cfg["TEST.DETECTIONS_PER_IMAGE"] == 100 and cfg["TEST.AUG.ENABLED"] is False,
                "fixed maxDets=100, no TTA required")
        require(cfg["INPUT.MIN_SIZE_TEST"] == cfg["INPUT.MAX_SIZE_TEST"] == 1024,
                "fixed 1024 test resolution required")
        require(cfg["SEED"] == 20261016 and cfg["MODEL.ROI_HEADS.NMS_THRESH_TEST"] == 0.5,
                "fixed seed=20261016 and NMS=0.5 required")
        require(named_path(cfg["OUTPUT_DIR"], directory), "config output path mismatch")
        best = read_json(directory / "best_metrics.json")
        require(best["selection_metric"] == "segm/APs" and best["tie_breaker"] == "segm/AP", "unexpected checkpoint selector")
        iteration = best["iteration"]
        require(type(iteration) is int and 0 <= iteration < max_iter and best["completed_iterations"] == iteration + 1,
                "best checkpoint iteration outside run budget")
        require(best["checkpoint"] == "best_segm_APs.pth", "unexpected best checkpoint name")
        checkpoint = directory / best["checkpoint"]
        require(checkpoint.stat().st_size > 0 and (directory / "model_final.pth").stat().st_size > 0,
                "selected/final checkpoint missing or empty")
        for kind in ("segm", "bbox"):
            for key in AP_KEYS:
                value = best["metrics"][kind][key]
                require(finite(value) and value <= 100, f"undefined/invalid {kind}/{key}")
        summary_path = directory / "report_artifacts/metrics.json"
        summary = read_json(summary_path)
        require(summary["smoke"] is False and summary["num_images"] == len(val_ids) == 2584,
                "full 2584-image evaluation summary required")
        require(summary["evaluation_scope"] == SCOPE and summary["primary_metric"] == "segm/APs"
                and summary["units"] == "percent (0-100)", "unrecognized evaluation scope/units")
        require(summary["metrics"] == best["metrics"] and summary["selection"] == best,
                "report metrics/best checkpoint mismatch")
        predictions = directory / "best_val_predictions.json"
        require(named_path(summary["source"], directory / "best_metrics.json")
                and named_path(summary["predictions"], predictions), "evaluation provenance path mismatch")
        rows = read_json(predictions)
        require(isinstance(rows, list) and len(rows) > 0, "missing saved validation predictions")
        require(all(row["image_id"] in val_ids for row in rows), "prediction ID outside validation split")
        latest = read_json(directory / "latest_eval_metrics.json")
        require(latest["iteration"] == max_iter - 1, "terminal full validation has not completed")
        logged_best, terminal_train, terminal_eval = False, False, False
        with (directory / "metrics.json").open() as stream:
            for line in stream:
                row = json.loads(line)
                idx = row.get("iteration", -1)
                if idx == max_iter - 1 and finite(row.get("total_loss")):
                    terminal_train = True
                if idx == max_iter and all(row.get(f"{kind}/{key}") == latest["metrics"][kind][key]
                                           for kind in ("segm", "bbox") for key in AP_KEYS):
                    terminal_eval = True
                if idx in (iteration, iteration + 1) and all(
                        row.get(f"{kind}/{key}") == best["metrics"][kind][key]
                        for kind in ("segm", "bbox") for key in AP_KEYS):
                    logged_best = True
        require(terminal_train and terminal_eval and logged_best, "training/evaluation log does not corroborate completion and best metrics")
        bench_path = directory / "report_artifacts/benchmark.json"
        bench = read_json(bench_path)
        count = bench["iterations"]
        require(type(count) is int and count >= 200 and type(bench["warmup"]) is int and bench["warmup"] >= 20
                and bench["smoke"] is False, "standard benchmark needs 20 warmup and >=200 timed images")
        require(bench["unique_source_images"] == count and count <= len(val_ids), "timed sources not distinct/full validation-backed")
        require(bench["input_shape"] == [1, 4 if architecture in ("b1", "m1") else 3, 1024, 1024], "benchmark shape mismatch")
        require(bench["tta"] is False and bench["tiling"] is False, "TTA/tiling differs from fixed policy")
        read_modality = "RGB+depth" if architecture in ("b1", "m1") else "RGB"
        expected_timing_scope = (MODEL_TIMING_SCOPE + "; read/preprocess includes " + read_modality
                                 + " decode and mapper; excludes JSON RLE encoding")
        require(bench["timing_scope"] == expected_timing_scope,
                "benchmark must share model/RLE timing boundary and declare the correct read modality")
        require(bench["score_threshold"] == cfg["MODEL.ROI_HEADS.SCORE_THRESH_TEST"] == 0.05,
                "benchmark threshold mismatch")
        require(named_path(bench["config"], directory / "config.yaml") and named_path(bench["weights"], checkpoint),
                "benchmark does not measure the selected config/checkpoint")
        require(bench["checkpoint_bytes"] == checkpoint.stat().st_size, "benchmark checkpoint byte count mismatch")
        for sample_key in ("model_ms_samples", "read_preprocess_ms_samples"):
            samples = bench[sample_key]
            require(isinstance(samples, list) and len(samples) == count
                    and all(finite(x, 0) and x > 0 for x in samples), "benchmark raw samples invalid/missing")
        model_mean, read_mean = (statistics.mean(bench[key]) for key in ("model_ms_samples", "read_preprocess_ms_samples"))
        for key, value in (("latency_model_mean_ms", model_mean), ("read_preprocess_mean_ms", read_mean),
                           ("fps_model", 1000 / model_mean), ("fps_read_plus_model", 1000 / (model_mean + read_mean)),
                           ("checkpoint_MiB", checkpoint.stat().st_size / 2**20)):
            require(finite(bench[key]) and math.isclose(bench[key], value, rel_tol=1e-9, abs_tol=1e-9),
                    f"benchmark aggregate inconsistent with samples/bytes: {key}")
        require(bench["device"].startswith("cuda") and bench["device_name"] and bench["torch_version"],
                "GPU benchmark device/software evidence missing")
        require(finite(bench["peak_allocated_MiB"], 0) and bench["peak_allocated_MiB"] > 0
                and finite(bench["peak_reserved_MiB"], bench["peak_allocated_MiB"]), "invalid GPU memory evidence")
        for key in ("parameters", "trainable_parameters", "model_state_bytes", "model_state_MiB"):
            require(finite(bench[key]) and bench[key] > 0, f"invalid model size evidence: {key}")
        require(bench["trainable_parameters"] <= bench["parameters"] and
                math.isclose(bench["model_state_MiB"], bench["model_state_bytes"] / 2**20, rel_tol=1e-9),
                "inconsistent model size evidence")
        result.update(verified=True, architecture=architecture, stage=stage, metrics=best["metrics"],
                      best_iteration=iteration, batch=batch, max_iter=max_iter, training_image_exposures=exposures,
                      seed=cfg["SEED"], nms_threshold=cfg["MODEL.ROI_HEADS.NMS_THRESH_TEST"],
                      selected_checkpoint_bytes=checkpoint.stat().st_size,
                      benchmark={key: bench[key] for key in (
                          "iterations", "warmup", "unique_source_images", "device", "device_name", "torch_version",
                          "latency_model_mean_ms", "read_preprocess_mean_ms", "fps_model", "fps_read_plus_model",
                          "parameters", "trainable_parameters", "model_state_MiB", "checkpoint_MiB",
                          "peak_allocated_MiB", "peak_reserved_MiB", "timing_scope")},
                      evidence_sha256={str(path): hash_file(path) for path in (
                          directory / "training_complete.json", directory / "config.yaml", directory / "best_metrics.json",
                          directory / "latest_eval_metrics.json", directory / "metrics.json", summary_path, bench_path, predictions)})
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        result["error"] = str(error)
    return result


def choose(candidates, baseline_ap):
    eligible = [row for row in candidates if row["metrics"]["segm"]["AP"] >= baseline_ap]
    score = lambda row: (row["metrics"]["segm"]["APs"], row["metrics"]["segm"]["AP"])
    winning_score = max(map(score, eligible))
    tied = [row for row in eligible if score(row) == winning_score]
    tied.sort(key=lambda row: ({"b1": 0, "m1": 1}.get(row["label"], 2), row["directory"]))
    return tied[0]["directory"], [row["directory"] for row in tied]


def audit(args):
    result = {"policy": POLICY, "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "incomplete", "test_labels_used": False, "issues": [], "runs": [],
              "scheduler_selection": {"status": "not_assessed_incomplete", "path": str(Path(args.scheduler_selection).resolve())},
              "scope_limitations": [
                  "Static evidence cross-check; no model/checkpoint deserialization, inference or independent COCO AP recomputation.",
                  "Validation coverage uses full-evaluation config/summary/log evidence; training did not save per-image coverage manifest.",
                  "Benchmark source count is recorded by benchmark.py, without original timed image IDs; shared-host contention is uncontrolled.",
                  "Final DDP can change batch/LR/training duration; cross-budget gains do not isolate architecture effects."]}
    try:
        metadata_path = Path(args.val_metadata).resolve()
        require(metadata_path.name == "instances_val.json", "only explicit instances_val.json metadata is allowed")
        metadata = read_json(metadata_path)
        images = metadata["images"]
        val_ids = {row["id"] for row in images}
        require(len(images) == len(val_ids) == 2584, "expected 2584 unique validation images")
        require(all(row["width"] == row["height"] == 1024 for row in images), "validation resolution mismatch")
        result["validation_metadata_sha256"] = hash_file(metadata_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["issues"].append(f"validation metadata: {error}")
        return result
    for label in ("b0", "b1", "m1", "a1"):
        result["runs"].append(inspect_run(Path(getattr(args, label)), label, val_ids))
    for directory in args.final:
        result["runs"].append(inspect_run(Path(directory), "final", val_ids, final=True))
    paths = [row["directory"] for row in result["runs"]]
    if len(set(paths)) != len(paths):
        result["issues"].append("duplicate run directories")
    for row in result["runs"]:
        if not row["verified"]:
            result["issues"].append(f"{row['directory']}: {row.get('error', 'unverified')}")
    if result["issues"]:
        return result
    b0, b1 = result["runs"][:2]
    hardware = {(row["benchmark"]["device_name"], row["benchmark"]["torch_version"])
                for row in result["runs"]}
    if len(hardware) != 1:
        result["issues"].append("benchmark hardware/software differs across formal runs")
        return result
    for row in result["runs"]:
        segm = row["metrics"]["segm"]
        row["delta_vs_1x_b0_pp"] = {key: segm[key] - b0["metrics"]["segm"][key] for key in ("AP", "APs")}
        row["delta_vs_1x_b1_pp"] = {key: segm[key] - b1["metrics"]["segm"][key] for key in ("AP", "APs")}
        row["ap_gate_pass"] = segm["AP"] >= b1["metrics"]["segm"]["AP"]
        row["eligible_rgbd_candidate"] = row["architecture"] in ("b1", "m1") and row["ap_gate_pass"]
        row["cost_ratio_vs_1x_b1"] = {key: row["benchmark"][key] / b1["benchmark"][key] for key in (
            "latency_model_mean_ms", "read_preprocess_mean_ms", "fps_model", "fps_read_plus_model",
            "parameters", "model_state_MiB", "checkpoint_MiB", "peak_allocated_MiB", "peak_reserved_MiB")}
    candidates = [row for row in result["runs"] if row["architecture"] in ("b1", "m1")]
    recommended, ties = choose(candidates, b1["metrics"]["segm"]["AP"])
    result.update(status="complete", recommended_directory=recommended, policy_optimal_ties=ties,
                  recommendation_only=True, overall_ap_floor=b1["metrics"]["segm"]["AP"])
    scheduler = {"status": "pending", "path": str(Path(args.scheduler_selection).resolve())}
    if Path(args.scheduler_selection).exists():
        try:
            original = read_json(args.scheduler_selection)
            selected = str(Path(original["selected_directory"]).resolve())
            expected_scores = {row["directory"]: [row["metrics"]["segm"]["APs"], row["metrics"]["segm"]["AP"]] for row in candidates}
            recorded_scores = {str(Path(path).resolve()): list(score) for path, score in original["candidates"].items()}
            selected_row = next((row for row in candidates if row["directory"] == selected), None)
            scheduler.update(status="reviewed", selected_directory=selected,
                             candidate_scores_match=recorded_scores == expected_scores,
                             selected_score_matches=original.get("selected_score_APs_AP") == expected_scores.get(selected),
                             ap_gate_pass=bool(selected_row and selected_row["ap_gate_pass"]),
                             policy_optimal=selected in ties,
                             test_labels_used_false=original.get("test_labels_used") is False)
            scheduler["conforms"] = all(scheduler[key] for key in ("candidate_scores_match", "selected_score_matches",
                                                                    "ap_gate_pass", "policy_optimal", "test_labels_used_false"))
        except (OSError, ValueError, TypeError, KeyError) as error:
            scheduler.update(status="invalid", conforms=False, error=str(error))
    result["scheduler_selection"] = scheduler
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("b0", "b1", "m1", "a1"):
        parser.add_argument("--" + label, required=True)
    parser.add_argument("--final", action="append", required=True, help="Explicit final_ddp_1x/2x_b1/m1 directory; repeat if needed")
    parser.add_argument("--val-metadata", required=True)
    parser.add_argument("--scheduler-selection", required=True, help="Read-only original scheduler decision; missing means pending")
    parser.add_argument("--output", help="Optional audit JSON outside any outputs directory")
    args = parser.parse_args()
    if args.output:
        output = Path(args.output).resolve()
        if "outputs" in output.parts or output.name == "final_selection.json":
            parser.error("audit output must not overwrite outputs or final_selection.json")
        sources = [Path(getattr(args, label)).resolve() for label in ("b0", "b1", "m1", "a1")]
        sources += [Path(path).resolve() for path in args.final]
        if any(output == source or source in output.parents for source in sources):
            parser.error("audit output must not be inside an input run")
        if output in (Path(args.val_metadata).resolve(), Path(args.scheduler_selection).resolve()):
            parser.error("audit output must not overwrite input evidence")
    result = audit(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

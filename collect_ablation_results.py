"""Publish auditable progress without altering training or immutable source.

Only fixed final-checkpoint/full-val metrics enter the comparison. Intermediate
scores are intentionally excluded. Safe to run while the sequential queue runs.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from course.common import atomic_json


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="runs/ablations_v1")
    args = parser.parse_args()
    suite = Path(args.suite).resolve()
    state = json.loads((suite / "suite.json").read_text())
    source = Path(state["source"])
    base = state["baseline"]["metrics"]
    report = dict(observed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  source_commit=state["commit"], suite_status=state["status"],
                  baseline=state["baseline"], experiments=[],
                  comparison="single seed 42; fixed final 30k checkpoints; full val 2584; scores 0-1",
                  caution="Small changes are not evidence of cross-seed statistical significance.")
    lines = ["# Ablation progress", "", f"Observed: {report['observed_at']}", "",
             f"Immutable training source: `{state['commit']}`. All scores 0–1.", "",
             "Only final 30k checkpoints on all 2,584 validation images appear below.", "",
             "| Experiment | Status | APS | Delta APS | Mask AP | Delta AP |",
             "| --- | --- | --- | --- | --- | --- |",
             f"| baseline | completed | {base['mask_aps']:.6f} | 0 | {base['mask_ap']:.6f} | 0 |"]
    for name in state["order"]:
        run = suite / name
        if not (run / "status.json").exists():
            item = dict(name=name, status="pending")
        else:
            status = json.loads((run / "status.json").read_text())
            if not (run / "protocol.json").exists() or not (run / "initialization_audit.json").exists():
                item = dict(name=name, status=status["status"], step=status["step"], error=status.get("error"))
                report["experiments"].append(item)
                lines.append(f"| {name} | {item['status']} ({item['step']}/30000) | — | — | — | — |")
                continue
            protocol = json.loads((run / "protocol.json").read_text())
            initialization = json.loads((run / "initialization_audit.json").read_text())
            changed = [f for f, value in protocol["source_sha256"].items() if digest(source / f) != value]
            assert not changed, changed
            assert protocol["source_revision"]["commit"] == state["commit"]
            assert protocol["train_val_never_merged"] and not protocol["test_used"]
            assert initialization["checked_tensors"] == 555 and not initialization["mismatches"]
            item = dict(name=name, status=status["status"], step=status["step"],
                        protocol=protocol, initialization_audit=initialization,
                        source_hash_audit_passed=True)
            if status["status"] == "completed":
                metrics = json.loads((run / "val_full_final.json").read_text())
                assert status["step"] == 30000 and metrics["validation_images"] == 2584
                assert metrics["mask_size"] == [1024, 1024]
                assert Path(metrics["checkpoint"]).name == "model_final.pth"
                checkpoint = run / "model_final.pth"
                item.update(metrics=metrics, checkpoint_sha256=digest(checkpoint),
                            delta_aps=metrics["mask_aps"]-base["mask_aps"],
                            delta_ap=metrics["mask_ap"]-base["mask_ap"])
                exported = run / "model_final_inference.json"
                if exported.exists():
                    item["inference_export"] = json.loads(exported.read_text())
                    assert item["inference_export"]["strict_load_passed"]
                    assert digest(run / "model_final_inference.pth") == item["inference_export"]["sha256"]
        report["experiments"].append(item)
        if item["status"] == "completed":
            m = item["metrics"]
            lines.append(f"| {name} | completed | {m['mask_aps']:.6f} | {item['delta_aps']:+.6f} | {m['mask_ap']:.6f} | {item['delta_ap']:+.6f} |")
        else:
            lines.append(f"| {name} | {item['status']} ({item.get('step', 0)}/30000) | — | — | — | — |")
    lines.extend(["", "All completed runs passed source-hash, COCO-transfer and split-use audits.",
                  "Training-only changes do not imply faster inference; small timing differences need controlled repeats.",
                  "No combinations have been tested. Record trade-offs and negative results.",
                  "These are single-seed observations, not statistical significance claims."])
    atomic_json("reports/ablations_v1_progress.json", report)
    Path("reports/ABLATIONS_V1_PROGRESS.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

"""CPU-only analysis recovery; never reruns capture, fitting, or COCOeval."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import launch_m2f_rle_passthrough as reused
from run_m2f_quality_select import DEADLINE, read, save, sha

ROOT = Path("/mnt/nfs/fenghaibin/AI_Class")
PARENT = ROOT/"outputs/course_m2f_quality_select_20261020_v1"
OUT = ROOT/"outputs/course_m2f_quality_analysis_20261020_v1"
OWNER = "ec404c7e3bf448959214a67a43b5ad7cd"


def main():
    assert time.time() < DEADLINE, "Original deadline; no extension"
    assert not (OUT/"plan.json").exists(), "No duplicate recovery"
    assert read(PARENT/"status.json") == dict(status="failed", phase="analyze",
                                            unix=read(PARENT/"status.json")["unix"])
    prior_plan = read(PARENT/"plan.json")
    assert all(sha(PARENT/"source"/n) == h for n, h in prior_plan["source_sha256"].items())
    assert all(sha(OUT/"source"/n) == h for n, h in prior_plan["source_sha256"].items()
               if n != "run_m2f_quality_select.py")
    assert read(PARENT/"full_val_metrics.json")["status"] == "passed"
    assert read(PARENT/"gpu_report.json")["strict_parity"]
    assert read(PARENT/"gpu_released.json")["original_lock_released"]
    names = ("cpu_acceptance.json", "train_manifest.json", "gpu_report.json", "selector.json",
             "full_val_metrics.json", "predictions_frozen.json", "predictions.segm.json",
             "decisions.json", "fit_diagnostics.json", "timing.json")
    preserved = {n: sha(PARENT/n) for n in names + ("first_error.json", "worker_first_error_analyze.json")}
    for n in names:
        (OUT/n).symlink_to(PARENT/n)
    plan = dict(parent=str(PARENT), deadline_unix=DEADLINE, source_commit=sys.argv[1],
                original_plan_sha256=sha(PARENT/"plan.json"), original_artifacts=preserved,
                source_sha256={p.name: sha(p) for p in (OUT/"source").glob("*.py")},
                gpu_used=False, model_loaded=False, fitting_repeated=False,
                full_val_evaluation_repeated=False, owner=OWNER, start_unix=time.time())
    save(OUT/"plan.json", plan)
    reused.ROOT, reused.OUT, reused.OWNER = ROOT, OUT, OWNER
    name, cmd = reused.command("analyze")
    name = "ai-m2f-quality-recover-" + OWNER[:10]
    cmd[cmd.index("--name")+1] = name
    cmd[cmd.index("--memory")+1] = "8g"
    cmd[cmd.index("m2f_rle_passthrough.py")] = "run_m2f_quality_select.py"
    assert "--gpus" not in cmd and cmd[cmd.index("--runtime")+1] == "runc"
    test = cmd[:cmd.index("python")] + ["python", "-m", "unittest", "-v",
                                               "test_m2f_quality_analysis_recovery"]
    try:
        save(OUT/"status.json", dict(status="running", phase="cpu_tests", unix=time.time()))
        with (OUT/"tests.log").open("x") as log:
            subprocess.run(test, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(120, DEADLINE-time.time()))
        save(OUT/"status.json", dict(status="running", phase="analyze", unix=time.time()))
        with (OUT/"analyze.log").open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=DEADLINE-time.time())
        report = read(OUT/"report.json")
        assert report["status"] == "passed" and report["finished_unix"] < DEADLINE
        assert report["complete_val_images"] == 2584 and report["paired_small"] == 1187
        assert report["diagnostic_images"] == 256 and report["images_with_small_gt"] == 142
        assert report["full_val_evaluations"] == 1
        assert all(sha(PARENT/n) == h for n, h in preserved.items())
        assert all(sha(OUT/"source"/n) == h for n, h in plan["source_sha256"].items())
        assert all(sha(OUT/n) == h for n, h in report["artifacts_sha256"].items())
        gpu = read(PARENT/"gpu_report.json")
        assert gpu["model_state_before"] == gpu["model_state_after"]
        assert gpu["calls"] == dict(eval_calls=2725, train_calls=0)
        assert all(sha(PARENT/n) == h for n, h in gpu["artifacts"].items())
        save(OUT/"terminal_acceptance.json", dict(status="passed", unix=time.time(),
            report_sha256=sha(OUT/"report.json"), plan_sha256=sha(OUT/"plan.json"),
            original_failure_preserved=True, original_outputs_unchanged=True,
            all_artifacts_verified=True, no_gpu_or_refitting_or_ap_rerun=True))
        save(OUT/"status.json", dict(status="complete", phase="complete", unix=time.time()))
    except BaseException as exc:
        save(OUT/"first_error.json", dict(error=repr(exc), unix=time.time()))
        try:
            item = json.loads(subprocess.check_output(["docker", "inspect", name], timeout=6))[0]
            assert item["Config"]["Labels"]["ai-course-directions-owner"] == OWNER
            subprocess.run(["docker", "stop", "-t", "5", name], timeout=12, check=True)
        except subprocess.CalledProcessError:
            pass
        save(OUT/"status.json", dict(status="failed", phase="analyze", unix=time.time()))
        raise


if __name__ == "__main__":
    main()

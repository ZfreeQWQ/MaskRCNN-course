#!/usr/bin/env python
"""Reuse full-state staging, also retain the predictions backing the parent best."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("prepare_algorithm_resume.py")),
               "--source-output", args.source_output, "--output", args.output]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, check=True)
    source, output = Path(args.source_output), Path(args.output)
    if not args.smoke:
        prediction = source / "best_val_predictions.json"
        if not prediction.exists():
            raise FileNotFoundError("Parent best predictions are required for independent validation")
        shutil.copy2(prediction, output / prediction.name)
        seed = json.loads((output / "resume_seed.json").read_text())
        seed["copied_parent_best"].append(prediction.name)
        (output / "resume_seed.json").write_text(json.dumps(seed, indent=2))

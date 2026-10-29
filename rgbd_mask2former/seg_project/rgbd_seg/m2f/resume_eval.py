"""Recover an evaluation interrupted by the controlled GPU-count expansion."""
import json
from pathlib import Path


def needs_expansion_evaluation(cfg,start_iter):
    output=Path(cfg.OUTPUT_DIR)
    period=cfg.TEST.EVAL_PERIOD
    if (not (output/'gpu_expansion.json').exists() or period<=0 or start_iter<=0 or
        start_iter%period or cfg.PROJECT.EVAL_LIMIT or cfg.PROJECT.SKIP_FINAL_EVAL):
        return False
    latest=output/'latest_eval_metrics.json'
    try:
        evaluated=json.loads(latest.read_text())['iteration']
    except (FileNotFoundError,json.JSONDecodeError,KeyError):
        return True
    return evaluated<start_iter-1

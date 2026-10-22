"""Record train-only augmentation coverage before long GPU experiments."""
import json
from pathlib import Path
import numpy as np
import torch

from course.common import atomic_json, configuration
from course.experiments import mapper


def main():
    torch.set_num_threads(2)
    index = json.loads(Path("data_cache/index.json").read_text())["train"]
    statistics = json.loads(Path("data_cache/train_small_stats.json").read_text())["records"]
    candidates = [i for i, r in enumerate(statistics) if r["small"]]
    chosen = [candidates[int(i)] for i in np.linspace(0, len(candidates)-1, 16, dtype=int)]
    report = dict(source="train ONLY; deliberately small-containing diagnostic examples, not AP measurements", variants={})
    for name in ["e02_small_crop", "e03_rgbd_copy_paste", "e04_small_loss", "e05_foreground_points"]:
        np.random.seed(42)
        obj = mapper(configuration(f"configs/experiments/{name}.yaml"), True)
        rows = []
        for i in chosen:
            output = obj(index[i])
            instances = output["instances"]
            assert len(instances) <= 100 and torch.isfinite(output["image"]).all()
            assert len(instances.gt_small) == len(instances)
            assert instances.gt_masks.sum((1, 2)).min() > 0 if len(instances) else True
            rows.append(dict(image_id=index[i]["image_id"], instances=len(instances),
                             small_instances=int(instances.gt_small.sum()),
                             pasted=output["copy_paste_accepted"]))
        if name == "e03_rgbd_copy_paste":
            assert sum(r["pasted"] for r in rows) > 0, "Copy-Paste is configured but never applied"
        report["variants"][name] = rows
        print(name, "mapped", len(rows), "pasted", sum(r["pasted"] for r in rows), flush=True)
    atomic_json("reports/augmentation_audit_v1.json", report)


if __name__ == "__main__":
    main()

"""Train-only area statistics, index integrity and immutable baseline manifest."""
import hashlib
import json
from pathlib import Path
import torch

from course.common import atomic_json
from course.data import AnnotationReader


def main():
    cache = Path("data_cache")
    index_path = cache / "index.json"
    index = json.loads(index_path.read_text())
    reader = AnnotationReader()
    records = []
    maximum = 0
    for i, record in enumerate(index["train"]):
        assert record["split"] == "train"
        annotations = [a for a in reader.get(record) if not a.get("iscrowd", 0)]
        count = sum(0 < a["area"] <= 1024 for a in annotations)
        maximum = max(maximum, len(annotations))
        records.append(dict(image_id=record["image_id"], instances=len(annotations), small=count,
                            repeat_factor=1 + min(count / 4, 1)))
        if (i + 1) % 2000 == 0:
            print("TRAIN_STATISTICS", i + 1, flush=True)
    assert maximum <= 100, "Existing query capacity needs a separate experiment first"
    atomic_json(cache / "train_small_stats.json", dict(
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        records=records, maximum_instances=maximum, source="supplied train annotations ONLY"))
    run = Path("runs/baseline_r50_rgbd")
    baseline = json.loads((run / "val_full_final.json").read_text())
    weights = run / "model_final.pth"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)["model"]
    parameter_count = 43953090
    atomic_json("reports/baseline_frozen.json", dict(
        metrics=baseline, checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        checkpoint_bytes=weights.stat().st_size, checkpoint_tensors=len(checkpoint),
        model_parameters=parameter_count,
        upstream_commit="9b0651c6c1d5b3af2e6da0589b719c514ec0d69a",
        tag="baseline-rgbd-r50-30k-v1", source_commit="463528b"))
    print("READY", len(records), "MAX_INSTANCES", maximum, flush=True)


if __name__ == "__main__":
    main()

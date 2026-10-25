"""Fixed scene-stratified screening panels. Validation labels never enter training."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def load_panel(path, records):
    payload = json.loads(Path(path).read_text())
    ids = payload["image_ids"]
    if payload.get("split") != "val" or not ids or len(ids) != len(set(ids)):
        raise ValueError("Panel must contain unique validation image IDs")
    by_id = {r["image_id"]: r for r in records}
    if any(i not in by_id for i in ids):
        raise ValueError("Panel contains IDs outside validation")
    return [by_id[i] for i in ids]


def make_panels(index, coco, seed=20261025, scenes_per_panel=100):
    images = {r["image_id"]: r for r in index["val"]}
    scenes, annotations = defaultdict(list), defaultdict(list)
    for r in images.values():
        scenes[Path(r["file_name"]).stem.split("_v")[0]].append(r["image_id"])
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)
    strata = defaultdict(list)
    for scene, ids in sorted(scenes.items()):
        small = [a for i in ids for a in annotations[i] if 0 < a["area"] <= 1024 and not a.get("iscrowd", 0)]
        tiny = any(a["area"] <= 64 for a in small)
        sparse = any(a["area"] / max(a["bbox"][2] * a["bbox"][3], 1) < .25 for a in small)
        strata[f"tiny{int(tiny)}_sparse{int(sparse)}"].append(scene)
    # Proportional scene allocation; no prediction/score-based panel selection.
    total = len(scenes)
    exact = {k: scenes_per_panel * len(v) / total for k, v in strata.items()}
    counts = {k: int(v) for k, v in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - counts[k]), k))[:scenes_per_panel - sum(counts.values())]:
        counts[k] += 1
    rng = random.Random(seed)
    chosen = [[], []]
    for k, names in sorted(strata.items()):
        rng.shuffle(names)
        n = counts[k]
        if len(names) < 2 * n:
            raise ValueError("Not enough independent scenes for both panels")
        chosen[0].extend(names[:n]); chosen[1].extend(names[n:2*n])
    assert not set(chosen[0]) & set(chosen[1])
    panels = []
    for label, names in zip(("screen", "confirmation"), chosen):
        ids = []
        for scene in sorted(names):
            ids.extend(sorted(rng.sample(sorted(scenes[scene]), 2)))
        small = [a for i in ids for a in annotations[i] if 0 < a["area"] <= 1024 and not a.get("iscrowd", 0)]
        panels.append(dict(split="val", panel=label, seed=seed, scenes=sorted(names), image_ids=ids,
                           small_instances=len(small), tiny_instances=sum(a["area"] <= 64 for a in small),
                           sparse_small_instances=sum(a["area"] / max(a["bbox"][2]*a["bbox"][3], 1) < .25 for a in small),
                           scene_strata_counts=counts,
                           selection="proportional scene strata (tiny/sparse presence), two random views/scene; no model scores",
                           index_sha256=hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()))
    return panels

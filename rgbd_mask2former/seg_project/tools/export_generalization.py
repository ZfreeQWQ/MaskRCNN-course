#!/usr/bin/env python
"""Export a full-val-selected model using exactly its validation inference path."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import read_json, write_json, load_images, image_record, require_depth_metadata
from run_algorithm_queue import sha

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "blender"
METADATA = ROOT / "datasets/blender/annotations/test_images_depth.json"


def shard(args):
    import train_m2f
    import torch
    from detectron2.modeling import build_model
    import rgbd_seg.m2f.frequency_fusion
    from rgbd_seg.m2f.generalization import setup_cfg
    from rgbd_seg.m2f.checkpoint import M2FCheckpointer
    from rgbd_seg.m2f.mapper import RGBDInstanceDatasetMapper
    from rgbd_seg.m2f.round2_inference import paired_prediction
    from common import encode_instances

    class StrictCheckpointer(M2FCheckpointer):
        def _load_model(self, checkpoint):
            incompatible = super()._load_model(checkpoint)
            if incompatible.missing_keys or incompatible.unexpected_keys or incompatible.incorrect_shapes:
                raise RuntimeError("Selected checkpoint mismatch: " + str(incompatible))
            return incompatible

    out = Path(args.output_dir)
    selected = read_json(out / "selected_model.json")
    if sha(Path(selected["weights"])) != selected["weights_sha256"]:
        raise RuntimeError("Selected weights changed")
    cfg = setup_cfg(out / "config.yaml")
    assert cfg.INPUT.TEST_IMAGE_SIZE == 1536
    torch.set_num_threads(cfg.PROJECT.CPU_THREADS)
    mapper = RGBDInstanceDatasetMapper(cfg, is_train=False)
    model = build_model(cfg).eval()
    StrictCheckpointer(model).load(selected["weights"])
    images = load_images(METADATA, "test")[args.shard::4]
    require_depth_metadata(images)
    folder = out / ("shard" + str(args.shard))
    folder.mkdir(exist_ok=True)
    if (folder / "complete.json").exists():
        raise FileExistsError("Completed shard exists")
    results = []
    started = time.time()
    with torch.inference_mode():
        for index, info in enumerate(images, 1):
            prediction = paired_prediction(model, mapper(image_record(info, DATA_ROOT)), chunk_size=4)
            results.extend(encode_instances(prediction["instances"], info["id"]))
            if index % 25 == 0 or index == len(images):
                write_json(folder / "progress.json", {"processed": index, "total": len(images),
                                                        "elapsed_seconds": time.time() - started})
                print(f"{index}/{len(images)}", flush=True)
    results.sort(key=lambda row: (row["image_id"], -row["score"]))
    manifest = dict(split="test", metadata=str(METADATA), config=str(out / "config.yaml"),
                    weights=selected["weights"], weights_sha256=selected["weights_sha256"],
                    image_ids=[im["id"] for im in images], num_images=len(images), num_instances=len(results),
                    size=1536, num_shards=4, shard_index=args.shard, score_threshold=0, ranking="mask-first",
                    tta=False, tiling=False, smoke=False, strict_checkpoint_load=True,
                    wall_seconds=time.time() - started)
    write_json(folder / "predictions.json", results)
    write_json(folder / "predictions.json.manifest.json", manifest)
    write_json(folder / "complete.json", manifest)


def manager(args):
    state = read_json(args.selection)
    selected = state["recommended_model"]
    if selected.get("full_val_images") != 2584:
        raise ValueError("Only a fully validated candidate can generate a new submission")
    assert sha(Path(selected["weights"])) == selected["weights_sha256"]
    out = Path(args.output_dir)
    out.mkdir(exist_ok=False)
    write_json(out / "selected_model.json", selected)
    shutil.copyfile(Path(selected["output"]) / "config.yaml", out / "config.yaml")

    def launch(index):
        folder = out / ("shard" + str(index))
        folder.mkdir()
        command = [str(ROOT / "run.sh"), "python", "-u", str(Path(__file__).resolve()),
                   "--selection", args.selection, "--output-dir", str(out), "--shard", str(index)]
        with (folder / "job.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT.parent, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index)),
                                       stdout=log, stderr=subprocess.STDOUT)
            write_json(folder / "launch.json", {"pid": process.pid, "command": command})
            code = process.wait()
        if code or not (folder / "complete.json").exists():
            raise RuntimeError(f"Export shard {index} failed: {code}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch, range(4)))
    prediction = out / "predictions.json"
    subprocess.run([str(ROOT / "run.sh"), "python", str(ROOT / "tools/merge_predictions.py"),
                    "--metadata", str(METADATA), "--split", "test", "--shards",
                    *[str(out / f"shard{i}/predictions.json") for i in range(4)],
                    "--output", str(prediction)], cwd=ROOT.parent, check=True)
    csv = out / "submission.csv"
    subprocess.run([str(ROOT / "run.sh"), "python", str(ROOT / "tools/export_submission.py"),
                    "--metadata", str(METADATA), "--predictions", str(prediction),
                    "--manifest", str(prediction) + ".manifest.json", "--template", str(DATA_ROOT / "submission.csv"),
                    "--output", str(csv)], cwd=ROOT.parent, check=True)
    validation = read_json(str(csv) + ".validation.json")
    assert validation["valid"] and validation["submission_ready"] and validation["csv_rows"] == 2112
    write_json(out / "test_complete.json", dict(selected=selected, submission=str(csv), sha256=sha(csv),
                                               validation=validation, hidden_test_score=None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard", type=int, choices=range(4))
    args = parser.parse_args()
    if args.shard is None:
        manager(args)
    else:
        shard(args)

"""Strip optimizer/trainer state and report actual model size, without GPU use."""
import argparse
import hashlib
from pathlib import Path

import torch
from detectron2.modeling import build_model
from course.common import atomic_json, configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/blender_rgbd_r50.yaml")
    args = parser.parse_args()
    source, destination = Path(args.weights), Path(args.output)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError(destination)
    torch.set_num_threads(4)
    state = torch.load(source, map_location="cpu", weights_only=False)["model"]
    cfg = configuration(args.config, ["MODEL.DEVICE", "cpu"])
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=state), destination)
    atomic_json(destination.with_suffix(".json"), dict(
        source_training_checkpoint=str(source.resolve()),
        inference_weights=str(destination.resolve()), parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        inference_file_bytes=destination.stat().st_size, training_file_bytes=source.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), strict_load_passed=True))
    print(destination.with_suffix(".json").read_text())


if __name__ == "__main__":
    main()

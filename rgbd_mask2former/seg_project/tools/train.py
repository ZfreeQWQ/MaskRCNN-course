#!/usr/bin/env python
"""Single device, Detectron2 launch, or torchrun entry point. No dependency installation."""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "detectron2"))

import torch
import torch.distributed as dist
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.utils import comm
from rgbd_seg.checkpoint import RGBDCheckpointer
from rgbd_seg.config import setup_cfg
from rgbd_seg.data import register_datasets
from rgbd_seg.provenance import snapshot
from rgbd_seg.metrics import json_safe
from rgbd_seg.trainer import Trainer


def main(args):
    cfg = setup_cfg(args.config_file, args.opts).clone()
    cfg.defrost()
    cfg.PROJECT.EVAL_LIMIT = args.eval_limit
    cfg.PROJECT.SKIP_FINAL_EVAL = args.skip_final_eval
    if cfg.MODEL.DEVICE == "cpu":
        torch.set_num_threads(cfg.PROJECT.CPU_THREADS)
    # Scale batch, LR and all sample-count-based schedules together, retaining 1 image/card.
    # Explicit SOLVER.IMS_PER_BATCH overrides remain respected unless --batch-per-gpu is given.
    explicit_batch = "SOLVER.IMS_PER_BATCH" in args.opts
    batch = cfg.SOLVER.IMS_PER_BATCH
    if args.batch_per_gpu is not None or not explicit_batch:
        cfg.SOLVER.IMS_PER_BATCH = (args.batch_per_gpu or 1) * comm.get_world_size()
    ratio = cfg.SOLVER.IMS_PER_BATCH / batch
    if ratio != 1:
        cfg.SOLVER.BASE_LR *= ratio
        cfg.SOLVER.MAX_ITER = max(1, round(cfg.SOLVER.MAX_ITER / ratio))
        cfg.SOLVER.STEPS = tuple(max(1, round(s / ratio)) for s in cfg.SOLVER.STEPS)
        cfg.SOLVER.WARMUP_ITERS = max(1, round(cfg.SOLVER.WARMUP_ITERS / ratio))
        cfg.SOLVER.CHECKPOINT_PERIOD = max(1, round(cfg.SOLVER.CHECKPOINT_PERIOD / ratio))
        cfg.TEST.EVAL_PERIOD = max(1, round(cfg.TEST.EVAL_PERIOD / ratio)) if cfg.TEST.EVAL_PERIOD else 0
    if args.stage:
        if args.stage == "pilot":
            cfg.SOLVER.MAX_ITER = max(1, round(3000 / cfg.SOLVER.IMS_PER_BATCH))
        else:
            epochs = {"1x": 10, "2x": 20}[args.stage]
            cfg.SOLVER.MAX_ITER = round(14432 * epochs / cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.STEPS = (round(cfg.SOLVER.MAX_ITER * 0.75), round(cfg.SOLVER.MAX_ITER * 0.9))
        cfg.SOLVER.WARMUP_ITERS = min(cfg.SOLVER.WARMUP_ITERS, max(1, cfg.SOLVER.MAX_ITER // 10))
    cfg.freeze()
    register_datasets(cfg)
    default_setup(cfg, args)
    if comm.is_main_process():
        snapshot(cfg, args)
    if args.eval_only:
        model = build_model(cfg)
        RGBDCheckpointer(model, save_dir=cfg.OUTPUT_DIR, depth_init=cfg.PROJECT.DEPTH_INIT).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        results = Trainer.test(cfg, model)
        if comm.is_main_process():
            import json
            (Path(cfg.OUTPUT_DIR) / "eval_only_metrics.json").write_text(json.dumps(json_safe(results), indent=2, allow_nan=False))
        return results
    # Protect earlier runs from accidental restart; --resume restores optimizer/scheduler/iteration.
    if not args.resume and (Path(cfg.OUTPUT_DIR) / "last_checkpoint").exists():
        raise FileExistsError("Existing training checkpoint: use --resume or a fresh OUTPUT_DIR")
    if args.resume and not (Path(cfg.OUTPUT_DIR) / "last_checkpoint").exists():
        raise FileNotFoundError("--resume requested but OUTPUT_DIR has no last_checkpoint")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def torchrun_main(args):
    world = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    local_rank = int(os.environ["LOCAL_RANK"])
    cpu = "MODEL.DEVICE" in args.opts and args.opts[args.opts.index("MODEL.DEVICE") + 1] == "cpu"
    backend = args.dist_backend or ("gloo" if cpu else "nccl")
    if not cpu:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    comm.create_local_process_group(local_world)
    try:
        main(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--skip-final-eval", action="store_true", help="smoke tests only: skip end-of-training validation")
    parser.add_argument("--eval-limit", type=int, default=0, help="smoke tests only: first N val images; subset metrics cannot select best checkpoint")
    parser.add_argument("--stage", choices=("pilot", "1x", "2x"), help="3000 training images, 10 epochs or 20 epochs; omit to honor config schedule")
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default=None, help="torchrun only; runtime must support chosen backend")
    args = parser.parse_args()
    if args.eval_limit < 0 or (args.batch_per_gpu is not None and args.batch_per_gpu < 1):
        parser.error("--eval-limit must be >=0 and --batch-per-gpu must be >=1")
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        torchrun_main(args)
    else:
        launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
               dist_url=args.dist_url, args=(args,))

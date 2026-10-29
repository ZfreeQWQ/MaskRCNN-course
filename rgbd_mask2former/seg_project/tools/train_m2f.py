#!/usr/bin/env python
"""Mask2Former training entry point (docs/plan.md §6). Mirrors tools/train.py's structure
(single device / detectron2 launch / torchrun, --stage epoch scaling, resume protections,
provenance snapshot) for the Mask2Former stack instead of Mask R-CNN.

--stage epoch counts follow docs/plan.md §5-§6: 6 epoch for screening runs (B/D/S/DS/DSH/SH),
12/16/24 epoch for the chosen configuration's longer training runs. STEPS (LR decay points) are
placed at 80%/93% of MAX_ITER, matching the plan's "8 epoch ... 80% / 93%" schedule.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "detectron2"))
sys.path.insert(0, str(PROJECT / "third_party" / "msda_shim"))
sys.path.insert(0, str(PROJECT / "third_party" / "Mask2Former"))

import torch
import torch.distributed as dist
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model
from detectron2.utils import comm

from rgbd_seg.data import register_datasets
from rgbd_seg.provenance import snapshot
from rgbd_seg.metrics import json_safe
from rgbd_seg.m2f.config import setup_cfg
from rgbd_seg.m2f.checkpoint import M2FCheckpointer
from rgbd_seg.m2f.trainer import Trainer, BestMaskCheckpoint
from rgbd_seg.m2f.resume_eval import needs_expansion_evaluation
import rgbd_seg.m2f.model  # noqa: F401 -- registers RGBDMaskFormer (I2/I3 meta-arch)
import rgbd_seg.m2f.depth_fusion  # noqa: F401 -- registers build_gated_rgbd_resnet_backbone (I1)

TRAIN_IMAGES = 14432
STAGE_EPOCHS = {"screen": 6, "1x": 12, "mid": 16, "2x": 24}


def main(args):
    cfg = setup_cfg(args.config_file, args.opts).clone()
    cfg.defrost()
    cfg.PROJECT.EVAL_LIMIT = args.eval_limit
    cfg.PROJECT.SKIP_FINAL_EVAL = args.skip_final_eval
    if cfg.MODEL.DEVICE == "cpu":
        torch.set_num_threads(cfg.PROJECT.CPU_THREADS)

    # Unlike tools/train.py (Mask R-CNN), batch is NOT auto-rescaled to "1 image/gpu x world
    # size" by default: configs/mask2former/*.yaml already specify the intended total
    # SOLVER.IMS_PER_BATCH (docs/plan.md's "全局 batch 8"), and AdamW's LR doesn't have SGD's
    # well-established linear-scaling-with-batch rule, so silently rescaling LR here would be
    # unprincipled. Only override when --batch-per-gpu is explicitly passed.
    if args.batch_per_gpu is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_per_gpu * comm.get_world_size()

    if args.stage:
        epochs = STAGE_EPOCHS[args.stage]
        cfg.SOLVER.MAX_ITER = max(1, round(TRAIN_IMAGES * epochs / cfg.SOLVER.IMS_PER_BATCH))
        cfg.SOLVER.STEPS = (round(cfg.SOLVER.MAX_ITER * 0.8), round(cfg.SOLVER.MAX_ITER * 0.93))
        cfg.SOLVER.WARMUP_ITERS = min(cfg.SOLVER.WARMUP_ITERS, max(1, cfg.SOLVER.MAX_ITER // 20))
        # Full validation every 2 epochs (docs/plan.md §7's "每 2 epoch 一次完整 val"), not the
        # official Mask2Former recipe's iteration-count-based EVAL_PERIOD.
        iters_per_epoch = max(1, round(TRAIN_IMAGES / cfg.SOLVER.IMS_PER_BATCH))
        cfg.TEST.EVAL_PERIOD = iters_per_epoch * 2
        cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch * 2

    cfg.freeze()
    register_datasets(cfg)
    default_setup(cfg, args)
    if comm.is_main_process():
        snapshot(cfg, args)

    if args.eval_only:
        model = build_model(cfg)
        M2FCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        results = Trainer.test(cfg, model)
        if comm.is_main_process():
            import json
            (Path(cfg.OUTPUT_DIR) / "eval_only_metrics.json").write_text(
                json.dumps(json_safe(results), indent=2, allow_nan=False))
        return results

    if not args.resume and (Path(cfg.OUTPUT_DIR) / "last_checkpoint").exists():
        raise FileExistsError("Existing training checkpoint: use --resume or a fresh OUTPUT_DIR")
    if args.resume and not (Path(cfg.OUTPUT_DIR) / "last_checkpoint").exists():
        raise FileNotFoundError("--resume requested but OUTPUT_DIR has no last_checkpoint")

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if args.resume and needs_expansion_evaluation(cfg,trainer.start_iter):
        # PeriodicCheckpoint precedes EvalHook. A controlled GPU expansion can
        # therefore resume from a saved boundary before its validation finished.
        # Complete that validation before the next optimization update.
        trainer.iter=trainer.start_iter-1
        trainer._last_eval_results=Trainer.test(cfg,trainer.model)
        for hook in trainer._hooks:
            if isinstance(hook,BestMaskCheckpoint):
                hook.before_train()
                hook._check()
        comm.synchronize()
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
    parser.add_argument("--stage", choices=tuple(STAGE_EPOCHS), help="screen=6 epoch, 1x=12, mid=16, 2x=24 (docs/plan.md SS5-6); omit to honor config schedule")
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

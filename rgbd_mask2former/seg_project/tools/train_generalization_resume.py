#!/usr/bin/env python
"""Full-state continuation of an unchanged generalization experiment."""
import os
import train_m2f as entry
import rgbd_seg.m2f.frequency_fusion
from rgbd_seg.m2f.generalization import setup_cfg
from rgbd_seg.m2f.generalization_continuation import GeneralizationContinuationTrainer


if __name__ == "__main__":
    entry.setup_cfg = setup_cfg
    entry.Trainer = GeneralizationContinuationTrainer
    parser = entry.default_argument_parser()
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--stage", choices=tuple(entry.STAGE_EPOCHS))
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default=None)
    args = parser.parse_args()
    if not args.resume or args.eval_limit < 0:
        parser.error("Requires --resume and nonnegative --eval-limit")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        entry.torchrun_main(args)
    else:
        entry.launch(entry.main, args.num_gpus, num_machines=args.num_machines,
                     machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))

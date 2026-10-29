#!/usr/bin/env python
"""Reuse established distributed training; replace only paired validation and head LR."""
import os

import train_m2f as entry
from rgbd_seg.m2f.round2_trainer import Round2Trainer


def main(args):
    Round2Trainer.head_lr_multiplier = args.head_lr_multiplier
    Round2Trainer.inference_chunk_size = args.inference_chunk_size
    entry.Trainer = Round2Trainer
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return entry.torchrun_main(args)
    return entry.main(args)


if __name__ == "__main__":
    parser = entry.default_argument_parser()
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--stage", choices=tuple(entry.STAGE_EPOCHS))
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default=None)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--inference-chunk-size", type=int, default=4)
    args = parser.parse_args()
    if args.eval_limit < 0 or args.head_lr_multiplier <= 0 or args.inference_chunk_size < 1:
        parser.error("Invalid limit, head LR multiplier or inference chunk size")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        main(args)
    else:
        entry.launch(main, args.num_gpus, num_machines=args.num_machines,
                     machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))

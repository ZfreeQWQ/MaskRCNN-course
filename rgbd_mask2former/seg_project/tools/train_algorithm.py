#!/usr/bin/env python
"""Frequency, MP+frequency, or algorithm continuation; reuse existing runtime."""
import os
import train_m2f as entry
import rgbd_seg.m2f.mpformer
import rgbd_seg.m2f.frequency_fusion
from rgbd_seg.m2f.algorithm_trainer import AlgorithmTrainer


def main(args):
    entry.Trainer = AlgorithmTrainer
    return entry.torchrun_main(args) if int(os.environ.get("WORLD_SIZE", "1")) > 1 else entry.main(args)


if __name__ == "__main__":
    parser = entry.default_argument_parser()
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--stage", choices=tuple(entry.STAGE_EPOCHS))
    parser.add_argument("--batch-per-gpu", type=int, default=None)
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default=None)
    args = parser.parse_args()
    if args.eval_limit < 0:
        parser.error("--eval-limit must be >= 0")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        main(args)
    else:
        entry.launch(main, args.num_gpus, num_machines=args.num_machines,
                     machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))

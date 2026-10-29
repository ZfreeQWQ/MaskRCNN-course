#!/usr/bin/env python
"""Real vendor-GPU DDP backward with unequal and globally empty pilot targets."""
import json
import os
from pathlib import Path
import train_m2f
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils import comm
from mask2former.modeling.matcher import HungarianMatcher
from rgbd_seg.m2f.mpformer import MPCriterion
from test_mpformer import make_decoder, make_targets


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    comm.create_local_process_group(int(os.environ["LOCAL_WORLD_SIZE"]))
    torch.manual_seed(19)
    model = DistributedDataParallel(make_decoder().cuda(), device_ids=[rank], broadcast_buffers=False)
    criterion = MPCriterion(1, HungarianMatcher(1, 1, 1, num_points=32), {}, .1,
                            ["labels", "masks"], 32, 3.0, .75).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    reports = []
    for step in range(3):
        counts = (0, 0) if step == 2 or rank == step else (1, 2)
        targets = [{k: v.cuda() for k, v in target.items()} for target in make_targets(counts)]
        x = [torch.randn(2, 32, size, size, device="cuda") for size in (4, 8, 16)]
        features = torch.randn(2, 16, 16, 16, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, features, targets=targets)
        losses = criterion(outputs, targets)
        if not all(torch.isfinite(loss) for loss in losses.values()):
            raise AssertionError("Nonfinite distributed pilot loss")
        loss = sum(losses.values())
        loss.backward()
        missing = [name for name, parameter in model.module.named_parameters() if parameter.grad is None]
        if missing:
            raise AssertionError("Missing gradients: " + repr(missing))
        optimizer.step()
        records = comm.gather({"rank": rank, "targets": list(counts), "loss": float(loss),
                               "loss_keys": sorted(losses), "missing_gradients": missing}, dst=0)
        if comm.is_main_process():
            if any(record["loss_keys"] != records[0]["loss_keys"] for record in records):
                raise AssertionError("Inconsistent loss keys across ranks")
            reports.append({"step": step, "ranks": records})
    if comm.is_main_process():
        folder = Path("/private/rgzndzy/seg_project/experiments/mask2former/round2_mpformer_checks_20261026")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ddp_empty_targets.json").write_text(json.dumps({"valid": True, "reports": reports}, indent=2))
        print("PASS: four-GPU DDP, three backwards including unequal and globally empty target batches", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

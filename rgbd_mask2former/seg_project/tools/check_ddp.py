"""Four-process communication and gradient smoke test on the existing backend."""
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
    torch.manual_seed(2026)
    model = DistributedDataParallel(torch.nn.Linear(8, 2).cuda(), device_ids=[rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for step in range(5):
        optimizer.zero_grad()
        x = torch.full((4, 8), float(rank + 1), device='cuda')
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
    weight = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.empty_like(weight) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, weight)
    assert all(torch.equal(weight, other) for other in gathered)
    assert weight.isfinite().all()
    if dist.get_rank() == 0:
        report = {'status': 'passed', 'world_size': dist.get_world_size(),
                  'backend': dist.get_backend(), 'steps': 5, 'identical_weights': True}
        Path('runs').mkdir(exist_ok=True)
        Path('runs/ddp_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

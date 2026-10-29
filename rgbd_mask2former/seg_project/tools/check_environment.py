"""Record existing packages and exercise all GPUs without installing anything."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback

import torch
from torchvision.ops import nms, roi_align


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/env_verified.json')
    args = parser.parse_args()
    report = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'torch_version': torch.__version__, 'cuda_api_version': torch.version.cuda,
              'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'), 'devices': [], 'packages': {}}
    for package in ('torch', 'torchvision', 'pycocotools', 'opencv-python', 'numpy', 'fvcore', 'iopath'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None
    for idx in range(torch.cuda.device_count()):
        result = {'id': idx, 'name': torch.cuda.get_device_name(idx)}
        try:
            with torch.cuda.device(idx):
                device = f'cuda:{idx}'
                x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
                conv = torch.nn.Conv2d(3, 8, 3).to(device)
                y = conv(x + x)
                rois = torch.tensor([[0., 1., 1., 15., 15.], [1., 2., 2., 20., 20.]], device=device)
                loss = roi_align(y, rois, (7, 7), aligned=True).square().mean()
                loss.backward()
                selected = nms(rois[:, 1:], torch.tensor([.9, .8], device=device), .5)
                torch.cuda.synchronize()
                assert x.grad.isfinite().all()
                result.update(status='passed', loss=loss.item(), nms_indices=selected.cpu().tolist(),
                              free_total_bytes=list(torch.cuda.mem_get_info()))
        except Exception:
            result.update(status='failed', traceback=traceback.format_exc())
        report['devices'].append(result)
        print(json.dumps(result), flush=True)
    proc = subprocess.run(['/usr/local/iluvatar/bin/ixsmi'], capture_output=True, text=True)
    report['ixsmi'] = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    with open('/proc/self/maps') as maps:
        report['loaded_gpu_libraries'] = sorted({line.split()[-1] for line in maps
            if any(name in line for name in ('libcuda', 'libixthunk', 'libcudnn', 'libnccl'))})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    if len(report['devices']) != 4 or any(d['status'] != 'passed' for d in report['devices']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

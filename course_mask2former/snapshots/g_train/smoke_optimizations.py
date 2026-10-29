"""Full pretrained model + synthetic backward checks for every loss combination.

Does not assess accuracy. Crop geometry and all eight switch combinations are
covered separately in tests/. This complements, not replaces, real-data tests.
"""
import argparse
import gc
import json
from pathlib import Path

from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
import numpy as np
import torch

from optimizations.weighted_boundary_head import WeightedBoundaryMask2FormerHead
from smoke_model import synthetic_train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/workspace/outputs/optimization_smoke_20260922.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new evidence path: {output}')
    register_all_modules(init_default_scope=True)
    manifest = json.loads(Path('/workspace/checkpoints/smoke_manifest.json').read_text())
    results = {}
    for small, boundary in [(False, False), (True, False), (False, True), (True, True)]:
        torch.manual_seed(20260922)
        np.random.seed(20260922)
        cfg = Config.fromfile(manifest['config'])
        cfg.model.backbone.init_cfg = None
        if small or boundary:
            cfg.model.panoptic_head.type = 'WeightedBoundaryMask2FormerHead'
            cfg.model.panoptic_head.small_object_loss = dict(enabled=small)
            cfg.model.panoptic_head.boundary_loss = dict(enabled=boundary)
        model = init_detector(cfg, manifest['checkpoint'], device='cuda:0', palette='coco')
        key = f'small_{int(small)}_boundary_{int(boundary)}'
        result = synthetic_train_step(model)
        result['parameters'] = sum(parameter.numel() for parameter in model.parameters())
        result['head'] = type(model.panoptic_head).__name__
        if boundary != ('loss_boundary' in result['losses']):
            raise AssertionError('Boundary loss presence does not match its switch')
        results[key] = result
        print(key, json.dumps(result), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if len({item['parameters'] for item in results.values()}) != 1:
        raise AssertionError('Training-only options unexpectedly changed model parameter count')
    report = dict(status='passed', tests=results,
                  note='Synthetic full-model functional validation, NOT AP/APs or accuracy improvement.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

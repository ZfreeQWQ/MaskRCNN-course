"""Full pretrained-model synthetic CPU integration for all 10 v3 recipes.

No GPU access; three SGD steps on 64px synthetic samples, then inference.
This is NOT the formal AdamW recipe, real-data GPU smoke, or accuracy evidence.
The one-shot workflow still requires its 512px real-data GPU smoke stage.
"""
import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import import_modules_from_strings
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules

from run_nyu_v3 import parser as training_parser, build_config
from run_nyu_v2 import load_initial, categories_from_export, map_coco_state
from run_coco_experiment import sha256
from nyu_v3_plan import experiment_jobs


def samples():
    items = []
    for index in range(2):
        masks = np.zeros((2, 64, 64), dtype=np.uint8)
        masks[0, 12:16, 10:14] = 1
        masks[1, 24:48, 24:48] = 1
        sample = DetDataSample(metainfo=dict(img_id=index, img_shape=(64, 64), ori_shape=(64, 64),
            scale_factor=(1., 1.), batch_input_shape=(64, 64), pad_shape=(64, 64)))
        sample.gt_instances = InstanceData(labels=torch.tensor([0, 2]), masks=BitmapMasks(masks, 64, 64),
            bboxes=torch.tensor([[10., 12., 14., 16.], [24., 24., 48., 48.]]))
        valid = torch.ones((1, 64, 64), dtype=torch.bool)
        valid[:, :4] = False
        sample.label_valid = PixelData(data=valid)
        rgb = torch.rand(3, 64, 64) * 255
        depth = torch.linspace(.5, 5, 64).reshape(1, 1, 64).expand(1, 64, 64)
        items.append(dict(inputs=torch.cat([rgb, depth, torch.ones_like(depth)]), data_samples=sample))
    return pseudo_collate(items)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    opts = p.parse_args()
    output = Path(opts.output)
    if output.exists():
        raise FileExistsError('Never overwrite integration evidence')
    if torch.cuda.is_available():
        raise RuntimeError('Run in a container without GPU access')
    torch.set_num_threads(1)
    MMLogger.get_instance('v3_cpu_integration', log_level='ERROR')
    register_all_modules(init_default_scope=True)
    args = training_parser().parse_args(['train', '--output', '/tmp/unused_v3_cpu', '--image-size', '64',
                                         '--iters', '3', '--val-interval', '3', '--workers', '0'])
    categories = categories_from_export(args.data_root)
    state = map_coco_state(torch.load(args.checkpoint, map_location='cpu')['state_dict'], categories)
    torch.manual_seed(1)
    batch = samples()
    records = []
    for job in experiment_jobs([20260923]):
        torch.manual_seed(job['seed'])
        current = copy.copy(args)
        vars(current).update(job['flags'])
        cfg = build_config(current, categories)
        import_modules_from_strings(**cfg.custom_imports)
        model = MODELS.build(cfg.model)
        model.init_weights()
        initial = load_initial(model, state)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        losses = []
        model.train()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            prepared = model.data_preprocessor(copy.deepcopy(batch), training=True)
            result = model(**prepared, mode='loss')
            loss, _ = model.parse_losses(result)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CPU integration loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        trained = model.state_dict()
        def changed(prefix):
            return any(not torch.equal(v, trained[k]) for k, v in initial.items() if k.startswith(prefix))
        checks = dict(projection=changed('spatial_fusion.projections.'),
                      encoder=changed(('spatial_fusion.stem.', 'spatial_fusion.stages.')),
                      gate=changed('spatial_fusion.gates.'))
        if checks != dict(projection=current.multiscale_depth, encoder=current.multiscale_depth,
                          gate=current.reliability_gate):
            raise RuntimeError(f'CPU gradient-path mismatch: {checks}')
        model.eval()
        counters_before = dict(model.advanced_counters)
        with torch.inference_mode():
            predictions = model.test_step(copy.deepcopy(batch))
        if model.advanced_counters != counters_before:
            raise RuntimeError('Training-only counters changed during inference')
        if len(predictions) != 2 or not all(torch.isfinite(x.pred_instances.scores).all() for x in predictions):
            raise RuntimeError('Nonfinite/incorrect inference output')
        records.append(dict(variant=job['variant'], flags=job['flags'], status='passed', losses=losses,
            changed=checks, advanced_depth_counters=dict(model.advanced_counters),
            small_matching_counters=dict(model.panoptic_head.matching_counters),
            inference_instances=[len(x.pred_instances) for x in predictions]))
        print('PASSED', job['variant'], checks, flush=True)
        del model, optimizer, initial, trained, prepared, result, loss, predictions
        gc.collect()
    report = dict(status='passed', device='cpu', steps=3, image_size=64, optimizer='SGD', results=records,
        source_checkpoint_sha256=sha256(Path(args.checkpoint)), script_sha256=sha256(Path(__file__)),
        note='Synthetic CPU full pretrained model + three backward/update steps + inference. '
             'Not real-data GPU integration, formal training recipe, or AP evidence.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('All 10 full-model CPU integrations passed', flush=True)


if __name__ == '__main__':
    main()

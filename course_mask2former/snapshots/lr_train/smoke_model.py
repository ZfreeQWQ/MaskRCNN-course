"""Real pretrained RGB inference plus a synthetic one-step training check.

The synthetic boxes/masks below are deliberately NOT dataset annotations.
This checks CUDA backward/optimizer behavior, not model accuracy or RGB-D fusion.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2
import mmcv
import mmengine
from mmengine.config import Config
from mmengine.optim import OptimWrapper
from mmengine.structures import InstanceData
import mmdet
from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer
import numpy as np
import torch

ROOT = Path('/workspace')


def synthetic_train_step(model):
    size = 256
    masks = np.zeros((2, size, size), dtype=np.uint8)
    masks[0, 32:128, 24:112] = 1
    masks[1, 156:172, 180:196] = 1
    image = np.full((size, size, 3), 96, dtype=np.uint8)
    image[masks[0] > 0] = (32, 64, 224)
    image[masks[1] > 0] = (224, 64, 32)
    gt = InstanceData(
        labels=torch.tensor([0, 2], dtype=torch.long),
        bboxes=torch.tensor([[24., 32., 112., 128.], [180., 156., 196., 172.]]),
        masks=BitmapMasks(masks, size, size),
    )
    sample = DetDataSample(metainfo=dict(
        img_id=0, img_shape=(size, size), ori_shape=(size, size),
        scale_factor=(1., 1.),
    ))
    sample.gt_instances = gt
    # Keep the full model and matching/loss logic, but reduce padding for this test.
    for augment in model.data_preprocessor.batch_augments:
        augment.size = (size, size)
    model.train()
    wrapper = OptimWrapper(torch.optim.AdamW(model.parameters(), lr=1e-5))
    tracked = model.panoptic_head.cls_embed.weight
    before = tracked.detach().clone()
    data = dict(inputs=[torch.from_numpy(image.transpose(2, 0, 1).copy())], data_samples=[sample])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # Use the real MMEngine optimizer context and the model's real loss implementation.
    with wrapper.optim_context(model):
        processed = model.data_preprocessor(data, training=True)
        losses = model(**processed, mode='loss')
    loss, logs = model.parse_losses(losses)
    if not torch.isfinite(loss).all():
        raise RuntimeError('Non-finite training loss')
    wrapper.backward(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError('Missing or non-finite gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
    wrapper.step()
    wrapper.zero_grad()
    torch.cuda.synchronize()
    changed = not torch.equal(before, tracked.detach())
    if not changed:
        raise RuntimeError('Classifier weights did not update')
    return dict(
        status='passed', synthetic=True, input_shape=[1, 3, size, size],
        loss=float(loss.detach()), losses={k: float(v.detach()) for k, v in logs.items()},
        gradient_tensor_count=len(gradients), gradient_norm_before_clip=float(norm),
        classifier_weights_changed=changed, seconds=time.perf_counter() - start,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        note='Synthetic rectangles, one optimizer step. No accuracy metric or trained checkpoint.',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/workspace/outputs/smoke')
    parser.add_argument('--train-step', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(8)
    register_all_modules(init_default_scope=True)
    manifest = json.loads((ROOT / 'checkpoints/smoke_manifest.json').read_text())
    cfg = Config.fromfile(manifest['config'])
    cfg.model.backbone.init_cfg = None
    # No GT annotations exist for the bundled demo image.
    cfg.test_dataloader.dataset.pipeline = [
        x for x in cfg.test_dataloader.dataset.pipeline if x['type'] != 'LoadAnnotations'
    ]
    model = init_detector(cfg, manifest['checkpoint'], device='cuda:0', palette='coco')
    print('Checkpoint loaded; running full-resolution demo inference.', flush=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = inference_detector(model, manifest['image'])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    instances = result.pred_instances.cpu()
    selected = instances[instances.scores >= 0.5]
    if not len(selected) or not selected.masks.any().item():
        raise RuntimeError('Pretrained demo produced no confident non-empty masks')
    classes = model.dataset_meta['classes']
    visualizer = DetLocalVisualizer(name='smoke_visualizer')
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'pretrained_mask2former', mmcv.imread(manifest['image'], channel_order='rgb'),
        data_sample=result.cpu(), draw_gt=False, pred_score_thr=0.5,
        out_file=str(output / 'prediction.jpg'),
    )
    np.savez_compressed(output / 'instances.npz',
        masks=selected.masks.numpy(), scores=selected.scores.numpy(),
        labels=selected.labels.numpy(), bboxes=selected.bboxes.numpy())
    report = dict(
        status='inference_passed', model=manifest['model'], checkpoint_sha256=manifest['sha256'],
        gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda, mmcv=mmcv.__version__,
                      mmengine=mmengine.__version__, mmdet=mmdet.__version__, cv2=cv2.__version__),
        input=manifest['image'], score_threshold=0.5,
        instances=len(selected), class_counts=dict(Counter(classes[i] for i in selected.labels.tolist())),
        first_inference_seconds=elapsed,
        inference_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        parameters=sum(p.numel() for p in model.parameters()),
        note='RGB COCO demo only. First-call time is NOT a benchmark; no AP/APs measured.',
    )
    (output / 'inference.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if args.train_step:
        print('Testing full backward + AdamW on a synthetic sample.', flush=True)
        report['training_smoke'] = synthetic_train_step(model)
    report['status'] = 'passed'
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    snapshot = subprocess.run(['python', '-m', 'pip', 'freeze'], check=True, text=True, capture_output=True)
    (output / 'pip-freeze.txt').write_text(snapshot.stdout)
    (output / 'resolved_config.py').write_text(cfg.pretty_text)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

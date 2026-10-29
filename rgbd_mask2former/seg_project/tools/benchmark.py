"""1024x1024, batch=1, synchronized latency; standard: 20 warmup + >=200 images."""
import argparse
import platform
import time
from pathlib import Path
import numpy as np
import torch
from common import (add_model_args, image_record, load_images, predictor_from_args,
                    write_json)


def benchmark(predictor, images, data_root, weights, warmup=20, iterations=200, smoke=False):
    if warmup < 0 or iterations < 1:
        raise ValueError('warmup >= 0 and iterations >= 1 required')
    if not smoke and (warmup < 20 or iterations < 200):
        raise ValueError('Standard benchmark requires warmup >=20 and iterations >=200; use --smoke explicitly')
    if not images or any((row['height'], row['width']) != (1024, 1024) for row in images):
        raise ValueError('Benchmark requires 1024x1024 metadata')
    if not smoke and len(images) < iterations:
        raise ValueError('Standard benchmark requires distinct source images for timed iterations')
    model = predictor.model
    model.eval()
    device = next(model.parameters()).device

    def sync():
        # CUDA API is exposed by the installed CoreX torch build, not a vendor assumption.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type != 'cpu':
            raise RuntimeError(f'No verified synchronization API for {device.type}')

    def mapped(index):
        item = predictor.mapper(image_record(images[index % len(images)], data_root))
        if tuple(item['image'].shape[-2:]) != (1024, 1024):
            raise ValueError('Mapper resized the input: use fixed 1024 inference configuration')
        return item

    with torch.inference_mode():
        for index in range(warmup):
            model([mapped(index)])
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        read_ms, model_ms = [], []
        for index in range(iterations):
            start = time.perf_counter()
            item = mapped(warmup + index)
            read_ms.append((time.perf_counter() - start) * 1000)
            sync()
            start = time.perf_counter()
            output = model([item])
            sync()
            model_ms.append((time.perf_counter() - start) * 1000)
            del output
    result = {
        'smoke': smoke, 'input_shape': [1, *item['image'].shape], 'warmup': warmup,
        'iterations': iterations, 'unique_source_images': min(len(images), iterations),
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'torch_version': torch.__version__, 'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'checkpoint_bytes': Path(weights).stat().st_size,
        'checkpoint_MiB': Path(weights).stat().st_size / 2**20,
        'model_state_bytes': sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()),
        'model_state_MiB': sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()) / 2**20,
        'size_scope': 'checkpoint is the complete file (may include optimizer/scheduler); model_state counts parameter/buffer tensor bytes only',
        'latency_model_mean_ms': float(np.mean(model_ms)),
        'latency_model_median_ms': float(np.median(model_ms)),
        'latency_model_p95_ms': float(np.percentile(model_ms, 95)),
        'read_preprocess_mean_ms': float(np.mean(read_ms)),
        'fps_model': float(1000 / np.mean(model_ms)),
        'fps_read_plus_model': float(1000 / (np.mean(read_ms) + np.mean(model_ms))),
        'peak_allocated_MiB': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
        'peak_reserved_MiB': torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else None,
        'tta': False, 'tiling': False,
        'timing_scope': ('model includes device transfer, forward, D2 postprocessing; '
                         'read/preprocess includes ' +
                         ('RGB+depth' if predictor.mapper.use_depth else 'RGB') +
                         ' decode and mapper; excludes JSON RLE encoding'),
        'model_ms_samples': model_ms, 'read_preprocess_ms_samples': read_ms}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument('--metadata', required=True, help='Use validation metadata/COCO JSON')
    p.add_argument('--data-root', default='/private/rgzndzy/blender')
    p.add_argument('--output', required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    images = load_images(args.metadata, 'val')
    result = benchmark(predictor_from_args(args, images), images, args.data_root, args.weights,
                       args.warmup, args.iterations, args.smoke)
    result.update(config=args.config, weights=args.weights, score_threshold=args.threshold)
    write_json(args.output, result)
    print({k: v for k, v in result.items() if not k.endswith('_samples')})


if __name__ == '__main__':
    main()

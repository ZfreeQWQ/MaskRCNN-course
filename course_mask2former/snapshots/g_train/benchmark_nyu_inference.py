"""Fixed-image, batch-one CUDA inference latency for an adapted NYUv2 model.

Example (inside the prepared container):
  python /workspace/scripts/benchmark_nyu_inference.py \
    --checkpoint /workspace/outputs/nyu_rgb/iter_100.pth \
    --output /workspace/outputs/nyu_rgb/inference_benchmark.json

The configuration defaults to config.py beside the checkpoint. This script
strictly loads the supplied checkpoint; it does not instantiate a Runner,
train, evaluate AP, use cfg.load_from, or overwrite an experiment directory.
Only trusted locally produced PyTorch checkpoints should be loaded.

Every timed call starts with a fresh CPU batch and measures model.test_step,
including data preprocessing, CPU-to-GPU transfer, inference, postprocessing,
and a final CUDA synchronization. Dataset loading, image decoding/resize,
CPU-batch cloning, model construction, and checkpoint loading are NOT timed.
This is not a GPU-kernel-only benchmark or full disk-to-result throughput.
The original-image void-aware AP metric is not run during this benchmark.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.utils import import_modules_from_strings
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_description(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', help='Defaults to config.py beside --checkpoint')
    parser.add_argument('--output', required=True, help='New JSON file; existing files are never overwritten')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error('warmup and repeats must both be at least one')
    checkpoint = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve() if args.config else checkpoint.parent / 'config.py'
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('Refusing to overwrite an existing benchmark output')
    if not checkpoint.is_file() or not config_path.is_file():
        parser.error('checkpoint and config must both be existing files')
    if not torch.cuda.is_available():
        parser.error('This benchmark requires a CUDA GPU')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    seed = 20260924
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config_path))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    torch.backends.cudnn.benchmark = bool(cfg.get('env_cfg', {}).get('cudnn_benchmark', False))

    saved = torch.load(checkpoint, map_location='cpu')
    if not isinstance(saved, dict) or 'state_dict' not in saved:
        raise ValueError('Expected a trained MMEngine checkpoint containing state_dict')
    metadata = saved.get('meta', {})
    previous_text = metadata.get('cfg', metadata.get('config'))
    if previous_text:
        previous = Config.fromstring(previous_text, '.py')
        if previous.model.to_dict() != cfg.model.to_dict():
            raise ValueError('Model config differs from the checkpoint metadata (including optimization switches)')
    # Constructor state is completely overwritten, including buffers. Calling
    # init_weights here could download an unrelated pretrained backbone.
    model = MODELS.build(cfg.model)
    model.load_state_dict(saved['state_dict'], strict=True)
    checkpoint_iteration = metadata.get('iter')
    del saved
    model.to(device).eval()

    loader_name = 'val_dataloader' if cfg.get('val_dataloader') else 'test_dataloader'
    loader_cfg = cfg.get(loader_name)
    if not loader_cfg or 'dataset' not in loader_cfg:
        raise ValueError('Config needs val_dataloader.dataset or test_dataloader.dataset')
    dataset_cfg = copy.deepcopy(loader_cfg.dataset)
    if not dataset_cfg.get('test_mode', False):
        raise ValueError('Benchmark input must use a non-augmented test-mode validation dataset')
    dataset = DATASETS.build(dataset_cfg)
    if len(dataset) == 0:
        raise ValueError('Validation dataset is empty')
    sample = dataset[0]
    if sample is None or not isinstance(sample.get('inputs'), torch.Tensor):
        raise ValueError('Expected a packed tensor sample at validation index zero')
    if sample['inputs'].device.type != 'cpu':
        raise ValueError('The fixed benchmark input must initially reside on CPU')
    fixed_batch = pseudo_collate([sample])
    input_meta = sample['data_samples'].metainfo
    image_id = input_meta.get('img_id')
    if isinstance(image_id, (np.integer, torch.Tensor)):
        image_id = int(image_id)
    description = dict(
        dataset_loader=loader_name, dataset_index=0, img_id=image_id,
        img_path=input_meta.get('img_path'),
        ori_shape=list(input_meta.get('ori_shape', ())),
        resized_img_shape=list(input_meta.get('img_shape', ())),
        cpu_input=tensor_description(sample['inputs']),
        cpu_tensor_sha256=hashlib.sha256(sample['inputs'].contiguous().numpy().tobytes()).hexdigest(),
        label_valid=tensor_description(sample['data_samples'].label_valid.data))

    parameters = list(model.parameters())
    parameter_count = sum(p.numel() for p in parameters)
    parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
    parameter_dtypes = sorted({str(p.dtype) for p in parameters})
    del parameters
    with torch.inference_mode():
        # Inspect preprocessing once outside the measurement and discard its
        # GPU tensors before warm-up. This does not call the neural network.
        prepared = model.data_preprocessor(copy.deepcopy(fixed_batch), training=False)
        description['preprocessed_input'] = tensor_description(prepared['inputs'])
        del prepared
        for _ in range(args.warmup):
            batch = copy.deepcopy(fixed_batch)
            outputs = model.test_step(batch)
            torch.cuda.synchronize(device)
            del outputs, batch
        gc.collect()
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        output_description = None
        for _ in range(args.repeats):
            # Fresh CPU objects prevent prior test_step mutations or GPU data
            # reuse from accidentally removing transfer/preprocessing costs.
            batch = copy.deepcopy(fixed_batch)
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            outputs = model.test_step(batch)
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter_ns() - started) / 1_000_000)
            if output_description is None:
                if len(outputs) != 1:
                    raise RuntimeError('Expected exactly one output for batch size one')
                pred = outputs[0].pred_instances
                output_description = dict(instances=len(pred), masks=tensor_description(pred.masks),
                                          scores=tensor_description(pred.scores))
            del outputs, batch
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    mean = statistics.mean(timings)
    model_flags = dict(
        depth_fusion=cfg.model.get('depth_fusion'),
        small_object_loss=cfg.model.panoptic_head.get('small_object_loss'),
        boundary_loss=cfg.model.panoptic_head.get('boundary_loss'),
        ignore_void=cfg.model.panoptic_head.get('ignore_void'))
    # ConfigDict is recursively JSON-compatible for these scalar options.
    report = dict(
        status='complete', protocol='fixed-first-validation-frame-cpu-to-prediction-batch1-v1',
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        checkpoint_iteration=checkpoint_iteration, config=str(config_path), config_sha256=sha256(config_path),
        script_sha256=sha256(Path(__file__)),
        seed=seed, batch_size=1, warmup=args.warmup, repeats=args.repeats,
        input=description, output=output_description, model_flags=model_flags,
        parameters=parameter_count, parameter_storage_bytes=parameter_bytes, parameter_dtypes=parameter_dtypes,
        environment=dict(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                         cudnn_version=torch.backends.cudnn.version(), gpu_name=torch.cuda.get_device_name(device),
                         gpu_index=0, visible_gpu_count=torch.cuda.device_count(),
                         cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                         cudnn_benchmark=torch.backends.cudnn.benchmark,
                         cudnn_deterministic=torch.backends.cudnn.deterministic,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         inference_mode=True, autocast=False),
        latency_ms=dict(samples=timings, mean=mean, median=statistics.median(timings),
                        p90=float(np.percentile(timings, 90)), p95=float(np.percentile(timings, 95)),
                        minimum=min(timings), maximum=max(timings), population_std=statistics.pstdev(timings)),
        images_per_second_reciprocal_mean_latency=1000 / mean,
        cuda_memory_mib=dict(allocated_before_measurement=allocated_before / 2**20,
                             reserved_before_measurement=reserved_before / 2**20,
                             peak_allocated=peak_allocated / 2**20, peak_reserved=peak_reserved / 2**20,
                             peak_allocated_above_start=(peak_allocated - allocated_before) / 2**20),
        timing_scope='Synchronized wall-clock model.test_step with a fresh CPU batch: preprocessing, '
                     'CPU-to-GPU transfer, inference and postprocessing. Excludes disk I/O, decode/resize, '
                     'CPU batch clone, setup and checkpoint loading. Excludes void-aware AP metric.',
        memory_scope='Peak PyTorch allocator counters after warm-up, without empty_cache: includes model '
                     'parameters/buffers and test_step tensors. Reserved memory may retain warm-up cache. '
                     'Not total device/NVML usage and not training memory.',
        limitations='One fixed frame, one process and one visible-device index; descriptive microbenchmark, '
                    'not dataset-wide latency or GPU-kernel-only speed. Shared GPU load and run order may '
                    'change latency; compare paired runs with identical config input/seed/budget.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), parameters=parameter_count, mean_ms=mean,
                          p95_ms=report['latency_ms']['p95'], peak_allocated_mib=peak_allocated / 2**20,
                          input_img_id=image_id), ensure_ascii=False))


if __name__ == '__main__':
    main()

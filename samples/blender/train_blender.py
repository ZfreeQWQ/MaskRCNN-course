"""训练 RGB-D Blender 数据集的 Matterport Mask R-CNN 入口。

这个脚本提供三类功能：

1. 建立 train/val 数据集并验证一个训练 batch；
2. 从随机初始化、COCO 权重或已有 RGB-D checkpoint 开始训练；
3. 用少量样本和少量 step 执行一次短跑，确认模型、数据和显存链路。

默认不下载权重，也不使用 test 集训练。测试推理和提交文件生成应另行实现。
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np

# 允许从仓库根目录或当前脚本目录直接执行。
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
# 服务器启动脚本通常设置 ROOT=$HOME/MaskRCNN；训练产物统一放在 ROOT/runs/member-a。
# 即使 ROOT 没有 export，运行时也会从数据集路径自动推导同一个目录。
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SAMPLE_DIR not in sys.path:
    sys.path.append(SAMPLE_DIR)

from blender import BlenderConfig, BlenderDataset
from mrcnn import model as modellib


COCO_EXCLUDED_LAYERS = [
    # 4 通道输入的首层和单类别任务不兼容的输出头需要单独处理。
    "conv1",
    "mrcnn_class_logits",
    "mrcnn_bbox_fc",
    "mrcnn_bbox",
    "mrcnn_mask",
]


def _decode_h5_name(value):
    """把 HDF5 属性中的 bytes 或字符串统一转换为字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _find_h5_layer_group(root, layer_name):
    """在旧版 Keras 权重文件中寻找指定层的 HDF5 分组。"""
    layer_names = root.attrs.get("layer_names", [])
    for raw_name in layer_names:
        name = _decode_h5_name(raw_name)
        if name == layer_name:
            return root[name]
    if layer_name in root:
        return root[layer_name]
    return None


def _read_h5_layer_weights(weights_path, layer_name):
    """读取一个旧版 Keras 层的权重数组。"""
    import h5py

    with h5py.File(weights_path, "r") as handle:
        root = handle["model_weights"] if "model_weights" in handle else handle
        group = _find_h5_layer_group(root, layer_name)
        if group is None:
            return None
        weight_names = group.attrs.get("weight_names", [])
        arrays = []
        for raw_name in weight_names:
            name = _decode_h5_name(raw_name)
            arrays.append(np.asarray(group[name]))
        return arrays


def _find_keras_layer(container, layer_name):
    """递归寻找模型中的 Keras 层，兼容单卡和封装模型。"""
    for layer in container.layers:
        if layer.name == layer_name:
            return layer
        if hasattr(layer, "layers"):
            found = _find_keras_layer(layer, layer_name)
            if found is not None:
                return found
    return None


def _initialize_rgbd_conv1(model, weights_path):
    """用 COCO 的 RGB 首层权重初始化四通道首层。

    新增的深度通道使用三个 RGB 输入通道的平均卷积核，避免随机首层
    破坏 COCO backbone 的初始特征分布。其余兼容层由 Mask R-CNN 的
    ``load_weights`` 完成加载。
    """
    source_weights = _read_h5_layer_weights(weights_path, "conv1")
    layer = _find_keras_layer(model.keras_model, "conv1")
    if source_weights is None or layer is None:
        raise RuntimeError("无法找到 COCO 权重或模型中的 conv1 层")
    target_weights = layer.get_weights()
    if len(source_weights) != 2 or len(target_weights) != 2:
        raise RuntimeError("conv1 权重格式不是预期的 kernel+bias")

    source_kernel, source_bias = source_weights
    target_kernel, target_bias = target_weights
    if source_kernel.ndim != 4 or target_kernel.ndim != 4:
        raise RuntimeError("conv1 卷积核维度错误")
    if source_kernel.shape[0:2] != target_kernel.shape[0:2]:
        raise RuntimeError("conv1 卷积核空间尺寸不一致")
    if source_kernel.shape[2] != 3 or target_kernel.shape[2] != 4:
        raise RuntimeError(
            "只支持从 3 通道 conv1 扩展到 4 通道，实际为 {} -> {}".format(
                source_kernel.shape[2], target_kernel.shape[2]))
    if source_kernel.shape[3] != target_kernel.shape[3]:
        raise RuntimeError("conv1 输出通道数不一致")
    if source_bias.shape != target_bias.shape:
        raise RuntimeError("conv1 偏置尺寸不一致")

    depth_kernel = np.mean(source_kernel, axis=2, keepdims=True)
    adapted_kernel = np.concatenate([source_kernel, depth_kernel], axis=2)
    layer.set_weights([adapted_kernel.astype(target_kernel.dtype),
                       source_bias.astype(target_bias.dtype)])


def load_initial_weights(model, weights_arg, coco_weights_path):
    """按命令行设置加载权重，并返回实际使用的权重说明。"""
    requested = (weights_arg or "none").strip()
    lower = requested.lower()
    if lower in ("none", "random", "scratch"):
        print("使用随机初始化，不加载预训练权重。")
        return "random"

    if lower == "coco":
        weights_path = coco_weights_path
        if not weights_path or not os.path.isfile(weights_path):
            raise IOError(
                "找不到 COCO 权重文件: {}。请上传 mask_rcnn_coco.h5，"
                "或用 --weights 指定已有 checkpoint。".format(weights_path))
        print("加载 COCO 权重: {}".format(weights_path))
        model.load_weights(weights_path, by_name=True, exclude=COCO_EXCLUDED_LAYERS)
        _initialize_rgbd_conv1(model, weights_path)
        print("已加载兼容层，并用 RGB 平均卷积核初始化深度输入通道。")
        return "coco"

    if lower == "last":
        weights_path = model.find_last()
    else:
        weights_path = requested
    if not os.path.isfile(weights_path):
        raise IOError("找不到权重文件: {}".format(weights_path))
    print("加载已有 RGB-D 权重: {}".format(weights_path))
    model.load_weights(weights_path, by_name=True)
    return weights_path


def _limit_dataset(dataset, limit):
    """将数据集的采样 ID 限制到前 limit 张，便于短跑测试。"""
    if limit and limit > 0:
        total = len(dataset.image_ids)
        dataset._image_ids = np.asarray(dataset.image_ids[:limit], dtype=np.int32)
        dataset.num_images = len(dataset._image_ids)
        print("数据集采样数量: {}/{}".format(dataset.num_images, total))


def _load_datasets(dataset_dir, train_limit=0, val_limit=0, depth_max_m=1.0):
    """读取并准备 train/val 数据集。"""
    train_dataset = BlenderDataset()
    train_dataset.load_blender(dataset_dir, "train", depth_max_m=depth_max_m)
    train_dataset.prepare()
    _limit_dataset(train_dataset, train_limit)

    val_dataset = BlenderDataset()
    val_dataset.load_blender(dataset_dir, "val", depth_max_m=depth_max_m)
    val_dataset.prepare()
    _limit_dataset(val_dataset, val_limit)
    return train_dataset, val_dataset


def _default_runs_dir(dataset_dir):
    """根据 .../MaskRCNN/data/blender 推导成员的运行产物目录。"""
    configured_root = os.environ.get("ROOT")
    if configured_root:
        root = os.path.abspath(configured_root)
    else:
        dataset_dir = os.path.abspath(dataset_dir)
        # 数据集通常位于 ROOT/data/blender；这样无需 export ROOT 也能定位。
        root = os.path.abspath(os.path.join(dataset_dir, os.pardir, os.pardir))
    return os.path.join(root, "runs", "member-a")


def check_data_batch(dataset, config, name):
    """用 Matterport 生成器读取一个 batch，检查四通道和训练目标形状。"""
    generator = modellib.data_generator(
        dataset, config, shuffle=False, batch_size=config.BATCH_SIZE)
    inputs, _ = next(generator)
    print(
        "{} batch: image={}, meta={}, rpn_match={}, gt_boxes={}, gt_masks={}".format(
            name, inputs[0].shape, inputs[1].shape, inputs[2].shape,
            inputs[5].shape, inputs[6].shape))
    if inputs[0].shape[-1] != 4:
        raise RuntimeError("数据生成器没有返回 4 通道 RGB-D 图像")
    return inputs


def make_parser():
    """建立训练命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 Blender RGB-D Mask R-CNN")
    parser.add_argument("dataset", help="Blender 数据集根目录")
    parser.add_argument(
        "--weights", default="none",
        help="none、coco、last 或已有 .h5 文件路径；默认 none",
    )
    parser.add_argument(
        "--coco-weights", default=None,
        help="--weights coco 时使用的 COCO 权重路径，默认位于 runs/member-a/weights",
    )
    parser.add_argument(
        "--logs", default=None,
        help="训练日志和 checkpoint 保存目录，默认是 ROOT/runs/member-a",
    )
    parser.add_argument(
        "--layers", choices=("auto", "heads", "heads+depth", "4+", "all"),
        default="auto",
        help="训练层；auto 会根据权重类型选择 all、heads+depth 或 heads",
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练总 epoch 数")
    parser.add_argument("--steps-per-epoch", type=int, default=0,
                        help="每 epoch step 数；0 表示按数据量自动设置")
    parser.add_argument("--validation-steps", type=int, default=0,
                        help="验证 step 数；0 表示按数据量自动设置")
    parser.add_argument("--train-limit", type=int, default=0,
                        help="只使用前 N 张训练图像，0 表示全部")
    parser.add_argument("--val-limit", type=int, default=0,
                        help="只使用前 N 张验证图像，0 表示全部")
    parser.add_argument("--depth-max-m", type=float, default=1.0,
                        help="深度归一化截断上限（米），默认 1.0；train/val 必须一致")
    parser.add_argument("--workers", type=int, default=0,
                        help="数据加载 worker 数，默认 0 便于调试")
    parser.add_argument("--multiprocessing", action="store_true",
                        help="使用 Keras 多进程数据加载；需同时设置 workers")
    parser.add_argument("--smoke-test", action="store_true",
                        help="使用 2 张训练图、1 张验证图、1 个 step 做短跑")
    parser.add_argument("--check-data-only", action="store_true",
                        help="只读取一个 train/val batch，不创建模型")
    return parser


def main(args=None):
    """执行数据检查或训练。"""
    args = make_parser().parse_args(args)
    if args.epochs < 1:
        raise ValueError("--epochs 必须大于等于 1")
    if args.workers < 0:
        raise ValueError("--workers 不能为负数")
    if args.depth_max_m <= 0:
        raise ValueError("--depth-max-m 必须大于 0")

    runs_dir = os.path.abspath(args.logs) if args.logs else _default_runs_dir(args.dataset)
    coco_weights_path = os.path.abspath(args.coco_weights) if args.coco_weights else os.path.join(
        runs_dir, "weights", "mask_rcnn_coco.h5")
    args.logs = runs_dir
    args.coco_weights = coco_weights_path
    print("运行产物目录: {}".format(args.logs))

    if args.smoke_test:
        if args.train_limit == 0:
            args.train_limit = 2
        if args.val_limit == 0:
            args.val_limit = 1
        if args.steps_per_epoch == 0:
            args.steps_per_epoch = 1
        if args.validation_steps == 0:
            args.validation_steps = 1

    train_dataset, val_dataset = _load_datasets(
        args.dataset, args.train_limit, args.val_limit, args.depth_max_m)
    config = BlenderConfig()
    if args.steps_per_epoch > 0:
        config.STEPS_PER_EPOCH = args.steps_per_epoch
    elif args.train_limit > 0:
        config.STEPS_PER_EPOCH = max(1, int(np.ceil(
            float(train_dataset.num_images) / config.BATCH_SIZE)))
    if args.validation_steps > 0:
        config.VALIDATION_STEPS = args.validation_steps
    elif args.val_limit > 0:
        config.VALIDATION_STEPS = max(1, int(np.ceil(
            float(val_dataset.num_images) / config.BATCH_SIZE)))

    print("训练图像数: {}, 验证图像数: {}".format(
        train_dataset.num_images, val_dataset.num_images))
    print("输入形状: {}, steps/epoch: {}, validation steps: {}".format(
        tuple(config.IMAGE_SHAPE), config.STEPS_PER_EPOCH,
        config.VALIDATION_STEPS))
    check_data_batch(train_dataset, config, "train")
    check_data_batch(val_dataset, config, "val")
    if args.check_data_only:
        print("数据 batch 检查完成，不创建模型。")
        return 0

    model = modellib.MaskRCNN(
        mode="training", config=config, model_dir=args.logs)
    weight_description = load_initial_weights(
        model, args.weights, args.coco_weights)
    layers = args.layers
    if layers == "auto":
        if weight_description == "random":
            layers = "all"
        elif weight_description == "coco":
            layers = "heads+depth"
        else:
            layers = "heads"
    print("训练层: {}".format(layers))
    if args.multiprocessing and args.workers == 0:
        raise ValueError("--multiprocessing 需要同时设置 --workers > 0")

    model.train(
        train_dataset,
        val_dataset,
        learning_rate=config.LEARNING_RATE,
        epochs=args.epochs,
        layers=layers,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
    )
    print("训练完成，日志和 checkpoint 位于: {}".format(model.log_dir))
    return 0


if __name__ == "__main__":
    main()

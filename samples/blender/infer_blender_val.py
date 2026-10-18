"""在 Blender 验证集上运行 RGB-D Mask R-CNN 推理。

脚本用于检查训练后的 checkpoint 是否能够完成完整推理链路：

* 从验证集读取 RGB-D 四通道图像；
* 加载 RGB-D Mask R-CNN checkpoint 并调用 ``model.detect``；
* 打印预测框、类别、分数和 mask 形状；
* 保存原始预测结果、RGB/深度输入、预测叠加图和 GT/预测对照图。

它只读取 val，不读取 test，也不计算正式的 Mask AP/APS。
"""

from __future__ import print_function

import argparse
import colorsys
import json
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

# 允许从仓库根目录或当前脚本目录直接执行。
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SAMPLE_DIR not in sys.path:
    sys.path.append(SAMPLE_DIR)

from blender import BlenderConfig, BlenderDataset
from mrcnn import model as modellib
from mrcnn import utils


class BlenderInferenceConfig(BlenderConfig):
    """验证推理配置；batch size 固定为 1，保留低分预测用于链路检查。"""

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.0


def _default_root(dataset_dir):
    """根据数据集路径推导项目根目录。"""
    configured_root = os.environ.get("ROOT")
    if configured_root:
        return os.path.abspath(configured_root)
    dataset_dir = os.path.abspath(dataset_dir)
    # 数据集通常位于 ROOT/data/blender。
    return os.path.abspath(os.path.join(dataset_dir, os.pardir, os.pardir))


def _parse_indices(text, num_images):
    """解析内部数据集索引，例如 ``0,1,2``。"""
    try:
        indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError:
        raise ValueError("--indices 必须是逗号分隔的整数，例如 0,1,2")
    if not indices:
        raise ValueError("至少需要一个验证集索引")
    if len(indices) > 10:
        raise ValueError("一次最多推理 10 张图像")
    if len(set(indices)) != len(indices):
        raise ValueError("--indices 中不能有重复索引")
    invalid = [index for index in indices if index < 0 or index >= num_images]
    if invalid:
        raise IndexError(
            "验证集索引超出范围: {}，合法范围为 0 到 {}".format(
                invalid, num_images - 1))
    return indices


def _colors(count):
    """生成可重复的 RGB 颜色。"""
    if count <= 0:
        return []
    return [tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(
        float(index) / count, 0.85, 1.0)) for index in range(count)]


def _overlay(image, boxes, masks, class_ids, scores=None, score_threshold=0.0,
             title=None):
    """把实例 mask、框和分数叠加到 RGB 图像上。"""
    image_array = np.asarray(image[..., :3], dtype=np.uint8)
    canvas = image_array.astype(np.float32).copy()
    if masks.ndim != 3:
        raise ValueError("mask 必须是 [H, W, N]，实际为 {}".format(masks.shape))
    if masks.shape[-1] != len(boxes):
        raise ValueError("框数量与 mask 数量不一致")
    colors = _colors(len(boxes))
    draw_items = []
    for index in range(len(boxes)):
        score = float(scores[index]) if scores is not None else None
        if score is not None and score < score_threshold:
            continue
        mask = masks[:, :, index].astype(bool)
        color = np.asarray(colors[index], dtype=np.float32)
        if np.any(mask):
            canvas[mask] = canvas[mask] * 0.55 + color * 0.45
        draw_items.append((index, color.astype(np.uint8), score))

    output = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    drawer = ImageDraw.Draw(output)
    if title:
        drawer.rectangle((0, 0, min(output.width, 520), 22), fill=(0, 0, 0))
        drawer.text((5, 4), title, fill=(255, 255, 255))
    for index, color, score in draw_items:
        y1, x1, y2, x2 = [int(value) for value in boxes[index]]
        drawer.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
                         outline=tuple(int(value) for value in color), width=2)
        label = "component"
        if score is not None:
            label += " {:.3f}".format(score)
        text_y = max(0, y1 - 12)
        drawer.rectangle((x1, text_y, min(output.width - 1, x1 + 105), y1),
                         fill=tuple(int(value) for value in color))
        drawer.text((x1 + 2, text_y), label, fill=(0, 0, 0))
    return output


def _save_prediction(output_dir, image_index, info, image, result,
                     ground_truth_masks, ground_truth_class_ids, score_threshold):
    """保存单张图像的原始预测和可视化文件。"""
    image_id = int(info["id"])
    stem = "val_{:06d}_id_{}".format(image_index, image_id)
    rgb = Image.fromarray(np.asarray(image[..., :3], dtype=np.uint8), mode="RGB")
    depth = Image.fromarray(np.asarray(image[..., 3], dtype=np.uint8), mode="L")
    rgb_path = os.path.join(output_dir, stem + "_rgb.png")
    depth_path = os.path.join(output_dir, stem + "_depth.png")
    rgb.save(rgb_path)
    depth.save(depth_path)

    boxes = np.asarray(result["rois"], dtype=np.int32)
    class_ids = np.asarray(result["class_ids"], dtype=np.int32)
    scores = np.asarray(result["scores"], dtype=np.float32)
    masks = np.asarray(result["masks"], dtype=np.bool_)
    mask_path = os.path.join(output_dir, stem + "_pred.npz")
    np.savez_compressed(mask_path, rois=boxes, class_ids=class_ids,
                        scores=scores, masks=masks)

    prediction_overlay = _overlay(
        image, boxes, masks, class_ids, scores=scores,
        score_threshold=score_threshold,
        title="prediction: {} detections".format(len(boxes)),
    )
    prediction_path = os.path.join(output_dir, stem + "_prediction.png")
    prediction_overlay.save(prediction_path)

    gt_boxes = utils.extract_bboxes(ground_truth_masks)
    gt_overlay = _overlay(
        image, gt_boxes, ground_truth_masks, ground_truth_class_ids,
        title="ground truth: {} instances".format(len(gt_boxes)),
    )
    gt_path = os.path.join(output_dir, stem + "_ground_truth.png")
    gt_overlay.save(gt_path)

    comparison = Image.new("RGB", (rgb.width * 2, rgb.height))
    comparison.paste(gt_overlay, (0, 0))
    comparison.paste(prediction_overlay, (rgb.width, 0))
    comparison_path = os.path.join(output_dir, stem + "_comparison.png")
    comparison.save(comparison_path)

    kept = scores >= score_threshold
    summary = {
        "image_id": image_id,
        "image_index": int(image_index),
        "file_name": info["path"],
        "ground_truth_count": int(len(gt_boxes)),
        "raw_prediction_count": int(len(boxes)),
        "visualized_prediction_count": int(np.sum(kept)),
        "score_threshold": float(score_threshold),
        "rois_y1_x1_y2_x2": boxes.tolist(),
        "class_ids": class_ids.tolist(),
        "scores": [float(score) for score in scores],
        "mask_shape": list(masks.shape),
        "prediction_npz": mask_path,
        "prediction_overlay": prediction_path,
        "ground_truth_overlay": gt_path,
        "comparison_overlay": comparison_path,
    }
    summary_path = os.path.join(output_dir, stem + ".json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def make_parser():
    """建立验证推理命令行参数。"""
    parser = argparse.ArgumentParser(description="验证 Blender RGB-D Mask R-CNN 推理")
    parser.add_argument("dataset", help="Blender 数据集根目录")
    parser.add_argument("--weights", required=True, help="RGB-D Mask R-CNN checkpoint 路径")
    parser.add_argument("--indices", default="0,1,2",
                        help="验证集内部索引，默认 0,1,2")
    parser.add_argument("--output", default=None,
                        help="输出目录，默认 ROOT/runs/member-a/inference/时间戳")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="叠加图显示阈值；默认 0，保留 smoke checkpoint 的低分预测")
    parser.add_argument("--detection-min-confidence", type=float, default=0.0,
                        help="模型内部最低置信度；默认 0，便于检查完整推理链路")
    parser.add_argument("--depth-max-m", type=float, default=1.0,
                        help="深度归一化截断上限（米），必须与训练一致")
    return parser


def main(args=None):
    """加载 checkpoint 并在指定验证图像上完成推理。"""
    args = make_parser().parse_args(args)
    if not os.path.isfile(args.weights):
        raise IOError("找不到 checkpoint: {}".format(args.weights))
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold 必须在 [0, 1] 范围内")
    if not 0.0 <= args.detection_min_confidence <= 1.0:
        raise ValueError("--detection-min-confidence 必须在 [0, 1] 范围内")
    if args.depth_max_m <= 0:
        raise ValueError("--depth-max-m 必须大于 0")

    dataset = BlenderDataset()
    dataset.load_blender(args.dataset, "val", depth_max_m=args.depth_max_m)
    dataset.prepare()
    indices = _parse_indices(args.indices, dataset.num_images)

    root_dir = _default_root(args.dataset)
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = os.path.join(root_dir, "runs", "member-a", "inference", timestamp)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    config = BlenderInferenceConfig()
    config.DETECTION_MIN_CONFIDENCE = args.detection_min_confidence
    config.display()
    print("验证图像数: {}".format(dataset.num_images))
    print("推理索引: {}".format(indices))
    print("输出目录: {}".format(output_dir))
    print("加载 checkpoint: {}".format(os.path.abspath(args.weights)))

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=output_dir)
    model.load_weights(args.weights, by_name=True)
    print("checkpoint 加载完成")

    all_summaries = []
    for image_index in indices:
        info = dataset.image_info[image_index]
        image = dataset.load_image(image_index)
        ground_truth_masks, ground_truth_class_ids = dataset.load_mask(image_index)
        result = model.detect([image], verbose=0)[0]
        summary = _save_prediction(
            output_dir, image_index, info, image, result,
            ground_truth_masks, ground_truth_class_ids, args.score_threshold)
        all_summaries.append(summary)

        print(
            "图像 index={} id={}：GT={}，raw_pred={}，显示={}，mask_shape={}".format(
                image_index, info["id"], summary["ground_truth_count"],
                summary["raw_prediction_count"],
                summary["visualized_prediction_count"], summary["mask_shape"]))
        for number, (box, class_id, score) in enumerate(zip(
                result["rois"], result["class_ids"], result["scores"]), 1):
            print("  pred {:03d}: class={} score={:.5f} roi={}".format(
                number, int(class_id), float(score),
                [int(value) for value in box]))

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(all_summaries, stream, ensure_ascii=False, indent=2)
    print("汇总文件: {}".format(summary_path))
    print("推理和可视化保存完成。")
    return 0


if __name__ == "__main__":
    main()

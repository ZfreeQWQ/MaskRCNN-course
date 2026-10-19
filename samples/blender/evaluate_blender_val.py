"""在 Blender RGB-D 验证集上计算 COCO Mask/BBox AP。

该脚本使用完整验证集，不依赖一个合并后的 COCO JSON：GT 由
``BlenderDataset.load_mask`` 解码得到，预测由 RGB-D Mask R-CNN 的
``model.detect`` 得到，再交给 pycocotools.COCOeval 计算指标。

示例（在服务器项目根目录执行）::

    python samples/blender/evaluate_blender_val.py \
        /home/liubowen26/MaskRCNN/data/blender \
        --weights /home/liubowen26/MaskRCNN/runs/member-a/\
blender_rgbd20261018T1021/mask_rcnn_blender_rgbd_0020.h5 \
        --output /home/liubowen26/MaskRCNN/runs/member-a/eval_baseline_0020

评估时建议将 ``--detection-min-confidence`` 保持为 0，让 COCOeval 在
不同 score 阈值上计算 AP；拥挤场景可单独搜索
``--detection-nms-threshold`` 和 ``--detection-max-instances``。
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SAMPLE_DIR not in sys.path:
    sys.path.append(SAMPLE_DIR)

from blender import BlenderConfig, BlenderDataset
from mrcnn import model as modellib


class BlenderEvaluationConfig(BlenderConfig):
    """推理配置；四通道输入和训练时的图像尺寸保持一致。"""

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.0


def _rle(mask):
    """将二值 mask 编码为 pycocotools 内部格式的 RLE。"""
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(encoded, list):
        encoded = encoded[0]
    return encoded


def _rle_json(mask):
    """将二值 mask 编码为 COCO JSON 可序列化的 RLE。"""
    encoded = _rle(mask)
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _xyxy_to_xywh(box, width, height):
    """将 Mask R-CNN 的 y1,x1,y2,x2 转成 COCO x,y,w,h。"""
    y1, x1, y2, x2 = [float(value) for value in box]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(x1, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(y1, min(float(height), y2))
    return [x1, y1, x2 - x1, y2 - y1]


def build_coco_ground_truth(dataset):
    """从 BlenderDataset 生成 COCOeval 所需的 GT API 对象。"""
    images = []
    annotations = []
    annotation_id = 1
    area_counts = {"small_lt_32": 0, "medium_32_to_96": 0, "large_gt_96": 0}
    for image_index, info in enumerate(dataset.image_info):
        image_id = int(info["id"])
        width = int(info["width"])
        height = int(info["height"])
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": str(info["path"]),
        })
        masks, class_ids = dataset.load_mask(image_index)
        for instance_index in range(masks.shape[-1]):
            if int(class_ids[instance_index]) != dataset.CLASS_ID:
                continue
            instance_mask = masks[:, :, instance_index].astype(np.uint8)
            encoded = _rle(instance_mask)
            rle = {
                "size": [int(value) for value in encoded["size"]],
                "counts": encoded["counts"].decode("ascii"),
            }
            area = float(mask_utils.area(encoded))
            if area < 32.0 * 32.0:
                area_counts["small_lt_32"] += 1
            elif area < 96.0 * 96.0:
                area_counts["medium_32_to_96"] += 1
            else:
                area_counts["large_gt_96"] += 1
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(dataset.CLASS_ID),
                "segmentation": rle,
                "area": area,
                "bbox": [float(value) for value in mask_utils.toBbox(encoded)],
                "iscrowd": 0,
            })
            annotation_id += 1

    coco = COCO()
    coco.dataset = {
        "info": {"description": "Blender RGB-D validation set"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": int(dataset.CLASS_ID),
            "name": dataset.CLASS_NAME,
            "supercategory": "component",
        }],
    }
    coco.createIndex()
    return coco, len(annotations), area_counts


def predict_dataset(model, dataset, progress_interval):
    """在完整数据集上推理并返回 COCO prediction 列表和耗时统计。"""
    predictions = []
    total_predictions = 0
    start = time.time()
    for image_index, info in enumerate(dataset.image_info):
        image = dataset.load_image(image_index)
        result = model.detect([image], verbose=0)[0]
        boxes = np.asarray(result["rois"])
        class_ids = np.asarray(result["class_ids"])
        scores = np.asarray(result["scores"])
        masks = np.asarray(result["masks"])
        image_id = int(info["id"])
        width = int(info["width"])
        height = int(info["height"])

        for index in range(len(boxes)):
            if int(class_ids[index]) != dataset.CLASS_ID:
                continue
            instance_mask = masks[:, :, index].astype(np.uint8)
            if not np.any(instance_mask):
                continue
            rle = _rle_json(instance_mask)
            predictions.append({
                "image_id": image_id,
                "category_id": int(class_ids[index]),
                "bbox": _xyxy_to_xywh(boxes[index], width, height),
                "score": float(scores[index]),
                "segmentation": rle,
            })
            total_predictions += 1

        if progress_interval and ((image_index + 1) % progress_interval == 0 or
                                  image_index + 1 == dataset.num_images):
            elapsed = time.time() - start
            rate = float(image_index + 1) / max(elapsed, 1e-9)
            print("进度: {}/{}，预测 {} 个实例，平均 {:.3f} 张/秒".format(
                image_index + 1, dataset.num_images, total_predictions, rate))

    elapsed = time.time() - start
    return predictions, elapsed


def evaluate(coco_gt, predictions, iou_type):
    """运行 COCOeval 并返回 COCO 的六个主指标。"""
    if not predictions:
        raise RuntimeError("模型没有输出任何有效预测，无法运行 COCOeval")
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }


def make_parser():
    parser = argparse.ArgumentParser(
        description="在完整 Blender val 上计算 RGB-D Mask R-CNN 的 COCO AP")
    parser.add_argument("dataset", help="Blender 数据集根目录")
    parser.add_argument("--weights", required=True, help="RGB-D checkpoint 路径")
    parser.add_argument("--output", required=True, help="评估输出目录")
    parser.add_argument("--depth-max-m", type=float, default=1.0,
                        help="深度归一化截断上限，必须与训练一致")
    parser.add_argument("--detection-min-confidence", type=float, default=0.0,
                        help="模型内部最低置信度；正式 COCO AP 建议保持 0")
    parser.add_argument("--detection-nms-threshold", type=float, default=0.3,
                        help="最终检测 NMS IoU 阈值；拥挤场景可试 0.4/0.5")
    parser.add_argument("--detection-max-instances", type=int, default=100,
                        help="每张图最多保留的最终实例数")
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="每多少张图打印一次进度，0 表示不打印")
    parser.add_argument("--save-predictions", action="store_true",
                        help="额外保存完整 COCO predictions.json，默认不保存")
    return parser


def main(args=None):
    args = make_parser().parse_args(args)
    if not os.path.isfile(args.weights):
        raise IOError("找不到 checkpoint: {}".format(args.weights))
    if args.depth_max_m <= 0:
        raise ValueError("--depth-max-m 必须大于 0")
    if not 0.0 <= args.detection_min_confidence <= 1.0:
        raise ValueError("--detection-min-confidence 必须在 [0, 1] 范围内")
    if not 0.0 < args.detection_nms_threshold <= 1.0:
        raise ValueError("--detection-nms-threshold 必须在 (0, 1] 范围内")
    if args.detection_max_instances < 1:
        raise ValueError("--detection-max-instances 必须大于 0")

    output_dir = os.path.abspath(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dataset = BlenderDataset()
    dataset.load_blender(args.dataset, "val", depth_max_m=args.depth_max_m)
    dataset.prepare()
    config = BlenderEvaluationConfig()
    config.DETECTION_MIN_CONFIDENCE = args.detection_min_confidence
    config.DETECTION_NMS_THRESHOLD = args.detection_nms_threshold
    config.DETECTION_MAX_INSTANCES = args.detection_max_instances
    config.display()
    print("验证图像数: {}".format(dataset.num_images))
    print("加载 checkpoint: {}".format(os.path.abspath(args.weights)))

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=output_dir)
    model.load_weights(args.weights, by_name=True)
    print("checkpoint 加载完成")

    coco_gt, gt_count, area_counts = build_coco_ground_truth(dataset)
    predictions, elapsed = predict_dataset(model, dataset, args.progress_interval)
    print("GT 实例数: {}，预测实例数: {}".format(gt_count, len(predictions)))
    print("推理耗时: {:.3f} 秒，平均 {:.3f} 张/秒".format(
        elapsed, float(dataset.num_images) / max(elapsed, 1e-9)))

    mask_metrics = evaluate(coco_gt, predictions, "segm")
    bbox_metrics = evaluate(coco_gt, predictions, "bbox")
    result = {
        "weights": os.path.abspath(args.weights),
        "weights_size_mb": float(os.path.getsize(args.weights) / (1024.0 * 1024.0)),
        "dataset": os.path.abspath(args.dataset),
        "images": int(dataset.num_images),
        "gt_instances": int(gt_count),
        "gt_area_counts_coco": area_counts,
        "pred_instances": int(len(predictions)),
        "depth_max_m": float(args.depth_max_m),
        "detection_min_confidence": float(args.detection_min_confidence),
        "detection_nms_threshold": float(args.detection_nms_threshold),
        "detection_max_instances": int(args.detection_max_instances),
        "inference_seconds": float(elapsed),
        "images_per_second": float(dataset.num_images / max(elapsed, 1e-9)),
        "mask": mask_metrics,
        "bbox": bbox_metrics,
    }
    if args.save_predictions:
        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as stream:
            json.dump(predictions, stream, ensure_ascii=False)
        result["predictions_file"] = predictions_path
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("指标文件: {}".format(os.path.join(output_dir, "metrics.json")))
    return 0


if __name__ == "__main__":
    main()

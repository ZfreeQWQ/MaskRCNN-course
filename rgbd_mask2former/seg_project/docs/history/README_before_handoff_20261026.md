# RGB-D 元器件实例分割（大作业）

当前计划：[docs/plan.md](docs/plan.md)（Mask2Former 基线 + RGB-D 小目标改进）。主指标是 **segm/APs**，同时报告整体 Mask AP、模型大小和推理速度。

所有命令从 `/private/rgzndzy` 执行，统一用 `./seg_project/run.sh` 启动。它只为子进程设置厂商驱动库路径和 PYTHONPATH，路径包括 `seg_project`、`seg_project/third_party/Mask2Former`、`detectron2`。禁止重装或升级 PyTorch、CoreX、驱动等底层环境。

## 目录结构

```text
/private/rgzndzy/
├── 图像分割.pdf  AGENTS.md
├── blender/                 原始数据（课程提供，只读）
├── detectron2/              Detectron2 源码（第三方）
└── seg_project/
    ├── README.md            本文件：结构、规则、常用命令
    ├── run.sh               进程级环境设置
    ├── docs/
    │   ├── plan.md          当前算法计划（Mask2Former）
    │   ├── migration_20261020.json   10-20 目录重组的旧→新路径对照
    │   └── history/         旧计划、Mask R-CNN 阶段的操作说明与总结
    ├── configs/
    │   ├── mask2former/     当前实验配置
    │   └── maskrcnn/        Mask R-CNN 阶段配置（B0/B1/M1/A1 及消融）
    ├── rgbd_seg/            项目库代码：数据注册、RGB-D mapper、训练器、checkpoint、推理
    ├── tools/
    │   ├── *.py             常用入口：prepare_data / prepare_test_eval / train / infer / evaluate /
    │   │                    benchmark / demo / error_analysis / merge_predictions / validate_submission
    │   ├── analysis/        数据与预测诊断脚本（结果写入 reports/analysis/）
    │   └── legacy/          Mask R-CNN 阶段的调度器与审计脚本（已归档，路径未维护，不再运行）
    ├── third_party/         Mask2Former 官方源码（待上传，见 docs/plan.md §8）
    ├── weights/             预训练权重（source.json 记录来源）
    ├── datasets/blender/annotations/
    │   ├── instances_{train,val,test}.json   合并后的 COCO 标注（test 只用于评估）
    │   ├── test_images_depth.json            test 推理输入元数据（含深度系数）
    │   ├── test_images.json                  旧版 test 元数据（深度系数为 null，保留）
    │   └── depth_stats.json                  训练集深度统计
    ├── experiments/
    │   ├── mask2former/     当前阶段实验输出
    │   ├── maskrcnn/        1x_{b0,b1,m1,a1}（权重、日志、val 预测）与 test_eval（test 指标）
    │   └── archive/         pilot / smoke 输出与旧调度器状态文件
    └── reports/
        ├── analysis/        数据组成、小目标形态、预测切片等诊断结果
        ├── data/            数据准备与校验报告
        ├── env/             环境、算子、DDP 检查记录（原 /private/rgzndzy/runs）
        └── maskrcnn/        Mask R-CNN 阶段报告、图表、旧审计记录
```

`docs/history/` 中旧文档的相对链接指向重组前的路径，查找对应文件请用 `docs/migration_20261020.json`。

## 数据与使用规则

| split | 图像 | 实例 | 用途 |
|---|---:|---:|---|
| train | 14,432 | 448,469 | 训练；深度归一化等统计只来自 train |
| val | 2,584 | 73,421 | 调参、checkpoint 选择、主要对比 |
| test | 2,112 | 75,462 | 最终推理；标注**只用于辅助模型选择** |

- test 规则（2026-10-20 用户决定）：test 标注可以在少量候选模型、checkpoint、后处理设置之间做选择，报告中要如实注明。test 图像和标注**一律不用于训练**，包括伪标签、统计量和采样权重。val 不并入 train。超参扫描以 val 为主。课程原始 `blender/README.md` 写明“针对 test 标注的试探视为违规”，该文件未改动。
- 深度：uint16 × `depth_scale_m`（全部为 0.0001）换算成米；`depth_invalid_code`（全部为 0）表示无效。test 的深度系数来自 test 分片 `images` 字段，见 `test_images_depth.json`。所有图像共享同一相机内参 K（fx = fy = 1910.81，cx = cy = 511.5）。
- 单类 `component`；`semantic_type`、`model_id` 只用于分析。三个划分的 CAD 型号两两不相交。

## 常用命令

```bash
# 数据（已完成；重新生成时使用）
./seg_project/run.sh python seg_project/tools/prepare_data.py
./seg_project/run.sh python seg_project/tools/prepare_test_eval.py

# 推理 + 评估（val 示例；test 把 --split 换成 test，metadata 换成 test_images_depth.json，annotations 换成 instances_test.json）
CUDA_VISIBLE_DEVICES=0 ./seg_project/run.sh python seg_project/tools/infer.py \
  --config <config.yaml> --weights <model.pth> \
  --metadata seg_project/datasets/blender/annotations/instances_val.json --split val \
  --output <dir>/val_predictions.json
./seg_project/run.sh python seg_project/tools/evaluate.py --split val \
  --annotations seg_project/datasets/blender/annotations/instances_val.json \
  --predictions <dir>/val_predictions.json --manifest <dir>/val_predictions.json.manifest.json \
  --output <dir>/val_metrics.json

# 速度与模型大小（1024×1024、batch=1、20 次预热、200 张计时）
CUDA_VISIBLE_DEVICES=0 ./seg_project/run.sh python seg_project/tools/benchmark.py \
  --config <config.yaml> --weights <model.pth> \
  --metadata seg_project/datasets/blender/annotations/instances_val.json \
  --warmup 20 --iterations 200 --output <dir>/benchmark.json

# 提交校验（test）
./seg_project/run.sh python seg_project/tools/validate_submission.py \
  --metadata seg_project/datasets/blender/annotations/test_images_depth.json \
  --predictions <submission.json> --csv <submission.csv> \
  --manifest <submission.json>.manifest.json --output <dir>/validation.json

# 诊断切片（碎片数 / 可见率 / 密度 / 相机距离 / 类型）
./seg_project/run.sh python seg_project/tools/analysis/pred_slices.py b0 a1
```

Mask2Former 的训练入口与配置会在实现后补充到这里，计划见 [docs/plan.md](docs/plan.md)。

## 提交格式（按 2026-10-25 用户提供的当前 Kaggle 要求）

- 提交 `submission.csv`，2,112 行加表头，列名为 `image_id,PredictionString`。
- `PredictionString` 是 JSON 列表；每个实例使用 `category_id=1, score, segmentation={size:[1024,1024],counts:<COCO compressed RLE 字符串>}`，每图最多 100 个实例。无预测必须写字符串 `[]`，不能留空或写 `null`。
- 主指标为 Mask APs@[0.50:0.95]，范围 0–1。Detectron2 日志中的百分制 APs 除以 100 后才是这一尺度；val 指标不等于 Kaggle 隐藏测试集分数。
- 保留完整 COCO results JSON，字段为 `image_id, category_id, bbox, score, segmentation`，用于复核、评估和再次导出。
- `blender/README.md` 的框坐标 CSV 说明已过时，原始数据保持不变。历史 `boxes.csv` 不可提交到当前掩码赛道。校验历史框文件时需显式指定 `--csv-format legacy-boxes`，该模式不会标记为可提交。
- `infer.py`、`infer_m2f.py`、`merge_predictions.py` 的 `--csv` 现在默认生成上述掩码格式。

已有完整推理 JSON 时，直接在 CPU 上转换，无需重新推理：

```bash
./seg_project/run.sh python seg_project/tools/export_submission.py \
  --metadata seg_project/datasets/blender/annotations/test_images_depth.json \
  --predictions seg_project/experiments/mask2former/submission_d/masks.json \
  --manifest seg_project/experiments/mask2former/submission_d/masks.json.manifest.json \
  --template blender/submission.csv \
  --output seg_project/experiments/mask2former/submission_d/submission.csv
```

导出器按模板图像顺序输出，逐个解码源 RLE 并核对 CSV、分片覆盖和实例数。输出已存在时拒绝覆盖。修复前的工具和说明位于 `seg_project/backups/submission_format_20261025_*`。

## 已有结果（Mask R-CNN R50-FPN 阶段，作为参考基线）

| 模型 | val APs | val AP | test APs | test AP | 推理 ms/图 |
|---|---:|---:|---:|---:|---:|
| B0 RGB | 26.75 | 55.61 | 12.48 | 60.82 | 85.9 |
| B1 RGB-D 4 通道 | 26.26 | 55.53 | 12.53 | 60.67 | 87.3 |
| M1 RGB-D + 小目标设置 | 27.52 | 54.92 | 13.93 | 59.62 | 136.1 |
| A1 RGB + 小目标设置 | 27.98 | 55.56 | 14.05 | 59.40 | 135.3 |

细节见 `reports/maskrcnn/report_1x.md` 与 `docs/history/`。

# ccc · MaskDINO

本目录对应 ccc 已选定的 `crop2048-extension` 版本，包含模型构建、几何融合、小实例裁剪续训、全图与裁剪融合推理及 CSV 导出。后续精修实验不在本分支中。

这里是便于公开提交的代码整理版：模型与数据变换计算保持原实现，服务器绝对路径改成环境变量，去掉历史队列、运行日志、数据索引和本地测试标签读取。原始实验归档与权重另行保存。精简版不会自行启动训练或调度其他 GPU。

## 实现

- 官方 MaskDINO R50，RGB 加深度派生的七通道几何信息；在 res2–res5 使用门控残差融合。
- 七通道包含深度、估计平面相对高度、三维法向、相对边缘和有效性。先按相机坐标计算，再同步变换 RGB、几何和标签。
- 小实例专门训练：训练集内小目标采样，原图 640 裁剪放大到 2048，FP32、单图微批次、累积两图、AdamW；恢复父模型与优化器、随机状态及 dropout 计数，1500 更新、每 50 步保存断点。
- 推理：2048 全图模型，加四角 640 裁剪、输入 2048 的专门模型。裁剪预测最大面积 2048，内部边界余量 8 像素，mask NMS IoU 0.5，每图最多 100 实例。
- 模型选择使用验证集；公开推理入口不读取测试标签。验证与比赛分数不得混用。

## 文件

| 文件 | 用途 |
|---|---|
| `geometry_signals.py`、`geometry_fusion.py` | 深度几何与门控融合 |
| `geometry_maskdino.py`、`train_maskdino_extra.py` | 严格权重加载、模型及优化器构建 |
| `focused_data.py`、`scale_data.py` | 小实例裁剪与同步增强 |
| `train_focused.py` | 本轮 1500 步续训、50 步断点和 OOM 恢复 |
| `mask_memory.py`、`mask_target_chunks.py`、`pixel_recompute.py`、`rgb_recompute_checked.py` | 显存优化 |
| `crop2048_impl.py`、`infer.py` | 原裁剪合并算法、参数化推理和 CSV 导出 |
| `course_data.py`、`independent_seg_data.py` | 训练/验证数据与仅图像测试元数据读取 |
| `model_config.yaml` | 对应的模型配置 |
| `settings.py` | 外部路径配置 |

## 环境与外部文件

Python 3.10、PyTorch 2.6.0 + CUDA 12.4、TorchVision 0.21.0，其余依赖见 `requirements.txt`。另外安装：

- [MaskDINO](https://github.com/IDEA-Research/MaskDINO/tree/3831d8514a3728535ace8d4ecc7d28044c42dd14)：commit `3831d8514a3728535ace8d4ecc7d28044c42dd14`。
- [Detectron2](https://github.com/facebookresearch/detectron2/tree/247691a1c292654867deb35cabb27f18d75a1363)：commit `247691a1c292654867deb35cabb27f18d75a1363`。
- 按 MaskDINO 说明编译 deformable-attention 扩展；PyTorch 2.6 的 scalar-dispatch 兼容修改见 `apply_torch_compat.py`。

必须自行提供数据、相机元数据、仅从训练集计算的几何归一化统计和权重。它们不在 Git 中。`CCC_DATA_ROOT` 使用课程 Blender 数据目录；训练、验证读取其对应标注。测试只接受 `CCC_IMAGE_METADATA` 中的 `images` 等非标注字段；每张图需要 `id/file_name/height/width/depth_file_name/depth_scale_m/depth_invalid_code/depth_semantics/camera.K`。

```bash
export CCC_MASKDINO_SRC=/path/to/MaskDINO
export CCC_DATA_ROOT=/path/to/blender
export CCC_WEIGHTS=/path/to/weights
export CCC_GEOMETRY_STATS=/path/to/geometry_stats.json
export CCC_IMAGE_METADATA=/path/to/image_only_metadata.json
export CCC_WORKDIR=/path/to/new_outputs
```

若依赖未安装到当前 Python，可用 `CCC_EXTRA_PYTHONPATH` 指定额外模块路径，分隔符同 `PYTHONPATH`。权重目录文件名与 SHA256：

| 文件名 | SHA256 |
|---|---|
| `official_instance_r50.pth` | `8c9df523ea5e53048b7b34af811d565917017009e1d22e3d9010cc00044d7d67` |
| `rgb_init.pth` | `4ab4d45d64b6400f0c52c16a3d2fe4c8e1b3c10f6b69dbc3a8f19dc7635fc2cd` |
| `global.pth` | `312654e16f6d413e06d91ef3674e21de3d3bfa2a8a3189f753ace04069da9268` |
| `crop.pth` | `472463ef32044a3525417bcfe4c5ff162af424aa6e704609799bde1242ec88de` |
| `parent.pth`（仅续训需要） | `9f26b613b6eaef04ac23350c56890f8bf57caa199f715699d12b734570c54ec7` |

初始化权重在模型构建时仍需提供；`parent.pth` 必须保留 AdamW 和 RNG 状态。不能将推理参数平均权重当作训练断点。

## 使用

在本目录、依赖与外部文件齐备后运行。示例 GPU 编号仅为示例，应先检查实际资源。

```bash
# 完整验证；两张工程检查可加 --limit 2
CUDA_VISIBLE_DEVICES=0 python infer.py --split val --output ./outputs/validation

# 比赛推理，按模板 ID 顺序导出，不读取测试标签
CUDA_VISIBLE_DEVICES=0 python infer.py --split test \
  --template /path/to/submission_template.csv --output ./outputs/submission

# 复现对应 1500 更新，使用新的输出目录
CUDA_VISIBLE_DEVICES=0 python train_focused.py --arm highres --phase train \
  --output ./outputs/training
# 中断后从同目录 latest.pth 恢复：添加 --resume
```

`submission.csv` 包含 `image_id,PredictionString`；掩码为 1024×1024 COCO 压缩 RLE，每图最多 100 个实例。生成结果及权重哈希记录在新输出目录中。此整理版的运行检查见 `CHECKS.md`，不将工程小样本检查当作完整验证或比赛成绩。

## 来源

使用官方 MaskDINO 与 Detectron2；上游许可证由对应项目管理，CUDA 兼容补丁保留其 Apache-2.0 声明。几何融合参考 SA-Gate/CMX 以及同组 RGB-D 经验，这里是任务适配，不声称提出新架构或完整复现论文。根目录原有 Matterport Mask R-CNN 实现保持原样。

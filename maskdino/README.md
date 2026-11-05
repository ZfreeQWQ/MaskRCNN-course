# ccc · MaskDINO / soft_quality

本目录包含课程实例分割的 RGB-D MaskDINO 实现，包括几何融合、全图与裁剪推理、边界修正和实例质量评分。

质量头使用 TRAIN 拟合，参数通过验证集选择。权重与输出文件哈希见 release.json。

## 推理过程

1. 全图和四角原生640×640裁剪均输入2048×2048，五个视野共用最终 `crop.pth`。每个视野保留原模型输出的100个预测，裁剪候选要求面积≤2048、距内部边界≥8像素。
2. 原始分数合并、mask NMS IoU=0.5，最多100个候选。在原生1024×1024上，学习到的 RGB-D 邻接亲和权驱动24步边界扩散；微小/细条实例使用分量种子和面积保护。
3. 冻结23维十阈值质量头选择最终掩码与NMS集合，再用43维头重新评分。新增20维来自原始mask logits、概率、熵、边界不确定性、类别概率和视野信息。
4. 对面积≤2048的入选预测，最终分数为原始模型分数乘43维头十个输出的均值；输出对应IoU 0.50:0.05:0.95的通过概率。较大预测沿用冻结父方案分数。最后排序，不额外改变掩码、阈值或再做NMS。

`infer.py` 是本版默认原图入口；`soft_quality.py` 生产真实特征，不需要历史实验缓存。旧版全图/裁剪不同权重的入口保存在 `infer_crop2048.py`，其输出对应较早的推理流程。未通过门槛的实验和自动训练队列没有包含在本提交中。

## 环境和权重

Python3.10、PyTorch2.6.0+CUDA12.4、TorchVision0.21.0，其余依赖见 `requirements.txt`。还需安装：

- [MaskDINO](https://github.com/IDEA-Research/MaskDINO/tree/3831d8514a3728535ace8d4ecc7d28044c42dd14)，commit `3831d8514a3728535ace8d4ecc7d28044c42dd14`，按上游说明编译 deformable-attention。
- [Detectron2](https://github.com/facebookresearch/detectron2/tree/247691a1c292654867deb35cabb27f18d75a1363)，commit `247691a1c292654867deb35cabb27f18d75a1363`。
- PyTorch2.6 scalar-dispatch兼容补丁见 `apply_torch_compat.py`。

三个小权重已放入 `artifacts/`，默认相对本目录加载，不依赖服务器目录。

| 文件 | SHA-256 |
|---|---|
| `edge_weights.pth` | `c06034a04101520359c02802ded3e3a9fd512737f2d70a8be467e786cb16533b` |
| `parent_ordinal_weights.pth` | `d433f356a495e46c51d33f6785d437237515e2a4e2133a6a062d0f6c09136680` |
| `soft_quality_weights.pth` | `dfa7ee8309a4b6a4528b3dc7730cbededd6370ec9b67fadbb04f124f8a9ce813` |

大型权重、课程数据和CSV单独交付，不纳入Git。主断点 `step_1500.pth` 在外部权重目录命名为 `crop.pth`。当前模型构建复用原训练初始化流程，还需要两个初始化文件，随后被最终checkpoint严格覆盖：

| 外部文件名 | SHA-256 |
|---|---|
| `crop.pth` | `472463ef32044a3525417bcfe4c5ff162af424aa6e704609799bde1242ec88de` |
| `official_instance_r50.pth` | `8c9df523ea5e53048b7b34af811d565917017009e1d22e3d9010cc00044d7d67` |
| `rgb_init.pth` | `4ab4d45d64b6400f0c52c16a3d2fe4c8e1b3c10f6b69dbc3a8f19dc7635fc2cd` |

另需原TRAIN几何归一化统计 `geometry_stats.json`、RGB-D数据和相机元数据。测试元数据JSON只接受 `images/categories/info/licenses`，不能含 `annotations`。每张图需有 `id/file_name/height/width/depth_file_name/depth_scale_m/depth_invalid_code/depth_semantics/camera.K`，图像路径相对数据根目录。

## 运行

```bash
cd maskdino
export CCC_MASKDINO_SRC=/path/to/MaskDINO
export CCC_DATA_ROOT=/path/to/blender
export CCC_WEIGHTS=/path/to/external_weights
export CCC_GEOMETRY_STATS=/path/to/geometry_stats.json
export CCC_IMAGE_METADATA=/path/to/image_only_metadata.json
export CCC_WORKDIR=/path/to/new_outputs
# 如需额外依赖路径可设CCC_EXTRA_PYTHONPATH，分隔符同PYTHONPATH。

# 两图工程检查；需要一张空闲、约24GB的CUDA显卡
CUDA_VISIBLE_DEVICES=0 python infer.py --split val --limit 2 --output ./outputs/check

# 完整验证；validation.json中segm指标量纲为0–100
CUDA_VISIBLE_DEVICES=0 python infer.py --split val --output ./outputs/validation

# 测试入口不读取测试标注、不计算测试AP，按模板ID顺序导出
CUDA_VISIBLE_DEVICES=0 python infer.py --split test \
  --template /path/to/submission_template.csv --output ./outputs/submission
```

可通过 `CCC_POSTPROCESS_WEIGHTS` 覆盖三个小权重目录。输入、代码、权重改变时需要新输出目录，避免混用缓存。`submission_io.py` 对所有mask解码再编码并回读CSV，要求两列 `image_id,PredictionString`、1024×1024压缩RLE、每图最多100实例、空预测为 `[]`。`--save-features` 可另存未评分43维缓存。

## 评分接口和核验

`FrozenFeatureScorer` 只对已有预测的原始特征缓存评分，原图推理使用上述入口；不能对已乘质量分的预测重复评分。

```python
from soft_feature_scoring import FrozenFeatureScorer
scorer = FrozenFeatureScorer(device='cpu')
predictions = scorer.score_file('outputs/submission/features/00000.json')
```

`check_release.py` 可比较缓存与独立归档，要求掩码/顺序一致、分数绝对误差≤1e-5：

```bash
python check_release.py --features /path/to/raw_43d_features \
  --expected /path/to/archived_predictions --device cuda --output replay.json
```

正式交付CSV为2112张图、203324实例，SHA-256为 `3accae775e8f0c26389d77523ba9c0fd28c454b6ca2717083c67c0c0b909c505`。核验范围见 [CHECKS.md](CHECKS.md)，生产源码与参数化代码对应关系见 [source_provenance.json](source_provenance.json)。

正式GPU评分回放在全部2584张验证图上完全一致。CPU评分可能在极接近的分数处交换次序：已观察到一张验证图中两项互换，分数误差小于1.2e-7；正式推理使用CUDA。全部2112张测试缓存的CPU回放仍保持掩码及顺序一致，最大分数误差约2.51e-7。

原主模型训练入口 `train_focused.py`、几何融合及显存优化模块保留。新43维头的固定拟合配置：TRAIN768图/留出256图、43→64→64→10，从23维父头等价归一化暖启动，新列权重为0，Adam学习率0.0003、weight decay0.0001、50epoch、batch512、面积均衡十阈值BCE。边界头10→32→32→1，只在TRAIN拟合。此处发布冻结已选权重，未重新训练或重跑完整AP评测。

## 来源

使用官方MaskDINO和Detectron2；上游许可证由对应项目管理。几何融合参考SA-Gate/CMX，质量评分参考Mask Scoring R-CNN思路，属于课程适配，不声称完整复现论文或提出新架构。根目录原Matterport Mask R-CNN保留。

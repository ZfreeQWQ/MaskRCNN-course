# RGB-D 小目标实例分割：代码与优化记录

此目录是 2026-10-29 从训练服务器整理的代码快照。项目以 Mask2Former R50、Detectron2 和 RGB-D 输入为基础，主要优化小目标实例分割。仓库其他目录保留原 Mask R-CNN 课程代码；本目录是独立的 Mask2Former 项目。

## 已完成的优化与实验

下表均为完整验证集 2,584 张图像的 COCO mask AP，按 0–100 计。APs 为小目标 AP，验证集结果不是隐藏测试集成绩。

| 模型或设置 | APs | AP |
| --- | ---: | ---: |
| RGB-D D 基线，1024 输入，legacy 推理 | 37.969 | 73.794 |
| 同一 D 模型，1536 输入，mask-first 推理 | 50.094 | 80.440 |
| D 追加 6 epoch | 52.916 | 81.603 |
| D + MP-Former 适配，6 epoch | 53.853 | 81.870 |
| D + 频率融合，6 epoch | 53.670 | 81.904 |
| D + MP-Former + 频率融合，6 epoch | 54.924 | 82.328 |
| 上述组合延长训练后的基线 | 55.367 | 82.501 |
| 基线续训，对照组最佳权重（第 11 epoch） | **56.611** | 82.805 |
| 基线续训，裁剪增强组最佳权重 | 56.462 | 83.100 |

- **RGB-D 几何融合**：编码深度及其有效性，通过门控方式融合 RGB 和深度特征；实现见 `seg_project/rgbd_seg/m2f/depth_fusion.py`、`geo_transforms.py`。
- **小目标推理**：输入缩放至 1536，恢复至原图尺寸；使用分类分数和掩码质量进行 mask-first 排序，最多保留 100 个实例。对应 `experimental_inference.py` 和 `round2_inference.py`。
- **MP-Former 适配与频率融合**：分别参考 [MP-Former](#ref-mp-former) 与 [FreqFusion](#ref-freqfusion)，实现 mask-piloted 训练适配及频率特征融合，完成单项和组合训练。适配不是对论文全部实验的原样复现。见 `mpformer.py`、`frequency_fusion.py` 和 `seg_project/docs/mpformer_execution_20261026.md`。
- **继续训练和泛化对照**：保留优化器状态和全局步数，比较普通续训、裁剪增强、局部监督及组合方案。裁剪和局部监督的设计分别参考 [SAHI](#ref-sahi) 和 [PointRend](#ref-pointrend) 的思路。对照组和裁剪组均完成累计 12 epoch；对照组最佳权重来自第 11 epoch。裁剪方案在 12 epoch 下未超过同预算对照组，局部监督也未在 6 epoch 时超过对照组。
- **高分辨率局部细化和多视角实验**：提供局部细化、裁剪与多尺度实验代码。多视角冒烟运行因 `ModuleNotFoundError: tools.common` 失败，未获得完整验证结果，不能视为已验证的收益。
- **提交生成和校验**：修正导出时权重路径类型错误；四卡完成 2,112 张 test 图像推理，生成 mask JSON 格式 submission，并验证了图像覆盖、预测及 RLE。结果共 207,797 个实例；CSV SHA-256 为 `5ca4c2b5c3ad3a71dfdf37a81e392c03ed3e6f647014702d3161e2b89c132115`。最终平台分数尚未确认。

## 目录和使用前提

```text
rgbd_mask2former/
├── detectron2/                 使用时的第三方源码快照（含本地兼容修改）
└── seg_project/
    ├── rgbd_seg/               模型、训练、数据和推理实现
    ├── tools/                  训练、评估、导出及测试工具
    ├── configs/                实验基础配置
    ├── third_party/            Mask2Former 和已有算子适配层
    ├── experiments/mask2former/ 选定实验配置快照
    └── docs/                   历史方案和交接资料
```

原环境为 Linux、4 张 Iluvatar BI-V150 32GB、Python 3.12.3、PyTorch `2.7.1+corex.4.4.0`。`seg_project/run.sh` 使用原服务器 CoreX 环境；该文件中的 `NCCL_P2P_DISABLE=1` 是原环境下四卡通信兼容设置。运行前应检查自己的硬件、依赖和路径，不要直接重装或升级服务器的 PyTorch、驱动或厂商运行时。

数据集、标注、模型权重、提交 CSV、编译算子和 SSH 凭据均不在此仓库。最佳权重原路径为 `/private/rgzndzy/seg_project/experiments/mask2former/round4_control_12ep_20261027/best_segm_APs.pth`，SHA-256 为 `989fcca52a1115c9d7564622d549c9d0ec51c0feec9ada6d0e5d8ad02accdedf`。配置与脚本保留原服务器绝对路径，迁移时需检查并修改；缺少外部数据、权重、厂商算子或依赖时不能直接复现。

服务器目录结构及运行入口参见 `seg_project/docs/HANDOFF_RUNBOOK_20261026.md`，其中历史实验状态以本文为准。核心导出入口是 `seg_project/tools/export_generalization.py`，需提供包含 `recommended_model` 的选择 JSON、外部权重及测试图像元数据；输出目录必须不存在。`export_submission.py` 和 `validate_submission.py` 负责最终格式和 RLE 校验。历史队列脚本会启动任务或修改状态，运行前需检查其路径和既有进程。

第三方源码的原许可证保留于 `detectron2/LICENSE` 和 `seg_project/third_party/Mask2Former/LICENSE`。Detectron2 上游快照版本为 `1e3e13bbf607b54f62205c4c33922521822fb298`，服务器曾修改 `detectron2/layers/wrappers.py`，此处保留该修改。

## 参考文献与代码来源

以下列出项目实际使用的基础框架，以及实验设计中借鉴的论文。论文在其他数据集上的结果不代表本项目验证集或测试集的收益。

1. <a id="ref-mask2former"></a> **Mask2Former**：Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar. *Masked-attention Mask Transformer for Universal Image Segmentation*. CVPR 2022. [论文](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html) · [官方代码](https://github.com/facebookresearch/Mask2Former)。本项目的基础实例分割框架；源码快照位于 `seg_project/third_party/Mask2Former/`。
2. <a id="ref-detectron2"></a> **Detectron2**：Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, Ross Girshick. [Detectron2 官方代码](https://github.com/facebookresearch/detectron2)。本项目使用其训练、数据结构和评估组件；随包源码包含服务器端的兼容修改。
3. <a id="ref-mp-former"></a> **MP-Former**：Hao Zhang, Feng Li, Huaizhe Xu, Shijia Huang, Shilong Liu, Lionel M. Ni, Lei Zhang. *MP-Former: Mask-Piloted Transformer for Image Segmentation*. CVPR 2023. [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_MP-Former_Mask-Piloted_Transformer_for_Image_Segmentation_CVPR_2023_paper.html) · [官方代码](https://github.com/IDEA-Research/MP-Former) · [参考版本](https://github.com/IDEA-Research/MP-Former/tree/a71f504a467d7079cf02726a05ff23bbb933607d)。本项目保留已有 RGB-D 主干和普通 query 位置编码，对 pilot 训练机制进行了适配，见 `seg_project/docs/mpformer_execution_20261026.md`。
4. <a id="ref-freqfusion"></a> **FreqFusion**：Linwei Chen, Ying Fu, Lin Gu, Chenggang Yan, Tatsuya Harada, Gao Huang. *Frequency-aware Feature Fusion for Dense Image Prediction*. IEEE TPAMI, 2024. [论文](https://arxiv.org/abs/2408.12879) · [作者代码](https://github.com/Linwei-Chen/FreqFusion)。本项目借鉴频率融合思想实现轻量模块，没有复现论文的完整模块；具体差异见 `seg_project/rgbd_seg/m2f/frequency_fusion.py`。
5. <a id="ref-sahi"></a> **SAHI**：Fatih Cagatay Akyon, Sinan Onur Altinuc, Alptekin Temizel. *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*. ICIP 2022. [论文](https://arxiv.org/abs/2202.06934) · [官方代码](https://github.com/obss/sahi)。本项目的目标保持裁剪借鉴切片与放大思路，并非 SAHI 的检测或融合流水线复现；多视角推理尚未完成验证。
6. <a id="ref-pointrend"></a> **PointRend**：Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick. *PointRend: Image Segmentation as Rendering*. CVPR 2020. [论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) · [官方代码](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)。本项目局部监督参考细节采样思路，没有新增 PointRend 细化头或复现其完整方法。
7. <a id="ref-coco"></a> **COCO**：Tsung-Yi Lin 等. *Microsoft COCO: Common Objects in Context*. ECCV 2014. [论文](https://arxiv.org/abs/1405.0312) · [COCO API](https://github.com/cocodataset/cocoapi)。本项目使用 COCO 风格的实例分割 AP/APs 和压缩 RLE 格式；项目数据集并非 COCO 数据集。

更多研究方向与其适用范围见 `seg_project/docs/optimization_round2_research_20261026.md`；其中的提议不代表已实现或已验证。

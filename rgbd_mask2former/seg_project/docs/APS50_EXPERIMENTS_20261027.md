# 面向 Kaggle 小目标 APs > 50 的第四轮实验

**20:46 起训练预算已修订：四组至少累计6 epoch，最佳改进组与同预算对照至少12 epoch，再按完整val平台判断是否延至18/24。**
详见 [充分训练与收敛判断修订](APS50_CONVERGENCE_REVISION_20261027.md)。下文2 epoch矩阵保留为初始阶段记录，不再是最终停止预算。

## 目标与固定基线

用户确认 `C:\workspace\ssh\deliverables` 中的提交在 Kaggle 约为 41。该目录唯一 CSV 是
`submission_mp_frequency_best_20261027.csv`，SHA256 为
`9c078dd897eefee4cd40644774b8d3316674262daff0dd36bade25b599ea3b5a`。
它来自 MP+频率融合的 12 epoch 延长训练，完整 val（2584 图）的 Mask APs 为 55.36656、AP 为 82.50141。
Kaggle 分数是用户反馈，当前无法读取线上评测明细。**val 超过 50 不表示 test 目标已达成。**

固定父权重：`experiments/mask2former/round3_best_algorithm_extend_20261026/best_segm_APs.pth`，
SHA256 `ac4429006e3021a1d20c34052a1d564a4749f97a9efe0e153fa82b8f6843fd58`。
所有新代码使用独立入口，不修改旧训练源文件、已有权重、依赖或驱动。

## 已完成的错误诊断

对父模型保存的 val 预测重新执行 COCO mask 评测，独立复现 APs=55.36656。
以下是 COCO 按置信度排序的一对一匹配召回（maxDets=100），并非 AP，也非任意候选覆盖。

| val 小目标分组 | 数量 | IoU≥0.50 | IoU≥0.75 | IoU≥0.90 |
|---|---:|---:|---:|---:|
| 所有小目标 | 10999 | 88.58% | 68.29% | 29.63% |
| 面积≤256 px² | 2185 | 65.17% | 38.76% | 10.71% |
| mask/bbox 填充率<0.25 | 4023 | 81.53% | 46.56% | 7.41% |
| 高密度图像中的小目标 | 5720 | 83.85% | 59.41% | 22.12% |

train/val 历史统计还显示：两者型号不重叠，train 小目标中 wire_board_connector 最多，val 则以
header_connector 为主。因此完整 val 总分可能低估某些形态和类型的困难；尚不能仅凭这些统计断言线上差距的原因。
本轮不读取 test 标注，不根据历史 test 标注统计选配方。

## 两项可独立检验的改进

### A：混合全图与目标保持裁剪

50% 概率尝试从 train 的小目标（原图面积≤1024）或低填充率遮挡目标（面积≤4096、填充率<0.25）中选择焦点。
对极小/低填充率目标温和提高采样权重，最高裁剪前面积权重 4 倍。
随机选 640/768/896 的方形区域，并保证所选目标完整可见掩码的包围框被保留；无法满足则回到原全图增强。
裁剪后放大至 1024，保留旋转、翻转和 RGB 颜色增强。其他实例同步裁剪、删除空实例并重算可见 mask 的框。
非焦点分支保留原 LSJ 配方。深度几何始终在原图计算，再与 RGB/GT 同步变换。

假设：增加目标有效像素和困难目标出现频率，有助于极小/遮挡目标泛化，同时保留全图上下文。
风险：部分裁剪会切断其他实例；完整 val 和所有形态切片都必须复核。
裁剪放大思路参考 [SAHI](https://arxiv.org/abs/2202.06934)，实现是本项目的训练适配，不是 SAHI 检测/融合流水线复现。

### B：追加局部掩码监督

保持原 matcher、标准 PointRend 风格采点、标准 CE/Dice、全部辅助层和 MP 分支不变。
仅对最后一层普通 query 的原有匹配结果添加局部损失；不重新匹配，不重复监督 pilot 分支。
选择变换后面积≤4096，或面积≤16384 且填充率<0.25 的实例，每 rank 最多 12 对。
每对额外采 96 个 GT 前景像素中心、96 个边界带像素中心、64 个扩展框上下文点。
GT 不先缩小，保留单像素碎片的被采样机会。追加 BCE 与 Dice 各权重 0.5，标准 mask/dice 权重为 5。
这是有意改变局部监督分布的辅助损失，不能把它解释为无偏的全像素 IoU 估计。
各 rank 无合格目标时仍返回相同 loss 键，分布式归一化使用全局选中实例数。

假设：提高已有 query 对稀疏前景和边界的刻画精度；它本身不能创造完全缺失的 query。
参考 [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)
对细节采样的思路；本实验没有新增 PointRend 细化头，也不是其论文复现。

## 实验矩阵和统一预算

| 实验 | 裁剪 A | 局部监督 B | 额外训练 |
|---|---|---|---|
| control | 关闭 | 关闭 | 2 epoch |
| crop | 开启 | 关闭 | 2 epoch |
| local | 关闭 | 开启 | 2 epoch |
| crop_local | 开启 | 开启 | 2 epoch |

四组都从同一父权重开始，fresh AdamW，四卡×每卡2图，全局 batch8，种子20261027，LR=1e-5、骨干0.1倍，训练1024。
频率模块/label embedding 已完成训练，本轮与其他成熟模块使用相同 LR，不再使用旧队列中新模块的10倍 LR。
每 epoch 完整验证2584图，1536输入恢复1024原图，mask-first排序、最多100实例、阈值0。
每902步存断点，计划每组3608步，总14432步；这是2 epoch机制筛选，不能当作充分收敛或种子稳定性结论。

先做组合分支60步四卡冒烟和16图推理。16图分数不能挑最终权重。
随后四组顺序运行，各组使用独立持久化目录；失败停止队列，不自动重试或覆盖结果。
每组训练完独立重算最佳 val 预测的 COCO 分数与形态/类型/密度召回切片，检查与 trainer 分数一致。
最终比较 A/control、B/control、A+B/control，以区分新增机制和额外训练的贡献。

候选若整体 AP 不低于父模型0.3点，且 APs 超过父模型0.2点，才作为新的提交候选；否则保留父模型。
只为选中的一个候选生成 test CSV，沿用与完整 val 相同的 `paired_prediction` 和 mapper，严格加载权重，验证2112行和RLE。
不会自动向 Kaggle 上传；生成后由线上评测检验是否接近/达到50。
若控制组胜出，结论只能是额外训练有效，不能宣称两个新增机制获益。

## 检查与运行

已通过：原标准/MP损失逐位一致、局部梯度只进入最后普通 mask、空目标有限值、像素中心采样、完整焦点保持、真实 train RGB-D mapper、RGB/深度裁剪翻转对齐。
首次四卡冒烟发现 bool GT 导致坐标常量错误转换，新增局部损失退化为常数；正式训练尚未开始时修正。
修正版增加 bool/uint8/float 回归检查；四卡60步验证前景标签均为1、采样logits有效，单卡峰值张量显存约13.63 GiB。
初次冒烟目录保留为审计记录，修正版目录为 `round4_crop_local_smoke_v2_20261027`，不混用其分数。
正式状态以服务器 `generalization_20261027/state.json` 为准。

```bash
cd /private/rgzndzy
./seg_project/run.sh python seg_project/tools/run_generalization_queue.py --status
tail -n 30 seg_project/experiments/mask2former/generalization_20261027/jobs/control_train/training.log
./seg_project/run.sh ixsmi
```

工作区同步结果和完成后的CSV：

```powershell
python sync_generalization_20261027.py
```

本地已配置有限12小时的后台同步，每5分钟更新 `outputs/generalization_20261027/status.md`，
若生成选优CSV则下载到 `deliverables/submission_generalization_best_20261027.csv` 并校验SHA256与2112行。
服务器训练队列独立于本地同步运行。实时阶段和已完成组别以状态文件为准，不保证所有改进都有效。

所有权重、日志、配置和状态均位于 `/private/rgzndzy/seg_project/experiments/mask2former/`。
本地 `outputs/generalization_20261027/` 保存同步的精简证据；本文件不把尚未完成的训练记为结果。

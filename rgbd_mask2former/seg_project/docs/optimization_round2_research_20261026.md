# 第二轮算法优化调研与实验建议

调研日期：2026-10-26。依据：项目源码、完整 val 结果、最新模型的全量候选覆盖诊断，以及论文和作者官方实现。本文件提出后续实验；本次未启动新的模型训练。

## 1. 当前基线与结论

建议以 `d_continue6_results_20261026` 的 `continued1536_mask_first` 为固定基线：完整 2,584 张 val，Mask APs=0.529162213（52.916%），Mask AP=0.816031070，ARs=0.598945359。它是原 D 权重追加 6 epoch 后的结果，不能称为严格从头训练的 12 epoch 对照。

| 对照 | 完整 val APs | 说明 |
|---|---:|---|
| 原 D，1024，legacy | 0.379694 | 第一轮复现 |
| 原 D，1536，mask-first | 0.500942 | 分辨率提升贡献最大 |
| 追加训练 D，1536，legacy | 0.528942 | 独立实验推理复核 |
| 追加训练 D，1536，mask-first | 0.529162 | 当前推荐基线 |

追加训练相对上一轮提高 2.822 个 APs 百分点；在追加训练模型上重排序只增加约 0.022 个百分点。训练器第 5/6 个追加 epoch 的 APs 为 0.524354 / 0.528937，最佳仍在最后一轮，目前证据不足以认定“50 是模型上限”或已经完全收敛。再延长训练可以保留为控制组，但值得开始有针对性的结构和训练机制实验。

新提交已完成：`outputs/d_continue6_results_20261026/submission.csv`，2,112 张图、210,375 个实例。服务器全量 RLE 和源预测一致性验证通过，本地下载 SHA256 为 `98f50cca289b0e767a72c95d8ca4b43427ee665e249f6dc90d300515de71949a`，与远程一致。上述分数全部为 val 指标，尚无这份提交的 Kaggle 分数。

## 2. 新模型仍缺什么：已完成的诊断

本次在服务器 CPU 上对最新完整 val 预测重新计算候选覆盖，未改动预测或模型。结果保存于 `reports/analysis/d_continue6_coverage_20261026.json`，包含预测与标注 SHA256。

下表表示每个 GT 是否存在任意一个保存候选达到指定 mask IoU。没有一对一匹配，不是 AP、不是 COCO AR，也不是评估分数的严格上界；只用于定位改进方向。

| 分组 / 条件 | 原 D 1536 | 追加训练 D 1536 |
|---|---:|---:|
| 所有小目标，IoU≥0.5 | 84.86% | 86.74% |
| 极小目标 ≤256 px²，IoU≥0.5 | 56.57% | 60.59% |
| 极小目标 ≤256 px²，IoU≥0.75 | 31.53% | 35.10% |
| 所有小目标，IoU≥0.75 | 61.65% | 64.72% |
| 所有小目标，IoU≥0.9 | 24.89% | 26.81% |
| 小目标 mask/bbox 面积比 <0.25，IoU≥0.75 | 37.31% | 41.41% |
| 同上，IoU≥0.9 | 4.40% | 5.64% |

约 39.4% 的极小目标仍没有 IoU≥0.5 的保存候选；细碎、低填充率目标在高 IoU 下也很弱。应分别检验“能否产生有效候选”和“已有候选能否精确刻画可见掩码”。这两项统计不能单独证明是 matcher、注意力还是边界头导致错误。

## 3. 已实现、已试验和尚未完成的区别

| 方向 | 实际状态 | 下一步判断 |
|---|---|---|
| 深度几何通道 + 多层门控残差融合 D | 已实现，当前最佳模型在用 | 保留为对照 |
| 200 候选最终分数排序、1024/1280/1536 | 已完整验证 | 重排序余量小，避免继续把它作为主攻方向 |
| 额外 6 epoch | 已完成并独立复核 | 可以延长作为控制，但尚未测随机种子稳定性 |
| S：小目标匹配 + 替换部分 loss 采样 | 已实现；现有 DS/DSH 配方整体差于 D | 未做 matcher-only / loss-only 的充分独立消融 |
| H：半分辨率残差细化头 | 已实现；已有 DSH 权重接入实验推理后 APs 仅增加约 0.000045 | 不能把这一个组合结果当作所有细化方法无效的证据 |
| **D+H，关闭 S 的独立消融** | 构造代码支持，但现有配置和已完成实验未包含这一组 | 最值得先补做的现有模块实验 |
| 全图 + 重叠切片 | 之前提出，M2F 实际推理仍记录 `tiling=False` | 可做低成本覆盖诊断，但需要实例掩码融合 |
| MP-Former、局部实例细化、独立前景/边界辅助 loss | 之前提出，尚未实现 | 仍有试验价值 |
| FreqFusion、几何注意力/可变形跨模态融合、DI-MaskDINO | 本轮新增的具体论文路线 | 见下面的接入方案 |

**D+H 的评估陷阱需要先处理。** `rgbd_seg/m2f/model.py` 的普通 eval 路径仍调用原 MaskFormer；真正使用 H 的路径是 `experimental_inference.py`。因此只打开训练配置的 H，再用原 trainer 的 coarse val 选 checkpoint，不能完成 H 的效果检验。D+H 应同时报告 coarse/refined 完整 val，并明确按实际采用的 refined 推理选头部 checkpoint。S 关闭后标准 matcher 和原 point loss 应保持不变。

另一个解释限制：12544×目标面积占比只适合描述均匀匹配点的期望覆盖，不能直接当作原 mask loss 的前景点数；原 loss 还包含不确定性采样。旧 S 会替换末尾采样点，独立辅助监督应保留原采样再额外加入，以便隔离因素。

## 4. 优先方案：MP-Former 训练分支

论文：[MP-Former: Mask-Piloted Transformer for Image Segmentation，CVPR 2023](https://arxiv.org/abs/2303.07336)。[作者实现](https://github.com/IDEA-Research/MP-Former)。论文针对 Mask2Former 相邻 decoder 层预测不一致、query 利用不足，增加 GT mask 引导的训练分支；该分支在推理时移除。

本项目接入建议（待验证的迁移设计）：保留 D 的骨干、深度融合、原 matcher/loss 和普通 200 queries，仅在训练时增加 mask-piloted queries、对应 attention 隔离和辅助监督。首先验证普通推理路径完全不接收 GT，然后记录层间匹配变化、极小 GT 的有效候选比例。

作者提供了 mask noise 为 0 的 all-layer 示例，可先建立明确的低扰动基线，再单独测试 mask 噪声。注意官方脚本虽然 `NOISE_SCALE=0`，仍设置 `LB_NOISE_RATIO=0.2`；单类任务的标签噪声需要显式定义，不能把示例误称为所有噪声都为零。[官方启动脚本](https://raw.githubusercontent.com/IDEA-Research/MP-Former/main/run_50ep_no_noise_all_ly.sh)

小碎片可能在粗尺度 GT mask 下采样或扰动后完全消失，需记录每个 GT 在每个注意力尺度上的有效像素。先测标准 resize；若增加 max-pool 保留小前景，必须作为额外消融，因为它也会扩大目标区域。GT 引导分支不能泄漏到普通 query 推理。该方案训练显存增加，推理图保持原状；具体显存和收益须在本机测量。

## 5. 新增、改动相对集中的方案：FreqFusion

论文：[Frequency-aware Feature Fusion for Dense Image Prediction，TPAMI 2024](https://arxiv.org/abs/2408.12879)。[作者实现](https://github.com/Linwei-Chen/FreqFusion)。它通过自适应低通、高通和重采样处理特征融合中的内部不一致及边界偏移。

源码接入点已经核对：当前 `msdeformattn.py:349` 使用 `cur_fpn + F.interpolate(..., mode="bilinear")`。建议第一版只替换最后一次 stride 8→4 的高分辨率融合，保留其余 pixel decoder、query decoder、D 骨干和监督。用可学习残差系数从原融合平滑过渡，避免直接换层破坏已有 checkpoint。

这是一个改善共享 mask 特征的独立实验，与给每个实例另加 H 头不同，便于单因素比较。优先观察细线/低填充率切片、APs75、高 IoU 覆盖及粘连错误。我们的所有实例同属一类，语义一致性增强不必然等于实例可分性增强，必须检查相邻元器件是否被进一步平滑粘连。

官方单文件含 PyTorch CARAFE fallback，已核对其使用 `unfold`、插值和加权求和。但它只在导入失败时回退；“厂商 mmcv 可以导入但对应 GPU kernel 不可用”不能靠默认分支解决。实施时先单独检查算子前后向，按实际情况显式选后端；`unfold` 可能显著增加内存，先从小图和实际 batch 做峰值验证。[官方核心代码](https://raw.githubusercontent.com/Linwei-Chen/FreqFusion/main/FreqFusion.py)

## 6. 主算法改进候选：query 条件化的 RGB-D 局部细化

依据：[Mask Transfiner，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html) 在易错稀疏区域进行细化；[RefineMask，CVPR 2021](https://arxiv.org/abs/2104.08569) 逐级引入精细特征和边界信息。它们支持局部细化思路，但没有证明我们这些极小 RGB-D 残片必然获益；RefineMask 原文尤其强调了大目标细节问题。

建议设计是本项目适配，不宣称为论文原样复现：保留每个 query 的实例身份，对预测 mask 的上下文区域取 128×128 或 256×256 的 RGB/深度细节、粗 logits 和 query embedding，预测残差与纠错置信度；或只在不确定点/边界带上计算。保留全图分支作为上下文。

当前 H 在半分辨率整图上复制实例特征，没有显式 query embedding，也不是预测 ROI 上的逐级精细化。局部头应从当前最佳 D 开始，先冻结主干，仅训练新头，再与相同额外训练预算的 D 对照。训练用预测区域及扰动区域，不能只用理想 GT 框。可额外对 GT 前景像素中心、各连通碎片和边界采点，但保留标准 mask loss；GT 只用于训练监督。

推理触发可依据预测面积、形状和不确定性；不能依据 GT 判断“小目标”。若粗 mask 已过大，单纯按预测面积≤1024 的门限会遗漏本应细化的目标，因此需要与不确定性条件做独立比较。一个实例的多个残片保持同一身份，不能自动拆成多个实例。局部细化仍不能产生完全漏掉的 query，需与候选召回方法分别检验。

## 7. 新增的 RGB-D 融合方向

**DA-Fusion，ICRA 2025。** 针对杂乱、遮挡、未见物体的 RGB-D 实例分割，使用多层可变形自注意力及跨模态注意力；任务比普通室内语义分割更接近本项目。[论文](https://arxiv.org/abs/2607.17754) · [作者项目](https://teamrobi.github.io/projects/da-fusion/) · [作者代码](https://github.com/teamROBI/DA-Fusion)。论文发表于 2025 年，arXiv 条目在 2026 年上传，两者不矛盾。

建议先保留 R50 与现有几何编码器，只在 res2/res3 的融合处增加轻量稀疏跨模态交互，以零残差初始化接到现有门控融合上，比较“原 gate / gate+cross-attention”。这属于借鉴机制的轻量改造，并非原论文双 Swin 架构的复现。其 Overlap/Boundary F 指标不能直接与我们的 COCO APs 比较。原仓库安装脚本针对另一套 PyTorch/CUDA 环境，不能直接运行。

**DFormerv2，CVPR 2025。** 使用深度与空间距离形成几何先验来分配注意力，而非必须为深度再学习一套对等的视觉编码器。[论文](https://arxiv.org/abs/2504.04701) · [作者代码](https://github.com/VCIP-RGBD/DFormer)。其主要结果是 RGB-D 语义分割，不能视作实例分割收益证明。

可借鉴为局部窗口/ROI 内的软几何注意力偏置，先与原 gate 独立比较。当前 D 输入的六个几何通道是高度、法向、不连续度和 valid，不能把其中的高度直接当作原始相机深度；若方法依赖米制深度/XYZ，需要从原始 depth 和 K 单独提供并与图像变换一致。深度无效时回退；不能仅凭深度相近把两个实例合并，也不能硬性阻断同一物体不连续残片之间的关联。

## 8. 较大改动的替代框架：DI-MaskDINO

[DI-MaskDINO，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html) 在 MaskDINO 上通过 DI 与 BATO 改善检测/分割任务不平衡及 query 初始化。[官方实现和 R50 模型](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO) 已公开。

它值得作为“更强候选定位是否能提高极小实例召回”的替代基线，但属于较大框架迁移，不能当作给当前 decoder 加一个简单模块。第一步应复现 RGB R50 基线，再加入当前 D 深度融合，并与相同训练预算的 Mask2Former 比较。需要重新核对 mask、bbox 匹配与去噪监督，验证国产 GPU 算子兼容性。论文在 COCO 的整体 AP 改善不能换算成我们的 APs 预期。旧安装说明要求其他版本 PyTorch/CUDA，实施应复用已有 CoreX 与算子适配层。

## 9. 低成本诊断与下一轮顺序

全图+切片仍未完成。可参考 [SAHI，ICIP 2022](https://arxiv.org/abs/2202.06934)，先在固定、按场景分组的 val 子集上做原全图 + 4 个 640×640 重叠块（起点 0/384、放大到 1024），观察极小 GT 的候选覆盖是否改善。几何通道先在原图计算再同步裁剪。论文主要验证检测框；本项目还要处理重复 mask、被裁断的实例和跨块残片，不能直接使用 bbox NMS 或把碎片当独立对象。

推荐执行顺序：

1. 固定当前基线、场景分组的开发子集与完整 val；保留本次全量诊断。先补 D+H 关闭 S 的独立消融，并打通 refined val 选择，判断已有 H 头是否值得继续投入。
2. 分别筛选 D+MP-Former 和 D+FreqFusion，不同时启用。统一从当前 D checkpoint 初始化，普通 D 也用相同额外更新数、学习率日程、batch、种子做控制。可先用额外 2 epoch 作工程筛选，每 epoch 做完整 val；该预算不能代替充分收敛结论。
3. 对局部细化先做冻结主干的头部可学习性检查，再做联合微调，并补相同额外预算的 D 对照。它是最贴合 RGB-D 残片问题、也较适合作为项目主算法贡献的方向。
4. 若诊断显示主要错误是相邻物体粘连、跨遮挡实例归属，再推进 DA-Fusion 式轻量跨模态注意力；若有效候选仍明显不足，再升级 MaskDINO/DI-MaskDINO 替代基线。
5. 只有单项验证有效后再测试组合，例如 MP+FreqFusion、MP+局部细化。不要同时改分辨率、训练长度、采样规则和新模块而失去因果解释。

全程以完整 val Mask APs 选优，额外报告整体 AP、small AP75/ARs、≤256 px² 与低填充率切片、ms/图和显存。可把相对同预算控制组 APs 增加 0.005、整体 AP 下降不超过 0.003 作为初步推进门槛；它只是实验判据，不是承诺收益。微小差异按场景重采样评估不确定性，主要结论补第二个训练种子。新方案只使用 train 学习、val 选择，不借助 test 标注或线上反复试探选参。

服务器继续复用现有 PyTorch/CoreX 和 `run.sh`。本轮仅完成检索、源码核查、CPU 诊断、报告与已完成提交下载；所有新算法仍需实现和受控验证。

# RGB-D 元器件实例分割：Mask2Former 基线与小目标改进计划（2026-10-20）

替代 `docs/history/20261020_算法优化计划_maskrcnn.md`。数据分析结论沿用该文件（完整表格见其第 2–3 节；结果 JSON 在 `reports/analysis/`），本文只保留与新方案相关的部分。

---

## 1. 作业要求与本计划的对应

| 作业要求（图像分割.pdf） | 本计划的做法 |
|---|---|
| 自选实例分割基线 | **Mask2Former R50**（COCO instance 预训练），Mask R-CNN 四组结果保留为参考基线 |
| 读取并融合 RGB 与深度 | I1：几何深度编码 + 零初始化门控融合；作为对照，也测试“修正尺度后的 4 通道早期融合” |
| 至少一项小目标改进，以 APs 为主指标 | I2：小目标感知的点采样匹配与损失（无推理代价）；I3：深度引导的高分辨率 mask 特征 |
| 报告整体 Mask AP、模型大小、推理速度 | 每个实验都报告 AP/APs/APm/APl、参数量、权重大小、1024 单卡 ms/图；另给“速度版”配置 |
| 改进前后对比 | 同为 Mask2Former R50、同配方、同预算：B → +I1 → +I2 → +I3 → 完整模型，以及去掉各模块的消融 |
| 漏分割、误分割、边界不准案例分析 | 固定切片（碎片数、可见率、密度、距离）+ `error_analysis.py` 案例图 |
| Demo：RGB-D 输入，展示 GT 与改进前后预测 | `demo.py` 扩展为 Mask2Former 输出，对比基线 B 与最终模型 |

主指标是 segm/APs；整体 Mask AP 不能明显下降；速度和大小必须如实报告。

---

## 2. 决定方案的数据事实

1. **“小目标”主要是被严重遮挡后剩下的残片。** 小目标可见率代理的中位数：train 0.08、val 0.25、test 0.08；55–62% 的小目标 mask 分成 ≥2 个不连通碎片。外接框等效边长中位数约 42 px，框内填充率约 0.3。
2. **瓶颈在 mask，不在框。** Mask R-CNN A1 在同一批小目标上 bbox APs 43.0，segm APs 28.0；≥4 碎片的切片 segm APs 只有 6.0。
3. **test 比 val 难得多。** 旧模型 test APs 只有 12.5–14.1，约为 val 的一半。test 中 ≤16² px 的极小碎片占小目标的 34%（val 为 20%），相机更近、遮挡更重。val 用来调参，最终在少量候选中用 test 做选择。
4. **每图最多 96 个实例**，密集图的检测数顶到 COCOeval 的 100 上限，排序质量直接影响 APs。
5. **深度可用且干净**：系数统一、无效像素约 2%、相机内参固定，可以反投影成点云，计算离桌面高度和法向。旧的 4 通道融合中，深度数值尺度只有 RGB 的约 1/60，等于被忽略，所以“深度无用”这一结论不成立。
6. 三个划分的 CAD 型号互不重叠，属于跨型号泛化。

### 2.1 Mask2Former 在本数据上的三个具体弱点（也是改进的切入点）

- **匹配与损失的点采样对小目标几乎“看不见”。** Mask2Former 在整张图的归一化坐标上随机采样 12,544 个点，用来计算匈牙利匹配代价和 mask 损失。小目标面积中位数约 490 px²，占 1024² 图的 0.047%，平均只有 **约 6 个采样点**落在目标上。≤256 px² 的碎片平均不到 3 个点。匹配会不稳定，mask 监督也很弱。
- **mask 分辨率是 1/4。** 42 px 的框在 mask 特征上只有约 10×10，2 px 宽的细丝（7–12% 的小目标）几乎无法表达。
- **默认 100 个 query**，接近每图 96 个实例的上限，需要增加到 200。

### 2.2 可行性（已在本机验证）

- 依赖：detectron2 0.6 源码、timm、scipy、shapely、h5py、scikit-image、Cython 均已安装；**不需要安装任何包**。
- 多尺度可变形注意力：厂商构建的 `mmcv 2.1.0+corex.4.4.0` 自带编译好的算子。本机 GPU 测试显示，其前向与纯 PyTorch 参考实现的差异为 5e-7，value / attention 权重的梯度相对误差约 1e-6，采样位置梯度最大相对误差 7%（双线性插值在整数网格点处的实现差异，常见现象），速度是纯 PyTorch 的 2.4 倍。计划用一个小的兼容层把 Mask2Former 的 `MultiScaleDeformableAttention` 扩展调用转到 mmcv 算子上，**不编译 Mask2Former 自带的 CUDA 扩展**。
- 需要用户上传 Mask2Former 源码与 COCO 预训练权重（见第 8 节）。

---

## 3. 基线定义

| 编号 | 说明 |
|---|---|
| **B** | Mask2Former R50，COCO instance 预训练；RGB 输入；200 queries，9 层 decoder；输入 1024；AdamW，LR 1e-4，weight decay 0.05，主干 LR 倍率 0.1，梯度裁剪 0.01；全局 batch 8（4 卡 × 2）；训练尺度随机放大到 0.8–1.6 倍后裁剪 1024（以放大为主，避免小目标被进一步缩小）；水平/垂直翻转、90° 旋转、轻度颜色扰动（仅 RGB）；测试每图取前 100 |
| B-early | B + 4 通道早期融合，深度按训练集统计标准化到与 RGB 相同的数值量级，第 4 通道零初始化（“修正后的简单融合”，用来说明 I1 的必要性） |
| 参考 | Mask R-CNN B0/B1/M1/A1（val 与 test 指标都已有） |

B 与所有改进共用同一配方、seed、预算和后处理。正式比较全部在**完整 val** 上进行。

---

## 4. 算法改进（以 APs 为核心）

### I1 几何深度编码 + 门控融合（RGB-D 融合，作业必需）

- **输入编码**（只依赖深度图与固定 K，train/val/test 一致）：用 K 反投影成点云；逐图 RANSAC 拟合桌面平面，得到每像素**离桌面高度**；由点云计算**表面法向**（3 通道）；**深度不连续强度**（相邻像素深度差 / 深度，表示遮挡边界）；valid 通道。各通道按训练集统计标准化。
- **融合**：轻量深度编码器（4 个 stage，通道 32/64/128/256，GroupNorm），分别对齐 R50 的 res2–res5。每级用 `F_rgb + σ(g) ⊙ Conv1×1(F_depth)` 融合后送进 pixel decoder；Conv1×1 零初始化，训练起点与 B 完全一致。门控 g 由两路特征共同预测。训练时以 p=0.2 整体丢弃深度，防止过度依赖。
- **验证深度确实被使用**：记录各级两路特征范数、梯度范数和门控均值；推理时把深度置零，指标应明显下降。
- 代价估计：参数 +2–3M，推理 +5–10%。

### I2 小目标感知的点采样匹配与损失（主改进之一，推理零代价）

针对 2.1 的第一个弱点：
- **匹配**：每个 GT 除了全图均匀点外，在其外接框（外扩 20%）内再采 K=256 个点，所有 query 都在这些点上计算该 GT 的 mask 代价。这样小目标的匹配代价由足够多的点决定。
- **损失**：对每个匹配上的 (query, GT) 对，在 Mask2Former 原有的“不确定性采样”点之外，固定拿出 30% 的点采在 GT 外接框内。BCE 和 Dice 在这些点上计算。
- **面积感知权重**（可选子项）：mask 损失乘以 `clip((1024/area)^0.25, 1, 2)`，适度强化小实例。
- 只改训练，不改推理结构，因此**模型大小与速度完全不变**，这一点正好符合作业对效率的要求。

### I3 深度引导的高分辨率 mask 特征（主改进之二）

针对 2.1 的第二个弱点：
- 在 pixel decoder 输出的 1/4 mask 特征之上，新增一个轻量上采样分支，输出 **1/2 分辨率** mask 特征。分支输入是 R50 stem 特征（1/2）和 I1 的几何深度编码（下采样到 1/2），两层 3×3 卷积后与上采样的 1/4 特征相加；最终 mask = query 嵌入 · 1/2 特征。
- 深度在这里的作用：同一实例的碎片处在同一高度/平面，遮挡者更高。高分辨率的深度通道能直接给出碎片边界和细丝。
- 训练时 mask 损失仍用点采样（配合 I2），显存可控；推理时只对排序前 100 的 query 计算 1/2 分辨率 mask。
- 可选子项：把 res2（1/4）加入可变形注意力编码器（默认只有 1/8–1/32）。代价较大，只有当 I3 主体有效、且时间允许时才测试。
- 代价估计：推理 +10–20%（需实测）。

### I4 训练与推理的小目标设置（随 B 一起固定，不单独算作创新）

- queries 200（每图最多 96 个实例）。
- 尺度增强以放大为主（0.8–1.6），测试尺度在 1024 / 1280 / 1536 中选（val 扫描、test 确认）。
- 水平翻转 TTA 作为单独的“精度版”结果，计入推理时间。
- 可选：遮挡式 RGB-D Copy-Paste，把其他训练图的实例（RGB、深度、mask）粘贴到局部表面，扣除被遮挡部分，人为制造重遮挡残片。只有在主线完成且时间允许时才做。

### 可选 I5 深度 3D 位置编码

在 decoder 的 masked attention 中，把像素的 3D 坐标 (X, Y, Z) 编码成位置嵌入，替代或叠加 2D 正弦编码，帮助 query 把同一深度平面上分散的碎片聚为一个实例。只有 I1–I3 完成后才考虑。

---

## 5. 实验矩阵

筛选实验都从 COCO 预训练开始，用 **6 epoch**（全局 batch 8，10,824 iter，LR 在 80% / 93% 处衰减）。最终模型训练 12 epoch。

| 编号 | 深度 | I2 | I3 | 作用 | 优先级 |
|---|---|---|---|---|---|
| B | – | – | – | Mask2Former RGB 基线（“改进前”） | 必需 |
| D | I1 | – | – | RGB-D 融合本身的收益 | 必需 |
| S | – | ✓ | – | 小目标点采样的收益（无推理代价） | 必需 |
| DS | I1 | ✓ | – | 深度与 I2 的组合 | 必需 |
| DSH | I1 | ✓ | ✓ | 完整模型（主结果） | 必需 |
| SH | – | ✓ | ✓（去掉深度通道） | 证明 I3 中深度引导的作用 | 推荐 |
| B-early | 4 通道早期融合 | – | – | 说明简单融合不够 | 时间允许 |

报告的核心对比链：**B → D → DS → DSH**，加上 S、SH 两个消融。作业要求的“改进前后”对比就是 B 与 DSH（同主干、同配方、同预算）。

Mask R-CNN 结果作为“不同框架的参考基线”单独列出，同时报告 val 与 test。它们也支持报告中的原因分析：小 anchors 对多碎片无效，旧 4 通道融合存在尺度问题。

---

## 6. 时间表（赶时间版，约 5 天）

| 时间 | GPU | 工作 |
|---|---|---|
| 第 0 天（等待上传期间） | – | 实现：Mask2Former 数据 mapper（RGB-D、bitmask、放大式尺度增强、旋转）、MSDeformAttn 兼容层、配置；I1 深度编码与融合、I2 点采样代码；CPU 单元检查（形状、零初始化时输出与 B 一致、50 图过拟合）|
| 第 1 天上午 | 4 卡 | 上传到位后：50 iter 冒烟测试；吞吐 profiling（单卡 / 四卡、每卡 batch 1/2、AMP 开关的数值检查）；LR 1e-4 / 2e-4 各 300 iter 稳定性检查 |
| 第 1 天 | 4 卡 DDP | **B**（6 epoch）；同时在 CPU 上实现 I3 |
| 第 2 天 | 2 卡 + 2 卡 | **S** 与 **D** 并行 |
| 第 2 天夜间 | 2 卡 + 2 卡 | **DS** 与 **DSH** 并行 |
| 第 3 天 | 2 卡 + 2 卡 | **SH**；测试尺度扫描（val）；按 val 选出 2–3 个候选，在 test 上确认 |
| 第 3 天夜间–第 4 天 | 4 卡 DDP | 最终配置从 COCO 预训练训练 12 epoch；同时在 CPU / 空闲时段做速度测试、错误分析 |
| 第 5 天 | 4 卡推理 | 最终 val/test 评估；TTA 开关（val 决定、test 确认）；生成并校验提交 JSON/CSV；Demo 与报告 |

**吞吐闸门**：如果第 1 天实测四卡 6 epoch 超过 7 小时，筛选实验改为 4 epoch，或训练裁剪尺寸改为 896（测试仍在 1024 以上）。改动对所有实验统一生效。

**结果闸门**：B 跑完后，如果 val APs 明显低于 Mask R-CNN A1（27.98），继续推进改进（改进收益以 B 为参照）。最终提交从“Mask2Former 最终模型”和“Mask R-CNN 最好模型”中按 val 主选、test 确认，报告两条线的结果。

---

## 7. 评估与选择协议

- **主表**：完整 val 的 segm AP/AP50/AP75/APs/APm/APl、bbox 指标、参数量、权重大小（纯模型与完整 checkpoint 分开）、1024 单卡 ms/图与 FPS、峰值显存。最终模型额外报告 test 指标。
- **固定诊断**：每个实验都运行 `tools/analysis/pred_slices.py`，输出碎片数、可见率、密度、相机距离、类型切片与“理想打分上界”，用来解释改进作用在哪里。
- **选择规则**：
  1. 超参和后处理只在 val 上扫描；按 val APs 排序，整体 AP 不得比 B 低 0.3 以上。
  2. 在 val 前 2–4 名中按 test APs 选最终模型、checkpoint 和 TTA 设置，并在报告中注明。
  3. 差异小于 0.5 时视为持平，优先选更快、更小的配置。
- **验证频率**：每 2 epoch 一次完整 val（DDP 分片推理）；不再每 4k iter 验证一次。
- **两个交付版本**：精度版（最高 APs，可含 TTA 与更大测试尺度）；速度版（1024 输入、无 TTA；必要时 decoder 6 层）。

---

## 8. 需要用户上传的内容

上传到 `/private/rgzndzy/seg_project/`，在 `weights/source.json` 中记录来源 URL 和 sha256。

| 优先级 | 内容 | 位置 | 来源 |
|---|---|---|---|
| **必需** | Mask2Former 官方源码（main 分支 zip 即可） | 解压到 `third_party/Mask2Former/`（应包含 `mask2former/`、`configs/`、`train_net.py`） | https://github.com/facebookresearch/Mask2Former |
| **必需** | Mask2Former R50 COCO instance 权重（`maskformer2_R50_bs16_50ep`） | `weights/` | Mask2Former 的 `MODEL_ZOO.md` → COCO → Instance Segmentation → R50；一般为 `https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl`（以 MODEL_ZOO.md 为准） |
| 推荐 | Mask2Former Swin-T COCO instance 权重（`maskformer2_swin_tiny_bs16_50ep`） | `weights/` | 同上表 Swin-T 一行；一般为 `.../coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl`。仅在时间富余时做主干扩展，且需同时重跑对应的 B |

服务器无外网，以上都需要手动下载上传。Mask2Former 的 `requirements.txt` 中，除 `submitit`（只用于集群提交，不需要）外均已安装。

---

## 9. 代码改动

| 位置 | 改动 |
|---|---|
| `third_party/Mask2Former/` | 只放官方源码，不修改 |
| `rgbd_seg/m2f/msda_compat.py`（新） | 把 Mask2Former 的 `MultiScaleDeformableAttention` 扩展调用转到 mmcv CoreX 算子 |
| `rgbd_seg/m2f/mapper.py`（新） | RGB-D instance mapper：bitmask GT、放大式尺度增强 + 裁剪、翻转/旋转、颜色扰动（仅 RGB）、几何深度编码 |
| `rgbd_seg/depth_geometry.py`（新） | 平面拟合、离桌面高度、法向、深度不连续、valid（CPU 预计算或 mapper 内计算，按 profiling 决定） |
| `rgbd_seg/m2f/depth_fusion.py`（新） | I1：深度编码器 + 零初始化门控融合 backbone 包装 |
| `rgbd_seg/m2f/criterion.py`（新） | I2：框内点采样的匹配器与 SetCriterion 子类、面积感知权重 |
| `rgbd_seg/m2f/hires_mask.py`（新） | I3：1/2 分辨率深度引导 mask 特征分支 |
| `configs/mask2former/*.yaml`（新） | B / D / S / DS / DSH / SH / B-early |
| `tools/train_m2f.py`（新） | Mask2Former 训练入口（DDP、按 epoch 验证、best checkpoint） |
| `tools/infer.py`、`benchmark.py`、`demo.py` | 支持 Mask2Former 模型与 RGB-D 几何编码输入 |
| `tools/analysis/pred_slices.py` | 接受任意预测路径与 split，纳入每个实验的固定报告 |

每个新模块先做单元检查：输出形状正确；零初始化时与 B 输出一致；50 张图能过拟合；DDP 下各卡参数一致；可以断点恢复。

---

## 10. 风险与应对

| 风险 | 应对 |
|---|---|
| Mask2Former 在本硬件上吞吐太低 | 吞吐闸门（第 6 节）：减 epoch 或减训练裁剪尺寸；必要时开 AMP（先做数值检查） |
| mmcv 算子与 Mask2Former 接口不兼容 | 退回 Mask2Former 自带的纯 PyTorch 实现（速度约为 1/2.4，已验证正确） |
| B 的 APs 低于 Mask R-CNN | 改进仍以 B 为参照；最终提交在两条线中按 val + test 选择 |
| I3 显存或速度代价过大 | 只对排序前 100 的 query 计算高分辨率 mask；仍过大则只保留 I1 + I2 |
| 时间不够 | 保留顺序：B → S → D → DS → 最终训练；I3、SH、B-early、Copy-Paste、I5 可依次砍掉 |
| test 选择导致对 test 过拟合 | 每轮最多 2–4 个候选上 test；超参扫描只在 val 上做 |

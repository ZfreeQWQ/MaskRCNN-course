# 交接进度：完成、正在进行与未开始

记录日期：2026-10-26。具体打包时间、迭代和存活状态见包根 `evidence/current_status.json`；后续实时状态须在原服务器查询。

## 已完成

| 工作 | 代码/证据 | 结论 |
|---|---|---|
| 数据准备与 RGB-D 几何编码 | `data.py`、`m2f/mapper.py`、深度几何模块与形态诊断 | 元数据、train-only 统计和同步增强已建立 |
| Mask R-CNN 阶段 | `configs/maskrcnn/`、`docs/history/` | 历史参考，非当前主模型 |
| 国产 GPU 适配 | `third_party/msda_shim/`、`run.sh`、checkpoint/DDP 检查 | 复用已有厂商 MMCV/CoreX 算子；未重装运行时 |
| D 深度门控融合 | `m2f/depth_fusion.py`、`d.yaml` | 当前已完成最佳模型使用的深度分支 |
| S 小目标匹配/损失采样 | `matcher.py`、`small_object_sampling.py`、`criterion.py` | 已实现；DS/DSH 低于 D，原因尚未充分拆分 |
| H 半分辨率残差细化 | `hires_mask.py`、`criterion.py` | 已实现并接通推理，旧 DSH 权重细化收益很小 |
| 阈值与提交修正 | 阈值扫描、RLE 导出、分片/CSV 一致性审计 | 当前阈值 0、最多 100 实例；最终 CSV 已验证 |
| 第一轮推理优化 | 1024/1280/1536、legacy/mask-first、DSH 细化完整 val | 原 D 1536 mask-first APs=50.0942352 |
| D 追加 6 epoch | `d_continue6_20261026` 及独立完整 val/test 后处理 | 已完成基线 APs=52.9162213、AP=81.6031070 |
| 极小/低填充率目标诊断 | `d_continue6_coverage_20261026.json` | 极小目标缺候选、细碎 mask 高 IoU 覆盖弱；覆盖率不是 AP/AR |
| H 验证流程修正 | `round2_inference.py`、`round2_trainer.py` | 同次比较 coarse/refined，按实际 refined val 选 H 权重 |
| MP 移植与工程检查 | `mpformer.py`、`mpformer_trainer.py`、CPU/DDP/真实图审计 | 5 项 CPU 检查、四卡空/不等目标数反向、3 图推理一致性、60 步/16 图短检验通过 |

最近已完成提交来自 D 基线：2112 张图、210375 个实例，SHA256 `98f50cca289b0e767a72c95d8ca4b43427ee665e249f6dc90d300515de71949a`。包中只带完成与校验报告，不带提交本体；没有该提交的 Kaggle 分数记录。

## 正在进行

**D + MP-Former 算法训练**，不是普通 D 或 D+H 消融。

- 从最佳 D 权重初始化，S/H 关闭，不用暂停的 H 权重。
- 4 GPU、batch 8、fresh AdamW，追加 3608 步（2 epoch）；train 1024 / val 1536，seed 20261020。
- 一组 all-layer GT mask-piloted queries，mask/label noise=0；普通 queries 不能读取 pilots；推理完全关闭 pilots。
- 主网络 LR=1e-5、backbone 乘 0.1，新标签 embedding LR=1e-4；warmup 100，2886/3355 降 LR。
- 每 1804 步完整验证 2584 图，按 mask-first APs、AP 次级选优。
- 管理器：`experiments/mask2former/round2_mpformer_priority_20261026/state.json`。
- 正式输出：`experiments/mask2former/round2_mpformer_20261026/`。

当前无该 MP 实验的完整 val 结论。新增辅助 loss 改变了总损失定义，不能直接与 D 的总 loss 比大小。

## 已实现但暂缓

用户要求算法优先后，D+H 在 `model_0000901.pth` 保存后停止（902 次更新）。已确认 model、iteration、387 项 optimizer state、scheduler hook 存在；恢复不保证 dataloader RNG/顺序逐位重现。普通 D 同预算对照尚未启动，已移出自动队列。

这些不是“尚未写代码”：D+H、普通 D 配置、成对验证和启动脚本均已实现，只是完整实验未完成。不要自动恢复旧的 `run_round2_dh_control.py` 队列。

## 有计划但未开始

下表是候选，不意味着全部必做。当前运行器不会自动实现或启动它们。

| 优先级 | 计划 | 尚缺工作 |
|---|---|---|
| 下一项 | D + FreqFusion | 未实现；已确认 stride 8→4 接入点，需算子/显存检查与训练 |
| 主算法方向 | query 条件化 RGB-D 局部细化 | 未实现 ROI/不确定点局部头；当前 H 不是此方案；先冻结主干验证可学习性 |
| 低成本诊断 | 全图 + 重叠切片推理 | 未实现 mask 融合/去重；不能直接套 bbox NMS 或拆连通残片 |
| 独立监督 | 前景/边界/残片辅助 loss | 未实现；保留原点损失再加监督，不重复旧 S 替换方式 |
| 解释旧结果 | S matcher-only / loss-only | 未完成充分拆分训练；当前消融已暂缓 |
| 较低优先级 | 学习式 mask 质量评分头 | 未实现 IoU/质量预测与排序校准 |
| 条件性方案 | 稀疏跨模态交互、几何注意力 | 未实现；借鉴 DA-Fusion/DFormerv2，按粘连/遮挡诊断决定 |
| 较大迁移 | MaskDINO / DI-MaskDINO | 未移植；需兼容检查、RGB 基线和深度融合迁移 |
| 数据侧备选 | 场景重采样、RGB-D 一致 Copy-Paste/遮挡增强 | 未实现；不能仅贴 RGB 或用 test 统计决定采样 |
| 独立基线备选 | 更大 backbone、TTA | 未执行；不与算法同时改动后混合解释 |
| 组合 | MP+FreqFusion、MP+局部细化 | 未开始；单项完整 val 有效后再组合 |

## 待完成的验证

- MP 首个/最终完整 val、最佳权重独立复核；若胜出，再生成并审计 test 提交。
- 固定按场景分层的开发子集；现有 first-N smoke 只用于工程检查。
- 统一整理 small AP75/ARs、≤256 px²、低填充率、速度与显存；覆盖诊断不替代 COCO 指标。
- 主要方案充分训练、第二 seed、按 scene 重采样的置信区间。
- 暂缓同预算控制后，相对基线的变化包含额外训练影响，不能单独归因算法；需要严谨结论时再补对照。

多尺度单尺度比较、mask-first、D 追加 6 epoch 和 MP 移植已经执行，不能再列为未实现。

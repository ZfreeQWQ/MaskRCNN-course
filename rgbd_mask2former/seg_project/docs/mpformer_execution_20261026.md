# 算法优先：D + MP-Former

2026-10-26 用户明确要求先暂缓消融，优先执行算法改进。已取消普通 D 控制组的自动启动，并将 D+H 停在已写完的 `model_0000901.pth`（902 次更新），保留其输出目录和可恢复 checkpoint。

## 方法与适配范围

采用 [MP-Former: Mask-Piloted Transformer for Image Segmentation，CVPR 2023](https://arxiv.org/abs/2303.07336) 的 mask-piloted 训练机制，[官方实现](https://github.com/IDEA-Research/MP-Former) 固定于 commit `a71f504a467d7079cf02726a05ff23bbb933607d`。官方参考文件和 SHA256 在本地 `references/mpformer_20261026/`，不执行其安装脚本。

本次实现是面向现有 D checkpoint 的适配，并非论文整套实验配方的原样复现：

- 保留当前 RGB-D 门控骨干、pixel decoder、200 个普通 query 及其已学习的位置编码。官方 DN decoder 不使用 query 位置编码，本版保留它以继续利用现有权重。
- 训练增加一组按 GT 顺序对应的 pilot queries，标签 embedding 为新增参数。
- 所有 cross-attention 层使用 area pooling 后 GT 支持区域引导 pilot attention，辅助输出接受固定一一对应监督；普通 queries 保留原 Hungarian matcher 和原点损失。
- 普通 queries 在每层 self-attention 中不能读取 pilot queries。与官方单组方式相同，pilots 可读取普通 queries。
- 首版 mask noise 为 0、label noise 显式设为 0；单类任务中官方“替换为随机类别”的标签噪声本来也只能得到类别 0。
- 验证直接调用原 decoder 的 eval forward，不生成 pilot queries、不读取 GT，继续使用 mask-first 最多 100 实例推理。
- S 与 H 均关闭；本批没有叠加 FreqFusion 或其他结构变化。

## 当前训练配方

初始权重为冻结的 `d_continue6_results_20261026/best_segm_APs.pth`，完整 val APs=52.9162213、AP=81.6031070（0–100 尺度）。不使用被暂停的 D+H 权重初始化。

4 GPU、全局 batch 8、训练 1024、验证 1536；追加 3608 次更新（2 epoch），fresh AdamW、主网络初始 LR=1e-5、backbone 乘 0.1、新标签 embedding LR=1e-4。warmup=100，2886/3355 降 LR，seed=20261020；每 1804 次更新做完整 2584 张 val，按 mask-first segm/APs 选优。

这是算法工程筛选预算，尚不能代表充分收敛。暂缓同预算控制组后，相对既有基线的分数变化包含额外训练的影响，不能单独归因于 MP 机制。按用户优先级先运行算法，需要时再补消融。

## 检查与已发现的问题

5 项 CPU 测试已通过：eval 与原 decoder 一致且忽略 GT、改变 GT 数量/位置不影响普通 query 输出、新 embedding 有有限且非零梯度、全空批次的 loss/反向图有效、单像素 GT 经 area pooling 保留支持。

四卡专项检查首次触发原 matcher 在空目标轴上的 CoreX TorchScript 错误 `Global alloc not supported yet`。已在 MP 专属 matcher 中跳过空目标配对，并为空 mask loss 返回带梯度的零；非空目标计算保持原方法。保留失败日志，修复后四卡三次反向检查通过，包含局部空目标、各卡目标数不同及全局空目标；未重装或升级底层依赖。

真实图像初始权重审计检查所有共享模型张量完全一致，新增参数只有 `sem_seg_head.predictor.label_enc.weight`；3 张真实 val 图的原图掩码逐像素一致、分数一致，推理 pilot 数量为 0。

正式启动前另跑四卡 60 步训练与 16 图验证，管理器核对 pilot 分支启用、标签 embedding 有有限非零梯度、验证输出没有 H 细化、子集不保存最佳模型。该短检验仅验证工程正确性。

## 持久化与状态

- 正式训练：`/private/rgzndzy/seg_project/experiments/mask2former/round2_mpformer_20261026/`。
- 自动验证和启动管理器：`round2_mpformer_priority_20261026/state.json`；任一检查失败即停止。
- 四卡边界及初始推理审计：`round2_mpformer_checks_20261026/`。
- 短检验：`round2_mpformer_smoke_20261026/`。
- 每 20 步记录各卡 pilot 数、GT 缩放丢失数、embedding 范数及梯度范数至 `mp_diagnostics.jsonl`。

本地检查：`python remote_exec.py --script round2_status.sh`；仅看当前算法：`python remote_exec.py "python /private/rgzndzy/seg_project/tools/mpformer_status.py"`。

同步报告并核对源码：`python sync_mpformer_results.py`。

2026-10-26 21:02 更新：本次 MP 两个 epoch 已完成，APs=51.288→52.112、AP=81.304；尚未超过 D。根据用户对短预算的担忧，[自动队列](AUTOMATIC_TRAINING_PLAN_20261026.md)已改为 MP 保留优化器续到累计 6 epoch，再跑频率融合 6 epoch、条件组合 6 epoch，最佳算法续到累计 12 epoch。当前已进入四卡完整状态恢复短测。6/12 epoch 仍是实验预算，不保证效果；不自动重启普通 D 对照。

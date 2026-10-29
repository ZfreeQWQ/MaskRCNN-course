# 算法改进自动训练队列

更新日期：2026-10-26 晚，训练预算修订版。根据用户对训练不足的担忧，已将早先的 2-epoch 筛选、fresh AdamW 再追加 4 epoch，改为单项 6 epoch、保留优化器续训至 12 epoch。算法改进优先，消融仍暂缓。

## 执行顺序

所有正式任务使用 4 张 Iluvatar BI-V150、全局 batch 8、train 1024、完整 val 1536、seed 20261020。train 14432 图，每 epoch 1804 步；每 epoch 完整验证 2584 图，以 mask-first segm/APs 为主、AP 为同分判据。新架构单项实验以模型权重初始化、fresh AdamW；同架构追加使用父阶段 final 完整 checkpoint，保留优化器动量及迭代计数，原始输出保留。

| 顺序 | 任务 | 训练预算 | 自动启动条件 |
|---|---|---|---|
| 0 | 目前 D+MP-Former | 当前 3608 步完成至 2 epoch | 已在运行；等待原控制器和训练进程退出 |
| 1 | MP 保留优化器续训 | 累计 10824 步，6 epoch；本轮后再跑 4 epoch | 当前完整训练与验证完成；四卡恢复短测通过 |
| 2 | D + 频率感知 FPN 融合 | 10824 步，统一 6 epoch | 从冻结 D 开始；初始化审计与四卡短测通过 |
| 3 | D + MP + 频率融合 | 10824 步，6 epoch | 两项单独均完成 6-epoch 预算；APs 均至少超过 D 0.2 点，AP 下降不超过 0.3 点 |
| 4 | 最佳有效算法候选续训 | 累计 21648 步，12 epoch，通常再跑 6 epoch | 按完整 val APs/AP 选择分支，从 final 完整状态续训 |

主网络基础 LR=1e-5，算法模块 LR×10、backbone LR×0.1。新单项采用统一 6-epoch 日程：第 8659/10066 步衰减，即约 4.8/5.58 epoch，替代原先约 1.6/1.86 epoch 就衰减的短日程。warmup 100 步，每 902 步保存 checkpoint。队列生成的 `configs/frequency.yaml` 覆盖静态模板继承的短预算，以实际任务配置为准。

续训保留 AdamW 状态、global iteration 和 scheduler 计数，扩展新的学习率时间范围，并使用 100 步过渡 warmup；12-epoch 阶段衰减点为 17318/20132。已经发生的短日程不能撤销，所以 MP 续到 6 epoch 不等价于从一开始设定 6 epoch；延长到 12 epoch 也不宣称与预先设定 12 epoch、完整恢复随机数及数据顺序逐位相同。

新组合从长 MP 的最佳权重初始化，不硬拼频率单项权重。同架构续训从 final 继续优化，同时复制、保留父阶段 best 指标与权重。长 MP 有效完成后，追加选择排除短 MP 父分支，避免继承的 best 分数同分时又退回 2 epoch 重训。各阶段计数从本算法分支初始化起计算，不包含此前 D 的训练历史。

每次正式训练前另跑 60 步、16 图验证短测；续训短测加载完整优化器并逐卡核对 AdamW moments、step 和 scheduler iteration。短测更新不用于正式初始化。

若两项长预算单独未同时达标，组合自动跳过，直接延长最佳算法候选，不以 2-epoch 初筛直接判定组合。如果候选仍低于冻结 D，追加属于延长筛选，不能宣称已改进。最终比较冻结 D、各阶段最佳 checkpoint 和续训；只有 APs/AP 选优胜过 D 且 AP 下降不超过 0.3 点才替换推荐。

正常完成时，当前 MP 后约 16 个额外 epoch，组合触发时约 22 个，另加短测；失败和恢复路径会改变实际预算。这是有限队列。6/12 epoch 是更充分的实验预算，不保证收敛或提升；以每 epoch 完整 val 曲线判断，保留最佳 checkpoint。实际耗时以四卡短测为准。验证与写盘也占流水线时间，不能保证每张卡始终满载。

## 频率融合实际实现范围

依据 [Frequency-aware Feature Fusion for Dense Image Prediction](https://arxiv.org/abs/2408.12879) 和 [作者实现](https://github.com/Linwei-Chen/FreqFusion)。本项目是**借鉴 FreqFusion 的轻量适配**，不称为完整论文复现：

- 只替换 pixel decoder 最后 stride 8→4 的融合，保留已有 D、query decoder、普通监督。
- HR/LR 压缩特征联合生成空间变化的 5×5 低通与 3×3 高通权重，softmax + Hamming 窗归一化。
- LR 用自适应邻域重组上采样，HR 使用高频残差；可学习 `tanh(alpha)` 系数以零初始化接入已有融合。
- 使用显式 PyTorch `unfold`、分通道计算和反向重算；不依赖新增 CARAFE 二进制算子，不安装底层依赖。边缘采用 replicate padding。
- 没有官方完整模块的 offset sampler、递归压缩特征细化。当前实现的效果与速度必须由后续实验验证。

主要代码：`rgbd_seg/m2f/frequency_fusion.py`、`algorithm_trainer.py`、`tools/train_algorithm.py`，独立配置 `configs/mask2former/round3_frequency_20261026.yaml`。完整状态续训新增 `algorithm_continuation.py`、`tools/prepare_algorithm_resume.py`、`tools/train_algorithm_resume.py`。原 MP 正在使用的源码和配置未改动。

## 已完成检查与待执行检查

CPU 已通过 13 项测试，覆盖融合及梯度、失败隔离、6-epoch 学习率预算、长分支选择、AdamW 动量恢复后下一步更新等价、拒绝损坏/空状态、完整 checkpoint staging，另通过配置注册检查。真实 MP final 的 377 份优化器状态、3608 个完成更新、scheduler 计数及扩展日程 LR 分组也已核对。

为避免占用当前 MP 的 GPU，真实图初始化等价审计、四卡前后向、实际显存和 16 图验证在当前训练结束后由队列自动运行，**目前尚不能称为 GPU 检查通过**。初始化审计检查所有共享权重、3 图 masks/scores 等价及推理无 pilot；短测检查四卡有限且非零梯度、新融合残差开始学习、完整短测图数和不保存子集最佳权重。

21:04 补充：MP 的四卡完整状态恢复 60 步及 16 图短测已通过，4 个 rank 均从 3608 开始，逐卡核对了 377 份优化器状态及抽样 moments；队列已进入正式续训准备。上段待检查状态仍适用于尚未运行的频率融合及新组合。

## 后台运行与状态

全部状态位于持久化目录：

```text
/private/rgzndzy/seg_project/experiments/mask2former/
  algorithm_queue_20261026/
    state.json                  队列计划、进度、决策、候选、来源哈希
    manager.log                 调度器日志
    manager.lock                防止重复调度器
    cpu_tests.json              CPU 检查和对应源码 SHA256
    preflight.json              启动前审计
    configs/                    每项任务的实际配置
    jobs/<任务>/
      command.json              实际 argv、时限、进度文件
      worker.json               子任务身份、心跳
      training.log              训练或检查日志
      result.json               退出码、状态、结束时间
  round3_mp_long_20261026/
  round3_frequency_20261026/
  round3_mp_frequency_20261026/           条件触发
  round3_best_algorithm_extend_20261026/
```

查看：

```bash
cd /private/rgzndzy
./seg_project/run.sh python seg_project/tools/run_algorithm_queue.py --status
python seg_project/tools/mpformer_status.py
./seg_project/run.sh ixsmi
```

本地已启动后台队列的命令为 `python remote_exec.py --script launch_algorithm_queue.sh`；查看使用 `python remote_exec.py --script algorithm_queue_status.sh`。队列已采用脱离 SSH 会话的进程启动方式，关闭 SSH 不影响运行。断电或服务器重启不会自动重建进程，需要重新运行启动脚本；不写系统服务或系统配置。

队列重启时复用已保存状态，等待仍存活的任务，不重新覆盖已有日志或从头重复已完成训练。若原 worker 消失且缺少退出凭据，该项标为失败，不盲目重复。正式训练异常后需要人工诊断；现有入口可用 `--resume` 从保存的训练状态恢复，但队列不自动重复失败训练。

## 衔接与异常策略

- 当前 MP 不被中断；每 15 秒检查前序完成。启动下一项前检查已有 `train_*.py` / torchrun 进程，两次确认退出后接续。发现其他手动训练则等待，不终止它。
- 每项任务有独立 runner，调度器意外退出时 runner 继续并写退出记录；控制器持有互斥锁，重复启动会被拒绝。
- 逐阶段核对冻结源码 SHA256、CPU 测试对应源码及持久盘空闲。队列运行期间不要修改受保护训练源码/配置；文档和状态读取可继续。
- 初始化审计或短测失败会跳过该算法正式训练；独立方案和已有有效候选的追加可继续。失败方案不会成为组合或追加的初始化来源。
- worker 每 15 秒记录心跳。90 分钟无进度或超过时限会终止自己的进程组并记录失败。短测时限 1 小时、初始化审计/准备 30 分钟、正式训练每项 24 小时。
- 全部完整验证通过后才接纳候选。失败记录、检查点和日志全部保留。GPU/依赖错误按原因报告，不通过改驱动或安装环境修复。

当前尚未排队：query 条件局部 RGB-D 细化、切片掩码融合、跨模态注意力和 MaskDINO 迁移，均需要后续实现和验证。消融及旧 D+H/普通 D 对照不自动恢复。

19:42 的交接 ZIP 是历史快照，不含本次新增队列与频率融合代码；本文件及服务器当前源码描述后续变更。

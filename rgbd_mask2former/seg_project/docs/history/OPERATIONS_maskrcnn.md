# 运行与续跑

工作目录：`/private/rgzndzy`。所有数据、日志、权重和输出均保存在持久目录；不安装或修改全局依赖。

## 当前自动实验流程

`tools/run_experiments.py` 已启动后台任务，PID 保存在 `outputs/pipeline.pid`，状态为 `outputs/pipeline_status.json`。

1. 四张卡独立运行 B0（RGB）、B1（RGB-D）、M1（RGB-D 小目标改进）、A1（M1 去深度），每组 pilot 3,000 次单图更新；每 1,000 步在全部 2,584 张 val 上评估。pilot 只占约 0.208 个训练 epoch，不能据此宣称正式性能。
2. 四组从同一 COCO 预训练源分别重新开始 1x 正式训练，每组 144,320 次单图更新，即 10 个训练 epoch；每 4,000 步完整验证。保存验证 `segm/APs` 最佳 checkpoint，AP 并列时按整体 Mask AP 选取。
3. 按完整验证集 APS 在 B1/M1 中选择 RGB-D 结构，四卡 DDP 从 COCO 预训练开始训练 2x：72,160 次更新、全局 batch=4，总计 20 个 epoch。B0/A1 始终保留为公平消融对照。B1 等同于 M1 去掉小目标改进（A2），无需重复训练同一配置。
4. 各实验自动执行 1024×1024、batch=1、20 次预热加 200 张计时的标准速度测试；生成真实指标报告。比较 1x RGB-D 与最终 2x DDP 模型，按验证 APS 选择提交权重。
5. 生成验证集错误分析候选与 RGB/深度/GT/B0/最终预测对比图。需要后续根据真实案例完成原因分析，自动候选不等于已证明的错误成因。
6. 当前缺少官方测试集逐图 `depth_scale_m` 和 `depth_invalid_code`。流程在完成训练分析后将记录 `training_complete_test_blocked`，不会猜标定值或读取测试标注。补齐合法测试输入元数据后可续跑最终 JSON/CSV 导出与校验。
   （2026-10-20 更新：用户决定允许用 test 标注辅助模型选择，test 深度系数改从 test 分片 `images` 字段读取，test 仍禁止用于训练。旧调度器本身的逻辑没有改，新实验按 `reports/算法优化计划_20261020.md` 执行。）

正式训练和反复完整验证可能持续数十小时，取决于实测吞吐与验证开销。训练日志 ETA 主要估算剩余训练更新，不包含后续全部完整验证、速度测试和实验阶段，不能当作整条流水线结束时间。M1 保留原本已有的 P2，不宣称新引入 P2；实际改动是更小 anchors、更多 proposals 和 28×28 mask ROI。

## 查看状态

```bash
cd /private/rgzndzy
cat seg_project/outputs/pipeline_status.json
python seg_project/tools/monitor.py --record
tail -n 30 seg_project/outputs/pilot_m1/console.log
./seg_project/run.sh ixsmi
```

每个训练目录包含 `config.yaml`、`metrics.json`（Detectron2 JSONL 训练日志）、checkpoint、`best_metrics.json`、`best_val_predictions.json` 和 `snapshots/`。`report_artifacts/metrics.json` 是可直接汇总的 JSON 指标，不能与训练 JSONL 混淆。

若失败，状态记录具体子进程、日志路径与退出码；不要通过重装底层环境解决。修复具体原因后重新运行下列命令，已有完成标记的实验跳过，未完成实验从 `last_checkpoint` 恢复模型、优化器、scheduler 和 iteration。加载器 worker RNG/样本顺序不保证与不中断训练逐位相同。

```bash
./seg_project/run.sh python seg_project/tools/run_experiments.py
```

`pipeline.lock` 防止重复启动。运行中不要修改配置、数据统计或源码；新实验使用独立输出目录。若要停止整个后台实验组，先确认 `pipeline.pid` 对应本项目进程，再对该进程组发送 SIGTERM；保留已有 checkpoint 后使用相同参数续跑。

## 环境兼容处理

- Python 3.12.3；原有 PyTorch/torchvision 为 CoreX 4.4.0 构建，未安装或升级。
- `run.sh` 仅对它启动的进程优先使用 `/usr/local/iluvatar/lib64` 中现有服务器驱动库，并使用该目录配套 `ixsmi`。原先 SDK 库优先会导致张量分配 `operation not supported`，SDK 的 `ixsmi` 还存在符号不匹配；未改系统配置。
- 本地 `detectron2/detectron2/layers/wrappers.py` 释放 CoreX Conv2d 的空 `activation` 参数槽，再设置 Detectron2 激活函数，避免名称冲突。补丁随每次训练源码快照保存。
- 常规 Mask R-CNN 使用现有 torchvision ROIAlign/NMS 和官方 Python COCO evaluator，无需重编译 Detectron2 扩展。
- 证据：`runs/env_verified.json`、`runs/ddp_smoke.json`、`runs/mask_rcnn_operator_smoke.log`、各 smoke 训练日志。smoke 指标仅用于验证链路。

## 产物入口

- 数据检查：`reports/data_report.md`
- 四组 pilot 完成后：`reports/report_pilot.md`
- 四组正式 1x 完成后：`reports/report_1x.md`
- 最终 DDP 完成后：`reports/report.md`、`outputs/final_selection.json`
- 验证错误候选：`reports/final_error_analysis/analysis.md`
- 合法测试标定补齐并推理完成后：`submissions/masks.json`、`submissions/boxes.csv`、`submissions/validation.json`

当前 `reports/report.md` 若仍显示待完成项，代表模板而非完整实验结论。状态文件仅在阶段切换时更新；用 `monitor.py --record` 核实 `/proc` 中实际进程、CPU累计时间增量和日志推进，再结合真实产物判断进度。COCO汇总使用CPU，此时GPU利用率可为0%，不代表训练进程已经停止。监测记录保存在 `reports/monitoring/latest.json` 和 `history.jsonl`。

## 补充逐项消融与同预算 DDP 对照

`tools/run_supplemental.py` 使用独立的 `supplemental.lock`、`supplemental.pid`、`supplemental_status.json` 和 `supplemental.log`。它等待主流程持有的 `pipeline.lock`，默认每 60 秒写入等待心跳；等待期间不启动 GPU 子进程。获得共享锁后，必须验证主流程成功结束、四组正式 1x 的完成标记与真实产物、最终选择和四卡正式 DDP 产物；主流程失败、崩溃或产物不完整都会使补充流程退出。

先检查完整条件计划（仅读配置和产物，不启动模型、子进程或写入调度状态）：

```bash
./seg_project/run.sh python seg_project/tools/run_supplemental.py --dry-run
```

后台启动命令如下；已有主流程运行时，补充流程将自动等待。不要重复启动。

```bash
nohup ./seg_project/run.sh python -u seg_project/tools/run_supplemental.py > seg_project/outputs/supplemental_launcher.log 2>&1 &
```

补充流程全程持有同一 `pipeline.lock`。先在 GPU 0/1/2 分别运行去掉小 anchors、额外 proposals、高分辨率 mask ROI 的三个 M1 组件消融，GPU 3 空闲。三组训练及标准速度测试全部完成后，再在四张卡运行最终所选 B1/M1 架构的独立 1x 对照。单卡每组 batch=1、LR=0.0025、144,320 次更新；四卡 batch=4、LR=0.01、36,080 次更新。图次预算均为 144,320，优化更新次数和轨迹仍不同，不能把性能差异单独归因于通信。

输出目录为 `outputs/1x_ablate_no_*` 和 `outputs/supplemental_ddp_1x_b1` 或 `_m1`。续跑使用同一后台命令，存在 `last_checkpoint` 时恢复训练；跳过已完成训练前仍校验权重、完整验证结果、末步日志及保存的配置契约。配置契约变化会拒绝复用目录。补充报告写入 `reports/report_supplemental.md`、`reports/report_supplemental_findings.md` 及同名 JSON；主流程的选择与状态仍由原流程维护。测试集元数据不参与本流程。

查看 `outputs/supplemental.pid` 对应进程和状态、日志可确认等待或执行阶段。停止时向该 PID 发送 SIGTERM；调度器会结束并等待自己启动的训练/DDP进程组，然后释放共享锁。保留 checkpoint 后可续跑。CPU 验证证据：`runs/supplemental_cpu_checks.json`；其中模拟分数和权重只存在已删除的临时测试夹具中，不属于实验结果。真实案例逐一复核和错误原因撰写仍需在实验结果产生后完成。

## 正式结果后的最终选择审核

正式结果产生前已固定 [最终选择政策](reports/final_selection_policy.md)：RGB-D 候选整体 AP 至少达到正式 1x B1，然后按 APs/AP 排序；同时报告相对 B0 的变化与速度、显存代价。原主调度器已经运行，其内存中的选择逻辑仍是 APs/AP 字典序。因此原 `final_selection.json` 是待审核的调度结果，不能自动视为满足整体 AP 判据。

四组 1x 及实际 final_ddp 完成后，按政策文档命令运行 `tools/audit_final_selection.py`，将 `--final` 改为真实完成的 B1 或 M1 目录。该工具只写独立报告；当前 `reports/final_selection_audit.json` 的 incomplete 表示正式证据尚未齐全。若审核建议与原选择不同，须在所有调度工作结束后另行核对并应用，随后重建最终报告、正式 Demo 和带独立 M1 面板的错误分析；不要在活跃调度期间覆盖其选择文件。正式提交/推理前必须通过此项审核。9 项 CPU 合成合同检查与真实配置解析交叉检查已通过，均不代表正式模型结果。


## User hold after the four current 1x runs

User requested finishing B0/B1/A1/M1 1x and stopping subsequent training pending a direction change. All four runs and final evaluations are now complete. `tools/finish_current_1x_hold.py` verified artifacts, terminated the old scheduler after its child exited successfully, and completed benchmarks/report only with `--stages 1x --final-stage none`. Finalization ended at 2026-10-17T10:19:31 UTC with status `paused_by_user_after_1x`; no final DDP or supplemental training launched. Supplemental PID 8517 was cancelled while waiting. The original scheduler and training processes have exited; stored PIDs are historical identifiers, not evidence of live jobs. `outputs/user_training_hold.json` still blocks new training in both schedulers. Do not remove the hold without a new user decision. Latest execution summary: [reports/execution_summary_1x.md](reports/execution_summary_1x.md). Finalizer status: `outputs/finish_current_1x_status.json`; log: `outputs/finish_current_1x.log`.

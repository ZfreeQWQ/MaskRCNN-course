# 最终选择政策 v1（正式结果前固定）

本政策于 2026-10-16 固定，落实执行计划 §8 的 segm/APs 优先与整体 AP 无不可接受下降要求。只读固定 blender_val 的正式证据，不读取 test 标签；pilot/smoke 只能验证工具，不参与正式选型。此文档与独立审核工具不改变正在运行的训练、两个调度器或 `outputs/final_selection.json`。

1. 必须先有四组 1x：B0（RGB）、B1（RGB-D）、M1（RGB-D 小目标）、A1（RGB 小目标）。训练预算为 14432 张 × 10 epochs = 144320 次图像曝光，必须核对 resolved config 的 batch × MAX_ITER、完成标记、最后验证和训练日志。B0/B1、A1/M1 是等预算配对；A1/B0 用于解释 RGB 结构收益。
2. 正式候选限于完成且证据合格的 1x B1、1x M1 及显式指定的 final_ddp_1x/2x_B1/M1。final_ddp 必须存在并完成；不能因它暂未生成而宣告最终选择完成。原始配置与实际 batch/迭代预算必须记录，final_ddp 的收益不能归因于结构本身。
3. 使用候选各自已有的 APs 最佳 checkpoint，不重新搜索 checkpoint。先要求候选完整验证 `segm/AP >= 1x B1 的 segm/AP`，不设任意容忍幅度，不把四舍五入后的数值用于判定。这是对“不可接受下降”的保守操作定义，并不代表统计显著性。若 APs 最佳 checkpoint 不合格，不回退到未预先指定的其他 checkpoint。
4. 在满足 AP 门槛的候选中，按 `(segm/APs, segm/AP)` 降序。完全同分时保留全部并列合格者，建议优先 1x B1、1x M1、再按 final 目录名排序，作为确定性的行政选择，不宣称模型性能差异。B1 本身构成安全参照；若只有它合格，保留 B1 并如实报告其他方案未通过门槛。
5. 同时列出所有正式组的 AP/APs、相对 1x B1 与 B0 的百分点变化，及延迟/FPS、参数量、模型张量大小、完整 checkpoint 大小、峰值显存和相对成本。必须核验 benchmark 原始样本与均值/FPS一致、至少 20 次 warmup 和 200 张不同来源的计时样本、batch=1、1024²、固定 seed=20261016、NMS=0.5、相同推理阈值和设备/软件、无 TTA/裁剪。RGB 对照的 read/preprocess 应标为 RGB，RGB-D 应标为 RGB+depth；共同模型计时边界必须包含传输/forward/D2 后处理、排除 RLE，不能把预期读入模态差异误判为测量不一致。没有事先规定速度/大小硬门槛，故只报告代价，不凭结果临时创造阈值或强称全面改善。共享机器负载会影响速度比较；测量不构成统计显著性证据。
6. 完整验证要联合核验 EVAL_LIMIT=0、SKIP_FINAL_EVAL=false、blender_val、maxDets=100、2584 张的评估摘要、best_metrics 与报告/日志一致、最终迭代完成证据及保存预测只包含 val image_id。预测没有检测的图像可以不产生预测行，因此不可用预测中不同 image_id 个数冒充覆盖率。现有训练路径未保存逐图覆盖 manifest，审核将覆盖性依据明确限制为 resolved config + 全量评估摘要 + 训练记录，不能把这些当作独立重算 COCO AP；本工具不运行推理或重算评估。
7. 原调度选择应只读记录：候选集合与分数是否完整一致、所选路径是否达到 AP 门槛、是否达到政策最优、是否使用 test 标签。未生成原选择时记录 pending。即使审核完成也只产出建议，应用政策由主 agent 单独审查，不覆盖原选择。

CPU 审核：

```bash
python seg_project/tools/audit_final_selection.py \
  --b0 seg_project/outputs/1x_b0 --b1 seg_project/outputs/1x_b1 \
  --m1 seg_project/outputs/1x_m1 --a1 seg_project/outputs/1x_a1 \
  --final seg_project/outputs/final_ddp_2x_m1 \
  --val-metadata seg_project/datasets/blender/annotations/instances_val.json \
  --scheduler-selection seg_project/outputs/final_selection.json \
  --output seg_project/reports/final_selection_audit.json
```

所有读取均须显式 CLI 指定；缺失或无效证据返回 `status=incomplete`、退出码 2，并列出原因。输出可省略（只打印 JSON）；工具拒绝写入 outputs 目录。`status=complete` 表示证据通过此静态审核，不代表作业报告、提交或部署完成。合成 CPU 检查可用 `python seg_project/tools/test_final_selection_audit.py`，临时文件位于持久化 /private 下并自动清理。

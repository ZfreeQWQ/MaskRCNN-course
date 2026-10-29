# 追加训练完成后的自动复核与提交导出

`tools/finalize_continue_results.py` 已接到现有 `d_continue6_20261026` 训练之后运行。它只等待和读取该训练的状态，不启动、重启或停止训练，不修改训练代码或配置，也不访问 test 标注。

输出：`/private/rgzndzy/seg_project/experiments/mask2former/d_continue6_results_20261026`。`workflow_state.json` 记录阶段和 PID，`workflow.log` 记录日志，`workflow.exit_code` 记录最终退出码。以成功的 `test_complete.json` 和 `submission.csv.validation.json` 为提交完成依据，不能仅凭进程存在或退出判定结果有效。

## 执行顺序

1. 等待训练 manager 成功退出、发布 `model_final.pth`，并确认训练 launcher 已退出。训练失败或 manager 异常消失时停止，保留错误，不尝试重装依赖或重新训练。
2. 冻结 `best_segm_APs.pth`、实际完整 config 与 best metrics；保存权重、源码、val 标注和原基线报告 SHA256。选的是训练器按完整 val APs 保存的最佳 checkpoint，未必是最后一次更新。
3. 等全部 GPU 空闲后，用现有 `infer_round1.py` 在 1536 尺度对 2,584 张完整 val 同时输出 legacy 与 mask-first。阈值 0，每图最多 100 个实例；原图坐标、几何通道与之前第一轮一致。
4. 使用同一 COCO segm 评估入口。新 legacy 与训练器最佳指标差异超过 0.001 时阻断自动选择，等待调查。
5. 比较追加训练的两个变体与第一轮优胜方案，按 val APs 优先、整体 AP 次之选择；完全相同时保留原方案。若原方案仍最好，复用其已校验提交；若新方案更好，重新对 2,112 张 test 图像/深度元数据推理并导出新 CSV。
6. 全量解码 RLE、核对 CSV 与源预测、图像覆盖和实例数；通过后才发布 `test_complete.json`。源文件或冻结权重发生变化时停止。

使用文件锁防止重复 workflow。分片有未完成的旧启动记录时拒绝静默重启或覆盖，须先检查进程与日志。所有原始训练、第一轮权重和提交保留。测试集没有参与模型或后处理选择，Kaggle 线上分数仍需提交后才能知道。

9 项测试覆盖训练未完成/失败/manager 消失、最终 checkpoint 与 launcher 检查、基线回退和指标相同时的选择、拒绝 smoke 与错误单位、legacy 对照漂移阻断。另包含隔离文件夹中的完整编排检查：无增益时保留基线且不重新推理 test、有增益时仅使用 test 元数据导出、导出校验失败时不发布完成标记；这些编排检查替换了外部 GPU 与 COCO 评估边界，真实全量复核仍由后台流程在训练完成后执行。测试在原有服务器 Python 环境运行，无额外依赖安装。

## 查看和同步

从服务器项目根目录执行：

```sh
./seg_project/run.sh python seg_project/tools/finalize_continue_results.py --inspect
cat seg_project/experiments/mask2former/d_continue6_results_20261026/workflow_state.json
```

已存在后台 workflow 时不要再次启动。首次启动由本地 `launch_continue_results.sh` 完成，已有 PID 文件时拒绝再次启动。

结果完成后，在本地 `C:\workspace\ssh` 执行：

```powershell
python sync_round1_results.py --experiment d_continue6_results_20261026
```

训练仍在运行时只同步状态与报告可加 `--reports-only`。最终文件下载到 `outputs/d_continue6_results_20261026`，CSV 下载后核对服务器 `test_complete.json` 中的 SHA256；预测大 JSON 和权重仍保存在服务器 `/private`。

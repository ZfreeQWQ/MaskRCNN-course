# 第一轮优化实验执行记录

实验目标：利用现有权重验证候选排序、输入分辨率和 DSH 细化头，最终依据完整 val 的 Mask APs 选择 test 提交方案。当前执行中的结果以远程 `val_results.json` 为准，不能把 smoke 指标作为完整验证结果。

实验目录：`/private/rgzndzy/seg_project/experiments/mask2former/optimization_round1_20261026`。

## 控制变量

- D 权重不变，输入尺寸分别为 1024、1280、1536；输出 mask 始终为原图 1024×1024。
- 每次前向同时输出 `legacy`（先按分类分数截取 100）及 `mask_first`（200 个候选计算最终分数后再截取 100），用于配对比较。
- DSH 使用原有 `best_segm_APs.pth`；另输出 `refined_small`。仅细化预测面积在 1–1024 px² 的已选候选，使用训练好的高分辨率 head；不读取 GT 来决定是否细化。
- 阈值统一为 0，最多 100 个实例；空 mask 不导出，空预测图片仍由 manifest 覆盖，CSV 填 `[]`。
- RGB-D 几何通道仍在原图上计算，再与 RGB 同步变换；不改变已有相机参数和训练统计量。
- COCOeval 使用 segm，loadRes 前删除 bbox，使 small 的面积由 mask 定义。完整 val 为 2,584 张；test 为 2,112 张且不使用标注。

## 正确性与环境检查

沿用 Python 3.12.3、PyTorch 2.7.1 + CoreX 4.4.0、4 张 Iluvatar BI-V150（每张 32GB），未安装或修改底层依赖。

3 项推理测试覆盖最终分数排序、带 padding 的分块 logits 还原及最多 100 个候选。4 项合并测试覆盖零预测图片、分辨率不一致、重复分片和 smoke 混入。

每组 12 张真实 val smoke 已通过：同一次前向的 legacy 掩码与原 `instance_inference` 完全一致，分数在 rtol=1e-6 / atol=1e-7 内一致。1024 / 1280 / 1536 / DSH 峰值已分配显存分别约 3.09 / 3.18 / 3.48 / 7.05 GiB。DSH head 残差末层权重范数为 1.0716；483 个候选进入细化，共改变 98,344 个像素。

这些检查证明实现路径生效；不能代替 APs 效果验证。

## 运行与复现

在 `/private/rgzndzy` 下运行：

```sh
./seg_project/run.sh python -m unittest discover -s seg_project/tools -p 'test_round1_*.py' -v
./seg_project/run.sh python -u seg_project/tools/run_round1.py val
# 检查 val_results.json 与完整基线重现情况后再生成 test
./seg_project/run.sh python -u seg_project/tools/run_round1.py test
./seg_project/run.sh python seg_project/tools/diagnose_round1.py
```

每个设置划分 8 个互不重叠的分片；每张 GPU 两个独立推理进程，重叠 CPU 预处理与 GPU 计算。默认使用 4 卡；用户要求并行启动长训练后，等 1536 分片全部完成，保留结果并将后续推理限制到 `ROUND1_GPUS=0,1`，GPU 2、3 启动 D 继续训练。不是 DDP 推理，不更改 GPU 通信环境。每个分片保存参数、PID、日志、已处理 image_id、实际 config、源码哈希和完成标记；首轮调度器停止前已写完的 1536 分片以 complete.json 为准。原调度日志保存在 val_initial.log。完成分片可复用；失败时先检查原因，不启动重复的活跃任务。

核心新增实现位于 `rgbd_seg/m2f/experimental_inference.py`、`tools/infer_round1.py`、`tools/run_round1.py`。原训练、模型及正式推理文件保持不变。`tools/evaluate_round1_setting.py` 可在某组 8 个分片全部完成后提前评估，主流程会复用其结果。

## 结果文件

- `smoke/`：真实图片路径及显存检查。
- `val/`、`merged_val/`、`evaluation/`：分片、合并预测与 COCOeval 日志。
- `val_results.json`：9 个对照、历史基线偏差、最终选择。差异超过 0.001 时阻断自动选择，先调查原因。
- `candidate_coverage.json`：极小目标与低填充率目标的候选覆盖诊断；不是 AP 或 COCO AR。
- `selected_test.json`、其 manifest、`submission.csv` 及验证报告：最终输出。CSV 在全量 RLE 解码与源预测一致性检查通过后才发布。

原阈值扫描和旧提交保留；新文件不覆盖旧实验。

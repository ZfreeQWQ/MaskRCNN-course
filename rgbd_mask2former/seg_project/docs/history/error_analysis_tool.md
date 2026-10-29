# 实例错误候选工具增强与复核约定

`tools/error_analysis.py` 保留原有 CLI 参数和 `analyze(data,predictions,threshold,max_cases)` 调用。新增 `--m1-predictions` 用于独立提供真实 M1 对照；原来的 `--predictions` 始终标为 **Selected**，不能据所选架构自动把它称为 M1。`--baseline-predictions` 对应 B0。未提供的 B0/M1 列明确显示 NOT PROVIDED。

```bash
seg_project/run.sh python seg_project/tools/error_analysis.py \
  --annotations seg_project/datasets/blender/annotations/instances_val.json \
  --predictions SELECTED_MODEL/best_val_predictions.json \
  --baseline-predictions seg_project/outputs/1x_b0/best_val_predictions.json \
  --m1-predictions seg_project/outputs/1x_m1/best_val_predictions.json \
  --output-dir seg_project/reports/final_error_analysis \
  --threshold 0.5 --max-cases 20 --render
```

这是验证集的诊断/显示阈值，不修改 AP、模型选择或最终提交阈值。工具拒绝 test split、图像路径中的 test 以及显式 test 标注路径；不读取 test 数据或使用 GPU。

## 统计、候选与稳定定位

四个通用候选类型是 `missed`、`false_positive`、`boundary_candidate` 和 `adjacent_merge_candidate`。所有类型都覆盖 S/M/L。按分数降序与原始预测索引进行 IoU=0.5 的一对一诊断匹配；该诊断不代替 COCO AP。

- missed/boundary 的面积分组依据 **GT mask area**。
- false positive/merge 的面积分组依据 **预测 mask area**；merge 同时保留被覆盖 GT 的面积与分组。
- 完全沿用 COCOeval 的闭区间：S=[0,1024]，M=[1024,9216]，L=[9216,1e10]。恰好 1024/9216 的实例在相邻两组计数，S/M/L 计数不能简单相加当作独立总数。
- 每个候选的 `area_groups` 保留所有匹配区间；`area_group` 是先匹配的 S/M/L，只用于无重复的候选采样与展示。
- 每类至多保存 **20 个不同 candidate ID**，默认最多10个S、5个M、5个L，缺额按S→M→L补足。计数覆盖全部候选，不受存储上限影响。排序为 image_id、GT id 和原始预测顺序，结果可复现。
- `prediction_index` 为输入 `--predictions` JSON 列表的 **0 基原始位置**，不因分数筛选或排序改变。`prediction_id` 若源预测未给 id 则为 null；不能用它代替原始 index。
- `case_id`（同值别名 `candidate_id`）由 kind/image_id/GT annotation id/原始 prediction index 组成。GT支持为空或预测支持为空时明确使用 none；漏检若所有预测 IoU 都为0，不关联任意不相交预测。

新 JSON 中：

| 字段 | 内容 |
|---|---|
| `all_area_counts[kind]` | 全部面积下、该类型的候选次数（不是已确认错误数） |
| `counts_by_area[kind].S/M/L` | 包含共享边界的分层候选计数 |
| `population_by_area.gt/predictions_above_threshold` | 各分层的 GT / 达阈值预测数，作为统计背景 |
| `stratified_cases[kind]` | 每类≤20的分层实例候选 |
| `selection_summary[kind]` | 可用/已保存候选数、保存的主分组计数；人工确认数量为null |
| `sources` | annotations、selected_predictions、B0_predictions、M1_predictions 的实际路径或null |
| `counts/cases` | 兼容旧 small_* 字段；cases 是上述已保存候选的小目标子集，不额外增添案例 |

每个实例包含 `annotation_id`、`semantic_type`、`gt_bbox`、`gt_area`、`prediction_index/id/bbox/score`、GT/预测 mask tight bbox 与面积、`best_iou` 和 matched/coverage 观察。merge 还含 `covered_annotations`（逐 GT id、面积、分组、bbox、覆盖率、IoU）、`covered_annotation_ids` 和被覆盖 GT bbox 的最小间距。这些值可以定位原标注和原预测，而不是只指向整张图。

`depth_evidence` 仅为保存的实例读取对应 val 深度，记录其明确支持区域（GT mask 或预测 mask）中的像素数、有效/无效比例、米制p10/median/p90，以及一像素外环的深度中位数和中位数差。缺少明确标定时记录 unavailable，不猜系数。外环差、无效比例、IoU 和 GT覆盖率均是观察，不能单独证明遮挡、边界问题或漏检原因。

## 图板与人工验收

`--render` 对每个 `case_id` 单独生成图板，不再按 image_id 去重。六列为 RGB、米制深度、GT、B0、M1、Selected。每列上方为全图，下方使用相同窗口放大局部（小实例不会仅在512缩图中出现）；红框和文字定位当前 GT/Selected prediction 及 merge 涉及的 GT，JSON保存 `roi_xyxy` 和图名。B0/M1列中的红框是当前候选的统一比较区域，不表示那些模型一定在红框中有匹配预测。

`analysis.md` 给出 S/M/L 表、全部实例索引、证据和对应图板链接。`human_review_template.json` 为每个实例生成待填写的 confirmed_error、reviewer、observations、cause_hypotheses、counterexamples_or_limits、ablation_support。所有自动结果均为 **pending_human_review**，工具不声称人工已复核，也不自动归因。

主任务需逐个看图，确认四类各10–20个真实错误；低IoU候选可能属于定位偏差、漏检、掩码形状等其他情况，覆盖多个GT也可能不是相邻实例粘连。若真实案例不足10个，必须如实报告数量和候选被否定的情况，不能复制或填充实例。原因应明确分开写观察、假设与真实消融证据，并记录反例/局限。同一实例可触发多个类型，类型数量也不能直接相加当作独立物体总数。

## CPU 验证

没有读取任何真实大 JSON。使用持久目录下的256×256合成 RGB、uint16 depth、RLE GT/预测验证：四类候选、S/M/L统计、1024/9216共享边界、分数过滤后的原始索引、稳定候选ID、30候选只保存20、GT掩码内精确0.5深度无效率、B0/M1/Selected来源区别、六列全图+放大ROI、pending复核状态和test拒绝。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python seg_project/reports/error_analysis_contract_checks.py
PYTHONPATH=seg_project/tools python -m unittest seg_project.tools.test_tools.ToolContracts.test_small_missed_and_false_positive_candidates -v
python -m py_compile seg_project/tools/error_analysis.py
```

实测通过日志：`reports/error_analysis_contract_checks.log`；合成输入、候选JSON和图板位于 `reports/error_analysis_synthetic/`，全部明确标为 synthetic，不可用作真实数据的错误分析或实验结论。

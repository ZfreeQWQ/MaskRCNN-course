# 执行计划完成性独立审查

审查时间：2026-10-16 12:50 UTC。依据：`大作业执行计划.md`、当前 `tools/run_experiments.py`、训练/评价/错误分析工具源码、实际日志和已有产物。审查期间没有启动 GPU、修改活跃训练代码或读取 test 标注。本文是当时的审查快照；新增配置不表示对应实验已经运行。

## 核心结论

当前流水线即使运行至 `training_complete_test_blocked`，也还不能说明“除 test 输入标定外，计划均已完成”。它安排了四组正式 1x、选定 RGB-D 架构的四卡 2x、完整验证、标准速度测量与错误候选出图，但尚未覆盖逐项模块消融、已复核错误原因和正式报告的实质性文字结论。同预算单卡/DDP 对照也未安排。

固定 0.05 的 AP/推理分数阈值在各组、benchmark 与最终提交之间一致；无须为了完成计划强行搜索阈值。错误图板使用的 0.5 是显示/诊断阈值，不能写成最终提交阈值。

## 当前真实证据与尚未完成的运行

- 已有环境与实际算子证据：`runs/env_verified.json`、`runs/ddp_smoke.json`、`runs/mask_rcnn_operator_smoke.log`。这是算子与通信可用性的证据，不是正式四卡训练的性能结果。
- 数据完整性记录 `reports/data_validation.json` 给出 train/val 全部 521,890 个实例解码、面积与 tight bbox 检查，train/val 场景不交叉；训练深度有效像素精确统计已保存。该数据验收不依赖流水线状态字段。
- 已有真实 smoke 训练、验证子集和验证 Demo；子集分数不能替代全 val 正式指标。
- 当前 `outputs/pilot_b0/metrics.json` 出现 iteration=999 的完整 bbox/segm 指标，`console.log` 有完整 inference、COCO bbox/segm evaluation 和 accumulation 记录，`best_metrics.json` 对应 completed_iterations=1000。因此 B0 至少完成了第一个 1000-step 验证节点，但这不是 pilot 完成，更不是 1x 完成。
- 审查时没有发现正式 `1x_*`/最终 `2x` 的完成实物或正式 benchmark 产物。现有 `reports/report.md` 仍含“待完成/待复核”表格，不能当作完成报告。

## 必须补足或在结束前实际验收的项目

| 计划要求 | 现有流程实际覆盖 | 必须补足/验收 |
|---|---|---|
| §2、§4 B0/B1/M1/A1 公平比较 | 同 seed、同预训练源、同输入尺度、同 stage，B0/B1 仅模态不同；A2 与 B1 配置完全等价 | 等四组正式 1x 的真实训练、full-val best 预测、指标和标准 benchmark 齐全后再下结论。A2 直接说明复用 B1，避免重复训练与伪造独立重复实验。 |
| §5.1 “逐项做消融” | B1/M1 同时改变 anchors、proposals、mask pool；只说明组合收益 | 在相同 1x 预算下补三个 leave-one-out：恢复标准 anchors、恢复标准 proposals、恢复 mask pool=14。每个记录 APS/AP/速度/显存，并与 M1 比较。三个新增配置见下文。 |
| §5.1 P2 贡献 | 所有模型本来已有 P2，配置未新增 P2 路径 | 报告明确 P2 是共同骨干结构，不归因于新模块；不需要制造“新增 P2”消融。 |
| §4.1 正式 1x/2x 两档 | 默认四组 1x，再只给所选 RGB-D 架构做四卡 2x | 至少实际完成四组 1x 和所选架构 2x，披露“2x 仅所选架构”。1x=10 epochs、2x=20 epochs 是本项目预算定义，不是 COCO 90k 原始日程。不能宣称所有结构都运行了 2x。 |
| §6.1 比较单卡/四卡 loss、吞吐、APS | 通信 smoke + 单卡 1x vs 四卡 2x，会同时改变卡数和预算 | 增加所选架构四卡 **1x** 对照，与同架构单卡 1x 比较图像预算、损失曲线、训练吞吐和 full-val APS；记录 batch/LR 缩放的影响，不能将差异全归于通信。 |
| §7.1 必报指标、大小和标准速度 | 工具默认 1024、20 warmup、200 timed 图；benchmark 在训练后安排 | 验收真实 JSON 的 warmup≥20、iterations≥200、shape=1024、smoke=false，并实际查看时间样本/设备名/峰值显存。报告区分完整 optimizer checkpoint 和纯模型张量容量，标明模型时间含传输及 D2 后处理。新增消融同样需要 benchmark。 |
| §7.2 四类错误各 10–20 案例 | 自动候选最多20条/类，不保证至少10条；图板按 image_id 去重；小目标候选是启发式 | 按实例明确编号并逐例查看 RGB/depth/GT/B0/M1。逐类确认10–20个实际错误；若某类真实不足10个，报告全部真实案例及不足原因，不能复制或把未确认候选凑数。 |
| §7.2 按 COCO area 分组 | 当前漏检/边界只处理 GT area<1024，FP 按预测 mask area<1024；没有 S/M/L 分层诊断表 | 明确 GT 与预测面积定义；补 S/M/L 分层统计，主案例仍聚焦 small。COCO small 边界使用与 evaluator 一致的约定，并避免把 GT 小目标和预测小 mask 混为一类。 |
| §7.2 案例定位与原因证据 | 候选 GT 有 annotation_id，但 FP/merge 仅保留 score/部分信息；整图缩到512可能看不清小实例 | 为每个 FP/merge 保留可复查的 prediction index/ID、bbox 或 RLE 关联，并在图板标出该实例；必要时放大局部。记录目标像素数、深度无效占比/深度边界、遮挡或相邻实例证据，区分观察和原因假设。 |
| §7.2 B0/M1 图板 | 流程画 B0 和 selected final，若所选为 B1 则没有 M1 | 无论最终是否选 M1，提供 B0 与 M1 的必需对照；若最终为 B1，增加 final 面板或另附 final 案例。 |
| §7.2 原因与消融支持 | `error_analysis.py` 写入“人工复核记录（待填写）”；`build_report.py` 也保留待填内容 | 补每类的真实观察、原因假设、反例/局限和对应消融支持。不能把低 IoU 直接解释为边界问题，也不能仅因深度无效就认定其导致漏检。 |
| §5.3 固定阈值/后处理 | AP、benchmark、test 默认0.05；NMS0.5、maxDets100统一。诊断/展示默认0.5 | 报告记录固定 score=0.05、NMS=0.5、maxDets=100，解释未做阈值搜索；诊断阈值0.5单独标记。若后续改变提交阈值，必须先在 val 评价，并用相同阈值重测最终模型速度/保存阈值决定，之后才读取 test RGB-D。 |
| §8 最终选择 APS 优先且整体 AP 无不可接受下降 | `score()` 仅按(APs,AP)字典序，AP只在APS相同时起作用 | 在最终选择前给出整体 AP 降幅可接受性的明确判据或真实取舍论证；最高 APS 不自动满足整体 AP 要求。速度/显存代价也应参与可行性判断。当前计划没有数值界限，不应擅自伪称已经满足。 |
| §7.3 Demo 与报告 | 静态 Demo 工具可用，已有 smoke 图；正式流程只自动出候选图，不自动完成科学解释 | 用正式 B0/M1/最终预测生成至少一份可复现 val RGB-D Demo，核对 GT 只来自 train/val。正式报告补训练预算/学习率/seed/预训练/无效深度语义、实际指标差异、消融、错误原因、限制、链接与复现命令。 |

## 训练方案的必要披露与可选项

1. 当前所有组统一使用 `[896, 1024]` 短边抖动，未实现 §5.1 举例的 768–1024 范围。保持统一增强有利于模块归因，但必须准确披露，不能声称已经覆盖768。该范围是推荐配方，不需要为修饰报告而中途更改活跃实验；若后续评估尺度策略，应作为独立、等预算实验。
2. 有限颜色扰动、深度噪声/失效增强、M2 门控/PointRend、第二 seed、TTA、ensemble 均为可选扩展；当前核心早期融合和小目标改进不依赖它们。没有启用就明确报告未使用。
3. 裁剪是全图显存/速度不可接受时的条件分支。已有全图算子/训练可运行证据，不必无条件引入裁剪；若后续采用，须独立报告训练设置、合并算法和全部计时。
4. 若将来宣称“小 anchors、extra proposals、high mask resolution各自都有收益”，必须真实逐项验证。leave-one-out只能说明在其他M1模块存在时的条件贡献，不能推出模块之间无交互；全因子实验可选。
5. 确保新报告使用实际 completed_iterations、有效图像预算、full-val评估数量、预测和权重文件。`training_complete.json`、`pipeline_status.json`、报告模板或候选图存在均不足以独立证明实验完成。

## 新增独立消融配置（仅落盘、未启动）

- `configs/ablate_no_small_anchors.yaml`：继承 M1，仅恢复 `[32,64,128,256,512]` anchors。
- `configs/ablate_no_extra_proposals.yaml`：继承 M1，仅恢复标准四项 RPN proposal 上限。
- `configs/ablate_no_high_maskpool.yaml`：继承 M1，仅恢复 mask pool 14。

CPU 配置解析和递归键比较已通过；除 experiment/output 元信息外，每个配置只改变所述模块。没有修改当前训练入口、主流水线或其配置。

## 可执行的补足顺序

以下命令是交由主任务在四卡空闲后调度的建议，本审查没有执行它们；不要与正在训练的任务争卡。采用与 `outputs/1x_m1` 相同的1x预算、seed和增强，从同一官方预训练独立启动。若目录已有 checkpoint，明确使用 `--resume`。

```bash
cd /private/rgzndzy
CUDA_VISIBLE_DEVICES=0 seg_project/run.sh python seg_project/tools/train.py --num-gpus 1 --config-file seg_project/configs/ablate_no_small_anchors.yaml --stage 1x OUTPUT_DIR /private/rgzndzy/seg_project/outputs/1x_ablate_no_small_anchors DATALOADER.NUM_WORKERS 2
CUDA_VISIBLE_DEVICES=1 seg_project/run.sh python seg_project/tools/train.py --num-gpus 1 --config-file seg_project/configs/ablate_no_extra_proposals.yaml --stage 1x OUTPUT_DIR /private/rgzndzy/seg_project/outputs/1x_ablate_no_extra_proposals DATALOADER.NUM_WORKERS 2
CUDA_VISIBLE_DEVICES=2 seg_project/run.sh python seg_project/tools/train.py --num-gpus 1 --config-file seg_project/configs/ablate_no_high_maskpool.yaml --stage 1x OUTPUT_DIR /private/rgzndzy/seg_project/outputs/1x_ablate_no_high_maskpool DATALOADER.NUM_WORKERS 2
```

上述三条分别是独立前台命令，主任务可按可用卡并行调度。每组训练实际完成后，对三个消融目录分别执行以下后处理（示例为 no_small_anchors）：

```bash
seg_project/run.sh python seg_project/tools/evaluate.py --annotations seg_project/datasets/blender/annotations/instances_val.json --predictions seg_project/outputs/1x_ablate_no_small_anchors/best_val_predictions.json --output seg_project/outputs/1x_ablate_no_small_anchors/report_artifacts/metrics.json
CUDA_VISIBLE_DEVICES=0 seg_project/run.sh python seg_project/tools/benchmark.py --config seg_project/outputs/1x_ablate_no_small_anchors/config.yaml --weights seg_project/outputs/1x_ablate_no_small_anchors/best_segm_APs.pth --metadata seg_project/datasets/blender/annotations/instances_val.json --output seg_project/outputs/1x_ablate_no_small_anchors/report_artifacts/benchmark.json
```

同预算DDP对照示例为 M1；若最终架构选 B1，则换为 b1 配置及不同输出目录。四卡均空闲后再执行，不复用或覆盖最终2x目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 seg_project/run.sh python seg_project/tools/train.py --num-gpus 4 --config-file seg_project/configs/m1_rgbd_small.yaml --stage 1x OUTPUT_DIR /private/rgzndzy/seg_project/outputs/ddp_1x_m1_comparator DATALOADER.NUM_WORKERS 2
```

随后汇总 `1x_m1`、三个消融与 `ddp_1x_m1_comparator` 的真实结果。错误候选生成可用已有工具，但之后必须逐例复核并填写原因与消融支持。保证 M1 参与图板的独立调用如下：

```bash
seg_project/run.sh python seg_project/tools/error_analysis.py --annotations seg_project/datasets/blender/annotations/instances_val.json --predictions seg_project/outputs/1x_m1/best_val_predictions.json --baseline-predictions seg_project/outputs/1x_b0/best_val_predictions.json --output-dir seg_project/reports/m1_error_analysis --threshold 0.5 --max-cases 20 --render
```

此调用仍只生成候选和待填表，不等于完成错误分析。若需改变候选标识、面积分层、定位高亮等工具行为，应由主任务在活跃训练代码冻结原则下安排独立工具修改。

## 到达“仅 test 元数据阻断”的最低验收

必须同时有：四组正式1x及选定架构2x真实产物；三个逐项消融及标准benchmark；同预算DDP诊断对照；完整验证与公平阈值记录；四类10–20已确认案例（不足时真实说明）、COCO面积分层和原因/消融证据；正式Demo；不再含实质“待填写”的最终报告。届时缺少test逐图 `depth_scale_m`/`depth_invalid_code` 才能作为唯一外部阻断。

test 输入标定缺失只阻断最终 RGB-D test 推理、2,112行CSV和掩码JSON以及提交验证；它不阻断上述训练、val评价、消融、速度、错误分析、报告或val Demo。不得用train中全部scale相同这一事实猜测test标定，也不得读取test标注补全。

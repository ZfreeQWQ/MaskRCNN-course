# 工具链集成验证（smoke，不能作为最终实验结论）

真实checkpoint：`outputs/smoke_m1/model_final.pth`，训练20 iterations。设备：GPU3 / Iluvatar BI-V150。该GPU在验证完毕后已释放给正式pipeline。

- CPU工具回归：8项通过，见 `tools_cpu_tests.log`。合成稀疏小mask验证官方COCO segm面积分组、APs满分与空预测0分；这些是夹具测试值，不是模型精度。
- 真实val前4张推理：400条compressed RLE实例，`outputs/smoke_m1/report_artifacts/val_predictions_smoke.json`。
- 官方COCO子集评价：`segm/APs=3.029869%`、`segm/AP=3.268901%`，所有指标明确 `smoke:true`，见同目录 `metrics_smoke.json`；不能用于正式模型选择或性能结论。
- JSON/xyxy CSV格式验证：4行val子集通过，`validation_smoke.json` 标明 `split:val, submission_ready:false`；不构成test提交。
- 真实GPU benchmark链路：1024×1024×4、20次预热、3次计时，约113.2ms模型均值，显式 `smoke:true`；次数不足200，不能用于正式速度表。见 `benchmark_smoke.json`。
- Demo：`figures/demo_smoke_m1_val.png` 展示RGB、按逐图系数换算的深度、GT、M1 smoke预测。显示阈值0.5下未出现高分预测，图像如实保留该结果。
- test缺失深度标定：真实恢复metadata的 `depth_scale_m:null` 在模型构建前触发明确错误；见 `test_metadata_guard.log`，未生成预测、未读取test标注、未占用GPU。直接Mapper入口也由训练模块实施缺值校验。

正式数据集精度、至少200张标准benchmark、正式test提交由长期pipeline执行；当前test RGB-D仍需原始允许的逐图标定元数据。训练日志metrics.json为JSONL，工具汇总产物保存report_artifacts/避免覆盖。

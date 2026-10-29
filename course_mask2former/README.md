# For Boo：G Mask2Former 实验交接

本目录交接已完成验收的 G-R50 单尺度模型、冻结九视图系统及实际 CSV 生产入口。最佳结果为开发验证集 **APs 56.68493022**。所有实验队列已经关闭；本次代码交接没有启动训练、推理或上传。

本目录独立于仓库原有 Matterport Mask R-CNN，采用 MMDetection 版 Mask2Former。保留已验收源码的原始字节，并通过清单记录来源。这里是实验源码和结果快照，**不是包含权重、数据及全部历史缓存的一键运行发行包**。

## 已接受结果

以下均为同一 2584 张开发验证集、0–100 分口径。单尺度为最终 1536 输入；九视图使用冻结四小头及原选择规则。

| 模型 | APs | AP | AP75 | 相对同层 G 的 APs |
| --- | ---: | ---: | ---: | ---: |
| G-R50 单尺度 | 55.747879 | 81.272463 | 85.722207 | 基线 |
| G-R50 九视图 | **56.684930** | **81.474842** | **85.796653** | 基线 |
| G+Q 单尺度 | 55.027742 | 80.775583 | 85.024932 | −0.720137 |
| G+Q 九视图 | 55.650331 | 80.972544 | 85.046903 | −1.034599 |

G 单尺度相对 LR3e-5 单尺度 45.03649818 提高 10.71138046 点。G 九视图相对原 LR 九视图 47.70413206 提高 8.98079816 点，相对 G 单尺度提高 0.93705158 点。收益分别比较，不能相加；重复开发验证不代表独立泛化或 Kaggle 分数。精确机器可读结果见 [results.json](results.json)。

Q 的两层最终结果均退化，保留 G。G+Swin-T 最后持久训练记录为 13300 步，因同卡外来计算进程触发保护而停止；10k 完整 1024 APs 为 42.5，对 G 同阶段 41.5 为 +1.0。没有 15k、最终 1536 或九视图结果。用户随后取消恢复及后续工作，因此不能把该项记作算法负结果。CNN 继续停止。

## 从哪里读代码

| 内容 | 入口 |
| --- | --- |
| 七通道几何计算 | [geometry_signal.py](snapshots/g_train/geometry_signal.py) |
| 几何变换与 Copy-Paste 对齐 | [course_geometry_data.py](snapshots/g_train/rgbd/course_geometry_data.py) |
| G 四尺度融合模型 | [course_geometry_model.py](snapshots/g_train/rgbd/course_geometry_model.py) |
| 已使用训练配置、生成模板 | [g_train_30k.py](configs/g_train_30k.py)、[g_formal_template.py](configs/g_formal_template.py) |
| 历史训练编排、检查 | [geometry_worker.py](snapshots/g_train/geometry_worker.py)、[geometry_cpu.py](snapshots/g_train/geometry_cpu.py) |
| G 九视图与几何裁剪适配 | [g_system_worker.py](snapshots/g_nineview/g_system_worker.py) |
| 冻结九视图主体与四小头 | [abc_a_worker.py](snapshots/g_nineview/abc_a_worker.py)、[runtime](snapshots/g_nineview/runtime) |
| test 入口与 CSV 编码核验 | [g_submit_worker.py](snapshots/submission/g_submit_worker.py)、[g_submit_codec.py](snapshots/submission/g_submit_codec.py) |
| 环境版本记录 | [requirements-lock.txt](environment/requirements-lock.txt)、[recorded-core-packages.txt](environment/recorded-core-packages.txt) |

`snapshots/` 共 538 个冻结源文件：G 训练 130、LR 继承源 122、tiles 继承源 106、几何运行依赖 3、G 九视图 86、提交入口快照 91。继承快照保留了原先一起冻结的辅助脚本和测试，有重复文件；不能把它们当成新增实验或新增模型组件。Q 和 Swin 本次只交接结果摘要，未纳入最佳生产路径。

## 配方与输入约定

G 从原 COCO Mask2Former-R50 初始化，训练 30k：1024、batch 2、FP32、seed 20261013，AdamW 基础 LR3e-5，backbone ×0.1、fusion ×10、weight decay0.05、clip0.01，在18k/24k衰减。沿用 CP、点噪声辅助监督和 tiny 匹配；每5k执行完整1024验证。G 的七通道为深度、相对估计平面高度、相机法向XYZ、相对深度边缘及有效位；只用 depth/valid/K，不把对象标签或世界姿态作为推理输入。几何分支独立20% dropout、零残差初始化，归一化统计仅来自固定训练子集。

生产系统采用该 30k 权重，保留原九视图窗口、四小头、特征定义、掩码修正、分数、选择器、NMS 和 top100，零重新拟合。640/512裁剪同步裁剪几何输入。两份冻结选择器参数和训练归一化统计位于 `configs/`；另两小头及主模型权重通过 [external_artifacts.json](external_artifacts.json) 的 SHA 定位，未放入 Git。

## 复核与复用边界

在仓库根目录执行以下命令仅检查源码 SHA、JSON 和 Python 语法，不导入模型，不使用 GPU：

```sh
python3 course_mask2former/verify_handoff.py
```

[source_manifest.json](source_manifest.json) 将每个快照文件映射到原项目相对路径，并绑定已接受 SHA。历史运行把 `/mnt/nfs/fenghaibin/AI_Class` 挂为 `/workspace`，并使用项目内 `mmdetection`。MMDetection 提交为 `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`，记录镜像为 `sha256:eff8b304fe898a9509ee7b6df88b17446519ec29c85d007c6f6ac709787e8140`。为复现 CPU 数值选择器，原生产环境使用 `OPENBLAS_CORETYPE=Prescott`。

历史 launcher/spec 保留了当时的机器路径、GPU UUID、锁、owner 和已到期预算；不应直接作为新任务启动命令。移植执行时须提供合法数据、真实相机K、列明的权重及运行依赖，适配路径和新的资源租约。完整历史验收重放还需要清单引用的旧报告和预测缓存，本分支未打包这些大文件。此次只验证了交接完整性，没有声称新路径下已重新训练或完成推理回归。

完整九视图同卡配对成本见 [paired_timing.json](evidence/g_nineview/paired_timing.json)：原 LR 约3262.334ms，G约4530.691ms；这是固定图各5次预热、20次完整IO至JSON测量，排除模型构建与结果落盘。small 召回改善同时伴随重复FP和大GT碎片代理增加，详见 [分析](evidence/g_nineview/independent_analysis.json)。

## 已交付 CSV

已生成并核验 2112 张真实 test 的 `submission.csv`，87946334 字节，SHA256：

```text
066b8098b1f8b9e6448d7117e69fe56abd1e7e99bf85b8e3fc635ebbbcf04342
```

CSV 包含 `image_id,PredictionString`，实例为 mask-only 的 `image_id/category_id/score/segmentation` 四键，无 bbox。每图最多100个1024 COCO压缩RLE，仅移除零面积槽位，保留其他mask、score和相对顺序；完整ID、RLE解码重编码及CSV/JSON往返已验证。该大文件不进入本分支，交付路径和来源绑定见 [外部文件清单](external_artifacts.json) 与 [生产报告](evidence/submission/report.json)。未自动上传 Kaggle。

## 来源与授权材料

`evidence/` 为已完成实验的精简验收与结果材料；[evidence_manifest.json](evidence_manifest.json) 记录原文件来源和 SHA。本分支未包含数据图像、完整标注、预测掩码缓存、模型二进制、完整日志、SSH凭据或个人会话记录。MMDetection、PyTorch等依赖代码不在本目录重新打包，使用时遵守各上游许可；原仓库 LICENSE 保持原样。

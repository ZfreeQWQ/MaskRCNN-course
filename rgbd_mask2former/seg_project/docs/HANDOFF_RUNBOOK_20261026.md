# 运行、恢复与迁移手册

面向已有 Linux / Iluvatar CoreX 环境。命令从 `/private/rgzndzy` 执行。代码包不包含数据、权重或二进制运行时。

## 检查与状态

```bash
cd /private/rgzndzy
pwd
ls
python --version
python -m pip show torch mmcv torchvision
./seg_project/run.sh ixsmi
python seg_project/tools/mpformer_status.py
python seg_project/tools/round2_status.py
tail -n 3 seg_project/experiments/mask2former/round2_mpformer_20261026/metrics.json
tail -n 1 seg_project/experiments/mask2former/round2_mpformer_20261026/mp_diagnostics.jsonl
```

四卡正在使用时不要另跑 GPU 测试或重复训练。`ixsmi` 进程列表有时为空，结合显存、利用率和进程/管理器状态判断。暂停消融的 manager_alive=false 是预期。包内 PID、日志和迭代是冻结记录，不是迁移后的实时状态。

包根 `evidence/environment.json` 是现有版本清单，不是 pip 安装列表。禁止擅自重装 torch/mmcv/CoreX 或运行第三方安装脚本。

## 新建与恢复训练

当前任务已在运行。下列命令仅供原任务停止后恢复，或明确新建实验时使用；先查 GPU 和输出目录。

```bash
# 新实验：D 权重初始化，fresh optimizer，独立输出目录
./seg_project/run.sh torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  seg_project/tools/train_mpformer.py \
  --config-file seg_project/configs/mask2former/round2_mpformer_20261026.yaml \
  OUTPUT_DIR /private/rgzndzy/seg_project/experiments/mask2former/mp_new_run

# 恢复原 MP 输出目录，保留 optimizer/scheduler/iteration
./seg_project/run.sh torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  seg_project/tools/train_mpformer.py \
  --config-file seg_project/configs/mask2former/round2_mpformer_20261026.yaml --resume
```

配置覆盖直接放末尾，不使用 `--opts`。resume 要求 OUTPUT_DIR 有 `last_checkpoint`；不带 resume 遇到已有 checkpoint 会拒绝覆盖。恢复不保证 dataloader RNG/顺序逐位相同。日期写死的控制器不是通用恢复器，勿重跑已完成的启动脚本。

D+H 暂缓，不自动恢复。若以后明确恢复，用 `train_round2.py --resume` 和 DH 配置，不能改用 coarse-only 验证来选 H。

## 完整验证与检查

```bash
# 有空闲资源时，对指定 MP 权重做完整 val，输出到新目录
./seg_project/run.sh torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  seg_project/tools/train_mpformer.py --eval-only \
  --config-file seg_project/configs/mask2former/round2_mpformer_20261026.yaml \
  MODEL.WEIGHTS /private/rgzndzy/seg_project/experiments/mask2former/round2_mpformer_20261026/best_segm_APs.pth \
  OUTPUT_DIR /private/rgzndzy/seg_project/experiments/mask2former/mp_eval_review

# CPU 协议检查，仍依赖已有项目环境
./seg_project/run.sh python seg_project/tools/test_mpformer.py
./seg_project/run.sh python seg_project/tools/test_round2.py
./seg_project/run.sh python seg_project/tools/test_continue_results.py
```

完整 val 不传 `--eval-limit`；子集不可选最终模型。`test_mpformer_ddp.py` 是占用四卡的专项检查，不在当前训练旁边再启动。当前 MP 第一个最佳权重尚未生成，示例中的 best 路径是训练验证后才会产生的资产。

## 已完成 D 的推理

```bash
./seg_project/run.sh python seg_project/tools/infer_round1.py \
  --config seg_project/experiments/mask2former/d_continue6_results_20261026/config.yaml \
  --weights seg_project/experiments/mask2former/d_continue6_results_20261026/best_segm_APs.pth \
  --metadata seg_project/datasets/blender/annotations/instances_val.json \
  --split val --size 1536 --output-dir seg_project/experiments/mask2former/d_review_val
```

该入口输出 legacy 与 mask_first；当前基线取 mask_first。阈值 0、最多 100，恢复原图坐标。不要用旧 `infer_m2f.py` 的默认阈值 0.5 来复现 52.916。

MP 完整 val 用上面的 MP eval 入口。**当前未提供直接注册 MP 的通用 test 导出入口**；需先复核最佳 val，再为 test 推理显式注册 MP、复用 mask-first/分片/manifest/RLE 审计。不能直接用普通 `infer_round1.py` 加载 MP 配置就认为已支持。

## 外部资产

| 资产 | 原服务器路径 |
|---|---|
| 原始 RGB/深度及提交模板 | `/private/rgzndzy/blender/` |
| 完整 train/val 标注、test 图像深度元数据 | `/private/rgzndzy/seg_project/datasets/blender/annotations/` |
| 预训练权重/来源 | `/private/rgzndzy/seg_project/weights/` |
| 已完成最佳 D 权重/提交 | `/private/rgzndzy/seg_project/experiments/mask2former/d_continue6_results_20261026/` |
| 暂停 H checkpoint | `/private/rgzndzy/seg_project/experiments/mask2former/round2_dh_20261026/model_0000901.pth` |
| 当前 MP 权重/日志 | `/private/rgzndzy/seg_project/experiments/mask2former/round2_mpformer_20261026/` |

部分 `_BASE_` 引用历史实验 `config.yaml`，包内保留这些已解析配置以免断链。MODEL.WEIGHTS、数据、上游配置和 vendor 路径仍含原服务器绝对路径。厂商算子不随包分发；`msda_shim` 复用原环境已有 `mmcv-2.1.0+corex.4.4.0`。

迁移时在新的持久化目录解压审查，调整路径并补齐资产；不整包覆盖运行目录，不做删除式同步，不通过重装底层环境处理路径错误。

## Windows 辅助工具

包根 `workspace_tools/` 保留 `remote_exec.py`、状态、同步和历史启动脚本。连接信息从外部 `ssh.txt` 读取，该文件不随包分发；本机 `.tools` 依赖目录也不分发。Paramiko 是工作区辅助依赖，不是服务器 GPU 依赖。

只读状态脚本可参考使用。`launch_*.sh`、暂停脚本和日期管理器需先读源码与状态，不重新执行历史任务。第三方原许可证保留；MP 固定参考版本见 `references/mpformer_20261026/manifest.json`。

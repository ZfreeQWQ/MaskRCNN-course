# 58.389版本整理核验

本次提交未重新训练模型，也未重新运行全部原图的AP评测。历史完整VAL=0.7332437038824104、公开本地TEST=0.58389045617875来自已经完成的正式评测；平台成绩未知。

- 全部Python文件通过语法解析，默认CLI帮助入口可用；发布文件不含服务器地址、主目录或凭据。
- 三个小型权重与正式归档的SHA-256一致。边界扩散、微小实例和23维特征模块保持原算法；参数化修改见[source_provenance.json](source_provenance.json)。
- 从原图重推理两张VAL和两张TEST：原始掩码/分数/顺序、43维特征、最终掩码/顺序/分数均与正式缓存完全一致。TEST检查中拦截读取annotations/test，未触发。见[raw_image_parity.json](checks/raw_image_parity.json)。
- 默认CLI分别完成两张VAL和TEST的五视野推理、特征保存、CSV导出和所有RLE回读，输出逐项等于正式预测。见[cli_val.json](checks/cli_val.json)、[cli_test.json](checks/cli_test.json)。两图AP不代表完整成绩。
- 全部2584张VAL、244087个预测：正式CUDA精度下缓存评分回放完全一致。见[cached_val_gpu.json](checks/cached_val_gpu.json)。
- 全部2112张TEST、203324个预测：CPU缓存评分的掩码与顺序完全一致，最大分数误差2.5002149684e-7。见[cached_test_cpu.json](checks/cached_test_cpu.json)。
- CPU与GPU可能对极近分数产生不同排序：VAL索引181的两项互换，掩码集合一致、最大分数误差1.1656853e-7，正式CUDA回放无此差异。见[cpu_val_tie_diagnostic.json](checks/cpu_val_tie_diagnostic.json)。

原生产测试标签只用于已授权的历史诊断；本版TEST推理仅使用图像元数据。数据、主模型断点和几何统计需按README提供，不能把仅含小权重的Git仓库称为自带所有依赖的安装包。

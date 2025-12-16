# Dataset 内存占用问题分析与优化方案

## 问题现象
在大规模训练集（高并发 worker / 大量样本）下，训练启动后系统内存会随着数据加载持续上升，最终触发 `worker is killed` 等 OOM 报错，导致训练中断。

## 原因分析
### 1. `keep_in_memory=True` 触发的常驻缓存
`utils/dataset.py` 在构建缓存时对 HuggingFace `Dataset` 使用了 `keep_in_memory=True`：

- 跳过已缓存分片时调用 `dataset.select(..., keep_in_memory=True)`。
- 展平 captions 时调用 `dataset.map(..., keep_in_memory=True)`。

`keep_in_memory=True` 会强制 Datasets 将处理后的分片全部留在 RAM 中直至整个 map/select 完成。对于千万级样本的大数据集，这些中间结果会不断堆积，worker 进程的内存曲线呈线性增长，最终压垮系统。

### 2. 训练阶段的表现
缓存生成发生在训练入口处，一旦数据量巨大或并行 worker 较多，map/select 的中间缓存会在训练开始后迅速吞噬可用内存，表现为“训练开始后内存逐渐上升直至爆满”。

## 解决方案
### 关闭内存常驻，改为流式处理
将上述 `keep_in_memory=True` 改为 `keep_in_memory=False`，让 HuggingFace Datasets 按需流式读取和写入磁盘缓存，而不是把整个结果保存在 RAM 中。这样即便数据规模巨大，内存占用也会保持平稳，不再随处理进度增长。

### 改动点
- `_map_and_cache`：跳过已缓存部分时使用 `keep_in_memory=False`，避免把剩余分片一次性加载到内存。
- `_cache_text_embeddings`：展平 captions 时使用 `keep_in_memory=False`，防止 caption 展开结果长期占用内存。

## 预期效果
- 训练开始后的系统内存占用维持稳定，不再随时间线性上涨。
- 大规模数据集生成缓存时，worker 进程的内存峰值显著降低，避免 `worker is killed` OOM。


# Multi-process Speaker Embedding Computation

## 概述

多进程版本的说话人embedding计算脚本通过并行处理多个说话人，大幅提升计算效率。相比单进程版本，可以获得接近CPU核心数倍的性能提升。

## 核心改进

### 🚀 性能提升
- **并行处理**: 同时处理多个说话人的embedding计算
- **负载均衡**: 智能分配说话人到不同进程
- **资源优化**: 充分利用多核CPU资源
- **进度监控**: 实时显示每个进程的处理状态

### 🔧 技术特性
- **多进程架构**: 使用`multiprocessing.Pool`进行并行计算
- **批处理**: 将说话人分组到不同的批次进行处理
- **错误隔离**: 单个进程错误不影响其他进程
- **内存管理**: 每个进程独立的内存空间

## 文件结构

```
├── compute_speaker_embeddings_multiprocess.py    # 多进程计算脚本
├── run_compute_speaker_embeddings_multiprocess.sh # 多进程运行脚本
├── test_speaker_embeddings_multiprocess.py       # 多进程测试脚本
└── MULTIPROCESS_USAGE.md                         # 使用说明文档
```

## 使用方法

### 1. 快速开始

```bash
# 使用默认配置运行
./run_compute_speaker_embeddings_multiprocess.sh
```

### 2. 自定义参数

```bash
# 直接运行Python脚本
python3 compute_speaker_embeddings_multiprocess.py \
    --utterances_dir /path/to/utterances \
    --speakers_dir /path/to/speakers \
    --num_processes 8 \
    --chunk_size 10 \
    --min_utterances 2 \
    --skip_existing
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--utterances_dir` | 预设路径 | 输入的utterance embeddings目录 |
| `--speakers_dir` | 预设路径 | 输出的speaker embeddings目录 |
| `--num_processes` | `nproc` | 使用的进程数量 |
| `--chunk_size` | 10 | 每个批次处理的说话人数量 |
| `--min_utterances` | 1 | 说话人最少utterance数量 |
| `--skip_existing` | True | 跳过已存在的说话人 |

## 性能优化建议

### 🎯 进程数量优化

```bash
# 获取CPU核心数
nproc
# 一般设置为CPU核心数或稍少
--num_processes $(nproc)
```

### 📦 批次大小优化

```bash
# 小批次 (适合内存限制)
--chunk_size 5

# 大批次 (适合高性能服务器)
--chunk_size 20
```

### 💾 I/O优化

- **SSD存储**: 使用SSD存储可以显著提升I/O性能
- **网络存储**: 避免使用网络存储，优先本地存储
- **并发I/O**: 多进程天然支持并发I/O操作

## 性能监控

### 系统资源监控

```bash
# 监控CPU使用率
htop

# 监控内存使用
free -h

# 监控磁盘I/O
iotop
```

### 进程状态监控

脚本会实时显示每个进程的处理状态：

```
Processing batches: 100%|████████████| 4/4 [00:30<00:00, 7.5s/it]
P0: ✅15 ⏭️3 ❌0  P1: ✅12 ⏭️5 ❌1  P2: ✅18 ⏭️2 ❌0  P3: ✅14 ⏭️4 ❌0
```

- ✅ 成功处理的说话人数量
- ⏭️ 跳过的说话人数量 (已存在)
- ❌ 处理失败的说话人数量

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少进程数量
   --num_processes 4
   
   # 减少批次大小
   --chunk_size 5
   ```

2. **磁盘空间不足**
   ```bash
   # 检查磁盘使用情况
   df -h
   
   # 清理临时文件
   find /tmp -name "*.pkl" -delete
   ```

3. **进程死锁**
   ```bash
   # 检查进程状态
   ps aux | grep compute_speaker
   
   # 强制终止
   pkill -f compute_speaker_embeddings_multiprocess
   ```

## 测试验证

### 运行测试

```bash
# 运行多进程测试
python3 test_speaker_embeddings_multiprocess.py
```

### 性能对比测试

```bash
# 单进程版本
time python3 compute_speaker_embeddings.py --utterances_dir /path/to/data

# 多进程版本
time python3 compute_speaker_embeddings_multiprocess.py --utterances_dir /path/to/data --num_processes 8
```

## 最佳实践

### 🎯 配置建议

1. **进程数量**: 设置为CPU核心数的80-100%
2. **批次大小**: 根据内存大小调整，一般5-20个说话人
3. **最小utterances**: 根据实际需求设置，避免过少的说话人数据

### 🔧 使用技巧

1. **预处理**: 确保输入数据格式正确且完整
2. **监控**: 运行时监控系统资源使用情况
3. **备份**: 重要数据务必做好备份
4. **日志**: 保留运行日志用于问题排查

### ⚡ 性能期望

- **理论加速比**: 接近进程数量倍数
- **实际加速比**: 受I/O限制，通常为进程数量的60-80%
- **内存使用**: 每个进程独立使用内存
- **磁盘I/O**: 多进程并发读写，需要高性能存储

## 输出格式

多进程版本的输出格式与单进程版本完全相同：

```python
# 说话人embedding文件结构
{
    'embedding': np.ndarray,           # 平均embedding向量
    'dataset': str,                    # 数据集名称
    'speaker_id': str,                 # 说话人ID
    'num_utterances': int,             # utterance数量
    'failed_utterances': int,          # 失败的utterance数量
    'utterance_list': List[str],       # utterance ID列表
    'original_paths': List[str],       # 原始文件路径列表
    'embedding_dim': int,              # embedding维度
    'embedding_stats': {               # embedding统计信息
        'mean': float,
        'std': float,
        'min': float,
        'max': float
    }
}
```

## 注意事项

1. **兼容性**: 确保Python版本支持multiprocessing
2. **资源限制**: 注意系统内存和CPU限制
3. **文件锁**: 避免同时写入同一文件
4. **错误处理**: 单个进程错误不会影响其他进程
5. **结果一致性**: 多进程结果与单进程结果完全一致 
# Speaker Embedding Computation

根据utterance级别的embedding计算每个说话人的平均embedding。

## 🎯 功能说明

这个脚本的作用是：
1. 读取所有utterance级别的embedding文件
2. 按照说话人分组
3. 计算每个说话人的平均embedding
4. 保存到speakers目录

## 📁 目录结构

### 输入目录 (utterances)
```
utterances/
├── dataset1/
│   ├── speaker1/
│   │   ├── utterance1.pkl
│   │   ├── utterance2.pkl
│   │   └── ...
│   └── speaker2/
│       ├── utterance1.pkl
│       └── ...
└── dataset2/
    └── ...
```

### 输出目录 (speakers)
```
speakers/
├── dataset1/
│   ├── speaker1.pkl
│   ├── speaker2.pkl
│   └── ...
└── dataset2/
    └── ...
```

## 🚀 使用方法

### 1. 快速使用
```bash
./run_compute_speaker_embeddings.sh
```

### 2. 测试功能
```bash
python test_speaker_embeddings.py
```

### 3. 自定义参数
```bash
python compute_speaker_embeddings.py \
    --utterances_dir "/path/to/utterances" \
    --speakers_dir "/path/to/speakers" \
    --min_utterances 3 \
    --skip_existing
```

## 📊 数据格式

### Utterance Embedding文件格式
每个 `.pkl` 文件包含：
```python
{
    'embedding': numpy.array,     # 256维的embedding向量
    'dataset': str,               # 数据集名称
    'speaker_id': str,            # 说话人ID
    'utterance_id': str,          # 语音ID
    'original_path': str          # 原始音频文件路径
}
```

### Speaker Embedding文件格式
每个说话人的 `.pkl` 文件包含：
```python
{
    'embedding': numpy.array,           # 平均embedding向量
    'dataset': str,                     # 数据集名称
    'speaker_id': str,                  # 说话人ID
    'num_utterances': int,              # 该说话人的utterance数量
    'failed_utterances': int,           # 加载失败的utterance数量
    'utterance_list': list,             # 所有utterance的ID列表
    'original_paths': list,             # 所有原始音频文件路径
    'embedding_dim': int,               # embedding维度
    'embedding_stats': {                # embedding统计信息
        'mean': float,                  # 平均值
        'std': float,                   # 标准差
        'min': float,                   # 最小值
        'max': float                    # 最大值
    }
}
```

## ⚙️ 参数说明

- `--utterances_dir`: utterance embedding目录路径
- `--speakers_dir`: speaker embedding输出目录路径
- `--min_utterances`: 一个说话人至少需要的utterance数量（默认：1）
- `--skip_existing`: 跳过已存在的speaker embedding文件

## 📈 处理统计

脚本运行后会显示：
- 扫描到的数据集和说话人数量
- 处理的说话人数量和速度
- 跳过的已存在文件数量
- 错误统计
- 每个数据集的说话人数量
- 示例说话人的统计信息

## 💡 使用示例

### 基本使用
```bash
# 1. 确保utterance embeddings已经生成
ls /path/to/utterances/*/*/*.pkl | head -5

# 2. 运行speaker embedding计算
./run_compute_speaker_embeddings.sh

# 3. 检查结果
ls /path/to/speakers/*/*.pkl | head -5
```

### 测试验证
```bash
# 测试功能
python test_speaker_embeddings.py

# 查看某个speaker的信息
python -c "
import pickle
with open('/path/to/speakers/dataset/speaker.pkl', 'rb') as f:
    data = pickle.load(f)
print('Speaker:', data['speaker_id'])
print('Utterances:', data['num_utterances'])
print('Embedding shape:', data['embedding'].shape)
print('Embedding norm:', np.linalg.norm(data['embedding']))
"
```

### 大批量处理
```bash
# 设置最小utterance数量，过滤说话人
python compute_speaker_embeddings.py \
    --min_utterances 5 \
    --skip_existing
```

## 🔍 质量检查

### 检查speaker数量
```bash
echo "Total speakers:"
find /path/to/speakers -name "*.pkl" | wc -l

echo "Speakers by dataset:"
for dataset in $(ls /path/to/speakers); do
    count=$(find "/path/to/speakers/$dataset" -name "*.pkl" | wc -l)
    echo "  $dataset: $count speakers"
done
```

### 检查embedding质量
```bash
python -c "
import pickle
import numpy as np
import glob

speaker_files = glob.glob('/path/to/speakers/**/*.pkl', recursive=True)
norms = []

for file_path in speaker_files[:100]:  # 检查前100个
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    norms.append(np.linalg.norm(data['embedding']))

print(f'Embedding norms - Mean: {np.mean(norms):.4f}, Std: {np.std(norms):.4f}')
print(f'Embedding norms - Min: {np.min(norms):.4f}, Max: {np.max(norms):.4f}')
"
```

## ⚠️ 注意事项

1. **内存使用**: 对于大量utterances的说话人，会同时加载所有embedding到内存
2. **磁盘空间**: 确保有足够空间存储speaker embeddings
3. **数据完整性**: 检查utterance embeddings的完整性
4. **并发处理**: 脚本是单线程的，大数据集可能需要较长时间

## 🛠️ 故障排除

### 1. 找不到utterance files
```bash
# 检查路径是否正确
ls -la /path/to/utterances

# 检查文件权限
find /path/to/utterances -name "*.pkl" | head -5
```

### 2. 内存不足
```bash
# 减少batch处理或增加swap空间
# 可以分批处理大的数据集
```

### 3. 结果验证失败
```bash
# 检查individual embedding文件格式
python test_speaker_embeddings.py
```

## 📋 完整工作流程

```bash
# 1. 生成utterance embeddings (如果还没有)
./run_wespeaker_embedding_extraction_optimized.sh

# 2. 测试speaker embedding计算
python test_speaker_embeddings.py

# 3. 计算speaker embeddings
./run_compute_speaker_embeddings.sh

# 4. 验证结果
python -c "
import pickle
import glob
files = glob.glob('/path/to/speakers/**/*.pkl', recursive=True)
print(f'Generated {len(files)} speaker embeddings')
"
```

---

**总结**: 这个脚本将utterance级别的embeddings聚合为speaker级别的embeddings，为后续的说话人识别、聚类或相似度分析提供基础。 
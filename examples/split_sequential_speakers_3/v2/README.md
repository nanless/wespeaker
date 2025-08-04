# 说话人边界检测 - 多GPU流水线

这是一个用于大规模音频文件说话人边界检测的多GPU处理流水线。

## 🚀 使用流程

### 第一步：提取Embedding（多GPU）

从音频文件提取speaker embedding，保持与原音频文件相同的目录结构：

```bash
python extract_embeddings_multigpu.py \
    --input_dir /path/to/audio/files \
    --output_dir /path/to/embeddings/output \
    --model_dir /path/to/speaker/model \
    --gpus "0,1,2,3"
```

**参数说明**:
- `--input_dir`: 输入音频文件目录
- `--output_dir`: embedding输出目录  
- `--model_dir`: WeSpeaker模型目录
- `--gpus`: 使用的GPU列表，用逗号分隔（可选，默认使用所有GPU）
- `--port`: 分布式通信端口（可选，默认12355）

### 第二步：边界检测

基于提取的embedding进行说话人边界检测，支持两种算法：

#### 方法1：传统余弦相似度算法

```bash
python detect_boundaries_from_embeddings.py \
    --embeddings_dir /path/to/embeddings/output \
    --output_dir /path/to/boundary/results \
    --segment_size 1000 \
    --boundary_window 10
```

#### 方法2：混合高斯模型(GMM)算法 🧠

```bash
python detect_boundaries_from_embeddings.py \
    --embeddings_dir /path/to/embeddings/output \
    --output_dir /path/to/boundary/results \
    --segment_size 1000 \
    --boundary_window 10 \
    --use_gmm \
    --gmm_components 2
```

**参数说明**:
- `--embeddings_dir`: embedding文件目录（第一步的输出）
- `--output_dir`: 边界检测结果输出目录
- `--segment_size`: 每段预期文件数量（默认1000）
- `--boundary_window`: 边界搜索窗口大小（默认10）
- `--use_gmm`: 使用混合高斯模型进行边界检测
- `--gmm_components`: GMM模型的组件数量（默认2个聚类中心）
- `--debug`: 开启调试模式（可选）

**🎭 GMM算法优势**:
- **多聚类中心**: 每个说话人段用2个聚类中心建模，更好地捕捉说话人变化性
- **概率化评估**: 使用概率衡量边界音频与相邻说话人的契合度
- **自适应组件数**: 根据样本数量自动调整GMM组件数
- **鲁棒性**: GMM训练失败时自动回退到余弦相似度方法

## 📁 输出结构

### Embedding提取输出
```
embeddings_output/
├── dataset1/
│   ├── speaker1/
│   │   ├── audio1.pkl
│   │   └── audio2.pkl
│   └── speaker2/
│       ├── audio3.pkl
│       └── audio4.pkl
└── extraction_stats.json
```

### 边界检测输出
```
boundary_results/
├── speaker_001/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── speaker_002/
│   ├── audio3.wav
│   └── ...
├── speaker_boundary_detection_result.json
└── boundary_detection_visualization.png
```

## ⚡ 性能特点

- **多GPU并行**: 自动将音频文件分配到多个GPU并行处理
- **目录结构保持**: 输出embedding文件保持与输入音频相同的目录结构
- **内存优化**: 分批处理避免内存溢出
- **进度监控**: 实时显示处理进度和统计信息
- **错误处理**: 完善的异常处理和恢复机制

## 📊 处理效率

| GPU数量 | 理论加速比 | 适用场景 |
|---------|------------|----------|
| 2个GPU  | 1.4-1.8x   | 中等规模 (< 10K文件) |
| 4个GPU  | 2.8-3.6x   | 大规模 (10K-50K文件) |
| 8个GPU  | 5.6-7.2x   | 超大规模 (> 50K文件) |

## 🔍 示例用法

```bash
# 1. 提取embedding（使用4个GPU）
python extract_embeddings_multigpu.py \
    --input_dir /data/audio_files \
    --output_dir /data/embeddings \
    --model_dir /models/wespeaker_samresnet \
    --gpus "0,1,2,3"

# 2. 边界检测
python detect_boundaries_from_embeddings.py \
    --embeddings_dir /data/embeddings \
    --output_dir /data/speaker_segments \
    --segment_size 1000 \
    --boundary_window 10

# 3. 查看结果
ls /data/speaker_segments/
cat /data/speaker_segments/speaker_boundary_detection_result.json
```

## 🛡️ 系统要求

- **硬件**: 多个NVIDIA GPU，每个GPU至少6GB显存
- **软件**: PyTorch with CUDA, WeSpeaker, sklearn, tqdm
- **存储**: 足够空间存储embedding文件（约为原音频文件大小的1-5%）

## ⚠️ 注意事项

1. **GPU内存**: 确保每个GPU有足够显存加载模型
2. **文件路径**: 使用绝对路径避免路径问题
3. **模型兼容**: 确保使用的WeSpeaker模型版本兼容
4. **端口占用**: 如果端口被占用，可以使用`--port`参数指定其他端口

## 🔧 故障排除

```bash
# 检查GPU状态
nvidia-smi

# 检查PyTorch CUDA支持  
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 测试小规模数据
python extract_embeddings_multigpu.py --input_dir /small/test/data --output_dir /test/output --model_dir /models/wespeaker --gpus "0"
```
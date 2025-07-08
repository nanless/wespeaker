# WeSpeaker Embedding Extraction

仿照 3D-Speaker 的结构，为 WeSpeaker 模型创建的音频 embedding 提取脚本。

## 文件说明

1. **`extract_wespeaker_embeddings.py`** - 主提取脚本，支持多GPU并行处理
2. **`run_wespeaker_embedding_extraction.sh`** - 运行脚本，包含所有配置
3. **`test_embedding_extraction.py`** - 测试脚本，验证单个文件的embedding提取

## 使用方法

### 1. 快速开始（推荐）

直接运行主脚本：

```bash
./run_wespeaker_embedding_extraction.sh
```

### 2. 自定义参数运行

```bash
python extract_wespeaker_embeddings.py \
    --data_root "/path/to/your/audio/data" \
    --model_dir "/path/to/your/model" \
    --output_dir "/path/to/output/embeddings" \
    --gpus "0,1,2,3" \
    --port "12355"
```

### 3. 测试功能

```bash
python test_embedding_extraction.py
```

## 配置说明

### 默认配置路径：

- **数据目录**: `/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments`
- **模型目录**: `/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100`
- **输出目录**: `/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual`

### 参数说明：

- `--data_root`: 音频数据根目录，包含 `dataset_name/speaker_id/audio_files` 结构
- `--model_dir`: WeSpeaker 模型目录，包含 `avg_model.pt` 和 `config.yaml`
- `--output_dir`: embedding 输出目录
- `--gpus`: 使用的GPU ID列表，用逗号分隔（如 "0,1,2,3"）
- `--port`: 分布式训练端口号

## 数据结构

### 输入数据结构：
```
data_root/
├── dataset1/
│   ├── speaker1/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   └── speaker2/
│       ├── audio1.wav
│       └── ...
└── dataset2/
    └── ...
```

### 输出数据结构：
```
output_dir/
├── dataset1/
│   ├── speaker1/
│   │   ├── audio1.pkl
│   │   ├── audio2.pkl
│   │   └── ...
│   └── speaker2/
│       ├── audio1.pkl
│       └── ...
└── dataset2/
    └── ...
```

## Embedding 文件格式

每个 `.pkl` 文件包含以下信息：

```python
{
    'embedding': numpy.array,     # 256维的embedding向量
    'dataset': str,               # 数据集名称
    'speaker_id': str,            # 说话人ID
    'utterance_id': str,          # 语音ID
    'original_path': str          # 原始音频文件路径
}
```

## 系统要求

- Python 3.7+
- PyTorch (支持CUDA)
- WeSpeaker 框架
- 多GPU支持（可选）

## 性能优化

- 支持多GPU并行处理
- 分布式处理，自动分配任务到不同GPU
- 进度显示和错误统计
- 自动创建输出目录结构

## 使用示例

### 测试单个文件
```bash
python test_embedding_extraction.py
```

### 提取所有embedding
```bash
./run_wespeaker_embedding_extraction.sh
```

### 使用特定GPU
```bash
python extract_wespeaker_embeddings.py \
    --data_root "/your/data/path" \
    --model_dir "/your/model/path" \
    --output_dir "/your/output/path" \
    --gpus "0,1"
```

## 注意事项

1. 确保模型文件 `avg_model.pt` 和配置文件 `config.yaml` 存在
2. 确保有足够的磁盘空间存储embedding文件
3. 支持的音频格式：.wav, .flac, .mp3
4. 每个embedding为256维向量（根据模型配置）

## 故障排除

如果遇到问题，请：

1. 首先运行测试脚本确认基本功能
2. 检查模型和数据路径是否正确
3. 确认GPU可用性和内存
4. 查看错误日志信息 
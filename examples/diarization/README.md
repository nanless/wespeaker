# WeSpeaker Diarization Pipeline

本目录包含使用WeSpeaker进行说话人分离（diarization）的脚本，完全仿照3D-Speaker的diarization工作流程。

## 概述

Diarization流程包括：
1. **VAD (语音活动检测)**: 使用TenVad检测语音段
2. **VAD后处理**: 平滑和形态学填充
3. **VAD边界细化**: 基于能量的边界细化
4. **特征提取**: 在VAD segments上提取fbank特征（不做滑窗）
5. **Embedding提取**: 从segments提取说话人embeddings
6. **聚类**: 使用AHC（Agglomerative Hierarchical Clustering）聚类识别说话人
7. **段合并**: 合并来自同一说话人的连续段

## 文件

- `wespeaker/bin/infer_diarization.py`: 核心diarization推理脚本，支持多GPU
- `local/run_diarization_simple.py`: 简单的Python包装脚本，用于批量处理
- `run_diarization.sh`: Bash脚本包装器，带配置选项
- `path.sh`: 环境设置脚本
- `tools/parse_options.sh`: 命令行选项解析器

## 使用方法

### 方法1: 使用Python包装脚本（推荐）

```bash
cd examples/diarization
python3 local/run_diarization_simple.py \
    --src_dir /path/to/audio/files \
    --out_dir /path/to/output \
    --model_dir /path/to/wespeaker/model \
    --nprocs 8 \
    --out_type json
```

### 方法2: 使用bash脚本

```bash
cd examples/diarization
bash run_diarization.sh \
    --stage 1 --stop_stage 1 \
    --DATA_ROOT /path/to/audio/files \
    --MODEL_DIR /path/to/wespeaker/model \
    --OUTPUT_DIR /path/to/output \
    --GPUS "0,1,2,3,4,5,6,7" \
    --OUT_TYPE json
```

### 方法3: 直接使用infer_diarization.py

```bash
# 创建wav列表文件
echo "/path/to/audio1.wav" > wav_list.txt
echo "/path/to/audio2.wav" >> wav_list.txt

# 运行diarization
python3 wespeaker/bin/infer_diarization.py \
    --wav wav_list.txt \
    --out_dir /path/to/output \
    --model_dir /path/to/wespeaker/model
```

## 参数说明

### 核心参数

- `--src_dir` / `--DATA_ROOT`: 包含音频文件的源目录
- `--out_dir` / `--OUTPUT_DIR`: 结果输出目录
- `--model_dir` / `--MODEL_DIR`: 包含WeSpeaker模型文件的目录（必须包含 `avg_model.pt` 和 `config.yaml`）
- `--nprocs`: 进程数（默认：根据GPU数量自动检测）
- `--GPUS`: 逗号分隔的GPU ID列表（例如："0,1,2,3"）

### Diarization参数

- `--vad_threshold`: TenVad阈值（默认：0.5）
- `--vad_min_speech_ms`: VAD后处理：最小语音段时长（毫秒，默认：200.0）
- `--vad_max_silence_ms`: VAD后处理：最大静音间隙（毫秒，默认：300.0）
- `--vad_energy_threshold`: VAD能量阈值（默认：0.05）
- `--vad_boundary_expansion_ms`: VAD边界扩展（毫秒，默认：10.0）
- `--vad_boundary_energy_percentile`: VAD边界能量百分位（默认：10.0）
- `--cluster_fix_cos_thr`: AHC聚类固定余弦阈值（默认：0.3）
- `--cluster_mer_cos`: AHC聚类合并余弦阈值（默认：0.3）
- `--cluster_min_cluster_size`: AHC聚类最小簇大小（默认：0）
- `--batch_size`: Embedding提取的batch size（默认：64）
- `--out_type`: 输出格式 - `json`（默认）或 `rttm`
- `--speaker_num`: 已知的说话人数量（可选）
- `--pattern`: 要匹配的文件模式（默认："*.wav"）

## 输出格式

对于每个音频文件，pipeline会生成：

1. **`<filename>.json`** 或 **`<filename>.rttm`**: Diarization结果
   - JSON格式: `{"segid": {"start": float, "stop": float, "speaker": int}, ...}`
   - RTTM格式: 标准RTTM格式，用于评估工具

2. **`<filename>.meta.json`**: 元数据
   ```json
   {
     "wav_path": "/path/to/audio.wav",
     "duration_sec": 120.5,
     "processing_time_sec": 15.2,
     "rtf": 0.126,
     "num_segments": 45
   }
   ```

## 模型要求

模型目录必须包含：
- `avg_model.pt`: 模型checkpoint文件
- `config.yaml`: 模型配置文件

示例模型目录结构：
```
model_dir/
├── avg_model.pt
└── config.yaml
```

## 性能优化建议

1. **多GPU处理**: 设置 `--nprocs` 为可用GPU数量以进行并行处理
2. **Batch Size**: 如果有更多GPU内存，可以增加 `--batch_size`（默认：64）
3. **VAD参数**: 
   - 调整 `--vad_threshold` 以平衡召回率和精确率
   - 调整 `--vad_min_speech_ms` 和 `--vad_max_silence_ms` 以过滤短段和填充间隙
4. **聚类参数**:
   - `--cluster_fix_cos_thr`: 控制聚类的严格程度（值越大，聚类越严格）
   - `--cluster_mer_cos`: 控制相似说话人的合并（值越大，合并越少）

## 示例

```bash
# 处理目录中的所有WAV文件
python3 local/run_diarization_simple.py \
    --src_dir /data/audio \
    --out_dir /data/diarization_results \
    --model_dir /models/wespeaker/voxblink2_samresnet100 \
    --nprocs 8 \
    --vad_threshold 0.5 \
    --vad_min_speech_ms 200.0 \
    --vad_max_silence_ms 300.0 \
    --cluster_fix_cos_thr 0.3 \
    --cluster_mer_cos 0.3 \
    --batch_size 64 \
    --out_type json \
    --pattern "*.wav"
```

## 注意事项

- Pipeline自动使用TenVad进行VAD
- 多GPU处理会将文件分配到不同GPU进行并行处理
- 脚本自动跳过非音频文件
- 默认显示进度条（可使用 `--disable_progress_bar` 禁用）
- **不做滑窗**：直接在VAD segments上提取embedding，每个segment一个embedding
- **使用AHC聚类**：采用Agglomerative Hierarchical Clustering进行说话人聚类

## 故障排除

1. **模型未找到**: 确保 `avg_model.pt` 和 `config.yaml` 存在于模型目录中
2. **CUDA内存不足**: 减少 `--batch_size` 或使用更少的GPU
3. **未生成结果**: 检查音频文件是否有效且包含语音
4. **处理速度慢**: 增加 `--nprocs` 以进行更多并行处理，或增加 `--batch_size`（如果GPU内存允许）
5. **TenVad未安装**: 确保已安装ten_vad包，或将其路径添加到sys.path

## 与3D-Speaker的对比

本实现完全仿照3D-Speaker的diarization流程：
- ✅ 使用TenVad进行VAD（而非Silero VAD）
- ✅ VAD后处理（平滑+形态学填充）
- ✅ VAD边界细化（基于能量）
- ✅ 不做滑窗，直接在VAD segments上提取embedding
- ✅ 使用AHC聚类（而非UMAP或Spectral）
- ✅ 使用WeSpeaker声纹模型（而非3D-Speaker的模型）

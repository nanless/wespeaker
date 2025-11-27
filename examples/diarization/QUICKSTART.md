# 快速开始指南

## 基本使用

### 1. 简单Python脚本（推荐）

```bash
cd examples/diarization
python3 local/run_diarization_simple.py \
    --src_dir /path/to/audio \
    --out_dir /path/to/output \
    --model_dir /path/to/model
```

### 2. Bash脚本

```bash
cd examples/diarization
bash run_diarization.sh \
    --stage 1 --stop_stage 1 \
    --DATA_ROOT /path/to/audio \
    --MODEL_DIR /path/to/model \
    --OUTPUT_DIR /path/to/output
```

### 3. 直接使用推理脚本

```bash
# 创建wav列表
echo "/path/to/audio1.wav" > wav_list.txt
echo "/path/to/audio2.wav" >> wav_list.txt

# 运行
python3 wespeaker/bin/infer_diarization.py \
    --wav wav_list.txt \
    --out_dir /path/to/output \
    --model_dir /path/to/model
```

## 必需文件

- 模型目录必须包含：
  - `avg_model.pt`
  - `config.yaml`

## 输出

对于每个音频文件：
- `<filename>.json` 或 `<filename>.rttm`: Diarization结果
- `<filename>.meta.json`: 元数据（时长、RTF等）

## 常用选项

- `--nprocs N`: 进程数（默认：自动检测GPU）
- `--out_type json|rttm`: 输出格式
- `--batch_size N`: Batch size（默认：64）
- `--vad_threshold FLOAT`: TenVad阈值（默认：0.5）
- `--vad_min_speech_ms FLOAT`: 最小语音段时长（默认：200.0）
- `--vad_max_silence_ms FLOAT`: 最大静音间隙（默认：300.0）
- `--cluster_fix_cos_thr FLOAT`: AHC聚类阈值（默认：0.3）
- `--cluster_mer_cos FLOAT`: AHC合并阈值（默认：0.3）

完整文档请参见 README.md。

# 相似度Pairs音频样本提取工具

根据说话人相似度分析结果，提取最相似和最不相似的说话人pairs的音频样本，用于人工验证和分析。

## 🎯 功能说明

这个脚本的作用是：
1. 读取相似度分析结果 (`extreme_similarity_pairs.json`)
2. 提取最相似和最不相似的前N个说话人pairs
3. 为每个pair找到对应的音频文件
4. 复制每个说话人的音频样本到专门的目录
5. 创建包含相似度信息的目录名，方便人工检查

## 📁 输出目录结构

```
similarity_pairs_audio_samples/
├── most_similar_pairs/
│   ├── most_similar_rank001_speaker1_vs_speaker2_sim0.9876/
│   │   ├── speaker1_utterance1_01.wav
│   │   ├── speaker1_utterance2_02.wav
│   │   ├── speaker2_utterance1_01.wav
│   │   ├── speaker2_utterance2_02.wav
│   │   └── pair_info.json
│   ├── most_similar_rank002_speaker3_vs_speaker4_sim0.9834/
│   │   └── ...
│   └── ...
├── least_similar_pairs/
│   ├── least_similar_rank001_speaker5_vs_speaker6_sim0.1234/
│   │   ├── speaker5_utterance1_01.wav
│   │   ├── speaker5_utterance2_02.wav
│   │   ├── speaker6_utterance1_01.wav
│   │   ├── speaker6_utterance2_02.wav
│   │   └── pair_info.json
│   └── ...
└── extraction_summary.json
```

## 🚀 使用方法

### 1. 快速使用（推荐）
```bash
cd examples/extract_and_conclude_similarities/v2/
./run_extract_similarity_pairs_audio.sh
```

### 2. 自定义参数
```bash
python extract_similarity_pairs_audio.py \
    --embeddings_dir "/path/to/embeddings" \
    --audio_data_dir "/path/to/original/audio" \
    --similarities_subdir "speaker_similarity_analysis" \
    --utterances_subdir "embeddings_individual/utterances" \
    --output_dir "similarity_pairs_audio_samples" \
    --num_samples_per_speaker 2 \
    --top_pairs 50 \
    --audio_extensions .wav .flac .mp3
```

## ⚙️ 参数说明

- `--embeddings_dir`: 包含embedding和相似度结果的基础目录
- `--audio_data_dir`: 原始音频文件所在目录
- `--similarities_subdir`: 相似度结果子目录名 (默认: speaker_similarity_analysis)
- `--utterances_subdir`: utterance embeddings子目录名
- `--output_dir`: 音频样本输出目录名
- `--num_samples_per_speaker`: 每个说话人提取的音频样本数量 (默认: 2)
- `--top_pairs`: 提取的最相似和最不相似pairs数量 (默认: 50)
- `--audio_extensions`: 支持的音频文件扩展名

## 📋 前置条件

运行此脚本前，请确保已完成：

1. **Embedding提取**: 
   ```bash
   # 确保utterance embeddings已生成
   ls /path/to/embeddings/embeddings_individual/utterances/*/*/*.pkl
   ```

2. **相似度计算**: 
   ```bash
   # 确保相似度分析已完成
   ls /path/to/embeddings/speaker_similarity_analysis/extreme_similarity_pairs.json
   ```

3. **原始音频文件**: 
   ```bash
   # 确保原始音频文件存在
   ls /path/to/original/audio/*/*/*.wav
   ```

## 📊 输出文件说明

### pair_info.json
每个pair目录中的信息文件：
```json
{
  "pair_info": {
    "rank": 1,
    "speaker1": "speaker_id_1",
    "speaker2": "speaker_id_2", 
    "similarity": 0.9876
  },
  "speaker1": {
    "speaker_key": "speaker_id_1",
    "audio_files": ["speaker_id_1_utterance1_01.wav", "speaker_id_1_utterance2_02.wav"]
  },
  "speaker2": {
    "speaker_key": "speaker_id_2", 
    "audio_files": ["speaker_id_2_utterance1_01.wav", "speaker_id_2_utterance2_02.wav"]
  },
  "total_files": 4
}
```

### extraction_summary.json
整体提取总结：
```json
{
  "extraction_summary": {
    "total_pairs_processed": 100,
    "most_similar_pairs": {
      "success": 48,
      "errors": 2,
      "success_rate": "96.0%"
    },
    "least_similar_pairs": {
      "success": 47,
      "errors": 3,
      "success_rate": "94.0%"
    },
    "total_success": 95,
    "total_errors": 5
  }
}
```

## 🔍 验证和检查

### 检查提取结果
```bash
# 查看最相似pairs
ls -la /path/to/output/most_similar_pairs/ | head -10

# 查看最不相似pairs  
ls -la /path/to/output/least_similar_pairs/ | head -10

# 检查某个pair的详细信息
cat /path/to/output/most_similar_pairs/*/pair_info.json | head -1
```

### 播放音频样本
```bash
# 播放最相似的一对说话人的音频
PAIR_DIR="/path/to/output/most_similar_pairs/most_similar_rank001_*"
ls $PAIR_DIR/*.wav

# 使用音频播放器检查
# aplay $PAIR_DIR/*.wav  # Linux
# afplay $PAIR_DIR/*.wav # macOS
```

## 💡 使用建议

1. **先运行小规模测试**: 
   ```bash
   python extract_similarity_pairs_audio.py --top_pairs 5
   ```

2. **检查路径映射**: 确保原始音频文件路径能正确解析

3. **验证相似度结果**: 
   - 听取最相似pairs的音频，验证是否确实相似
   - 听取最不相似pairs的音频，验证是否确实不同

4. **批量处理**: 
   ```bash
   # 提取更多pairs进行详细分析
   python extract_similarity_pairs_audio.py --top_pairs 100
   ```

## 🔧 故障排除

### 常见问题

1. **音频文件未找到**:
   - 检查 `--audio_data_dir` 路径是否正确
   - 确认原始音频文件的目录结构
   - 检查embedding文件中的 `original_path` 字段

2. **相似度文件缺失**:
   ```bash
   # 确保相似度分析已完成
   python compute_speaker_similarities_fast.py
   ```

3. **Utterance文件缺失**:
   ```bash  
   # 确保embedding提取已完成
   python extract_wespeaker_embeddings.py
   ```

4. **权限问题**:
   ```bash
   chmod +x run_extract_similarity_pairs_audio.sh
   ```

### 调试模式
```bash
# 启用详细日志
python extract_similarity_pairs_audio.py --top_pairs 1 2>&1 | tee debug.log
```

## 📈 应用场景

1. **模型验证**: 验证说话人相似度计算的正确性
2. **数据分析**: 分析哪些说话人容易被混淆
3. **模型调优**: 为模型改进提供数据支持
4. **质量控制**: 检查数据集中的异常情况

## 🎵 音频分析建议

听取音频样本时，可以关注：
- **音色相似性**: 声音特征是否相似
- **语言风格**: 说话方式、语调等
- **录音质量**: 音质、噪声等技术因素
- **说话人特征**: 年龄、性别、口音等

通过人工验证，可以：
- 确认相似度计算的准确性
- 发现可能的数据问题
- 为模型改进提供方向 
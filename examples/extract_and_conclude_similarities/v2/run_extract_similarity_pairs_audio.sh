#!/bin/bash

# 提取最相似和最不相似说话人pairs的音频样本
# 用于人工验证和分析相似度计算结果

set -e

# 默认配置
EMBEDDINGS_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet"
AUDIO_DATA_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
SIMILARITIES_SUBDIR="speaker_similarity_analysis"
UTTERANCES_SUBDIR="embeddings_individual/utterances"
OUTPUT_DIR="similarity_pairs_audio_samples"
NUM_SAMPLES_PER_SPEAKER=2
TOP_PAIRS=1000
AUDIO_EXTENSIONS=".wav .flac .mp3"

# 打印配置信息
echo "==========================================="
echo "相似度Pairs音频样本提取"
echo "==========================================="
echo "Embeddings directory: $EMBEDDINGS_DIR"
echo "Audio data directory: $AUDIO_DATA_DIR"
echo "Similarities subdirectory: $SIMILARITIES_SUBDIR"
echo "Utterances subdirectory: $UTTERANCES_SUBDIR"
echo "Output directory: $OUTPUT_DIR"
echo "Samples per speaker: $NUM_SAMPLES_PER_SPEAKER"
echo "Top pairs to extract: $TOP_PAIRS"
echo "Audio extensions: $AUDIO_EXTENSIONS"
echo "==========================================="

# 检查必要目录
SIMILARITIES_DIR="$EMBEDDINGS_DIR/$SIMILARITIES_SUBDIR"
UTTERANCES_DIR="$EMBEDDINGS_DIR/$UTTERANCES_SUBDIR"

if [ ! -d "$SIMILARITIES_DIR" ]; then
    echo "❌ Error: Similarities directory not found: $SIMILARITIES_DIR"
    echo "💡 Please run the similarity computation first"
    exit 1
fi

if [ ! -d "$UTTERANCES_DIR" ]; then
    echo "❌ Error: Utterances directory not found: $UTTERANCES_DIR"
    echo "💡 Please run the embedding extraction first"
    exit 1
fi

if [ ! -d "$AUDIO_DATA_DIR" ]; then
    echo "❌ Error: Audio data directory not found: $AUDIO_DATA_DIR"
    echo "💡 Please check the audio data path"
    exit 1
fi

# 检查必要文件
EXTREME_PAIRS_FILE="$SIMILARITIES_DIR/extreme_similarity_pairs.json"
if [ ! -f "$EXTREME_PAIRS_FILE" ]; then
    echo "❌ Error: Extreme similarity pairs file not found: $EXTREME_PAIRS_FILE"
    echo "💡 Please run the similarity analysis first"
    exit 1
fi

echo "✅ All required directories and files found"
echo ""

# 运行音频提取脚本
echo "🚀 Starting audio extraction..."
python extract_similarity_pairs_audio.py \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --audio_data_dir "$AUDIO_DATA_DIR" \
    --similarities_subdir "$SIMILARITIES_SUBDIR" \
    --utterances_subdir "$UTTERANCES_SUBDIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples_per_speaker $NUM_SAMPLES_PER_SPEAKER \
    --top_pairs $TOP_PAIRS \
    --audio_extensions $AUDIO_EXTENSIONS

# 检查结果
OUTPUT_PATH="$EMBEDDINGS_DIR/$OUTPUT_DIR"
if [ -d "$OUTPUT_PATH" ]; then
    echo ""
    echo "🎉 Audio extraction completed!"
    echo "📁 Output directory: $OUTPUT_PATH"
    
    # 统计结果
    MOST_SIMILAR_DIR="$OUTPUT_PATH/most_similar_pairs"
    LEAST_SIMILAR_DIR="$OUTPUT_PATH/least_similar_pairs"
    
    if [ -d "$MOST_SIMILAR_DIR" ]; then
        MOST_COUNT=$(find "$MOST_SIMILAR_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "📊 Most similar pairs extracted: $MOST_COUNT"
        
        # 显示前几个样本目录
        echo "📂 Sample most similar pairs:"
        find "$MOST_SIMILAR_DIR" -mindepth 1 -maxdepth 1 -type d | head -3 | while read dir; do
            echo "  $(basename "$dir")"
            audio_count=$(find "$dir" -name "*.wav" -o -name "*.flac" -o -name "*.mp3" | wc -l)
            echo "    Audio files: $audio_count"
        done
    fi
    
    if [ -d "$LEAST_SIMILAR_DIR" ]; then
        LEAST_COUNT=$(find "$LEAST_SIMILAR_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "📊 Least similar pairs extracted: $LEAST_COUNT"
        
        # 显示前几个样本目录
        echo "📂 Sample least similar pairs:"
        find "$LEAST_SIMILAR_DIR" -mindepth 1 -maxdepth 1 -type d | head -3 | while read dir; do
            echo "  $(basename "$dir")"
            audio_count=$(find "$dir" -name "*.wav" -o -name "*.flac" -o -name "*.mp3" | wc -l)
            echo "    Audio files: $audio_count"
        done
    fi
    
    # 显示提取总结
    SUMMARY_FILE="$OUTPUT_PATH/extraction_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        echo ""
        echo "📋 Extraction Summary:"
        python -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)

summary = data['extraction_summary']
print(f\"  Total pairs processed: {summary['total_pairs_processed']}\")
print(f\"  Most similar - Success: {summary['most_similar_pairs']['success']}, Errors: {summary['most_similar_pairs']['errors']}\")
print(f\"  Least similar - Success: {summary['least_similar_pairs']['success']}, Errors: {summary['least_similar_pairs']['errors']}\")
print(f\"  Overall success rate: {summary['total_success']}/{summary['total_success'] + summary['total_errors']} ({summary['total_success']/(summary['total_success'] + summary['total_errors'])*100:.1f}%)\")
"
    fi
    
    echo ""
    echo "💡 Usage instructions:"
    echo "  1. Navigate to: $OUTPUT_PATH"
    echo "  2. Check 'most_similar_pairs' and 'least_similar_pairs' directories"
    echo "  3. Each pair directory contains:"
    echo "     - Audio files from both speakers"
    echo "     - pair_info.json with detailed information"
    echo "  4. Directory names include similarity scores for easy identification"
    echo ""
    echo "🔍 Example commands to explore results:"
    echo "  # List all pair directories"
    echo "  ls -la \"$OUTPUT_PATH/most_similar_pairs/\" | head -10"
    echo "  ls -la \"$OUTPUT_PATH/least_similar_pairs/\" | head -10"
    echo ""
    echo "  # Check a specific pair"
    echo "  ls -la \"$OUTPUT_PATH/most_similar_pairs/\"*/ | head -1"
    echo "  cat \"$OUTPUT_PATH/most_similar_pairs/\"*/pair_info.json | head -1"
    
else
    echo "❌ Error: Output directory was not created"
    exit 1
fi

echo ""
echo "✅ All done! You can now manually inspect the audio samples to verify similarity computation results." 
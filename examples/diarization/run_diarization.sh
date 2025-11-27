#!/bin/bash

set -e
. ./path.sh || exit 1

# 配置
DATA_ROOT="${DATA_ROOT:-/root/code/own/download_next_online_audio_for_speakerdetection_1125/original_audios_SC_CausalMelBandRNN_EDA_16k_resume3_variable_length_narrowgap_E0001_B030000}"
MODEL_DIR="${MODEL_DIR:-/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"  # 可用GPU

# Diarization参数（仿照3D-Speaker流程）
VAD_THRESHOLD="${VAD_THRESHOLD:-0.5}"
VAD_MIN_SPEECH_MS="${VAD_MIN_SPEECH_MS:-200.0}"
VAD_MAX_SILENCE_MS="${VAD_MAX_SILENCE_MS:-300.0}"
VAD_ENERGY_THRESHOLD="${VAD_ENERGY_THRESHOLD:-0.05}"
VAD_BOUNDARY_EXPANSION_MS="${VAD_BOUNDARY_EXPANSION_MS:-10.0}"
CLUSTER_FIX_COS_THR="${CLUSTER_FIX_COS_THR:-0.55}"
CLUSTER_MER_COS="${CLUSTER_MER_COS:-0.55}"
CLUSTER_MIN_CLUSTER_SIZE="${CLUSTER_MIN_CLUSTER_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
OUT_TYPE="${OUT_TYPE:-json}"  # json or rttm
PATTERN="${PATTERN:-*_speech_estimate.wav}"

# Parse command line arguments
stage=1
stop_stage=1

. tools/parse_options.sh || exit 1

echo "=== WeSpeaker Diarization Pipeline（仿照3D-Speaker流程）==="
echo "数据根目录: $DATA_ROOT"
echo "模型目录: $MODEL_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU: $GPUS"
echo "文件模式: $PATTERN"
echo ""
echo "VAD参数:"
echo "  - VAD阈值: $VAD_THRESHOLD"
echo "  - 最小语音段时长: $VAD_MIN_SPEECH_MS ms"
echo "  - 最大静音间隙: $VAD_MAX_SILENCE_MS ms"
echo "  - 能量阈值: $VAD_ENERGY_THRESHOLD"
echo "  - 边界扩展: $VAD_BOUNDARY_EXPANSION_MS ms"
echo ""
echo "聚类参数:"
echo "  - 固定余弦阈值: $CLUSTER_FIX_COS_THR"
echo "  - 合并余弦阈值: $CLUSTER_MER_COS"
echo "  - 最小簇大小: $CLUSTER_MIN_CLUSTER_SIZE"
echo ""
echo "其他参数:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - 输出格式: $OUT_TYPE"
echo "  - Stage: $stage"
echo "  - Stop stage: $stop_stage"
echo "======================================"

# 检查模型是否存在
if [ ! -f "$MODEL_DIR/avg_model.pt" ]; then
    echo "错误: 在 $MODEL_DIR/avg_model.pt 找不到模型文件"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$MODEL_DIR/config.yaml" ]; then
    echo "错误: 在 $MODEL_DIR/config.yaml 找不到配置文件"
    exit 1
fi

# 检查数据目录是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "错误: 在 $DATA_ROOT 找不到数据目录"
    exit 1
fi

# Set output directory if not provided
if [ -z "$OUTPUT_DIR" ]; then
    base_name=$(basename "$DATA_ROOT")"_wespeaker_diarization"
    OUTPUT_DIR=$(dirname "$DATA_ROOT")/$base_name
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "阶段 1: 在音频文件上运行diarization..."
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 统计音频文件
    echo "正在统计音频文件..."
    audio_count=$(find "$DATA_ROOT" -name "$PATTERN" | wc -l)
    echo "找到 $audio_count 个匹配模式 $PATTERN 的音频文件"
    
    if [ $audio_count -eq 0 ]; then
        echo "错误: 在 $DATA_ROOT 中未找到匹配模式 $PATTERN 的音频文件"
        exit 1
    fi
    
    # 设置GPU环境
    export CUDA_VISIBLE_DEVICES="$GPUS"
    
    # 运行diarization
    echo "正在启动diarization..."
    python3 local/run_diarization_simple.py \
        --src_dir "$DATA_ROOT" \
        --out_dir "$OUTPUT_DIR" \
        --model_dir "$MODEL_DIR" \
        --pattern "$PATTERN" \
        --vad_threshold "$VAD_THRESHOLD" \
        --vad_min_speech_ms "$VAD_MIN_SPEECH_MS" \
        --vad_max_silence_ms "$VAD_MAX_SILENCE_MS" \
        --vad_energy_threshold "$VAD_ENERGY_THRESHOLD" \
        --vad_boundary_expansion_ms "$VAD_BOUNDARY_EXPANSION_MS" \
        --cluster_fix_cos_thr "$CLUSTER_FIX_COS_THR" \
        --cluster_mer_cos "$CLUSTER_MER_COS" \
        --cluster_min_cluster_size "$CLUSTER_MIN_CLUSTER_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --out_type "$OUT_TYPE"
    
    echo "阶段 1 完成。"
    
    # 检查结果
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Diarization结果已保存到: $OUTPUT_DIR"
        
        # 统计输出文件
        result_count=$(find "$OUTPUT_DIR" -name "*.$OUT_TYPE" | wc -l)
        meta_count=$(find "$OUTPUT_DIR" -name "*.meta.json" | wc -l)
        
        echo "Diarization结果总数: $result_count"
        echo "元数据文件总数: $meta_count"
        
        if [ $result_count -gt 0 ]; then
            echo ""
            echo "✓ Diarization成功完成！"
            echo "✓ 结果已保存到: $OUTPUT_DIR"
            echo ""
            echo "输出文件类型:"
            echo "  - <filename>.$OUT_TYPE: Diarization结果"
            echo "  - <filename>.meta.json: 元数据（时长、处理时间、RTF、成对相似度统计）"
            echo "  - <filename>.pairs.json: 成对相似度详情"
            echo "  - <filename>.vad.png: VAD可视化图片（raw/processed/refined三层）"
            echo "  - <filename>.vad_masked.wav: VAD掩码后的音频"
            echo "  - <filename>.vad_info.json: VAD详细信息（raw/processed/refined）"
        else
            echo ""
            echo "⚠ 警告: 未生成diarization结果。"
        fi
    else
        echo "错误: 未创建输出目录。"
        exit 1
    fi
fi

echo "完成。"


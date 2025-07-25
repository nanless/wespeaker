#!/bin/bash

set -e
. ./path.sh || exit 1

# Configuration
DATA_ROOT="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
MODEL_DIR="/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"
OUTPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual"
MASTER_PORT=29503
GPUS="0,1,2,3"  # Available GPUs (comma-separated for this script)

# Parse command line arguments
stage=1
stop_stage=1

. tools/parse_options.sh || exit 1

echo "=== WeSpeaker Embedding Extraction Pipeline ==="
echo "Data root: $DATA_ROOT"
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Master port: $MASTER_PORT"
echo "======================================"

# Check if model exists
if [ ! -f "$MODEL_DIR/avg_model.pt" ]; then
    echo "Error: Model file not found at $MODEL_DIR/avg_model.pt"
    exit 1
fi

# Check if config exists
if [ ! -f "$MODEL_DIR/config.yaml" ]; then
    echo "Error: Config file not found at $MODEL_DIR/config.yaml"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory not found at $DATA_ROOT"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Extracting embeddings using multi-GPU processing..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Count number of audio files
    echo "Counting audio files..."
    total_files=$(find "$DATA_ROOT" -name "*.wav" -o -name "*.flac" -o -name "*.mp3" | wc -l)
    echo "Total audio files to process: $total_files"
    
    # Run embedding extraction
    echo "Starting embedding extraction..."
    python extract_wespeaker_embeddings.py \
        --data_root "$DATA_ROOT" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --gpus "$GPUS" \
        --port "$MASTER_PORT"
    
    echo "Stage 1 completed."
    
    # Check results
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Individual embedding files saved to: $OUTPUT_DIR"
        
        # Count extracted embeddings
        extracted_count=$(find "$OUTPUT_DIR" -name "*.pkl" | wc -l)
        echo "Total embeddings extracted: $extracted_count"
        
        # Show directory structure sample
        echo ""
        echo "Directory structure (first few examples):"
        find "$OUTPUT_DIR" -name "*.pkl" | head -5 | while read file; do
            echo "  $file"
        done
        
        # Show some statistics
        echo ""
        echo "Dataset statistics:"
        for dataset in $(ls "$OUTPUT_DIR" 2>/dev/null); do
            if [ -d "$OUTPUT_DIR/$dataset" ]; then
                dataset_count=$(find "$OUTPUT_DIR/$dataset" -name "*.pkl" | wc -l)
                speaker_count=$(find "$OUTPUT_DIR/$dataset" -type d -mindepth 1 | wc -l)
                echo "  $dataset: $dataset_count embeddings from $speaker_count speakers"
            fi
        done
        
        if [ $extracted_count -gt 0 ]; then
            echo ""
            echo "✓ Embedding extraction completed successfully!"
            echo "✓ Results saved in: $OUTPUT_DIR"
        else
            echo ""
            echo "⚠ Warning: No embeddings were extracted. Please check the logs for errors."
        fi
    else
        echo "Error: Output directory was not created."
        exit 1
    fi
fi

echo "Done." 
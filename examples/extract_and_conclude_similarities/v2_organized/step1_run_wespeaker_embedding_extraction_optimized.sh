#!/bin/bash

set -e
. ./path.sh || exit 1

# Configuration - Optimized version
DATA_ROOT="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments_20250808/merged_datasets_20250610_vad_segments/audio"
MODEL_DIR="/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"
OUTPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments_20250808/merged_datasets_20250610_vad_segments/audioembeddings_wespeaker_samresnet100/embeddings_utterances"
MASTER_PORT=29503
GPUS="0,1,2,3,4,5,6,7"  # Available GPUs (8 cards)

# Optimization parameters
BATCH_SIZE=24        # Increase batch size for better GPU utilization
NUM_WORKERS=6        # I/O worker threads per GPU
SKIP_EXISTING=true   # Skip files that already have embeddings
RANDOM_SHUFFLE=true  # Randomly shuffle files for better load balancing
RANDOM_SEED=42       # Random seed for reproducible shuffling

# Parse command line arguments
stage=1
stop_stage=1

. tools/parse_options.sh || exit 1

echo "=== WeSpeaker Embedding Extraction Pipeline (Optimized) ==="
echo "Data root: $DATA_ROOT"
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Master port: $MASTER_PORT"
echo "Batch size: $BATCH_SIZE"
echo "I/O workers per GPU: $NUM_WORKERS"
echo "Skip existing files: $SKIP_EXISTING"
echo "Random shuffle for load balancing: $RANDOM_SHUFFLE"
echo "Random seed: $RANDOM_SEED"
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
    echo "Stage 1: Extracting embeddings using optimized multi-GPU processing..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Quick count of audio files (sample-based estimate for speed)
    echo "Estimating total audio files..."
    sample_count=$(find "$DATA_ROOT" -name "*.wav" | head -1000 | wc -l)
    if [ $sample_count -eq 1000 ]; then
        total_estimate=$(find "$DATA_ROOT" -name "*.wav" -o -name "*.flac" -o -name "*.mp3" | wc -l)
        echo "Estimated total audio files: $total_estimate"
    else
        echo "Total audio files found: $sample_count"
    fi
    
    # Run optimized embedding extraction
    echo "Starting optimized embedding extraction..."
    echo "Optimization features:"
    echo "  - Batch processing with size $BATCH_SIZE"
    echo "  - Parallel I/O with $NUM_WORKERS workers per GPU"
    echo "  - Skip existing embeddings: $SKIP_EXISTING"
    echo "  - Random shuffle for load balancing: $RANDOM_SHUFFLE"
    echo "  - Multi-threaded file scanning"
    
    # Add memory optimization environment variables
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export OMP_NUM_THREADS=2
    
    python local/extract_wespeaker_embeddings_optimized.py \
        --data_root "$DATA_ROOT" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --gpus "$GPUS" \
        --port "$MASTER_PORT" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --random_seed "$RANDOM_SEED" \
        $([ "$SKIP_EXISTING" = true ] && echo "--skip_existing") \
        $([ "$RANDOM_SHUFFLE" = true ] && echo "--random_shuffle")
    
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
        for dataset in $(ls "$OUTPUT_DIR" 2>/dev/null | head -10); do
            if [ -d "$OUTPUT_DIR/$dataset" ]; then
                dataset_count=$(find "$OUTPUT_DIR/$dataset" -name "*.pkl" | wc -l)
                speaker_count=$(find "$OUTPUT_DIR/$dataset" -type d -mindepth 1 | wc -l)
                echo "  $dataset: $dataset_count embeddings from $speaker_count speakers"
            fi
        done
        
        if [ $extracted_count -gt 0 ]; then
            echo ""
            echo "✓ Optimized embedding extraction completed successfully!"
            echo "✓ Results saved in: $OUTPUT_DIR"
            echo ""
            echo "Performance tips for next run:"
            echo "  - The script will automatically skip existing embeddings"
            echo "  - You can increase batch_size for faster processing if you have more GPU memory"
            echo "  - You can adjust num_workers based on your I/O performance"
        else
            echo ""
            echo "⚠ Warning: No new embeddings were extracted."
            echo "  This might be because all files were already processed (skip_existing=true)"
            echo "  Use --skip_existing=false to reprocess all files"
        fi
    else
        echo "Error: Output directory was not created."
        exit 1
    fi
fi

echo "Done." 
#!/bin/bash

# Utterance Similarity Computation Per Speaker Script
# This script computes similarity between all utterances within each speaker using optimized multiprocessing

set -e

# Configuration - matching step1, step2, and step3 paths
EMBEDDINGS_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_wespeaker_samresnet100"
UTTERANCES_SUBDIR="embeddings_utterances"
OUTPUT_SUBDIR="utterance_similarities_per_speaker"
NUM_WORKERS=1
NUM_WORKERS_INTERNAL=64
BATCH_SIZE=10
MIN_UTTERANCES=2
SKIP_EXISTING=true
MAX_SPEAKERS=
MAX_UTTERANCES=
MAX_UTTERANCES_LIMIT=5000
SIMILARITY_THRESHOLD=0.7

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Utterance Similarity Computation Per Speaker ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  📂 Embeddings directory: ${EMBEDDINGS_DIR}"
echo -e "  📁 Utterances subdirectory: ${UTTERANCES_SUBDIR}"
echo -e "  📁 Output subdirectory: ${OUTPUT_SUBDIR}"
echo -e "  ⚡ Number of workers (speakers): ${NUM_WORKERS}"
if [ -n "$NUM_WORKERS_INTERNAL" ]; then
    echo -e "  ⚡ Number of workers (internal): ${NUM_WORKERS_INTERNAL}"
else
    echo -e "  ⚡ Number of workers (internal): auto (min(8, cpu_count))"
fi
echo -e "  📦 Batch size: ${BATCH_SIZE}"
echo -e "  🔢 Minimum utterances: ${MIN_UTTERANCES}"
echo -e "  ⏭️  Skip existing: ${SKIP_EXISTING}"
if [ -n "$MAX_UTTERANCES" ]; then
    echo -e "  🔢 Max utterances per speaker: ${MAX_UTTERANCES}"
else
    echo -e "  🔢 Max utterances per speaker: no limit"
fi
if [ -n "$MAX_SPEAKERS" ]; then
    echo -e "  🔢 Max speakers: ${MAX_SPEAKERS}"
fi
echo -e "  🔢 Max utterances limit: ${MAX_UTTERANCES_LIMIT} (speakers with more will be skipped, matrix too large)"
echo -e "  📊 Similarity threshold: ${SIMILARITY_THRESHOLD} (only pairs >= threshold will be saved)"
echo -e "${BLUE}===============================================${NC}"

# Check if embeddings directory exists
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo -e "${RED}❌ Error: Embeddings directory does not exist: $EMBEDDINGS_DIR${NC}"
    exit 1
fi

# Check if utterances subdirectory exists
UTTERANCES_FULL_PATH="$EMBEDDINGS_DIR/$UTTERANCES_SUBDIR"
if [ ! -d "$UTTERANCES_FULL_PATH" ]; then
    echo -e "${RED}❌ Error: Utterances directory does not exist: $UTTERANCES_FULL_PATH${NC}"
    echo -e "${YELLOW}💡 Hint: Run step1 first to extract utterance embeddings${NC}"
    exit 1
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/local/compute_utterance_similarities_per_speaker.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ Error: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Print system information
echo -e "${BLUE}💻 System Information:${NC}"
echo -e "  CPU cores: $(nproc)"
echo -e "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo -e "  Python version: $(python3 --version)"

# Get initial statistics
echo -e "${BLUE}📊 Initial Statistics:${NC}"
if [ -d "$UTTERANCES_FULL_PATH" ]; then
    TOTAL_UTTERANCES=$(find "$UTTERANCES_FULL_PATH" -name "*.pkl" | wc -l)
    TOTAL_SPEAKERS=$(find "$UTTERANCES_FULL_PATH" -mindepth 2 -maxdepth 2 -type d | wc -l)
    echo -e "  🎤 Total utterance files: ${TOTAL_UTTERANCES}"
    echo -e "  👥 Total speakers: ${TOTAL_SPEAKERS}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$UTTERANCES_FULL_PATH"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            speaker_count=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
            utterance_count=$(find "$dataset_dir" -name "*.pkl" | wc -l)
            echo -e "    ${dataset_name}: ${speaker_count} speakers, ${utterance_count} utterances"
        fi
    done
else
    echo -e "  ${YELLOW}⚠️  Cannot access utterances directory${NC}"
fi

# Check if output directory already exists
OUTPUT_FULL_PATH="$EMBEDDINGS_DIR/$OUTPUT_SUBDIR"
if [ -d "$OUTPUT_FULL_PATH" ]; then
    EXISTING_RESULTS=$(find "$OUTPUT_FULL_PATH" -name "*_utterance_similarities.json" | wc -l)
    echo -e "  📊 Existing similarity results: ${EXISTING_RESULTS} files"
else
    echo -e "  📊 Existing similarity results: 0 files"
fi

# Start computation
echo -e "${GREEN}🚀 Starting utterance similarity computation per speaker...${NC}"
echo -e "${GREEN}⏰ Start time: $(date)${NC}"

START_TIME=$(date +%s)

# Build command
CMD_ARGS=(
    "--embeddings_dir" "$EMBEDDINGS_DIR"
    "--utterances_subdir" "$UTTERANCES_SUBDIR"
    "--output_subdir" "$OUTPUT_SUBDIR"
    "--num_workers" "$NUM_WORKERS"
    "--batch_size" "$BATCH_SIZE"
    "--min_utterances" "$MIN_UTTERANCES"
)

if [ -n "$MAX_UTTERANCES" ]; then
    CMD_ARGS+=("--max_utterances" "$MAX_UTTERANCES")
fi

if [ -n "$NUM_WORKERS_INTERNAL" ]; then
    CMD_ARGS+=("--num_workers_internal" "$NUM_WORKERS_INTERNAL")
fi

if [ "$SKIP_EXISTING" = true ]; then
    CMD_ARGS+=("--skip_existing")
fi

CMD_ARGS+=("--similarity_threshold" "$SIMILARITY_THRESHOLD")
CMD_ARGS+=("--max_utterances_limit" "$MAX_UTTERANCES_LIMIT")

if [ -n "$MAX_SPEAKERS" ]; then
    CMD_ARGS+=("--max_speakers" "$MAX_SPEAKERS")
fi

# Set environment variables for better performance
export PYTHONPATH=../../../:$PYTHONPATH
export PYTHONIOENCODING=UTF-8

# Run the Python script
echo -e "${YELLOW}⚡ Running utterance similarity computation with optimized multiprocessing...${NC}"
python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ Utterance similarity computation completed!${NC}"
echo -e "${GREEN}⏰ End time: $(date)${NC}"
echo -e "${GREEN}⏱️  Total execution time: ${EXECUTION_TIME} seconds ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Final statistics
echo -e "${BLUE}📈 Final Statistics:${NC}"
if [ -d "$OUTPUT_FULL_PATH" ]; then
    FINAL_RESULTS=$(find "$OUTPUT_FULL_PATH" -name "*_utterance_similarities.json" | wc -l)
    echo -e "  📊 Total similarity result files: ${FINAL_RESULTS}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$OUTPUT_FULL_PATH"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            json_count=$(find "$dataset_dir" -name "*_utterance_similarities.json" | wc -l)
            echo -e "    ${dataset_name}: ${json_count} similarity files"
        fi
    done
    
    # Calculate processing rate
    if [ $EXECUTION_TIME -gt 0 ] && [ $FINAL_RESULTS -gt 0 ]; then
        RATE=$(echo "scale=2; $FINAL_RESULTS / $EXECUTION_TIME" | bc -l)
        echo -e "  🚀 Processing rate: ${RATE} speakers/second"
    fi
    
    # Show sample output structure
    echo -e "  🔍 Sample output structure:"
    SAMPLE_FILE=$(find "$OUTPUT_FULL_PATH" -name "*_utterance_similarities.json" | head -1)
    if [ -n "$SAMPLE_FILE" ]; then
        dataset_name=$(basename "$(dirname "$SAMPLE_FILE")")
        filename=$(basename "$SAMPLE_FILE")
        speaker_id=${filename%_utterance_similarities.json}
        echo -e "    📄 ${dataset_name}/${filename}"
        
        # Show file size
        file_size=$(du -h "$SAMPLE_FILE" | cut -f1)
        echo -e "    📦 File size: ${file_size}"
        
        # Show basic info from JSON (if jq is available)
        if command -v jq &> /dev/null; then
            num_utterances=$(jq -r '.num_utterances' "$SAMPLE_FILE" 2>/dev/null || echo "unknown")
            mean_sim=$(jq -r '.statistics.mean' "$SAMPLE_FILE" 2>/dev/null || echo "unknown")
            echo -e "    📊 Utterances: ${num_utterances}, Mean similarity: ${mean_sim}"
        fi
    fi
else
    echo -e "  ${RED}❌ Cannot access output directory${NC}"
fi

# Show disk usage
echo -e "${BLUE}💾 Disk Usage:${NC}"
if [ -d "$OUTPUT_FULL_PATH" ]; then
    OUTPUT_SIZE=$(du -sh "$OUTPUT_FULL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "  Output directory: ${OUTPUT_SIZE}"
fi

# Show next steps
echo -e "${YELLOW}🔄 Next Steps:${NC}"
echo -e "  • Check similarity results in: $OUTPUT_FULL_PATH"
echo -e "  • Each dataset directory contains: {speaker_id}_utterance_similarities.json files"
echo -e "  • JSON file includes: similarity matrix, pairs, and statistics"
echo -e "  • Use the similarity data for speaker consistency analysis"

# Performance tips
echo -e "${YELLOW}💡 Performance Tips:${NC}"
echo -e "  • Adjust --num_workers based on your CPU cores and memory"
echo -e "  • Use --batch_size to balance load distribution"
echo -e "  • Use --skip_existing to resume from previous progress"
echo -e "  • Use --max_speakers for testing with smaller datasets"
echo -e "  • Adjust --min_utterances to filter speakers with too few utterances"

echo -e "${GREEN}🎉 Utterance similarity computation per speaker finished!${NC}"
echo -e "${GREEN}📊 Results available in: $OUTPUT_FULL_PATH${NC}"


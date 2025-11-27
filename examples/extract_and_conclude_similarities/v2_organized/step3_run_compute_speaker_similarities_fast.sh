#!/bin/bash

# Fast Speaker Similarity Computation Script
# This script computes speaker-to-speaker similarities using optimized multiprocessing

set -e

# Configuration - matching step1 and step2 paths
EMBEDDINGS_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/embeddings_wespeaker_samresnet100"
UTTERANCES_SUBDIR="embeddings_utterances"
SPEAKERS_OUTPUT_SUBDIR="embeddings_speakers"
SIMILARITIES_OUTPUT_SUBDIR="speaker_similarity_analysis"
NUM_WORKERS=32
BATCH_SIZE=100
TOP_K=100
SKIP_SIMILARITY=false
RESUME=false
MAX_SPEAKERS=

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Fast Speaker Similarity Computation ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  📂 Embeddings directory: ${EMBEDDINGS_DIR}"
echo -e "  📁 Utterances subdirectory: ${UTTERANCES_SUBDIR}"
echo -e "  📁 Speakers output subdirectory: ${SPEAKERS_OUTPUT_SUBDIR}"
echo -e "  📁 Similarities output subdirectory: ${SIMILARITIES_OUTPUT_SUBDIR}"
echo -e "  ⚡ Number of workers: ${NUM_WORKERS}"
echo -e "  📦 Batch size: ${BATCH_SIZE}"
echo -e "  🔝 Top-K similar speakers: ${TOP_K}"
echo -e "  ⏭️  Skip similarity computation: ${SKIP_SIMILARITY}"
echo -e "  🔄 Resume from previous progress: ${RESUME}"
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
    exit 1
fi

# Check if speakers subdirectory exists
SPEAKERS_FULL_PATH="$EMBEDDINGS_DIR/$SPEAKERS_OUTPUT_SUBDIR"
if [ ! -d "$SPEAKERS_FULL_PATH" ]; then
    echo -e "${RED}❌ Error: Speakers directory does not exist: $SPEAKERS_FULL_PATH${NC}"
    echo -e "${YELLOW}💡 Hint: Run step2 first to compute speaker embeddings${NC}"
    exit 1
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/local/compute_speaker_similarities_fast.py"
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
    echo -e "  🎤 Total utterance files: ${TOTAL_UTTERANCES}"
else
    echo -e "  ${YELLOW}⚠️  Cannot access utterances directory${NC}"
fi

if [ -d "$SPEAKERS_FULL_PATH" ]; then
    TOTAL_SPEAKERS=$(find "$SPEAKERS_FULL_PATH" -name "*.pkl" | wc -l)
    echo -e "  👥 Total speaker embeddings: ${TOTAL_SPEAKERS}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$SPEAKERS_FULL_PATH"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            speaker_count=$(find "$dataset_dir" -name "*.pkl" | wc -l)
            echo -e "    ${dataset_name}: ${speaker_count} speakers"
        fi
    done
else
    echo -e "  ${RED}❌ Cannot access speakers directory${NC}"
fi

# Check if similarities output directory already exists
SIMILARITIES_FULL_PATH="$EMBEDDINGS_DIR/$SIMILARITIES_OUTPUT_SUBDIR"
if [ -d "$SIMILARITIES_FULL_PATH" ]; then
    EXISTING_RESULTS=$(find "$SIMILARITIES_FULL_PATH" -name "*.json" -o -name "*.npy" | wc -l)
    echo -e "  📊 Existing similarity results: ${EXISTING_RESULTS} files"
else
    echo -e "  📊 Existing similarity results: 0 files"
fi

# Start computation
echo -e "${GREEN}🚀 Starting fast speaker similarity computation...${NC}"
echo -e "${GREEN}⏰ Start time: $(date)${NC}"

START_TIME=$(date +%s)

# Build command
CMD_ARGS=(
    "--embeddings_dir" "$EMBEDDINGS_DIR"
    "--utterances_subdir" "$UTTERANCES_SUBDIR"
    "--speakers_output_subdir" "$SPEAKERS_OUTPUT_SUBDIR"
    "--similarities_output_subdir" "$SIMILARITIES_OUTPUT_SUBDIR"
    "--num_workers" "$NUM_WORKERS"
    "--batch_size" "$BATCH_SIZE"
    "--top_k" "$TOP_K"
)

if [ "$SKIP_SIMILARITY" = true ]; then
    CMD_ARGS+=("--skip_similarity")
fi

if [ "$RESUME" = true ]; then
    CMD_ARGS+=("--resume")
fi

if [ -n "$MAX_SPEAKERS" ]; then
    CMD_ARGS+=("--max_speakers" "$MAX_SPEAKERS")
fi

# Set environment variables for better performance
export PYTHONPATH=../../../:$PYTHONPATH
export PYTHONIOENCODING=UTF-8

# Run the Python script
echo -e "${YELLOW}⚡ Running similarity computation with optimized multiprocessing...${NC}"
python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ Fast speaker similarity computation completed!${NC}"
echo -e "${GREEN}⏰ End time: $(date)${NC}"
echo -e "${GREEN}⏱️  Total execution time: ${EXECUTION_TIME} seconds ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Final statistics
echo -e "${BLUE}📈 Final Statistics:${NC}"
if [ -d "$SIMILARITIES_FULL_PATH" ]; then
    FINAL_RESULTS=$(find "$SIMILARITIES_FULL_PATH" -name "*.json" -o -name "*.npy" | wc -l)
    echo -e "  📊 Total similarity result files: ${FINAL_RESULTS}"
    
    # Show key output files
    echo -e "  📄 Key output files:"
    key_files=("speaker_similarities.json" "similarity_matrix.npy" "speaker_keys_mapping.json" "similarity_statistics.json" "analysis_summary.json")
    for file in "${key_files[@]}"; do
        if [ -f "$SIMILARITIES_FULL_PATH/$file" ]; then
            file_size=$(du -h "$SIMILARITIES_FULL_PATH/$file" | cut -f1)
            echo -e "    ✅ $file (${file_size})"
        else
            echo -e "    ❌ $file (missing)"
        fi
    done
    
    # Show analysis files
    echo -e "  📊 Analysis files:"
    analysis_files=("upper_triangular_statistics.json" "extreme_similarity_pairs.json" "speaker_top_similarities.json" "threshold_statistics.json")
    for file in "${analysis_files[@]}"; do
        if [ -f "$SIMILARITIES_FULL_PATH/$file" ]; then
            file_size=$(du -h "$SIMILARITIES_FULL_PATH/$file" | cut -f1)
            echo -e "    ✅ $file (${file_size})"
        else
            echo -e "    ❌ $file (missing)"
        fi
    done
    
    # Calculate processing rate
    if [ $EXECUTION_TIME -gt 0 ] && [ $TOTAL_SPEAKERS -gt 0 ]; then
        RATE=$(echo "scale=2; $TOTAL_SPEAKERS / $EXECUTION_TIME" | bc -l)
        echo -e "  🚀 Processing rate: ${RATE} speakers/second"
        
        # Total pairs processed
        TOTAL_PAIRS=$(echo "scale=0; $TOTAL_SPEAKERS * ($TOTAL_SPEAKERS - 1) / 2" | bc -l)
        PAIRS_RATE=$(echo "scale=0; $TOTAL_PAIRS / $EXECUTION_TIME" | bc -l)
        echo -e "  🔄 Pairs processed: ${TOTAL_PAIRS} pairs (${PAIRS_RATE} pairs/second)"
    fi
else
    echo -e "  ${RED}❌ Cannot access similarities output directory${NC}"
fi

# Show disk usage
echo -e "${BLUE}💾 Disk Usage:${NC}"
if [ -d "$SIMILARITIES_FULL_PATH" ]; then
    SIMILARITIES_SIZE=$(du -sh "$SIMILARITIES_FULL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "  Similarities directory: ${SIMILARITIES_SIZE}"
fi

# Show next steps
echo -e "${YELLOW}🔄 Next Steps:${NC}"
echo -e "  • Check analysis results in: $SIMILARITIES_FULL_PATH"
echo -e "  • Review similarity statistics in: analysis_summary.json"
echo -e "  • Examine top similar pairs in: extreme_similarity_pairs.json"
echo -e "  • Use similarity_matrix.npy for further analysis"

# Performance tips
echo -e "${YELLOW}💡 Performance Tips:${NC}"
echo -e "  • Adjust --num_workers based on your CPU cores and memory"
echo -e "  • Use --batch_size to balance memory usage and performance"
echo -e "  • Use --resume to continue from previous progress"
echo -e "  • Use --max_speakers for testing with smaller datasets"
echo -e "  • Monitor memory usage during large-scale computation"

echo -e "${GREEN}🎉 Fast speaker similarity computation finished!${NC}"
echo -e "${GREEN}📊 Results available in: $SIMILARITIES_FULL_PATH${NC}" 
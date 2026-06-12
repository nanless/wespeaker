#!/bin/bash

# Multi-process Speaker Embedding Computation Script
# This script computes speaker-level embeddings by averaging utterance embeddings using multiple processes

set -e

# Configuration
UTTERANCES_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_wespeaker_samresnet100/embeddings_utterances"
SPEAKERS_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_wespeaker_samresnet100/embeddings_speakers"
MIN_UTTERANCES=1
NUM_PROCESSES=$(nproc)  # Use all available CPU cores
CHUNK_SIZE=10
SKIP_EXISTING=false
# Exclude utterance embedding files whose basename starts with "voiceprint"
EXCLUDE_VOICEPRINT_PREFIX=true
# Exclude TTS clone audio files
EXCLUDE_CLONE_PATTERN="*_clone_text_*"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Multi-process Speaker Embedding Computation ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  📂 Utterances directory: ${UTTERANCES_DIR}"
echo -e "  📁 Speakers directory: ${SPEAKERS_DIR}"
echo -e "  🔢 Minimum utterances: ${MIN_UTTERANCES}"
echo -e "  ⚡ Number of processes: ${NUM_PROCESSES}"
echo -e "  📦 Chunk size: ${CHUNK_SIZE}"
echo -e "  ⏭️  Skip existing: ${SKIP_EXISTING}"
echo -e "  🚫 Exclude voiceprint* files: ${EXCLUDE_VOICEPRINT_PREFIX}"
echo -e "  🗑️  Exclude clone TTS pattern: ${EXCLUDE_CLONE_PATTERN}"
echo -e "${BLUE}===============================================${NC}"

# Check if utterances directory exists
if [ ! -d "$UTTERANCES_DIR" ]; then
    echo -e "${RED}❌ Error: Utterances directory does not exist: $UTTERANCES_DIR${NC}"
    exit 1
fi

# Create speakers directory if it doesn't exist
echo -e "${YELLOW}📁 Creating speakers directory if needed...${NC}"
mkdir -p "$SPEAKERS_DIR"

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/local/compute_speaker_embeddings_multiprocess.py"
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
if [ -d "$UTTERANCES_DIR" ]; then
    TOTAL_UTTERANCES=$(find "$UTTERANCES_DIR" -name "*.pkl" | wc -l)
    TOTAL_SPEAKERS=$(find "$UTTERANCES_DIR" -mindepth 2 -maxdepth 2 -type d | wc -l)
    echo -e "  🎤 Total utterance files: ${TOTAL_UTTERANCES}"
    echo -e "  👥 Total speakers: ${TOTAL_SPEAKERS}"
else
    echo -e "  ${YELLOW}⚠️  Cannot access utterances directory${NC}"
fi

if [ -d "$SPEAKERS_DIR" ]; then
    EXISTING_SPEAKERS=$(find "$SPEAKERS_DIR" -name "*.pkl" | wc -l)
    echo -e "  ✅ Existing speaker embeddings: ${EXISTING_SPEAKERS}"
else
    echo -e "  ✅ Existing speaker embeddings: 0"
fi

# Start computation
echo -e "${GREEN}🚀 Starting multi-process speaker embedding computation...${NC}"
echo -e "${GREEN}⏰ Start time: $(date)${NC}"

START_TIME=$(date +%s)

# Build command
CMD_ARGS=(
    "--utterances_dir" "$UTTERANCES_DIR"
    "--speakers_dir" "$SPEAKERS_DIR"
    "--min_utterances" "$MIN_UTTERANCES"
    "--num_processes" "$NUM_PROCESSES"
    "--chunk_size" "$CHUNK_SIZE"
)

if [ "$SKIP_EXISTING" = true ]; then
    CMD_ARGS+=("--skip_existing")
fi

if [ "$EXCLUDE_VOICEPRINT_PREFIX" = true ]; then
    CMD_ARGS+=("--exclude_filename_prefix" "voiceprint")
fi

if [ -n "$EXCLUDE_CLONE_PATTERN" ]; then
    CMD_ARGS+=("--exclude_filename_pattern" "$EXCLUDE_CLONE_PATTERN")
fi

# Run the Python script
python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ Multi-process speaker embedding computation completed!${NC}"
echo -e "${GREEN}⏰ End time: $(date)${NC}"
echo -e "${GREEN}⏱️  Total execution time: ${EXECUTION_TIME} seconds ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Final statistics
echo -e "${BLUE}📈 Final Statistics:${NC}"
if [ -d "$SPEAKERS_DIR" ]; then
    FINAL_SPEAKERS=$(find "$SPEAKERS_DIR" -name "*.pkl" | wc -l)
    echo -e "  👥 Total speaker embeddings: ${FINAL_SPEAKERS}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$SPEAKERS_DIR"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            speaker_count=$(find "$dataset_dir" -name "*.pkl" | wc -l)
            echo -e "    ${dataset_name}: ${speaker_count} speakers"
        fi
    done
    
    # Calculate processing rate
    if [ $EXECUTION_TIME -gt 0 ]; then
        RATE=$(echo "scale=2; $FINAL_SPEAKERS / $EXECUTION_TIME" | bc -l)
        echo -e "  🚀 Processing rate: ${RATE} speakers/second"
        
        # Theoretical speedup
        THEORETICAL_SPEEDUP=$(echo "scale=1; $NUM_PROCESSES" | bc -l)
        echo -e "  ⚡ Theoretical speedup: ${THEORETICAL_SPEEDUP}x"
    fi
else
    echo -e "  ${RED}❌ Cannot access speakers directory${NC}"
fi

# Show disk usage
echo -e "${BLUE}💾 Disk Usage:${NC}"
SPEAKERS_SIZE=$(du -sh "$SPEAKERS_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo -e "  Speakers directory: ${SPEAKERS_SIZE}"

# Show sample output
echo -e "${BLUE}🔍 Sample Output Files:${NC}"
SAMPLE_FILES=$(find "$SPEAKERS_DIR" -name "*.pkl" | head -3)
if [ -n "$SAMPLE_FILES" ]; then
    echo "$SAMPLE_FILES" | while read -r file; do
        echo -e "  📄 $file"
    done
else
    echo -e "  ${YELLOW}No sample files found${NC}"
fi

# Performance tips
echo -e "${YELLOW}💡 Performance Tips:${NC}"
echo -e "  • Adjust --num_processes based on your CPU cores and I/O capacity"
echo -e "  • Use --chunk_size to balance load distribution"
echo -e "  • Monitor system resources during execution"
echo -e "  • Use SSD storage for better I/O performance"

echo -e "${GREEN}🎉 Multi-process speaker embedding computation finished!${NC}" 
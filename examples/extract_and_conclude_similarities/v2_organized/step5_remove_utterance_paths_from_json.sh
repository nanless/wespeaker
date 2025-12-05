#!/bin/bash

# Remove utterance_paths from existing JSON files
# This script removes the utterance_paths field from all existing JSON files
# to reduce file size (since path_to_id mapping already contains all necessary information)

set -e

# Configuration
BASE_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_wespeaker_samresnet100/utterance_similarities_per_speaker"
NUM_WORKERS=8
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Remove utterance_paths from JSON files ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  📂 Base directory: ${BASE_DIR}"
echo -e "  ⚡ Number of workers: ${NUM_WORKERS}"
if [ "$DRY_RUN" = true ]; then
    echo -e "  🔍 Dry run mode: ${YELLOW}ENABLED${NC} (files will not be modified)"
else
    echo -e "  🔍 Dry run mode: ${GREEN}DISABLED${NC} (files will be modified)"
fi
echo -e "${BLUE}===============================================${NC}"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}❌ Error: Base directory does not exist: $BASE_DIR${NC}"
    exit 1
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/local/remove_utterance_paths_from_json.py"
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
if [ -d "$BASE_DIR" ]; then
    TOTAL_FILES=$(find "$BASE_DIR" -name "*_utterance_similarities.json" | wc -l)
    echo -e "  📄 Total JSON files: ${TOTAL_FILES}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$BASE_DIR"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            json_count=$(find "$dataset_dir" -name "*_utterance_similarities.json" | wc -l)
            if [ $json_count -gt 0 ]; then
                echo -e "    ${dataset_name}: ${json_count} JSON files"
            fi
        fi
    done
else
    echo -e "  ${YELLOW}⚠️  Cannot access base directory${NC}"
fi

# Start processing
echo -e "${GREEN}🚀 Starting to remove utterance_paths from JSON files...${NC}"
echo -e "${GREEN}⏰ Start time: $(date)${NC}"

START_TIME=$(date +%s)

# Build command
CMD_ARGS=(
    "--base_dir" "$BASE_DIR"
    "--num_workers" "$NUM_WORKERS"
)

if [ "$DRY_RUN" = true ]; then
    CMD_ARGS+=("--dry_run")
fi

# Set environment variables
export PYTHONPATH=../../../:$PYTHONPATH
export PYTHONIOENCODING=UTF-8

# Run the Python script
echo -e "${YELLOW}⚡ Running removal script...${NC}"
python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ Processing completed!${NC}"
echo -e "${GREEN}⏰ End time: $(date)${NC}"
echo -e "${GREEN}⏱️  Total execution time: ${EXECUTION_TIME} seconds ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Show final statistics
echo -e "${BLUE}📈 Final Statistics:${NC}"
if [ -d "$BASE_DIR" ]; then
    FINAL_FILES=$(find "$BASE_DIR" -name "*_utterance_similarities.json" | wc -l)
    echo -e "  📄 Total JSON files: ${FINAL_FILES}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$BASE_DIR"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            json_count=$(find "$dataset_dir" -name "*_utterance_similarities.json" | wc -l)
            if [ $json_count -gt 0 ]; then
                echo -e "    ${dataset_name}: ${json_count} JSON files"
            fi
        fi
    done
fi

# Show disk usage
echo -e "${BLUE}💾 Disk Usage:${NC}"
if [ -d "$BASE_DIR" ]; then
    OUTPUT_SIZE=$(du -sh "$BASE_DIR" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "  Output directory: ${OUTPUT_SIZE}"
fi

# Show next steps
echo -e "${YELLOW}🔄 Next Steps:${NC}"
echo -e "  • JSON files have been updated (utterance_paths field removed)"
echo -e "  • Files now only contain path_to_id mapping"
echo -e "  • Use path_to_id to reconstruct paths if needed:"
echo -e "    paths = [path for path, _ in sorted(path_to_id.items(), key=lambda x: x[1])]"

echo -e "${GREEN}🎉 Removal of utterance_paths completed!${NC}"


#!/bin/bash

# Audio Resampling Script
# This script checks all audio files and resamples them to 16000Hz if needed

set -e

# Configuration - matching step1 paths
DATA_ROOT="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio"
TARGET_SR=16000
RES_TYPE="fft"
NUM_WORKERS=16
SKIP_EXISTING=false
BACKUP=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Audio Resampling to 16000Hz ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  📂 Data root: ${DATA_ROOT}"
echo -e "  🎵 Target sample rate: ${TARGET_SR}Hz"
echo -e "  🔧 Resampling type: ${RES_TYPE}"
echo -e "  ⚡ Number of workers: ${NUM_WORKERS}"
echo -e "  ⏭️  Skip existing: ${SKIP_EXISTING}"
echo -e "  💾 Backup original: ${BACKUP}"
echo -e "${BLUE}====================================${NC}"

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}❌ Error: Data root does not exist: $DATA_ROOT${NC}"
    exit 1
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/local/resample_audio_to_16k.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ Error: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Print system information
echo -e "${BLUE}💻 System Information:${NC}"
echo -e "  CPU cores: $(nproc)"
echo -e "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo -e "  Python version: $(python3 --version)"

# Check if librosa and soundfile are available
echo -e "${BLUE}📦 Checking dependencies...${NC}"
if ! python3 -c "import librosa" 2>/dev/null; then
    echo -e "${RED}❌ Error: librosa is not installed${NC}"
    echo -e "${YELLOW}💡 Install with: pip install librosa${NC}"
    exit 1
fi

if ! python3 -c "import soundfile" 2>/dev/null; then
    echo -e "${RED}❌ Error: soundfile is not installed${NC}"
    echo -e "${YELLOW}💡 Install with: pip install soundfile${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies available${NC}"

# Get initial statistics
echo -e "${BLUE}📊 Initial Statistics:${NC}"
if [ -d "$DATA_ROOT" ]; then
    TOTAL_AUDIO=$(find "$DATA_ROOT" -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) | wc -l)
    TOTAL_DATASETS=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo -e "  🎵 Total audio files: ${TOTAL_AUDIO}"
    echo -e "  📂 Total datasets: ${TOTAL_DATASETS}"
    
    # Show dataset breakdown
    echo -e "  📂 Dataset breakdown:"
    for dataset_dir in "$DATA_ROOT"/*; do
        if [ -d "$dataset_dir" ]; then
            dataset_name=$(basename "$dataset_dir")
            audio_count=$(find "$dataset_dir" -type f \( -name "*.wav" -o -name "*.flac" -o -name "*.mp3" \) | wc -l)
            speaker_count=$(find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
            echo -e "    ${dataset_name}: ${speaker_count} speakers, ${audio_count} audio files"
        fi
    done
else
    echo -e "  ${YELLOW}⚠️  Cannot access data root${NC}"
fi

# Start resampling
echo -e "${GREEN}🚀 Starting audio resampling...${NC}"
echo -e "${GREEN}⏰ Start time: $(date)${NC}"

START_TIME=$(date +%s)

# Build command
CMD_ARGS=(
    "--data_root" "$DATA_ROOT"
    "--target_sr" "$TARGET_SR"
    "--res_type" "$RES_TYPE"
    "--num_workers" "$NUM_WORKERS"
)

if [ "$SKIP_EXISTING" = true ]; then
    CMD_ARGS+=("--skip_existing")
fi

if [ "$BACKUP" = true ]; then
    CMD_ARGS+=("--backup")
fi

# Set environment variables
export PYTHONPATH=../../../:$PYTHONPATH
export PYTHONIOENCODING=UTF-8

# Run the Python script
echo -e "${YELLOW}⚡ Running audio resampling with multiprocessing...${NC}"
python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

# Calculate execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}✅ Audio resampling completed!${NC}"
echo -e "${GREEN}⏰ End time: $(date)${NC}"
echo -e "${GREEN}⏱️  Total execution time: ${EXECUTION_TIME} seconds ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Show disk usage
echo -e "${BLUE}💾 Disk Usage:${NC}"
if [ -d "$DATA_ROOT" ]; then
    DATA_SIZE=$(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "  Data directory: ${DATA_SIZE}"
fi

# Show next steps
echo -e "${YELLOW}🔄 Next Steps:${NC}"
echo -e "  • All audio files have been checked and resampled if needed"
echo -e "  • Files are now at ${TARGET_SR}Hz sample rate"
echo -e "  • You can now proceed with step1 to extract embeddings"

# Performance tips
echo -e "${YELLOW}💡 Performance Tips:${NC}"
echo -e "  • Adjust --num_workers based on your CPU cores and I/O capacity"
echo -e "  • Use --skip_existing to skip files already at target sample rate"
echo -e "  • Use --backup to create backups before resampling"
echo -e "  • Monitor disk space if using --backup option"

echo -e "${GREEN}🎉 Audio resampling finished!${NC}"


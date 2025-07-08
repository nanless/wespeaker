#!/bin/bash

set -e
. ./path.sh || exit 1

# Configuration
UTTERANCES_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual/utterances"
SPEAKERS_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual/speakers"

# Parameters
MIN_UTTERANCES=1      # Minimum utterances required for a speaker
SKIP_EXISTING=true    # Skip speakers that already have embeddings

# Parse command line arguments
stage=1
stop_stage=1

. tools/parse_options.sh || exit 1

echo "=== Speaker Embedding Computation Pipeline ==="
echo "Utterances directory: $UTTERANCES_DIR"
echo "Speakers directory: $SPEAKERS_DIR"
echo "Minimum utterances: $MIN_UTTERANCES"
echo "Skip existing: $SKIP_EXISTING"
echo "=============================================="

# Check if utterances directory exists
if [ ! -d "$UTTERANCES_DIR" ]; then
    echo "❌ Error: Utterances directory not found at $UTTERANCES_DIR"
    echo "💡 Make sure you have run the embedding extraction first"
    exit 1
fi

# Check if there are any utterance files
utterance_count=$(find "$UTTERANCES_DIR" -name "*.pkl" 2>/dev/null | wc -l)
if [ $utterance_count -eq 0 ]; then
    echo "❌ Error: No utterance embedding files found in $UTTERANCES_DIR"
    echo "💡 Make sure the embedding extraction has completed successfully"
    exit 1
fi

echo "📊 Found $utterance_count utterance embedding files"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Computing speaker-level embeddings..."
    
    # Create output directory
    mkdir -p "$SPEAKERS_DIR"
    
    # Run speaker embedding computation
    echo "🚀 Starting speaker embedding computation..."
    
    python compute_speaker_embeddings.py \
        --utterances_dir "$UTTERANCES_DIR" \
        --speakers_dir "$SPEAKERS_DIR" \
        --min_utterances $MIN_UTTERANCES \
        $([ "$SKIP_EXISTING" = true ] && echo "--skip_existing")
    
    echo "✅ Stage 1 completed."
    
    # Check results
    if [ -d "$SPEAKERS_DIR" ]; then
        echo "📁 Speaker embeddings saved to: $SPEAKERS_DIR"
        
        # Count speaker embeddings
        speaker_count=$(find "$SPEAKERS_DIR" -name "*.pkl" | wc -l)
        echo "📈 Total speaker embeddings: $speaker_count"
        
        # Show directory structure
        echo ""
        echo "📂 Directory structure:"
        for dataset in $(ls "$SPEAKERS_DIR" 2>/dev/null | head -5); do
            if [ -d "$SPEAKERS_DIR/$dataset" ]; then
                dataset_speakers=$(find "$SPEAKERS_DIR/$dataset" -name "*.pkl" | wc -l)
                echo "  $dataset: $dataset_speakers speakers"
                
                # Show sample files
                find "$SPEAKERS_DIR/$dataset" -name "*.pkl" | head -3 | while read file; do
                    echo "    $(basename "$file")"
                done
                if [ $dataset_speakers -gt 3 ]; then
                    echo "    ... and $((dataset_speakers - 3)) more"
                fi
            fi
        done
        
        # Show some statistics using Python
        echo ""
        echo "🔍 Sample speaker statistics:"
        python -c "
import pickle
import glob
import numpy as np
from pathlib import Path

speakers_dir = '$SPEAKERS_DIR'
sample_files = glob.glob(f'{speakers_dir}/**/*.pkl', recursive=True)[:5]

for file_path in sample_files:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        path = Path(file_path)
        dataset = path.parent.name
        speaker = path.stem
        
        print(f'  📄 {dataset}/{speaker}:')
        print(f'     Utterances: {data.get(\"num_utterances\", \"unknown\")}')
        print(f'     Embedding dim: {data.get(\"embedding_dim\", \"unknown\")}')
        print(f'     Avg embedding norm: {np.linalg.norm(data.get(\"embedding\", [0])):.4f}')
        
        stats = data.get('embedding_stats', {})
        print(f'     Stats - mean: {stats.get(\"mean\", 0):.4f}, std: {stats.get(\"std\", 0):.4f}')
        
    except Exception as e:
        print(f'     Error reading {file_path}: {e}')
"
        
        if [ $speaker_count -gt 0 ]; then
            echo ""
            echo "🎉 Speaker embedding computation completed successfully!"
            echo "✅ Results saved in: $SPEAKERS_DIR"
            echo ""
            echo "📋 Next steps:"
            echo "  - Use speaker embeddings for similarity analysis"
            echo "  - Compute speaker similarity matrices"
            echo "  - Perform speaker clustering or verification"
        else
            echo ""
            echo "⚠️  Warning: No speaker embeddings were created."
            echo "💡 Check if utterance files have the correct format"
        fi
    else
        echo "❌ Error: Output directory was not created."
        exit 1
    fi
fi

echo "Done." 
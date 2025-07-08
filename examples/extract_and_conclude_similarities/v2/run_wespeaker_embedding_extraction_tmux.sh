#!/bin/bash

set -e
. ./path.sh || exit 1

# Configuration
DATA_ROOT="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
MODEL_DIR="/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"
OUTPUT_DIR="/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual"
MASTER_PORT=29503
GPUS="0,1,2,3"  # Available GPUs (comma-separated for this script)

# Tmux configuration
SESSION_NAME="wespeaker_embedding_extraction"
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/embedding_extraction_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/extraction.pid"

# Parse command line arguments
stage=1
stop_stage=1
action="start"  # start, stop, status, attach, logs

. tools/parse_options.sh || exit 1

# Create log directory
mkdir -p "$LOG_DIR"

# Function to check if tmux session exists
session_exists() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

# Function to start the extraction in tmux
start_extraction() {
    if session_exists; then
        echo "❌ Tmux session '$SESSION_NAME' already exists!"
        echo "Use --action status to check status, or --action stop to stop it first."
        exit 1
    fi
    
    echo "=== WeSpeaker Embedding Extraction Pipeline (Tmux Background) ==="
    echo "Data root: $DATA_ROOT"
    echo "Model directory: $MODEL_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo "GPUs: $GPUS"
    echo "Master port: $MASTER_PORT"
    echo "Session name: $SESSION_NAME"
    echo "Log file: $LOG_FILE"
    echo "======================================"
    
    # Check if model exists
    if [ ! -f "$MODEL_DIR/avg_model.pt" ]; then
        echo "❌ Error: Model file not found at $MODEL_DIR/avg_model.pt"
        exit 1
    fi
    
    # Check if config exists
    if [ ! -f "$MODEL_DIR/config.yaml" ]; then
        echo "❌ Error: Config file not found at $MODEL_DIR/config.yaml"
        exit 1
    fi
    
    # Check if data directory exists
    if [ ! -d "$DATA_ROOT" ]; then
        echo "❌ Error: Data directory not found at $DATA_ROOT"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    echo "🚀 Starting embedding extraction in tmux session '$SESSION_NAME'..."
    echo "📝 Logs will be saved to: $LOG_FILE"
    
    # Create tmux session and run the extraction
    tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" bash -c "
        set -e
        
        # Setup logging
        exec > >(tee -a '$LOG_FILE') 2>&1
        
        echo '=== WeSpeaker Embedding Extraction Started at $(date) ==='
        echo 'Session: $SESSION_NAME'
        echo 'PID: \$\$'
        echo 'Working directory: $(pwd)'
        echo ''
        
        # Save PID
        echo \$\$ > '$PID_FILE'
        
        # Count audio files
        echo '📊 Counting audio files...'
        total_files=\$(find '$DATA_ROOT' -name '*.wav' -o -name '*.flac' -o -name '*.mp3' | wc -l)
        echo \"Total audio files to process: \$total_files\"
        echo ''
        
        # Run embedding extraction
        echo '🔥 Starting embedding extraction...'
        python extract_wespeaker_embeddings.py \\
            --data_root '$DATA_ROOT' \\
            --model_dir '$MODEL_DIR' \\
            --output_dir '$OUTPUT_DIR' \\
            --gpus '$GPUS' \\
            --port '$MASTER_PORT' || {
            echo '❌ Embedding extraction failed!'
            exit 1
        }
        
        echo ''
        echo '✅ Stage 1 completed.'
        
        # Check results
        if [ -d '$OUTPUT_DIR' ]; then
            echo '📁 Individual embedding files saved to: $OUTPUT_DIR'
            
            # Count extracted embeddings
            extracted_count=\$(find '$OUTPUT_DIR' -name '*.pkl' | wc -l)
            echo \"📈 Total embeddings extracted: \$extracted_count\"
            
            # Show directory structure sample
            echo ''
            echo '📂 Directory structure (first few examples):'
            find '$OUTPUT_DIR' -name '*.pkl' | head -5 | while read file; do
                echo \"  \$file\"
            done
            
            # Show statistics
            echo ''
            echo '📊 Dataset statistics:'
            for dataset in \$(ls '$OUTPUT_DIR' 2>/dev/null | head -10); do
                if [ -d '$OUTPUT_DIR/\$dataset' ]; then
                    dataset_count=\$(find '$OUTPUT_DIR/\$dataset' -name '*.pkl' | wc -l)
                    speaker_count=\$(find '$OUTPUT_DIR/\$dataset' -type d -mindepth 1 | wc -l)
                    echo \"  \$dataset: \$dataset_count embeddings from \$speaker_count speakers\"
                fi
            done
            
            if [ \$extracted_count -gt 0 ]; then
                echo ''
                echo '🎉 Embedding extraction completed successfully at $(date)!'
                echo '✅ Results saved in: $OUTPUT_DIR'
            else
                echo ''
                echo '⚠️  Warning: No embeddings were extracted. Please check the logs for errors.'
            fi
        else
            echo '❌ Error: Output directory was not created.'
            exit 1
        fi
        
        echo ''
        echo '🏁 All done at $(date)!'
        
        # Clean up PID file
        rm -f '$PID_FILE'
        
        # Keep session alive for a while to allow user to check results
        echo ''
        echo '💤 Session will remain active for 60 seconds for you to check results...'
        echo 'Use: tmux attach -t $SESSION_NAME to view this session'
        echo 'Use: ./run_wespeaker_embedding_extraction_tmux.sh --action logs to view logs'
        sleep 60
    "
    
    # Wait a moment for session to start
    sleep 2
    
    echo "✅ Tmux session '$SESSION_NAME' started successfully!"
    echo ""
    echo "📋 Management commands:"
    echo "  Check status:  ./run_wespeaker_embedding_extraction_tmux.sh --action status"
    echo "  View logs:     ./run_wespeaker_embedding_extraction_tmux.sh --action logs"
    echo "  Attach session: ./run_wespeaker_embedding_extraction_tmux.sh --action attach"
    echo "  Stop session:  ./run_wespeaker_embedding_extraction_tmux.sh --action stop"
    echo ""
    echo "📁 Log file: $LOG_FILE"
}

# Function to stop the extraction
stop_extraction() {
    if ! session_exists; then
        echo "❌ Tmux session '$SESSION_NAME' does not exist."
        exit 1
    fi
    
    echo "🛑 Stopping tmux session '$SESSION_NAME'..."
    tmux kill-session -t "$SESSION_NAME"
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    echo "✅ Session stopped."
}

# Function to show status
show_status() {
    if session_exists; then
        echo "✅ Tmux session '$SESSION_NAME' is running."
        
        if [ -f "$PID_FILE" ]; then
            pid=$(cat "$PID_FILE")
            echo "📊 Process ID: $pid"
            
            # Check if process is actually running
            if kill -0 "$pid" 2>/dev/null; then
                echo "✅ Process is active."
                
                # Show GPU usage
                echo ""
                echo "🖥️  GPU Status:"
                nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=',' read -r gpu_id name util mem_used mem_total; do
                    echo "  GPU $gpu_id ($name): ${util}% utilization, ${mem_used}MB/${mem_total}MB memory"
                done
            else
                echo "⚠️  Process may have finished or crashed."
            fi
        else
            echo "⚠️  PID file not found, but session exists."
        fi
        
        # Show recent log entries
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "📝 Recent log entries (last 10 lines):"
            tail -n 10 "$LOG_FILE"
        fi
    else
        echo "❌ Tmux session '$SESSION_NAME' is not running."
        
        # Check for recent log files
        recent_log=$(ls -t $LOG_DIR/embedding_extraction_*.log 2>/dev/null | head -1)
        if [ -n "$recent_log" ]; then
            echo "📝 Most recent log file: $recent_log"
        fi
    fi
}

# Function to attach to session
attach_session() {
    if ! session_exists; then
        echo "❌ Tmux session '$SESSION_NAME' does not exist."
        exit 1
    fi
    
    echo "🔗 Attaching to tmux session '$SESSION_NAME'..."
    echo "💡 Press Ctrl+B then D to detach from session"
    tmux attach-session -t "$SESSION_NAME"
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "📝 Showing logs from: $LOG_FILE"
        echo "💡 Press Ctrl+C to exit, or use 'tail -f' to follow"
        echo "----------------------------------------"
        less "$LOG_FILE"
    else
        # Try to find the most recent log
        recent_log=$(ls -t $LOG_DIR/embedding_extraction_*.log 2>/dev/null | head -1)
        if [ -n "$recent_log" ]; then
            echo "📝 No current log found, showing most recent: $recent_log"
            echo "----------------------------------------"
            less "$recent_log"
        else
            echo "❌ No log files found in $LOG_DIR"
        fi
    fi
}

# Main logic
case "$action" in
    "start")
        start_extraction
        ;;
    "stop")
        stop_extraction
        ;;
    "status")
        show_status
        ;;
    "attach")
        attach_session
        ;;
    "logs")
        show_logs
        ;;
    *)
        echo "Usage: $0 --action {start|stop|status|attach|logs}"
        echo ""
        echo "Actions:"
        echo "  start   - Start embedding extraction in tmux background"
        echo "  stop    - Stop the running extraction session"  
        echo "  status  - Show current status and recent logs"
        echo "  attach  - Attach to the tmux session"
        echo "  logs    - View full logs"
        echo ""
        echo "Examples:"
        echo "  $0 --action start"
        echo "  $0 --action status"
        echo "  $0 --action logs"
        exit 1
        ;;
esac 
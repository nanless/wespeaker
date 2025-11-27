#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to run WeSpeaker diarization on all audio files in a directory.

Usage:
    python run_diarization_simple.py [--src_dir DIR] [--out_dir DIR] [--model_dir DIR] [--nprocs N]
"""

import os
import sys
import subprocess
import argparse
import tempfile
from pathlib import Path

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Path to the inference script
INFER_SCRIPT = Path(__file__).parent.parent.parent / "wespeaker" / "bin" / "infer_diarization.py"

# Default directories
DEFAULT_SRC_DIR = "/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio"
DEFAULT_MODEL_DIR = "/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"


def find_audio_files(src_dir, extensions=None, pattern=None):
    """Find all audio files in the directory."""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    audio_files = []
    src_path = Path(src_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    
    if pattern:
        # Use pattern-based search
        from glob import glob
        audio_files = sorted(glob(os.path.join(src_dir, pattern)))
        audio_files = [Path(f) for f in audio_files]
    else:
        # Find all audio files
        for ext in extensions:
            audio_files.extend(src_path.glob(f"*{ext}"))
            audio_files.extend(src_path.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(
        description="Run WeSpeaker diarization on all audio files in a directory."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default=DEFAULT_SRC_DIR,
        help=f"Source directory containing audio files (default: {DEFAULT_SRC_DIR})"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for results (default: <src_dir_parent>/<basename>_wespeaker_diarization)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing WeSpeaker model files (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=None,
        help="Number of processes to use (default: auto-detect based on GPU count)"
    )
    parser.add_argument(
        "--speaker_num",
        type=int,
        default=None,
        help="Oracle number of speakers if known (optional)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="File pattern to match (default: *.wav)"
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=['umap', 'spectral'],
        default='umap',
        help="Clustering method: umap (default) or spectral"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.255,
        help="Minimum VAD segment duration (seconds, default: 0.255)"
    )
    parser.add_argument(
        "--window_secs",
        type=float,
        default=1.5,
        help="Window duration for subsegmentation (seconds, default: 1.5)"
    )
    parser.add_argument(
        "--period_secs",
        type=float,
        default=0.75,
        help="Period/step for subsegmentation (seconds, default: 0.75)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)"
    )
    parser.add_argument(
        "--apply_vad",
        action='store_true',
        help="Apply VAD before diarization"
    )
    parser.add_argument(
        "--out_type",
        type=str,
        choices=['rttm', 'json'],
        default='json',
        help="Output format: json (default) or rttm"
    )
    
    args = parser.parse_args()
    
    src_dir = os.path.abspath(args.src_dir)
    
    # Set output directory
    if args.out_dir is None:
        base_name = os.path.basename(src_dir) + "_wespeaker_diarization"
        out_dir = os.path.join(os.path.dirname(src_dir), base_name)
    else:
        out_dir = os.path.abspath(args.out_dir)
    
    print(f"[INFO] Starting diarization on directory: {src_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Model directory: {args.model_dir}")
    print(f"[INFO] Pattern: {args.pattern}")
    
    # Find all audio files
    audio_files = find_audio_files(src_dir, pattern=args.pattern)
    
    if len(audio_files) == 0:
        print(f"[ERROR] No audio files found in {src_dir} matching pattern {args.pattern}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(audio_files)} audio files")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Create temporary wav list file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        wav_list_path = f.name
        for audio_file in audio_files:
            f.write(str(audio_file) + "\n")
    
    try:
        # Build command
        cmd = [
            sys.executable,
            str(INFER_SCRIPT),
            "--wav", wav_list_path,
            "--out_dir", out_dir,
            "--model_dir", args.model_dir,
            "--out_type", args.out_type,
            "--cluster_method", args.cluster_method,
            "--min_duration", str(args.min_duration),
            "--window_secs", str(args.window_secs),
            "--period_secs", str(args.period_secs),
            "--batch_size", str(args.batch_size),
        ]
        
        # Note: disable_progress_bar is handled by infer_diarization.py internally
        
        if args.nprocs is not None:
            cmd.extend(["--nprocs", str(args.nprocs)])
        
        if args.speaker_num is not None:
            cmd.extend(["--speaker_num", str(args.speaker_num)])
        
        if args.apply_vad:
            cmd.append("--apply_vad")
        
        print(f"[INFO] Running diarization...")
        print(f"[INFO] Command: {' '.join(cmd)}")
        print()
        
        # Run diarization
        subprocess.run(cmd, check=True)
        
        print()
        print(f"[INFO] Diarization completed successfully!")
        print(f"[INFO] Results saved to: {out_dir}")
        print(f"[INFO] Each audio file has:")
        print(f"  - <filename>.{args.out_type}: Diarization results")
        print(f"  - <filename>.meta.json: Metadata (duration, processing time, RTF)")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Diarization failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted by user")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if os.path.exists(wav_list_path):
            os.unlink(wav_list_path)


if __name__ == "__main__":
    main()


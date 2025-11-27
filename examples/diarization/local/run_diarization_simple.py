#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的脚本，用于在目录中的所有音频文件上运行WeSpeaker diarization。
完全仿照3D-Speaker流程：使用TenVad、VAD后处理、不做滑窗、AHC聚类。

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
# 从 examples/diarization/local/ 到项目根目录需要往上4层
INFER_SCRIPT = Path(__file__).parent.parent.parent.parent / "wespeaker" / "bin" / "infer_diarization.py"

# Default directories
DEFAULT_SRC_DIR = "/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone/audio"
DEFAULT_MODEL_DIR = "/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"


def find_audio_files(src_dir, extensions=None, pattern=None):
    """在目录中查找所有音频文件。"""
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
        description="在目录中的所有音频文件上运行WeSpeaker diarization（仿照3D-Speaker流程）。"
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default=DEFAULT_SRC_DIR,
        help=f"包含音频文件的源目录（默认: {DEFAULT_SRC_DIR}）"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="结果输出目录（默认: <src_dir_parent>/<basename>_wespeaker_diarization）"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"包含WeSpeaker模型文件的目录（默认: {DEFAULT_MODEL_DIR}）"
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=None,
        help="使用的进程数（默认：根据GPU数量自动检测）"
    )
    parser.add_argument(
        "--speaker_num",
        type=int,
        default=None,
        help="已知的说话人数量（可选）"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_speech_estimate.wav",
        help="要匹配的文件模式（默认: *_speech_estimate.wav）"
    )
    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=0.5,
        help="TenVad阈值（默认：0.5）"
    )
    parser.add_argument(
        "--vad_min_speech_ms",
        type=float,
        default=200.0,
        help="VAD后处理：最小语音段时长（毫秒，默认：200.0）"
    )
    parser.add_argument(
        "--vad_max_silence_ms",
        type=float,
        default=300.0,
        help="VAD后处理：最大静音间隙（毫秒，默认：300.0）"
    )
    parser.add_argument(
        "--vad_energy_threshold",
        type=float,
        default=0.05,
        help="VAD能量阈值（默认：0.05）"
    )
    parser.add_argument(
        "--vad_boundary_expansion_ms",
        type=float,
        default=10.0,
        help="VAD边界扩展（毫秒，默认：10.0）"
    )
    parser.add_argument(
        "--cluster_fix_cos_thr",
        type=float,
        default=0.3,
        help="AHC聚类固定余弦阈值（默认：0.3）"
    )
    parser.add_argument(
        "--cluster_mer_cos",
        type=float,
        default=0.3,
        help="AHC聚类合并余弦阈值（默认：0.3）"
    )
    parser.add_argument(
        "--cluster_min_cluster_size",
        type=int,
        default=0,
        help="AHC聚类最小簇大小（默认：0）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="embedding提取的batch size（默认：64）"
    )
    parser.add_argument(
        "--out_type",
        type=str,
        choices=['rttm', 'json'],
        default='json',
        help="输出格式: json（默认）或 rttm"
    )
    
    args = parser.parse_args()
    
    src_dir = os.path.abspath(args.src_dir)
    
    # Set output directory
    if args.out_dir is None:
        base_name = os.path.basename(src_dir) + "_wespeaker_diarization"
        out_dir = os.path.join(os.path.dirname(src_dir), base_name)
    else:
        out_dir = os.path.abspath(args.out_dir)
    
    print(f"[信息] 开始在目录上运行diarization: {src_dir}")
    print(f"[信息] 输出目录: {out_dir}")
    print(f"[信息] 模型目录: {args.model_dir}")
    print(f"[信息] 文件模式: {args.pattern}")
    
    # Find all audio files
    audio_files = find_audio_files(src_dir, pattern=args.pattern)
    
    if len(audio_files) == 0:
        print(f"[错误] 在 {src_dir} 中未找到匹配模式 {args.pattern} 的音频文件")
        sys.exit(1)
    
    print(f"[信息] 找到 {len(audio_files)} 个音频文件")
    
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
            "--vad_threshold", str(args.vad_threshold),
            "--vad_min_speech_ms", str(args.vad_min_speech_ms),
            "--vad_max_silence_ms", str(args.vad_max_silence_ms),
            "--vad_energy_threshold", str(args.vad_energy_threshold),
            "--vad_boundary_expansion_ms", str(args.vad_boundary_expansion_ms),
            "--cluster_fix_cos_thr", str(args.cluster_fix_cos_thr),
            "--cluster_mer_cos", str(args.cluster_mer_cos),
            "--cluster_min_cluster_size", str(args.cluster_min_cluster_size),
            "--batch_size", str(args.batch_size),
        ]
        
        if args.nprocs is not None:
            cmd.extend(["--nprocs", str(args.nprocs)])
        
        if args.speaker_num is not None:
            cmd.extend(["--speaker_num", str(args.speaker_num)])
        
        print(f"[信息] 正在运行diarization...")
        print(f"[信息] 命令: {' '.join(cmd)}")
        print()
        
        # 运行diarization
        subprocess.run(cmd, check=True)
        
        print()
        print(f"[信息] Diarization成功完成！")
        print(f"[信息] 结果已保存到: {out_dir}")
        print(f"[信息] 每个音频文件包含:")
        print(f"  - <filename>.{args.out_type}: Diarization结果")
        print(f"  - <filename>.meta.json: 元数据（时长、处理时间、RTF）")
        
    except subprocess.CalledProcessError as e:
        print(f"[错误] Diarization失败，错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n[信息] 用户中断")
        sys.exit(1)
    finally:
        # Clean up temporary file
        if os.path.exists(wav_list_path):
            os.unlink(wav_list_path)


if __name__ == "__main__":
    main()


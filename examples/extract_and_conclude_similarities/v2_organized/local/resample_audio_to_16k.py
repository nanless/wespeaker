#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
检查并重采样音频文件到16000Hz
如果音频文件的采样率不是16000，使用librosa重采样到16000，res_type='fft'
"""

import os
import sys
import argparse
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Resample audio files to 16000Hz')
    parser.add_argument('--data_root', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio',
                        help='Root directory containing audio files')
    parser.add_argument('--target_sr', type=int, default=16000,
                        help='Target sample rate (default: 16000)')
    parser.add_argument('--res_type', type=str, default='fft',
                        help='Resampling type for librosa (default: fft)')
    parser.add_argument('--num_workers', type=int, default=min(32, mp.cpu_count()),
                        help='Number of worker processes')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='Skip files that are already at target sample rate')
    parser.add_argument('--backup', action='store_true', default=False,
                        help='Create backup of original files before resampling')
    parser.add_argument('--audio_extensions', type=str, nargs='+', 
                        default=['.wav', '.flac', '.mp3'],
                        help='Audio file extensions to process')
    
    return parser.parse_args()

def check_and_resample_audio(file_info):
    """检查并重采样单个音频文件"""
    file_path, target_sr, res_type, skip_existing, backup = file_info
    
    try:
        # First, quickly check sample rate using soundfile (faster than librosa)
        try:
            import soundfile as sf_check
            info = sf_check.info(file_path)
            sr = info.samplerate
        except Exception:
            # Fallback to librosa if soundfile fails
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True, duration=0.1)
            except Exception as e:
                return {
                    'file': str(file_path),
                    'status': 'error',
                    'error': f'Failed to load audio: {e}'
                }
        
        # Check if already at target sample rate
        if sr == target_sr:
            if skip_existing:
                return {
                    'file': str(file_path),
                    'status': 'skipped',
                    'reason': f'already at {target_sr}Hz'
                }
            else:
                return {
                    'file': str(file_path),
                    'status': 'skipped',
                    'reason': f'already at {target_sr}Hz (no resampling needed)'
                }
        
        # Create backup if requested
        if backup:
            backup_path = str(file_path) + '.backup'
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(file_path, backup_path)
        
        # Load full audio for resampling
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
        except Exception as e:
            return {
                'file': str(file_path),
                'status': 'error',
                'error': f'Failed to load audio for resampling: {e}'
            }
        
        # Resample to target sample rate
        try:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type=res_type)
        except Exception as e:
            return {
                'file': str(file_path),
                'status': 'error',
                'error': f'Failed to resample: {e}'
            }
        
        # Save resampled audio
        try:
            # Use soundfile to preserve original format
            sf.write(file_path, y_resampled, target_sr)
        except Exception as e:
            return {
                'file': str(file_path),
                'status': 'error',
                'error': f'Failed to save resampled audio: {e}'
            }
        
        return {
            'file': str(file_path),
            'status': 'success',
            'original_sr': sr,
            'target_sr': target_sr
        }
        
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'error': f'Unexpected error: {e}'
        }

def scan_audio_files(data_root, audio_extensions):
    """扫描所有音频文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning audio files in: {data_root}")
    
    audio_files = []
    data_path = Path(data_root)
    
    if not data_path.exists():
        logger.error(f"Data root does not exist: {data_root}")
        return audio_files
    
    for dataset_dir in data_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        logger.info(f"Processing dataset: {dataset_name}")
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            
            for audio_file in speaker_dir.iterdir():
                if audio_file.suffix.lower() in [ext.lower() for ext in audio_extensions]:
                    audio_files.append({
                        'path': str(audio_file),
                        'dataset': dataset_name,
                        'speaker_id': speaker_id,
                        'utterance_id': audio_file.stem
                    })
    
    logger.info(f"Found {len(audio_files)} audio files")
    return audio_files

def main():
    args = parse_args()
    logger = setup_logging()
    
    logger.info("=== Audio Resampling to 16000Hz ===")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Target sample rate: {args.target_sr}Hz")
    logger.info(f"Resampling type: {args.res_type}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info(f"Backup original: {args.backup}")
    logger.info(f"Audio extensions: {args.audio_extensions}")
    logger.info("====================================")
    
    # Check input directory
    if not os.path.exists(args.data_root):
        logger.error(f"Data root does not exist: {args.data_root}")
        sys.exit(1)
    
    # Scan audio files
    audio_files = scan_audio_files(args.data_root, args.audio_extensions)
    
    if not audio_files:
        logger.error("No audio files found!")
        sys.exit(1)
    
    # Prepare file info for processing
    file_infos = [
        (file_info['path'], args.target_sr, args.res_type, args.skip_existing, args.backup)
        for file_info in audio_files
    ]
    
    # Process files in batches to avoid memory issues
    start_time = time.time()
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    error_messages = []
    
    # Process in batches to avoid creating too many futures at once
    batch_size = args.num_workers * 10  # Process 10x worker count at a time
    
    logger.info(f"Processing {len(file_infos)} files in batches of {batch_size}")
    
    for batch_start in range(0, len(file_infos), batch_size):
        batch_end = min(batch_start + batch_size, len(file_infos))
        batch = file_infos[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(file_infos) + batch_size - 1)//batch_size} "
                   f"(files {batch_start+1}-{batch_end} of {len(file_infos)})")
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(check_and_resample_audio, file_info): file_info[0] 
                      for file_info in batch}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"Batch {batch_start//batch_size + 1}"):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    
                    if result['status'] == 'success':
                        total_processed += 1
                        if total_processed % 100 == 0:
                            logger.debug(f"Resampled {result['file']}: {result['original_sr']}Hz -> {result['target_sr']}Hz")
                    elif result['status'] == 'skipped':
                        total_skipped += 1
                    elif result['status'] == 'error':
                        total_errors += 1
                        error_msg = result.get('error', 'Unknown error')
                        error_messages.append(f"{result['file']}: {error_msg}")
                        if len(error_messages) <= 20:
                            logger.warning(f"Error processing {result['file']}: {error_msg}")
                    
                except TimeoutError:
                    total_errors += 1
                    file_path = futures[future]
                    error_msg = "Timeout (5 minutes)"
                    error_messages.append(f"{file_path}: {error_msg}")
                    logger.error(f"Timeout processing {file_path}")
                except Exception as e:
                    total_errors += 1
                    file_path = futures[future]
                    error_msg = f"Unexpected error: {e}"
                    error_messages.append(f"{file_path}: {error_msg}")
                    logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        
        # Log progress
        logger.info(f"Progress: {total_processed} processed, {total_skipped} skipped, {total_errors} errors "
                   f"out of {batch_end} files")
    
    # Final statistics
    total_time = time.time() - start_time
    
    logger.info(f"\n🎉 Audio resampling completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"  ✅ Processed: {total_processed} files")
    logger.info(f"  ⏭️  Skipped: {total_skipped} files")
    logger.info(f"  ❌ Errors: {total_errors} files")
    logger.info(f"  ⏱️  Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    if total_processed > 0:
        logger.info(f"  🚀 Processing rate: {total_processed/total_time:.2f} files/sec")
    
    if error_messages:
        logger.info(f"\n⚠️  Error Summary (showing up to 20 errors):")
        for msg in error_messages[:20]:
            logger.warning(f"  {msg}")
        if len(error_messages) > 20:
            logger.warning(f"  ... and {len(error_messages) - 20} more errors")
    
    # Show dataset breakdown
    logger.info(f"\n📂 Dataset breakdown:")
    dataset_stats = {}
    for file_info in audio_files:
        dataset = file_info['dataset']
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'total': 0, 'processed': 0, 'skipped': 0, 'errors': 0}
        dataset_stats[dataset]['total'] += 1
    
    # Note: This is approximate since we don't track per-dataset in results
    for dataset, stats in sorted(dataset_stats.items()):
        logger.info(f"  {dataset}: {stats['total']} audio files")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


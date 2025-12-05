#!/usr/bin/env python3
"""
移除所有JSON文件中的utterance_paths字段，减少文件大小
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_single_json(json_file, dry_run=False):
    """处理单个JSON文件，移除utterance_paths字段"""
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否包含utterance_paths字段
        if 'utterance_paths' not in data:
            return {
                'file': str(json_file),
                'status': 'skipped',
                'reason': 'no utterance_paths field'
            }
        
        # 计算原始文件大小
        original_size = json_file.stat().st_size
        
        # 移除utterance_paths字段
        utterance_paths_count = len(data.get('utterance_paths', []))
        del data['utterance_paths']
        
        if dry_run:
            # 模拟计算新文件大小（估算）
            new_data_str = json.dumps(data, indent=2, ensure_ascii=False)
            estimated_size = len(new_data_str.encode('utf-8'))
            size_reduction = original_size - estimated_size
            return {
                'file': str(json_file),
                'status': 'dry_run',
                'original_size': original_size,
                'estimated_size': estimated_size,
                'size_reduction': size_reduction,
                'utterance_paths_count': utterance_paths_count
            }
        
        # 保存到临时文件
        temp_file = json_file.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # 获取新文件大小
        new_size = temp_file.stat().st_size
        
        # 原子性替换
        temp_file.replace(json_file)
        
        size_reduction = original_size - new_size
        
        return {
            'file': str(json_file),
            'status': 'success',
            'original_size': original_size,
            'new_size': new_size,
            'size_reduction': size_reduction,
            'utterance_paths_count': utterance_paths_count
        }
        
    except json.JSONDecodeError as e:
        return {
            'file': str(json_file),
            'status': 'error',
            'error': f'JSON decode error: {e}'
        }
    except Exception as e:
        return {
            'file': str(json_file),
            'status': 'error',
            'error': str(e)
        }

def find_json_files(base_dir):
    """查找所有utterance_similarities.json文件"""
    base_path = Path(base_dir)
    json_files = list(base_path.rglob('*_utterance_similarities.json'))
    return json_files

def main():
    parser = argparse.ArgumentParser(description='Remove utterance_paths field from JSON files')
    parser.add_argument('--base_dir', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_wespeaker_samresnet100/utterance_similarities_per_speaker',
                        help='Base directory containing JSON files')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes')
    parser.add_argument('--dry_run', action='store_true',
                        help='Dry run mode: only show what would be done without actually modifying files')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("=== Remove utterance_paths from JSON files ===")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Dry run mode: {args.dry_run}")
    logger.info("=" * 50)
    
    # 查找所有JSON文件
    logger.info("Scanning for JSON files...")
    json_files = find_json_files(args.base_dir)
    logger.info(f"Found {len(json_files)} JSON files")
    
    if not json_files:
        logger.warning("No JSON files found!")
        return
    
    # 处理文件
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    total_size_reduction = 0
    total_original_size = 0
    total_new_size = 0
    
    error_messages = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_json, json_file, args.dry_run): json_file 
                   for json_file in json_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            json_file = futures[future]
            try:
                result = future.result()
                
                if result['status'] == 'success':
                    total_processed += 1
                    total_original_size += result['original_size']
                    total_new_size += result['new_size']
                    total_size_reduction += result['size_reduction']
                elif result['status'] == 'dry_run':
                    total_processed += 1
                    total_original_size += result['original_size']
                    total_new_size += result['estimated_size']
                    total_size_reduction += result['size_reduction']
                elif result['status'] == 'skipped':
                    total_skipped += 1
                elif result['status'] == 'error':
                    total_errors += 1
                    error_msg = result.get('error', 'Unknown error')
                    error_messages.append(f"{result['file']}: {error_msg}")
                    if len(error_messages) <= 20:
                        logger.warning(f"Error processing {result['file']}: {error_msg}")
                
            except Exception as e:
                total_errors += 1
                error_messages.append(f"{json_file}: {e}")
                logger.error(f"Error processing {json_file}: {e}", exc_info=True)
    
    # 输出统计信息
    logger.info("\n" + "=" * 50)
    logger.info("📊 Processing Summary:")
    logger.info(f"  ✅ Processed: {total_processed} files")
    logger.info(f"  ⏭️  Skipped: {total_skipped} files")
    logger.info(f"  ❌ Errors: {total_errors} files")
    
    if total_processed > 0:
        logger.info(f"\n💾 Size Statistics:")
        logger.info(f"  Original total size: {total_original_size / (1024**2):.2f} MB")
        logger.info(f"  New total size: {total_new_size / (1024**2):.2f} MB")
        logger.info(f"  Total size reduction: {total_size_reduction / (1024**2):.2f} MB")
        logger.info(f"  Average size reduction per file: {total_size_reduction / total_processed / 1024:.2f} KB")
        if total_original_size > 0:
            reduction_percent = (total_size_reduction / total_original_size) * 100
            logger.info(f"  Size reduction percentage: {reduction_percent:.2f}%")
    
    if error_messages:
        logger.info(f"\n⚠️  Error Summary (showing up to 20 errors):")
        for msg in error_messages[:20]:
            logger.warning(f"  {msg}")
        if len(error_messages) > 20:
            logger.warning(f"  ... and {len(error_messages) - 20} more errors")
    
    if args.dry_run:
        logger.info("\n💡 This was a dry run. Use without --dry_run to actually modify files.")
    else:
        logger.info("\n✅ All files processed successfully!")

if __name__ == "__main__":
    main()


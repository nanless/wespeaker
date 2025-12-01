#!/usr/bin/env python3
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
计算每个说话人内部utterances之间的相似度
对每个说话人的所有utterances计算相似度矩阵，并保存为JSON文件
"""

import os
import sys
import json
import pickle
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import logging
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Compute utterance similarities per speaker')
    parser.add_argument('--embeddings_dir', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_wespeaker_samresnet100',
                        help='Base embeddings directory')
    parser.add_argument('--utterances_subdir', type=str, default='embeddings_utterances',
                        help='Subdirectory containing utterance embeddings')
    parser.add_argument('--output_subdir', type=str, default='utterance_similarities_per_speaker',
                        help='Subdirectory for output (will be created under embeddings_dir)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of worker processes for processing speakers (default: 1 to process one speaker at a time)')
    parser.add_argument('--num_workers_internal', type=int, default=None,
                        help='Number of worker processes for internal computation within each speaker (default: min(8, cpu_count))')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for processing speakers')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip speakers that already have similarity files')
    parser.add_argument('--min_utterances', type=int, default=2,
                        help='Minimum number of utterances required for similarity computation')
    parser.add_argument('--max_speakers', type=int, default=None,
                        help='Maximum number of speakers to process (for testing)')
    parser.add_argument('--max_utterances', type=int, default=None,
                        help='Maximum number of utterances to process per speaker (None for no limit)')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                        help='Only save similarity pairs above this threshold (default: 0.7)')
    parser.add_argument('--max_utterances_limit', type=int, default=5000,
                        help='Maximum number of utterances per speaker to process (speakers with more will be skipped, default: 5000)')
    
    return parser.parse_args()

def load_utterance_embedding(file_path):
    """加载单个utterance的embedding"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        embedding = data.get('embedding', None)
        if embedding is None:
            return None, None
            
        # Ensure embedding is numpy array and flatten if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        # Check for invalid values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return None, None
            
        return embedding, data
        
    except Exception as e:
        return None, None

def load_embeddings_chunk(file_paths_chunk):
    """多进程加载一批utterance embeddings"""
    embeddings_chunk = []
    utterance_data_chunk = []
    utterance_paths_chunk = []
    
    for file_path in file_paths_chunk:
        embedding, data = load_utterance_embedding(file_path)
        if embedding is not None and data is not None:
            embeddings_chunk.append(embedding)
            utterance_data_chunk.append(data)
            utterance_paths_chunk.append(file_path)
    
    return embeddings_chunk, utterance_data_chunk, utterance_paths_chunk

def compute_similarity_chunk(args):
    """多进程计算相似度矩阵的一个chunk"""
    chunk_embeddings, all_embeddings, start_idx, end_idx = args
    chunk_similarities = cosine_similarity(chunk_embeddings, all_embeddings)
    return start_idx, end_idx, chunk_similarities

def compute_speaker_utterance_similarities(speaker_info, max_utterances=None, num_workers_internal=None, similarity_threshold=0.7):
    """计算单个说话人内部utterances的相似度（完整信息，使用多进程加速）"""
    dataset_name, speaker_id, utterance_files = speaker_info
    
    try:
        # Process all utterances (no limit unless max_utterances is explicitly set)
        files_to_process = utterance_files
        if max_utterances is not None and len(utterance_files) > max_utterances:
            # Only limit if max_utterances is explicitly set
            import random
            files_to_process = random.sample(utterance_files, max_utterances)
        
        # Load embeddings using multiprocessing
        num_workers_load = num_workers_internal or min(8, mp.cpu_count())
        chunk_size_load = max(1, len(files_to_process) // num_workers_load)
        
        embeddings = []
        utterance_data_list = []
        utterance_paths = []  # Store full paths for mapping
        
        if len(files_to_process) > 100:  # Use multiprocessing for large speakers
            # Split files into chunks for parallel loading
            file_chunks = [files_to_process[i:i+chunk_size_load] 
                          for i in range(0, len(files_to_process), chunk_size_load)]
            
            with ProcessPoolExecutor(max_workers=num_workers_load) as executor:
                futures = [executor.submit(load_embeddings_chunk, chunk) for chunk in file_chunks]
                for future in futures:
                    emb_chunk, data_chunk, paths_chunk = future.result()
                    embeddings.extend(emb_chunk)
                    utterance_data_list.extend(data_chunk)
                    utterance_paths.extend(paths_chunk)
        else:
            # For small speakers, load sequentially
            for file_path in files_to_process:
                embedding, data = load_utterance_embedding(file_path)
                if embedding is not None and data is not None:
                    embeddings.append(embedding)
                    utterance_data_list.append(data)
                    utterance_paths.append(file_path)
        
        # Create mapping: path -> numeric id (0-indexed)
        path_to_id = {path: idx for idx, path in enumerate(utterance_paths)}
        
        if len(embeddings) < 2:
            return None, f"Speaker {dataset_name}/{speaker_id} has only {len(embeddings)} valid utterances (min: 2)"
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        num_utterances = len(embeddings)
        
        # Compute similarity matrix using multiprocessing
        num_workers_compute = num_workers_internal or min(8, mp.cpu_count())
        chunk_size = max(100, num_utterances // num_workers_compute)
        
        if num_utterances > 200:  # Use multiprocessing for computation
            # For large speakers, compute similarity matrix in chunks using multiprocessing
            similarity_matrix = np.zeros((num_utterances, num_utterances), dtype=np.float32)
            
            # Create chunks for parallel computation
            compute_chunks = []
            for i in range(0, num_utterances, chunk_size):
                end_i = min(i + chunk_size, num_utterances)
                chunk_embeddings_i = embeddings_array[i:end_i]
                compute_chunks.append((chunk_embeddings_i, embeddings_array, i, end_i))
            
            # Compute chunks in parallel
            with ProcessPoolExecutor(max_workers=num_workers_compute) as executor:
                futures = [executor.submit(compute_similarity_chunk, chunk) for chunk in compute_chunks]
                for future in futures:
                    start_idx, end_idx, chunk_similarities = future.result()
                    similarity_matrix[start_idx:end_idx, :] = chunk_similarities
                    del chunk_similarities
                
        else:
            # For smaller speakers, compute full matrix at once
            similarity_matrix = cosine_similarity(embeddings_array)
        
        # Create similarity pairs (upper triangular matrix, excluding diagonal)
        # Only save pairs above threshold to reduce file size
        # Use numeric ids instead of utterance_id strings to reduce file size
        similarity_pairs = []
        all_similarities_flat = []
        
        for i in range(num_utterances):
            for j in range(i + 1, num_utterances):
                sim_value = float(similarity_matrix[i, j])
                all_similarities_flat.append(sim_value)
                
                # Only save pairs above threshold
                if sim_value >= similarity_threshold:
                    similarity_pairs.append({
                        'id_1': i,  # Use numeric id instead of utterance_id
                        'id_2': j,  # Use numeric id instead of utterance_id
                        'similarity': sim_value
                    })
        
        # Sort pairs by similarity (descending)
        similarity_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Calculate statistics from all pairs (for accurate statistics)
        similarities_flat = np.array(all_similarities_flat)
        
        # Build result dictionary with complete information
        result = {
            'dataset': dataset_name,
            'speaker_id': speaker_id,
            'num_utterances': num_utterances,
            'num_utterances_total': len(utterance_files),
            'utterance_paths': utterance_paths,  # List of paths, index corresponds to numeric id
            'path_to_id': path_to_id,  # Mapping: path -> numeric id (for reverse lookup)
            'similarity_pairs': similarity_pairs,  # Pairs with numeric ids only (id_1, id_2, similarity)
            'statistics': {
                'mean': float(np.mean(similarities_flat)),
                'std': float(np.std(similarities_flat)),
                'min': float(np.min(similarities_flat)),
                'max': float(np.max(similarities_flat)),
                'median': float(np.median(similarities_flat))
            },
            'num_pairs_total': len(all_similarities_flat),
            'num_pairs_saved': len(similarity_pairs),
            'similarity_threshold': similarity_threshold
        }
        
        # Free memory
        del embeddings_array, embeddings, similarity_matrix, similarities_flat
        
        return result, None
        
    except MemoryError as e:
        return None, f"Memory error for {dataset_name}/{speaker_id}: {e}"
    except Exception as e:
        return None, f"Error processing {dataset_name}/{speaker_id}: {str(e)}"

def process_single_speaker(speaker_info, output_dir, skip_existing, min_utterances, max_utterances, num_workers_internal, similarity_threshold, max_utterances_limit):
    """处理单个说话人（独立进程，避免相互影响）"""
    dataset_name, speaker_id, utterance_files = speaker_info
    
    try:
        # Check if output file already exists
        if skip_existing:
            output_file = Path(output_dir) / dataset_name / f'{speaker_id}_utterance_similarities.json'
            if output_file.exists():
                return {
                    'speaker_key': f"{dataset_name}/{speaker_id}",
                    'status': 'skipped',
                    'reason': 'already exists'
                }
        
        # Check minimum utterances
        if len(utterance_files) < min_utterances:
            return {
                'speaker_key': f"{dataset_name}/{speaker_id}",
                'status': 'skipped',
                'reason': f'only {len(utterance_files)} utterances (min: {min_utterances})'
            }
        
        # Check maximum utterances (skip if too many to avoid huge similarity matrix)
        if len(utterance_files) > max_utterances_limit:
            return {
                'speaker_key': f"{dataset_name}/{speaker_id}",
                'status': 'skipped',
                'reason': f'too many utterances ({len(utterance_files)} > {max_utterances_limit}), similarity matrix would be too large'
            }
        
        # Compute similarities with retry mechanism
        result = None
        error_msg = None
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                # Compute similarities (timeout handled by ProcessPoolExecutor)
                result, error_msg = compute_speaker_utterance_similarities(
                    speaker_info,
                    max_utterances=max_utterances,
                    num_workers_internal=num_workers_internal,
                    similarity_threshold=similarity_threshold
                )
                
                if result is not None:
                    break
                # If result is None but no exception, it's a valid skip case
                if retry == 0 and error_msg and "only" in error_msg.lower():
                    break
                    
            except TimeoutError as e:
                error_msg = f"Timeout (retry {retry+1}/{max_retries}): {e}"
                if retry < max_retries - 1:
                    # Try with reduced utterances if max_utterances was set
                    if max_utterances is not None:
                        max_utterances_reduced = max(1000, max_utterances // 2)
                        speaker_info_reduced = (dataset_name, speaker_id, utterance_files[:max_utterances_reduced])
                        speaker_info = speaker_info_reduced
                    import gc
                    gc.collect()
                    continue
            except MemoryError as e:
                error_msg = f"Memory error (retry {retry+1}/{max_retries}): {e}"
                if retry < max_retries - 1:
                    # Try with reduced utterances if max_utterances was set, otherwise use 50000 as fallback
                    if max_utterances is not None:
                        max_utterances_reduced = max(1000, max_utterances // 2)
                    else:
                        # If no limit was set, use a reasonable fallback to avoid memory issues
                        max_utterances_reduced = min(50000, len(utterance_files))
                    speaker_info_reduced = (dataset_name, speaker_id, utterance_files[:max_utterances_reduced])
                    speaker_info = speaker_info_reduced
                    import gc
                    gc.collect()
                    continue
            except Exception as e:
                error_msg = f"Error (retry {retry+1}/{max_retries}): {e}"
                if retry < max_retries - 1:
                    import gc
                    gc.collect()
                    continue
        
        if result is None:
            return {
                'speaker_key': f"{dataset_name}/{speaker_id}",
                'status': 'error',
                'error': error_msg or 'Unknown error'
            }
        
        # Save result with error handling
        try:
            output_file = Path(output_dir) / dataset_name / f'{speaker_id}_utterance_similarities.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use temporary file to ensure atomic write
            temp_file = output_file.with_name(f'{speaker_id}_utterance_similarities.json.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(output_file)
            
            # Free memory
            del result
            import gc
            gc.collect()
            
            return {
                'speaker_key': f"{dataset_name}/{speaker_id}",
                'status': 'success'
            }
            
        except MemoryError as e:
            return {
                'speaker_key': f"{dataset_name}/{speaker_id}",
                'status': 'error',
                'error': f"Memory error saving: {e}"
            }
        except Exception as e:
            # Clean up temp file if exists
            temp_file = Path(output_dir) / dataset_name / f'{speaker_id}_utterance_similarities.json.tmp'
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            return {
                'speaker_key': f"{dataset_name}/{speaker_id}",
                'status': 'error',
                'error': f"Error saving: {e}"
            }
        
    except Exception as e:
        return {
            'speaker_key': f"{dataset_name}/{speaker_id}",
            'status': 'error',
            'error': f"Unexpected error: {e}"
        }

def scan_speaker_utterances(utterances_dir, max_speakers=None):
    """扫描所有说话人的utterance文件"""
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning utterance files in: {utterances_dir}")
    
    speaker_utterances = []
    utterances_path = Path(utterances_dir)
    
    if not utterances_path.exists():
        logger.error(f"Utterances directory does not exist: {utterances_dir}")
        return speaker_utterances
    
    speaker_count = 0
    
    for dataset_dir in utterances_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        logger.info(f"Processing dataset: {dataset_name}")
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            if max_speakers and speaker_count >= max_speakers:
                break
            
            speaker_id = speaker_dir.name
            
            # Collect all pkl files for this speaker
            utterance_files = [str(f) for f in speaker_dir.glob('*.pkl')]
            
            if utterance_files:
                speaker_utterances.append((dataset_name, speaker_id, utterance_files))
                speaker_count += 1
        
        if max_speakers and speaker_count >= max_speakers:
            break
    
    logger.info(f"Found {len(speaker_utterances)} speakers with utterances")
    total_utterances = sum(len(files) for _, _, files in speaker_utterances)
    logger.info(f"Total utterances: {total_utterances}")
    
    return speaker_utterances


def main():
    args = parse_args()
    logger = setup_logging()
    
    logger.info("=== Utterance Similarity Computation Per Speaker ===")
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Utterances subdirectory: {args.utterances_subdir}")
    logger.info(f"Output subdirectory: {args.output_subdir}")
    logger.info(f"Number of workers (speakers): {args.num_workers}")
    logger.info(f"Number of workers (internal): {args.num_workers_internal or min(8, mp.cpu_count())}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info(f"Minimum utterances: {args.min_utterances}")
    if args.max_utterances is not None:
        logger.info(f"Maximum utterances per speaker: {args.max_utterances}")
    else:
        logger.info(f"Maximum utterances per speaker: no limit")
    logger.info(f"Maximum utterances limit: {args.max_utterances_limit} (speakers with more will be skipped, matrix too large)")
    logger.info(f"Similarity threshold: {args.similarity_threshold} (only pairs >= threshold will be saved)")
    logger.info("=====================================================")
    
    # Check input directory
    utterances_dir = Path(args.embeddings_dir) / args.utterances_subdir
    if not utterances_dir.exists():
        logger.error(f"Utterances directory does not exist: {utterances_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.embeddings_dir) / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan speaker utterances
    speaker_utterances = scan_speaker_utterances(utterances_dir, args.max_speakers)
    
    if not speaker_utterances:
        logger.error("No speaker utterances found!")
        sys.exit(1)
    
    # Process each speaker individually (one process per speaker)
    # This ensures that one speaker failure doesn't affect others
    start_time = time.time()
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    error_messages = []
    
    # Use ProcessPoolExecutor with one speaker per process
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all speakers as individual tasks
        futures = {}
        for speaker_info in speaker_utterances:
            dataset_name, speaker_id, _ = speaker_info
            future = executor.submit(
                process_single_speaker,
                speaker_info,
                output_dir,
                args.skip_existing,
                args.min_utterances,
                args.max_utterances,
                args.num_workers_internal,
                args.similarity_threshold,
                args.max_utterances_limit
            )
            futures[future] = f"{dataset_name}/{speaker_id}"
        
        # Collect results with progress tracking
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing speakers"):
            speaker_key = futures[future]
            try:
                result = future.result(timeout=14400)  # 4 hours timeout per speaker (for very large speakers)
                
                if result['status'] == 'success':
                    total_processed += 1
                elif result['status'] == 'skipped':
                    total_skipped += 1
                elif result['status'] == 'error':
                    total_errors += 1
                    error_msg = result.get('error', 'Unknown error')
                    error_messages.append(f"{speaker_key}: {error_msg}")
                    if len(error_messages) <= 20:  # Log first 20 errors
                        logger.warning(f"Error processing {speaker_key}: {error_msg}")
                
            except TimeoutError:
                total_errors += 1
                error_msg = f"Timeout after 4 hours"
                error_messages.append(f"{speaker_key}: {error_msg}")
                logger.error(f"Timeout processing {speaker_key}")
            except Exception as e:
                total_errors += 1
                error_msg = f"Unexpected error: {e}"
                error_messages.append(f"{speaker_key}: {error_msg}")
                logger.error(f"Error processing {speaker_key}: {e}", exc_info=True)
    
    # Final statistics
    total_time = time.time() - start_time
    
    logger.info(f"\n🎉 Utterance similarity computation completed!")
    logger.info(f"📊 Statistics:")
    logger.info(f"  ✅ Processed: {total_processed} speakers")
    logger.info(f"  ⏭️  Skipped: {total_skipped} speakers")
    logger.info(f"  ❌ Errors: {total_errors} speakers")
    logger.info(f"  ⏱️  Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    if total_processed > 0:
        logger.info(f"  🚀 Processing rate: {total_processed/total_time:.2f} speakers/sec")
    
    if error_messages:
        logger.info(f"\n⚠️  Error Summary (showing up to 20 errors):")
        for msg in error_messages[:20]:
            logger.warning(f"  {msg}")
        if len(error_messages) > 20:
            logger.warning(f"  ... and {len(error_messages) - 20} more errors")
    
    # Show dataset breakdown
    logger.info(f"\n📂 Dataset breakdown:")
    for dataset_dir in output_dir.iterdir():
        if dataset_dir.is_dir():
            json_count = len(list(dataset_dir.glob('*_utterance_similarities.json')))
            logger.info(f"  {dataset_dir.name}: {json_count} similarity files")
    
    logger.info(f"\n📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()


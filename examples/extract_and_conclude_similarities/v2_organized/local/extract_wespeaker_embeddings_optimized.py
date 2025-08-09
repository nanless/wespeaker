#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import time
import threading
import queue
import random

# Add wespeaker to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from wespeaker.cli.speaker import load_model_local

def get_args():
    parser = argparse.ArgumentParser(description='Extract embeddings for all audio files in dataset directories (Optimized)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing dataset_names/speaker_ids/audio_files structure')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the wespeaker model files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for individual embedding files')
    parser.add_argument('--gpus', default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')
    parser.add_argument('--port', default='12355',
                        help='Port number for distributed training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing (higher = faster)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of file I/O worker threads per GPU')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='Skip files that already have embeddings')
    parser.add_argument('--random_shuffle', action='store_true', default=True,
                        help='Randomly shuffle audio files for better load balancing (default: True)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
    
    return parser.parse_args()

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def scan_audio_files_optimized(data_root, output_dir=None, skip_existing=True):
    """Optimized scanning with optional skip existing files."""
    audio_files = []
    skipped_count = 0
    
    print("Scanning audio files...")
    start_time = time.time()
    
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == 'embeddings':
            continue
            
        dataset_name = dataset_dir.name
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            
            for audio_file in speaker_dir.iterdir():
                if audio_file.suffix.lower() in ['.wav', '.flac', '.mp3']:
                    # Check if embedding already exists
                    if skip_existing and output_dir:
                        embedding_path = os.path.join(output_dir, dataset_name, speaker_id, f"{audio_file.stem}.pkl")
                        if os.path.exists(embedding_path):
                            skipped_count += 1
                            continue
                    
                    audio_files.append({
                        'path': str(audio_file),
                        'dataset': dataset_name,
                        'speaker_id': speaker_id,
                        'utterance_id': audio_file.stem,
                        'filename': audio_file.name
                    })
    
    scan_time = time.time() - start_time
    print(f"Scanning completed in {scan_time:.2f}s. Found {len(audio_files)} new files to process, skipped {skipped_count} existing files.")
    
    return audio_files

def save_embedding_batch(embedding_batch, file_info_batch, output_dir):
    """Save multiple embeddings in batch for better I/O performance."""
    save_paths = []
    
    for embedding, file_info in zip(embedding_batch, file_info_batch):
        dataset = file_info['dataset']
        speaker_id = file_info['speaker_id']
        utterance_id = file_info['utterance_id']
        
        # Create directory structure: output_dir/dataset/speaker_id/
        embedding_dir = os.path.join(output_dir, dataset, speaker_id)
        os.makedirs(embedding_dir, exist_ok=True)
        
        # Save embedding as pickle file
        save_path = os.path.join(embedding_dir, f"{utterance_id}.pkl")
        
        embedding_data = {
            'embedding': embedding.flatten() if isinstance(embedding, np.ndarray) else embedding.detach().cpu().numpy().flatten(),
            'dataset': dataset,
            'speaker_id': speaker_id,
            'utterance_id': utterance_id,
            'original_path': file_info['path']
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        save_paths.append(save_path)
    
    return save_paths

def io_worker(save_queue, output_dir):
    """Worker thread for handling file I/O operations."""
    while True:
        item = save_queue.get()
        if item is None:
            break
        
        embedding_batch, file_info_batch = item
        try:
            save_embedding_batch(embedding_batch, file_info_batch, output_dir)
        except Exception as e:
            print(f"I/O Error: {e}")
        finally:
            save_queue.task_done()

def process_subset_optimized(rank, world_size, args):
    """Optimized processing with batch inference and parallel I/O."""
    setup(rank, world_size, args.port)
    
    # Load model on specific GPU
    print(f"GPU {rank}: Loading model...")
    model = load_model_local(args.model_dir)
    model.set_device(f'cuda:{rank}')
    
    # Scan all audio files (only rank 0 does this to avoid race conditions)
    if rank == 0:
        all_audio_files = scan_audio_files_optimized(args.data_root, args.output_dir, args.skip_existing)
        
        # Randomly shuffle files for better load balancing
        if args.random_shuffle:
            print(f"Shuffling audio files with seed {args.random_seed} for load balancing...")
            random.seed(args.random_seed)
            random.shuffle(all_audio_files)
            print("Audio files shuffled successfully.")
        
        # Save to shared file for other ranks
        with open('/tmp/audio_files_list.pkl', 'wb') as f:
            pickle.dump(all_audio_files, f)
    
    # Wait for rank 0 to finish scanning
    dist.barrier()
    
    # Load audio files list
    if rank != 0:
        with open('/tmp/audio_files_list.pkl', 'rb') as f:
            all_audio_files = pickle.load(f)
    
    if rank == 0:
        print(f"Total audio files to process: {len(all_audio_files)}")
    
    # Split data among GPUs
    num_files = len(all_audio_files)
    start_idx = (num_files * rank) // world_size
    end_idx = (num_files * (rank + 1)) // world_size
    subset_files = all_audio_files[start_idx:end_idx]
    
    print(f"GPU {rank}: Processing {len(subset_files)} files (from {start_idx} to {end_idx-1})")
    
    if len(subset_files) == 0:
        cleanup()
        return
    
    # Set up I/O worker threads
    save_queue = queue.Queue(maxsize=args.num_workers * 2)
    io_threads = []
    for _ in range(args.num_workers):
        t = threading.Thread(target=io_worker, args=(save_queue, args.output_dir))
        t.daemon = True
        t.start()
        io_threads.append(t)
    
    # Process files in batches
    success_count = 0
    error_count = 0
    batch_size = args.batch_size
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(subset_files), batch_size), desc=f'GPU {rank}'):
        batch_files = subset_files[i:i+batch_size]
        batch_embeddings = []
        batch_info = []
        
        # Extract embeddings in batch
        for file_info in batch_files:
            try:
                wav_path = file_info['path']
                embedding = model.extract_embedding(wav_path)
                
                if embedding is not None:
                    batch_embeddings.append(embedding)
                    batch_info.append(file_info)
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                if rank == 0 and error_count < 10:  # Only show first 10 errors
                    print(f"GPU {rank}: Error processing {file_info['path']}: {e}")
                error_count += 1
        
        # Send batch to I/O workers
        if batch_embeddings:
            save_queue.put((batch_embeddings, batch_info))
        
        # Progress update
        if success_count % 500 == 0 and success_count > 0:
            elapsed = time.time() - start_time
            rate = success_count / elapsed
            print(f"GPU {rank}: Processed {success_count} files successfully at {rate:.1f} files/sec")
    
    # Wait for all I/O operations to complete
    save_queue.join()
    
    # Stop I/O workers
    for _ in io_threads:
        save_queue.put(None)
    for t in io_threads:
        t.join()
    
    total_time = time.time() - start_time
    print(f"GPU {rank} completed: {success_count} successful, {error_count} errors in {total_time:.2f}s ({success_count/total_time:.1f} files/sec)")
    
    cleanup()

def main():
    args = get_args()
    
    # Set up GPU configuration
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs available!")
        sys.exit(1)
    
    print(f"Using {world_size} GPUs")
    print(f"Data root: {args.data_root}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"I/O workers per GPU: {args.num_workers}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Random shuffle for load balancing: {args.random_shuffle}")
    if args.random_shuffle:
        print(f"Random seed: {args.random_seed}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Start multiple processes
    if world_size > 1:
        mp.spawn(
            process_subset_optimized,
            args=(world_size, args),
            nprocs=world_size,    
            join=True
        )
    else:
        # Single GPU case
        process_subset_optimized(0, 1, args)
    
    total_time = time.time() - start_time
    
    print(f"Embedding extraction completed in {total_time:.2f}s! Results saved in {args.output_dir}")
    
    # Print summary
    total_embeddings = 0
    for root, dirs, files in os.walk(args.output_dir):
        total_embeddings += len([f for f in files if f.endswith('.pkl')])
    
    print(f"Total embeddings extracted: {total_embeddings}")
    print(f"Average processing rate: {total_embeddings/total_time:.1f} files/sec")

if __name__ == "__main__":
    main() 
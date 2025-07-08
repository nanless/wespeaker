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

# Add wespeaker to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wespeaker.cli.speaker import load_model_local

def get_args():
    parser = argparse.ArgumentParser(description='Extract embeddings for all audio files in dataset directories')
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing (recommended: 1 for individual file extraction)')
    
    return parser.parse_args()

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def scan_audio_files(data_root):
    """Scan the directory structure to find all audio files."""
    audio_files = []
    
    for dataset_dir in Path(data_root).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"Scanning dataset: {dataset_name}")
        
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            
            for audio_file in speaker_dir.iterdir():
                if audio_file.suffix.lower() in ['.wav', '.flac', '.mp3']:
                    audio_files.append({
                        'path': str(audio_file),
                        'dataset': dataset_name,
                        'speaker_id': speaker_id,
                        'utterance_id': audio_file.stem,
                        'filename': audio_file.name
                    })
    
    return audio_files

def save_individual_embedding(embedding, file_info, output_dir):
    """Save individual embedding file maintaining directory structure."""
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
    
    return save_path

def process_subset(rank, world_size, args):
    """Process a subset of audio files on a specific GPU."""
    setup(rank, world_size, args.port)
    
    # Load model on specific GPU
    print(f"GPU {rank}: Loading model...")
    model = load_model_local(args.model_dir)
    model.set_device(f'cuda:{rank}')
    
    # Scan all audio files
    if rank == 0:
        print("Scanning audio files...")
    all_audio_files = scan_audio_files(args.data_root)
    
    if rank == 0:
        print(f"Total audio files found: {len(all_audio_files)}")
    
    # Split data among GPUs
    num_files = len(all_audio_files)
    start_idx = (num_files * rank) // world_size
    end_idx = (num_files * (rank + 1)) // world_size
    subset_files = all_audio_files[start_idx:end_idx]
    
    print(f"GPU {rank}: Processing {len(subset_files)} files (from {start_idx} to {end_idx-1})")
    
    # Process files
    success_count = 0
    error_count = 0
    
    for file_info in tqdm(subset_files, desc=f'GPU {rank}'):
        try:
            wav_path = file_info['path']
            
            # Extract embedding
            embedding = model.extract_embedding(wav_path)
            
            if embedding is not None:
                # Save individual embedding
                save_path = save_individual_embedding(embedding, file_info, args.output_dir)
                success_count += 1
                
                if success_count % 100 == 0:
                    print(f"GPU {rank}: Processed {success_count} files successfully")
            else:
                print(f"GPU {rank}: Failed to extract embedding from {wav_path}")
                error_count += 1
                
        except Exception as e:
            print(f"GPU {rank}: Error processing {file_info['path']}: {e}")
            error_count += 1
    
    print(f"GPU {rank} completed: {success_count} successful, {error_count} errors")
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start multiple processes
    if world_size > 1:
        mp.spawn(
            process_subset,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU case
        process_subset(0, 1, args)
    
    print(f"Embedding extraction completed! Results saved in {args.output_dir}")
    
    # Print summary
    total_embeddings = 0
    for root, dirs, files in os.walk(args.output_dir):
        total_embeddings += len([f for f in files if f.endswith('.pkl')])
    
    print(f"Total embeddings extracted: {total_embeddings}")

if __name__ == "__main__":
    main() 
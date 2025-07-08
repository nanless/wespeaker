#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import kaldiio

from wespeaker.cli.speaker import load_model_local

def get_args():
    parser = argparse.ArgumentParser(description='Extract speaker embeddings using multiple GPUs')
    parser.add_argument('--model_dir', required=True,
                      help='Directory containing the model files')
    parser.add_argument('--data_dir', required=True,
                      help='Directory containing wav.scp')
    parser.add_argument('--exp_dir', required=True,
                      help='Directory to save embeddings')
    parser.add_argument('--gpus', default=None,
                      help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3"). If not specified, use all available GPUs.')
    parser.add_argument('--port', default='12355',
                      help='Port number for distributed training')
    args = parser.parse_args()
    return args

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def process_subset(rank, world_size, args):
    setup(rank, world_size, args.port)
    
    # Load model on specific GPU
    model = load_model_local(args.model_dir)
    model.set_device(f'cuda:{rank}')
    
    # Create output directory
    os.makedirs(os.path.join(args.exp_dir, 'embeddings'), exist_ok=True)
    
    # Read wav.scp
    wav_scp = os.path.join(args.data_dir, 'wav.scp')
    with open(wav_scp, 'r') as f:
        lines = f.readlines()
    
    # Split data among GPUs
    num_utts = len(lines)
    start_idx = (num_utts * rank) // world_size
    end_idx = (num_utts * (rank + 1)) // world_size
    subset_lines = lines[start_idx:end_idx]
    
    # Process files
    embeddings_dict = {}
    for line in tqdm(subset_lines, desc=f'GPU {rank}'):
        utt_id, wav_path = line.strip().split()
        embedding = model.extract_embedding(wav_path)
        if embedding is not None:
            embeddings_dict[utt_id] = embedding.detach().cpu().numpy()
    
    # Save embeddings
    ark_path = os.path.join(args.exp_dir, 'embeddings', f'xvector.{rank}.ark')
    scp_path = os.path.join(args.exp_dir, 'embeddings', f'xvector.{rank}.scp')
    
    with kaldiio.WriteHelper(f'ark,scp:{ark_path},{scp_path}') as writer:
        for utt_id, embedding in embeddings_dict.items():
            writer(utt_id, embedding)
    
    cleanup()

def main():
    args = get_args()
    
    # Check wav.scp exists
    wav_scp = os.path.join(args.data_dir, 'wav.scp')
    if not os.path.exists(wav_scp):
        print(f"Error: {wav_scp} does not exist!")
        sys.exit(1)
    
    # Set up GPU configuration
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs available!")
        sys.exit(1)
    
    print(f"Using {world_size} GPUs")
    
    # Start multiple processes
    mp.spawn(
        process_subset,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
    
    # Merge scp files
    merged_scp_path = os.path.join(args.exp_dir, 'embeddings', 'xvector.scp')
    with open(merged_scp_path, 'w') as merged_scp:
        for rank in range(world_size):
            scp_path = os.path.join(args.exp_dir, 'embeddings', f'xvector.{rank}.scp')
            with open(scp_path, 'r') as f:
                merged_scp.write(f.read())
    
    print(f"Embeddings extracted successfully. Results saved in {os.path.join(args.exp_dir, 'embeddings')}")

if __name__ == "__main__":
    main() 
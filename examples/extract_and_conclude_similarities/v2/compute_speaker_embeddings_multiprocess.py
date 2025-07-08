 #!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial

def get_args():
    parser = argparse.ArgumentParser(description='Compute speaker-level embeddings by averaging utterance embeddings (Multi-process)')
    parser.add_argument('--utterances_dir', type=str, 
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual/utterances',
                        help='Directory containing utterance embeddings')
    parser.add_argument('--speakers_dir', type=str,
                        default='/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual/speakers',
                        help='Output directory for speaker embeddings')
    parser.add_argument('--min_utterances', type=int, default=1,
                        help='Minimum number of utterances required for a speaker')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip speakers that already have embeddings')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Number of processes to use (default: CPU count)')
    parser.add_argument('--chunk_size', type=int, default=10,
                        help='Number of speakers to process per chunk')
    
    return parser.parse_args()

def scan_utterance_files(utterances_dir):
    """Scan utterance directory and group files by speaker."""
    print("🔍 Scanning utterance embedding files...")
    
    speaker_files = defaultdict(list)
    total_files = 0
    
    utterances_path = Path(utterances_dir)
    if not utterances_path.exists():
        print(f"❌ Error: Utterances directory does not exist: {utterances_dir}")
        return speaker_files, 0
    
    # Scan all datasets
    for dataset_dir in utterances_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        print(f"  📂 Scanning dataset: {dataset_name}")
        
        # Scan all speakers in this dataset
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            speaker_key = (dataset_name, speaker_id)
            
            # Scan all utterance files for this speaker
            utterance_files = []
            for file_path in speaker_dir.iterdir():
                if file_path.suffix.lower() == '.pkl':
                    utterance_files.append(str(file_path))
                    total_files += 1
            
            if utterance_files:
                speaker_files[speaker_key] = utterance_files
    
    print(f"📊 Found {len(speaker_files)} speakers with {total_files} total utterance files")
    return speaker_files, total_files

def load_utterance_embedding(file_path):
    """Load embedding from a single utterance file."""
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
            
        return embedding, data
        
    except Exception as e:
        return None, None

def compute_speaker_embedding(utterance_files, speaker_key):
    """Compute average embedding for a speaker from all their utterances."""
    dataset_name, speaker_id = speaker_key
    
    embeddings = []
    utterance_data = []
    failed_files = 0
    
    for file_path in utterance_files:
        embedding, data = load_utterance_embedding(file_path)
        
        if embedding is not None and data is not None:
            embeddings.append(embedding)
            utterance_data.append(data)
        else:
            failed_files += 1
    
    if not embeddings:
        return None
    
    # Compute average embedding
    embeddings_array = np.array(embeddings)
    avg_embedding = np.mean(embeddings_array, axis=0)
    
    # Collect metadata
    utterance_ids = [data.get('utterance_id', 'unknown') for data in utterance_data]
    original_paths = [data.get('original_path', 'unknown') for data in utterance_data]
    
    # Create speaker embedding data
    speaker_data = {
        'embedding': avg_embedding,
        'dataset': dataset_name,
        'speaker_id': speaker_id,
        'num_utterances': len(embeddings),
        'failed_utterances': failed_files,
        'utterance_list': utterance_ids,
        'original_paths': original_paths,
        'embedding_dim': len(avg_embedding),
        'embedding_stats': {
            'mean': float(np.mean(avg_embedding)),
            'std': float(np.std(avg_embedding)),
            'min': float(np.min(avg_embedding)),
            'max': float(np.max(avg_embedding))
        }
    }
    
    return speaker_data

def save_speaker_embedding(speaker_data, speakers_dir):
    """Save speaker embedding to file."""
    dataset_name = speaker_data['dataset']
    speaker_id = speaker_data['speaker_id']
    
    # Create output directory
    output_dir = Path(speakers_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save speaker embedding
    output_file = output_dir / f"{speaker_id}.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(speaker_data, f)
    
    return str(output_file)

def process_speaker_batch(speaker_batch, speakers_dir, skip_existing, process_id):
    """Process a batch of speakers in a single process."""
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for speaker_key, utterance_files in speaker_batch:
        dataset_name, speaker_id = speaker_key
        
        try:
            # Check if speaker embedding already exists
            if skip_existing:
                speaker_file = Path(speakers_dir) / dataset_name / f"{speaker_id}.pkl"
                if speaker_file.exists():
                    skipped_count += 1
                    continue
            
            # Compute speaker embedding
            speaker_data = compute_speaker_embedding(utterance_files, speaker_key)
            
            if speaker_data is not None:
                # Save speaker embedding
                output_file = save_speaker_embedding(speaker_data, speakers_dir)
                processed_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
    
    return {
        'process_id': process_id,
        'processed': processed_count,
        'skipped': skipped_count,
        'errors': error_count
    }

def create_speaker_batches(filtered_speakers, num_processes, chunk_size):
    """Create balanced batches of speakers for multiprocessing."""
    speaker_items = list(filtered_speakers.items())
    
    # Calculate optimal batch size
    total_speakers = len(speaker_items)
    
    if num_processes >= total_speakers:
        # More processes than speakers, assign one speaker per process
        batches = [[item] for item in speaker_items]
    else:
        # Create balanced batches
        speakers_per_process = total_speakers // num_processes
        remainder = total_speakers % num_processes
        
        batches = []
        start_idx = 0
        
        for i in range(num_processes):
            # Add one extra speaker to first 'remainder' processes
            batch_size = speakers_per_process + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            
            if start_idx < total_speakers:
                batches.append(speaker_items[start_idx:end_idx])
                start_idx = end_idx
    
    # Remove empty batches
    batches = [batch for batch in batches if batch]
    
    return batches

def main():
    args = get_args()
    
    # Set number of processes
    if args.num_processes is None:
        args.num_processes = mp.cpu_count()
    
    print("=== Speaker Embedding Computation (Multi-process) ===")
    print(f"Utterances directory: {args.utterances_dir}")
    print(f"Speakers directory: {args.speakers_dir}")
    print(f"Minimum utterances: {args.min_utterances}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Number of processes: {args.num_processes}")
    print(f"Chunk size: {args.chunk_size}")
    print("=====================================================")
    
    # Check input directory
    if not os.path.exists(args.utterances_dir):
        print(f"❌ Error: Utterances directory does not exist: {args.utterances_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.speakers_dir, exist_ok=True)
    
    # Scan utterance files
    speaker_files, total_utterances = scan_utterance_files(args.utterances_dir)
    
    if not speaker_files:
        print("❌ No speaker files found!")
        sys.exit(1)
    
    # Filter speakers by minimum utterances
    filtered_speakers = {}
    for speaker_key, files in speaker_files.items():
        if len(files) >= args.min_utterances:
            filtered_speakers[speaker_key] = files
        else:
            dataset_name, speaker_id = speaker_key
            print(f"⚠️  Skipping {dataset_name}/{speaker_id}: only {len(files)} utterances (min: {args.min_utterances})")
    
    print(f"📈 Processing {len(filtered_speakers)} speakers (filtered from {len(speaker_files)})")
    
    if not filtered_speakers:
        print("❌ No speakers meet the minimum utterances requirement!")
        sys.exit(1)
    
    # Create speaker batches for multiprocessing
    speaker_batches = create_speaker_batches(filtered_speakers, args.num_processes, args.chunk_size)
    actual_processes = len(speaker_batches)
    
    print(f"🚀 Created {actual_processes} process batches")
    for i, batch in enumerate(speaker_batches):
        print(f"  Process {i}: {len(batch)} speakers")
    
    start_time = time.time()
    
    # Process speakers using multiprocessing
    print(f"⚡ Starting multi-process computation with {actual_processes} processes...")
    
    # Add process IDs to batches with all required arguments
    batch_with_args = [
        (batch, args.speakers_dir, args.skip_existing, i) 
        for i, batch in enumerate(speaker_batches)
    ]
    
    with mp.Pool(processes=actual_processes) as pool:
        # Submit all batches to the pool
        tasks = [pool.apply_async(process_speaker_batch, args) for args in batch_with_args]
        
        # Monitor progress
        results = []
        with tqdm(total=len(tasks), desc="Processing batches") as pbar:
            for task in tasks:
                result = task.get()
                results.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    f"P{result['process_id']}": f"✅{result['processed']} ⏭️{result['skipped']} ❌{result['errors']}"
                })
    
    # Aggregate results
    total_processed = sum(r['processed'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    # Final statistics
    total_time = time.time() - start_time
    
    print(f"\n🎉 Multi-process speaker embedding computation completed!")
    print(f"📊 Statistics:")
    print(f"  ✅ Processed: {total_processed} speakers")
    print(f"  ⏭️  Skipped (existing): {total_skipped} speakers")
    print(f"  ❌ Errors: {total_errors} speakers")
    print(f"  ⏱️  Total time: {total_time:.2f} seconds")
    print(f"  🚀 Processing rate: {total_processed/total_time:.1f} speakers/sec")
    print(f"  ⚡ Speedup: ~{args.num_processes}x (theoretical)")
    
    # Show per-process statistics
    print(f"\n📈 Per-process statistics:")
    for result in results:
        print(f"  Process {result['process_id']}: "
              f"✅{result['processed']} ⏭️{result['skipped']} ❌{result['errors']}")
    
    # Show dataset statistics
    print(f"\n📂 Dataset statistics:")
    speakers_path = Path(args.speakers_dir)
    for dataset_dir in speakers_path.iterdir():
        if dataset_dir.is_dir():
            speaker_count = len(list(dataset_dir.glob('*.pkl')))
            print(f"  {dataset_dir.name}: {speaker_count} speakers")
    
    # Show sample speaker embedding info
    print(f"\n🔍 Sample speaker embedding info:")
    sample_files = list(speakers_path.rglob('*.pkl'))[:3]
    for sample_file in sample_files:
        try:
            with open(sample_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"  📄 {sample_file.parent.name}/{sample_file.stem}:")
            print(f"     Utterances: {data.get('num_utterances', 'unknown')}")
            print(f"     Embedding dim: {data.get('embedding_dim', 'unknown')}")
            print(f"     Mean value: {data.get('embedding_stats', {}).get('mean', 'unknown'):.4f}")
            
        except Exception as e:
            print(f"     Error reading sample: {e}")

if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
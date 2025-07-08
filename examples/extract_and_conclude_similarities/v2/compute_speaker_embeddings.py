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

def get_args():
    parser = argparse.ArgumentParser(description='Compute speaker-level embeddings by averaging utterance embeddings')
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
        print(f"⚠️  Warning: Failed to load {file_path}: {e}")
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
        print(f"❌ No valid embeddings found for {dataset_name}/{speaker_id}")
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

def main():
    args = get_args()
    
    print("=== Speaker Embedding Computation ===")
    print(f"Utterances directory: {args.utterances_dir}")
    print(f"Speakers directory: {args.speakers_dir}")
    print(f"Minimum utterances: {args.min_utterances}")
    print(f"Skip existing: {args.skip_existing}")
    print("=====================================")
    
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
    
    # Process each speaker
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    start_time = time.time()
    
    for speaker_key, utterance_files in tqdm(filtered_speakers.items(), desc="Processing speakers"):
        dataset_name, speaker_id = speaker_key
        
        # Check if speaker embedding already exists
        if args.skip_existing:
            speaker_file = Path(args.speakers_dir) / dataset_name / f"{speaker_id}.pkl"
            if speaker_file.exists():
                skipped_count += 1
                continue
        
        try:
            # Compute speaker embedding
            speaker_data = compute_speaker_embedding(utterance_files, speaker_key)
            
            if speaker_data is not None:
                # Save speaker embedding
                output_file = save_speaker_embedding(speaker_data, args.speakers_dir)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    print(f"  ✅ Processed {processed_count} speakers at {rate:.1f} speakers/sec")
            else:
                error_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {dataset_name}/{speaker_id}: {e}")
            error_count += 1
    
    # Final statistics
    total_time = time.time() - start_time
    
    print(f"\n🎉 Speaker embedding computation completed!")
    print(f"📊 Statistics:")
    print(f"  ✅ Processed: {processed_count} speakers")
    print(f"  ⏭️  Skipped (existing): {skipped_count} speakers")
    print(f"  ❌ Errors: {error_count} speakers")
    print(f"  ⏱️  Total time: {total_time:.2f} seconds")
    print(f"  🚀 Processing rate: {processed_count/total_time:.1f} speakers/sec")
    
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
    main() 
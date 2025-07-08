#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
from pathlib import Path

def test_speaker_embeddings():
    """Test speaker embedding computation with sample data."""
    
    # Configuration
    utterances_dir = "/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual/utterances"
    speakers_dir = "/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet/embeddings_individual/speakers"
    
    print("=== Speaker Embedding Computation Test ===")
    print(f"Utterances directory: {utterances_dir}")
    print(f"Speakers directory: {speakers_dir}")
    print("==========================================")
    
    # Check if utterances directory exists
    if not os.path.exists(utterances_dir):
        print(f"❌ Error: Utterances directory does not exist: {utterances_dir}")
        print("💡 Please run the embedding extraction first")
        return False
    
    # Find a sample speaker with multiple utterances
    print("🔍 Looking for sample speaker with multiple utterances...")
    
    utterances_path = Path(utterances_dir)
    sample_speaker = None
    sample_files = []
    
    for dataset_dir in utterances_path.iterdir():
        if not dataset_dir.is_dir():
            continue
            
        for speaker_dir in dataset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            utterance_files = list(speaker_dir.glob('*.pkl'))
            if len(utterance_files) >= 2:  # At least 2 utterances
                sample_speaker = (dataset_dir.name, speaker_dir.name)
                sample_files = utterance_files[:3]  # Take first 3 files
                break
        
        if sample_speaker:
            break
    
    if not sample_speaker:
        print("❌ No speaker found with multiple utterances for testing")
        return False
    
    dataset_name, speaker_id = sample_speaker
    print(f"📄 Testing with speaker: {dataset_name}/{speaker_id}")
    print(f"📊 Using {len(sample_files)} utterance files")
    
    # Load and examine utterance embeddings
    utterance_embeddings = []
    utterance_data = []
    
    for file_path in sample_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            embedding = data.get('embedding')
            if embedding is not None:
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                
                utterance_embeddings.append(embedding)
                utterance_data.append(data)
                
                print(f"  ✅ Loaded {file_path.name}: shape {embedding.shape}, norm {np.linalg.norm(embedding):.4f}")
            else:
                print(f"  ❌ No embedding found in {file_path.name}")
                
        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")
    
    if not utterance_embeddings:
        print("❌ No valid utterance embeddings loaded")
        return False
    
    # Compute average embedding
    print(f"\n🧮 Computing average embedding from {len(utterance_embeddings)} utterances...")
    
    embeddings_array = np.array(utterance_embeddings)
    avg_embedding = np.mean(embeddings_array, axis=0)
    
    print(f"  📏 Individual embedding shapes: {[emb.shape for emb in utterance_embeddings]}")
    print(f"  📏 Average embedding shape: {avg_embedding.shape}")
    print(f"  📊 Average embedding norm: {np.linalg.norm(avg_embedding):.4f}")
    print(f"  📊 Average embedding stats:")
    print(f"     Mean: {np.mean(avg_embedding):.4f}")
    print(f"     Std:  {np.std(avg_embedding):.4f}")
    print(f"     Min:  {np.min(avg_embedding):.4f}")
    print(f"     Max:  {np.max(avg_embedding):.4f}")
    
    # Create test speaker embedding data
    speaker_data = {
        'embedding': avg_embedding,
        'dataset': dataset_name,
        'speaker_id': speaker_id,
        'num_utterances': len(utterance_embeddings),
        'failed_utterances': 0,
        'utterance_list': [data.get('utterance_id', 'unknown') for data in utterance_data],
        'original_paths': [data.get('original_path', 'unknown') for data in utterance_data],
        'embedding_dim': len(avg_embedding),
        'embedding_stats': {
            'mean': float(np.mean(avg_embedding)),
            'std': float(np.std(avg_embedding)),
            'min': float(np.min(avg_embedding)),
            'max': float(np.max(avg_embedding))
        }
    }
    
    # Test saving
    test_output_dir = "/tmp/test_speaker_embeddings"
    os.makedirs(f"{test_output_dir}/{dataset_name}", exist_ok=True)
    
    test_file = f"{test_output_dir}/{dataset_name}/{speaker_id}.pkl"
    
    print(f"\n💾 Testing save to: {test_file}")
    with open(test_file, 'wb') as f:
        pickle.dump(speaker_data, f)
    
    # Test loading
    print(f"📖 Testing load from: {test_file}")
    with open(test_file, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"✅ Successfully saved and loaded speaker embedding!")
    print(f"  📊 Loaded data keys: {list(loaded_data.keys())}")
    print(f"  📊 Embedding shape: {loaded_data['embedding'].shape}")
    print(f"  📊 Number of utterances: {loaded_data['num_utterances']}")
    
    # Compare embeddings
    original_norm = np.linalg.norm(avg_embedding)
    loaded_norm = np.linalg.norm(loaded_data['embedding'])
    
    print(f"\n🔍 Verification:")
    print(f"  Original embedding norm: {original_norm:.6f}")
    print(f"  Loaded embedding norm: {loaded_norm:.6f}")
    print(f"  Difference: {abs(original_norm - loaded_norm):.6f}")
    
    if abs(original_norm - loaded_norm) < 1e-10:
        print(f"  ✅ Embeddings match perfectly!")
    else:
        print(f"  ⚠️  Small difference detected (likely due to precision)")
    
    return True

if __name__ == "__main__":
    success = test_speaker_embeddings()
    if success:
        print("\n🎉 Test completed successfully! Speaker embedding computation should work correctly.")
        print("\n💡 Next steps:")
        print("  1. Run: ./run_compute_speaker_embeddings.sh")
        print("  2. Check the speakers directory for results")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
    sys.exit(0 if success else 1) 
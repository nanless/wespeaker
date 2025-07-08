#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import pickle
import numpy as np
from pathlib import Path

# Add wespeaker to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wespeaker.cli.speaker import load_model_local

def test_single_file():
    """Test embedding extraction on a single audio file."""
    
    # Configuration
    model_dir = "/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100"
    data_root = "/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
    
    print("=== WeSpeaker Embedding Extraction Test ===")
    print(f"Model directory: {model_dir}")
    print(f"Data root: {data_root}")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, "avg_model.pt")):
        print(f"Error: Model not found at {model_dir}/avg_model.pt")
        return False
    
    try:
        # Load model
        print("Loading model...")
        model = load_model_local(model_dir)
        model.set_device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Model loaded successfully on {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
        
        # Find a test audio file
        test_file = None
        for dataset_dir in Path(data_root).iterdir():
            if dataset_dir.is_dir():
                for speaker_dir in dataset_dir.iterdir():
                    if speaker_dir.is_dir():
                        for audio_file in speaker_dir.iterdir():
                            if audio_file.suffix.lower() in ['.wav', '.flac', '.mp3']:
                                test_file = str(audio_file)
                                dataset_name = dataset_dir.name
                                speaker_id = speaker_dir.name
                                utterance_id = audio_file.stem
                                break
                        if test_file:
                            break
                if test_file:
                    break
        
        if not test_file:
            print("Error: No audio files found in the data directory")
            return False
        
        print(f"Test file: {test_file}")
        print(f"Dataset: {dataset_name}, Speaker: {speaker_id}, Utterance: {utterance_id}")
        
        # Extract embedding
        print("Extracting embedding...")
        embedding = model.extract_embedding(test_file)
        
        if embedding is not None:
            print(f"✓ Embedding extracted successfully!")
            print(f"  Shape: {embedding.shape}")
            print(f"  Type: {type(embedding)}")
            print(f"  Dtype: {embedding.dtype}")
            print(f"  Min value: {embedding.min():.4f}")
            print(f"  Max value: {embedding.max():.4f}")
            print(f"  Mean value: {embedding.mean():.4f}")
            
            # Test saving
            test_output_dir = "/tmp/test_embeddings"
            os.makedirs(f"{test_output_dir}/{dataset_name}/{speaker_id}", exist_ok=True)
            
            embedding_data = {
                'embedding': embedding.detach().cpu().numpy().flatten(),
                'dataset': dataset_name,
                'speaker_id': speaker_id,
                'utterance_id': utterance_id,
                'original_path': test_file
            }
            
            save_path = f"{test_output_dir}/{dataset_name}/{speaker_id}/{utterance_id}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            print(f"✓ Test embedding saved to: {save_path}")
            
            # Test loading
            with open(save_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            print(f"✓ Test embedding loaded successfully!")
            print(f"  Loaded embedding shape: {loaded_data['embedding'].shape}")
            
            return True
        else:
            print("✗ Failed to extract embedding")
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_file()
    if success:
        print("\n🎉 Test completed successfully! The embedding extraction should work.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")
    sys.exit(0 if success else 1) 
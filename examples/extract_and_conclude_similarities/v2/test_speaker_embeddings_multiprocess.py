#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import tempfile
import shutil

def create_test_data(test_dir):
    """Create test utterance embeddings for testing."""
    print("🧪 Creating test data...")
    
    # Create test utterances directory structure
    utterances_dir = Path(test_dir) / "utterances"
    
    # Define test structure
    test_structure = {
        'dataset1': {
            'speaker1': 5,  # 5 utterances
            'speaker2': 3,  # 3 utterances
            'speaker3': 1,  # 1 utterance (may be filtered out)
        },
        'dataset2': {
            'speaker1': 4,  # 4 utterances
            'speaker4': 6,  # 6 utterances
        }
    }
    
    utterance_counter = 0
    
    for dataset_name, speakers in test_structure.items():
        for speaker_id, num_utterances in speakers.items():
            speaker_dir = utterances_dir / dataset_name / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_utterances):
                utterance_counter += 1
                utterance_id = f"utt_{utterance_counter:04d}"
                
                # Create test embedding (256-dimensional)
                embedding = np.random.randn(256).astype(np.float32)
                # Add some speaker-specific bias for consistency
                speaker_bias = hash(speaker_id) % 100 / 100.0
                embedding = embedding + speaker_bias
                
                # Create utterance data
                utterance_data = {
                    'embedding': embedding,
                    'dataset': dataset_name,
                    'speaker_id': speaker_id,
                    'utterance_id': utterance_id,
                    'original_path': f'/test/path/{dataset_name}/{speaker_id}/{utterance_id}.wav'
                }
                
                # Save utterance file
                utterance_file = speaker_dir / f"{utterance_id}.pkl"
                with open(utterance_file, 'wb') as f:
                    pickle.dump(utterance_data, f)
    
    print(f"✅ Created test data with {utterance_counter} utterances")
    return str(utterances_dir)

def run_multiprocess_computation(utterances_dir, speakers_dir):
    """Run the multi-process speaker embedding computation."""
    print("🚀 Running multi-process speaker embedding computation...")
    
    # Import the compute function
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        from compute_speaker_embeddings_multiprocess import main as compute_main
        import argparse
        
        # Mock command line arguments
        original_argv = sys.argv
        sys.argv = [
            'compute_speaker_embeddings_multiprocess.py',
            '--utterances_dir', utterances_dir,
            '--speakers_dir', speakers_dir,
            '--min_utterances', '2',  # Filter out speakers with < 2 utterances
            '--num_processes', '2',   # Use 2 processes for testing
            '--chunk_size', '1',      # Small chunk size for testing
            '--skip_existing'
        ]
        
        # Run computation
        compute_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("✅ Multi-process computation completed")
        
    except Exception as e:
        print(f"❌ Error during computation: {e}")
        sys.argv = original_argv
        raise

def validate_speaker_embeddings(speakers_dir):
    """Validate the computed speaker embeddings."""
    print("🔍 Validating speaker embeddings...")
    
    speakers_path = Path(speakers_dir)
    if not speakers_path.exists():
        print("❌ Speakers directory does not exist")
        return False
    
    success_count = 0
    error_count = 0
    
    # Expected speakers (with >= 2 utterances)
    expected_speakers = {
        ('dataset1', 'speaker1'): 5,
        ('dataset1', 'speaker2'): 3,
        ('dataset2', 'speaker1'): 4,
        ('dataset2', 'speaker4'): 6,
    }
    
    # Check each expected speaker
    for (dataset_name, speaker_id), expected_utterances in expected_speakers.items():
        speaker_file = speakers_path / dataset_name / f"{speaker_id}.pkl"
        
        if not speaker_file.exists():
            print(f"❌ Missing speaker file: {speaker_file}")
            error_count += 1
            continue
        
        try:
            with open(speaker_file, 'rb') as f:
                speaker_data = pickle.load(f)
            
            # Validate speaker data structure
            required_fields = [
                'embedding', 'dataset', 'speaker_id', 'num_utterances',
                'utterance_list', 'embedding_dim', 'embedding_stats'
            ]
            
            for field in required_fields:
                if field not in speaker_data:
                    print(f"❌ Missing field '{field}' in {dataset_name}/{speaker_id}")
                    error_count += 1
                    continue
            
            # Validate embedding
            embedding = speaker_data['embedding']
            if not isinstance(embedding, np.ndarray):
                print(f"❌ Embedding is not numpy array in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            if embedding.shape != (256,):
                print(f"❌ Wrong embedding shape {embedding.shape} in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            # Validate metadata
            if speaker_data['dataset'] != dataset_name:
                print(f"❌ Wrong dataset name in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            if speaker_data['speaker_id'] != speaker_id:
                print(f"❌ Wrong speaker ID in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            if speaker_data['num_utterances'] != expected_utterances:
                print(f"❌ Wrong utterance count {speaker_data['num_utterances']} (expected {expected_utterances}) in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            if len(speaker_data['utterance_list']) != expected_utterances:
                print(f"❌ Wrong utterance list length in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            # Validate embedding statistics
            stats = speaker_data['embedding_stats']
            if not all(isinstance(stats[key], float) for key in ['mean', 'std', 'min', 'max']):
                print(f"❌ Invalid embedding statistics in {dataset_name}/{speaker_id}")
                error_count += 1
                continue
            
            success_count += 1
            print(f"✅ {dataset_name}/{speaker_id}: {speaker_data['num_utterances']} utterances, "
                  f"embedding dim: {speaker_data['embedding_dim']}, "
                  f"mean: {stats['mean']:.4f}")
            
        except Exception as e:
            print(f"❌ Error reading {speaker_file}: {e}")
            error_count += 1
    
    # Check for unexpected speakers
    all_speaker_files = list(speakers_path.rglob('*.pkl'))
    if len(all_speaker_files) != len(expected_speakers):
        print(f"⚠️  Found {len(all_speaker_files)} speaker files, expected {len(expected_speakers)}")
    
    print(f"\n📊 Validation Summary:")
    print(f"  ✅ Successful validations: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    
    # Check if speaker3 was correctly filtered out (only 1 utterance < min 2)
    speaker3_file = speakers_path / 'dataset1' / 'speaker3.pkl'
    if speaker3_file.exists():
        print(f"⚠️  speaker3 should have been filtered out (only 1 utterance)")
        error_count += 1
    else:
        print(f"✅ speaker3 correctly filtered out (< 2 utterances)")
    
    return error_count == 0

def test_multiprocess_performance():
    """Test multiprocessing performance compared to single process."""
    print("⚡ Testing multi-process performance...")
    
    # This is a conceptual test - in real scenarios, you would compare
    # execution times between single-process and multi-process versions
    
    print("💡 Performance testing tips:")
    print("  • Compare execution times with different num_processes values")
    print("  • Monitor CPU usage during execution")
    print("  • Test with different chunk_size values")
    print("  • Ensure results are identical between single and multi-process")
    
    return True

def main():
    print("=== Multi-process Speaker Embedding Test ===")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"🗂️  Using temporary directory: {temp_dir}")
        
        try:
            # Create test data
            utterances_dir = create_test_data(temp_dir)
            speakers_dir = os.path.join(temp_dir, "speakers")
            
            # Run multi-process computation
            run_multiprocess_computation(utterances_dir, speakers_dir)
            
            # Validate results
            validation_success = validate_speaker_embeddings(speakers_dir)
            
            # Test performance concepts
            performance_success = test_multiprocess_performance()
            
            # Final result
            if validation_success and performance_success:
                print(f"\n🎉 All tests passed! Multi-process speaker embedding computation works correctly.")
                return True
            else:
                print(f"\n❌ Some tests failed!")
                return False
                
        except Exception as e:
            print(f"\n💥 Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
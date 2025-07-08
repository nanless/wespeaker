#!/usr/bin/env python3

"""
测试音频提取功能
用于验证相似度pairs音频样本提取脚本的功能
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def test_audio_extraction():
    """测试音频提取功能"""
    
    print("=== 音频提取功能测试 ===")
    
    # 配置路径
    embeddings_dir = "/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments/embeddings_wespeaker_samresnet"
    audio_data_dir = "/root/group-shared/voiceprint/data/speech/speech_enhancement/audio_segments/merged_datasets_20250610_vad_segments"
    similarities_subdir = "speaker_similarity_analysis"
    utterances_subdir = "embeddings_individual/utterances"
    
    # 设置完整路径
    similarities_dir = os.path.join(embeddings_dir, similarities_subdir)
    utterances_dir = os.path.join(embeddings_dir, utterances_subdir)
    
    print(f"Embeddings directory: {embeddings_dir}")
    print(f"Audio data directory: {audio_data_dir}")
    print(f"Similarities directory: {similarities_dir}")
    print(f"Utterances directory: {utterances_dir}")
    print("=" * 50)
    
    # 检查必要目录和文件
    checks = [
        (embeddings_dir, "Embeddings directory"),
        (audio_data_dir, "Audio data directory"),
        (similarities_dir, "Similarities directory"),
        (utterances_dir, "Utterances directory"),
    ]
    
    missing_paths = []
    for path, name in checks:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
            missing_paths.append((path, name))
    
    if missing_paths:
        print(f"\n❌ Missing {len(missing_paths)} required paths:")
        for path, name in missing_paths:
            print(f"  - {name}: {path}")
        return False
    
    # 检查关键文件
    extreme_pairs_file = os.path.join(similarities_dir, 'extreme_similarity_pairs.json')
    if os.path.exists(extreme_pairs_file):
        print(f"✅ Extreme pairs file: {extreme_pairs_file}")
        
        # 读取并显示文件内容摘要
        try:
            with open(extreme_pairs_file, 'r') as f:
                data = json.load(f)
            
            most_similar = data.get('most_similar_pairs', [])
            least_similar = data.get('least_similar_pairs', [])
            
            print(f"  📊 Most similar pairs: {len(most_similar)}")
            print(f"  📊 Least similar pairs: {len(least_similar)}")
            
            if most_similar:
                print(f"  🔍 Sample most similar pair:")
                sample = most_similar[0]
                print(f"    Rank: {sample.get('rank', 'N/A')}")
                print(f"    Speaker1: {sample.get('speaker1', 'N/A')}")
                print(f"    Speaker2: {sample.get('speaker2', 'N/A')}")
                print(f"    Similarity: {sample.get('similarity', 'N/A'):.4f}")
            
            if least_similar:
                print(f"  🔍 Sample least similar pair:")
                sample = least_similar[0]
                print(f"    Rank: {sample.get('rank', 'N/A')}")
                print(f"    Speaker1: {sample.get('speaker1', 'N/A')}")
                print(f"    Speaker2: {sample.get('speaker2', 'N/A')}")
                print(f"    Similarity: {sample.get('similarity', 'N/A'):.4f}")
        
        except Exception as e:
            print(f"  ❌ Error reading extreme pairs file: {e}")
            return False
    else:
        print(f"❌ Extreme pairs file: {extreme_pairs_file} (NOT FOUND)")
        return False
    
    # 检查utterance文件
    print(f"\n🔍 Checking utterance files...")
    utterance_count = 0
    dataset_count = 0
    speaker_count = 0
    
    for dataset_dir in Path(utterances_dir).iterdir():
        if dataset_dir.is_dir():
            dataset_count += 1
            print(f"  📂 Dataset: {dataset_dir.name}")
            
            dataset_speaker_count = 0
            dataset_utterance_count = 0
            
            for speaker_dir in dataset_dir.iterdir():
                if speaker_dir.is_dir():
                    speaker_count += 1
                    dataset_speaker_count += 1
                    
                    pkl_files = list(speaker_dir.glob('*.pkl'))
                    utterance_count += len(pkl_files)
                    dataset_utterance_count += len(pkl_files)
            
            print(f"    Speakers: {dataset_speaker_count}, Utterances: {dataset_utterance_count}")
    
    print(f"  📊 Total: {dataset_count} datasets, {speaker_count} speakers, {utterance_count} utterances")
    
    # 检查原始音频文件
    print(f"\n🔍 Checking original audio files...")
    audio_count = 0
    audio_dataset_count = 0
    
    for dataset_dir in Path(audio_data_dir).iterdir():
        if dataset_dir.is_dir():
            audio_dataset_count += 1
            dataset_audio_count = 0
            
            for speaker_dir in dataset_dir.iterdir():
                if speaker_dir.is_dir():
                    audio_files = list(speaker_dir.glob('*.wav')) + list(speaker_dir.glob('*.flac')) + list(speaker_dir.glob('*.mp3'))
                    dataset_audio_count += len(audio_files)
                    audio_count += len(audio_files)
            
            if dataset_audio_count > 0:
                print(f"  📂 Dataset {dataset_dir.name}: {dataset_audio_count} audio files")
    
    print(f"  📊 Total: {audio_dataset_count} datasets, {audio_count} audio files")
    
    # 测试运行建议
    print(f"\n💡 测试运行建议:")
    print(f"1. 小规模测试 (推荐先运行):")
    print(f"   python extract_similarity_pairs_audio.py --top_pairs 3")
    print(f"")
    print(f"2. 标准测试:")
    print(f"   python extract_similarity_pairs_audio.py --top_pairs 10")
    print(f"")
    print(f"3. 使用脚本:")
    print(f"   ./run_extract_similarity_pairs_audio.sh")
    print(f"")
    print(f"4. 自定义参数:")
    print(f"   python extract_similarity_pairs_audio.py \\")
    print(f"       --embeddings_dir \"{embeddings_dir}\" \\")
    print(f"       --audio_data_dir \"{audio_data_dir}\" \\")
    print(f"       --top_pairs 5")
    
    if missing_paths:
        print(f"\n❌ 测试失败: 缺少必要的文件或目录")
        return False
    else:
        print(f"\n✅ 测试通过: 所有必要文件和目录都存在")
        print(f"🚀 可以开始运行音频提取脚本")
        return True

if __name__ == "__main__":
    success = test_audio_extraction()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
多GPU音频embedding提取脚本
将所有音频文件提取为embedding并保存为单独的文件
目录结构与原音频文件保持一致
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import time
import pickle

# 添加WeSpeaker路径
wespeaker_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, wespeaker_root)

try:
    from wespeaker.cli.speaker import load_model_local
    print("✅ 成功导入wespeaker模块")
except ImportError as e:
    print(f"❌ 导入wespeaker模块失败: {e}")
    sys.exit(1)

def setup(rank, world_size, port='12355'):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def scan_audio_files(input_dir):
    """扫描所有音频文件并保持目录结构"""
    audio_files = []
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    
    print(f"📂 扫描音频文件: {input_dir}")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                full_path = os.path.join(root, file)
                # 计算相对路径，用于保持目录结构
                rel_path = os.path.relpath(full_path, input_dir)
                audio_files.append({
                    'full_path': full_path,
                    'relative_path': rel_path,
                    'filename': file
                })
    
    print(f"📊 找到 {len(audio_files)} 个音频文件")
    return audio_files

def extract_embeddings_on_gpu(rank, world_size, args, audio_files):
    """在特定GPU上提取embeddings"""
    setup(rank, world_size, args.port)
    
    print(f"🖥️ GPU {rank}: 开始加载模型...")
    
    try:
        # 加载模型到特定GPU
        speaker_model = load_model_local(args.model_dir)
        speaker_model.set_device(f'cuda:{rank}')
        
        # 计算这个GPU需要处理的文件范围
        total_files = len(audio_files)
        files_per_gpu = (total_files + world_size - 1) // world_size
        start_idx = rank * files_per_gpu
        end_idx = min(start_idx + files_per_gpu, total_files)
        
        gpu_audio_files = audio_files[start_idx:end_idx]
        
        print(f"🎯 GPU {rank}: 处理文件 {start_idx}-{end_idx-1} ({len(gpu_audio_files)} 个文件)")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 提取embeddings
        success_count = 0
        failed_count = 0
        
        for audio_info in tqdm(gpu_audio_files, desc=f"GPU {rank} 提取embedding"):
            try:
                audio_path = audio_info['full_path']
                relative_path = audio_info['relative_path']
                
                if not os.path.exists(audio_path):
                    print(f"⚠️ GPU {rank}: 文件不存在: {audio_path}")
                    failed_count += 1
                    continue
                
                # 提取embedding
                embedding = speaker_model.extract_embedding(audio_path)
                
                if embedding is not None:
                    # 构建输出路径，保持目录结构
                    output_rel_path = Path(relative_path).with_suffix('.pkl')
                    output_path = os.path.join(args.output_dir, output_rel_path)
                    
                    # 创建输出目录
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # 保存embedding和元数据
                    embedding_data = {
                        'embedding': embedding,
                        'original_path': audio_path,
                        'relative_path': relative_path,
                        'filename': audio_info['filename'],
                        'embedding_dim': len(embedding),
                        'gpu_rank': rank
                    }
                    
                    with open(output_path, 'wb') as f:
                        pickle.dump(embedding_data, f)
                    
                    success_count += 1
                else:
                    print(f"❌ GPU {rank}: 提取embedding失败: {audio_path}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ GPU {rank}: 处理文件出错 {audio_info['full_path']}: {e}")
                failed_count += 1
                continue
        
        print(f"✅ GPU {rank}: 完成处理，成功 {success_count} 个，失败 {failed_count} 个")
        
        # 保存GPU处理结果统计
        stats = {
            'gpu_rank': rank,
            'total_processed': len(gpu_audio_files),
            'success_count': success_count,
            'failed_count': failed_count,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        
        stats_file = os.path.join(args.output_dir, f'gpu_{rank}_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
    except Exception as e:
        print(f"❌ GPU {rank}: 发生严重错误: {e}")
    finally:
        cleanup()

def merge_gpu_stats(output_dir, world_size):
    """合并所有GPU的统计信息"""
    print("🔄 合并GPU统计信息...")
    
    total_stats = {
        'total_files': 0,
        'total_success': 0,
        'total_failed': 0,
        'gpu_stats': []
    }
    
    for rank in range(world_size):
        stats_file = os.path.join(output_dir, f'gpu_{rank}_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                gpu_stats = json.load(f)
            
            total_stats['total_files'] += gpu_stats['total_processed']
            total_stats['total_success'] += gpu_stats['success_count']
            total_stats['total_failed'] += gpu_stats['failed_count']
            total_stats['gpu_stats'].append(gpu_stats)
            
            # 删除临时统计文件
            os.remove(stats_file)
    
    # 保存合并后的统计信息
    final_stats_file = os.path.join(output_dir, 'extraction_stats.json')
    with open(final_stats_file, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    print(f"📊 总体统计: 处理 {total_stats['total_files']} 个文件")
    print(f"   成功: {total_stats['total_success']} 个")
    print(f"   失败: {total_stats['total_failed']} 个")
    print(f"   成功率: {total_stats['total_success']/total_stats['total_files']*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="多GPU音频embedding提取")
    parser.add_argument('--input_dir', type=str, default="/root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_files",
                       help='输入音频目录路径')
    parser.add_argument('--output_dir', type=str, default="/root/group-shared/voiceprint/data/speech/speech_enhancement/child-2.07M/wav_embeddings_samresnet100",
                       help='输出embedding目录路径')
    parser.add_argument('--model_dir', type=str, default="/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100",
                       help='说话人模型目录路径')
    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                       help='使用的GPU列表，用逗号分隔 (e.g., "0,1,2,3"). 如果不指定则使用所有可用GPU')
    parser.add_argument('--port', type=str, default='12355',
                       help='分布式通信端口')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 检查模型目录
    if not os.path.exists(args.model_dir):
        print(f"❌ 模型目录不存在: {args.model_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 扫描音频文件
    audio_files = scan_audio_files(args.input_dir)
    
    if not audio_files:
        print("❌ 没有找到音频文件")
        return
    
    # 设置GPU配置
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        available_gpus = [int(x) for x in args.gpus.split(',')]
        world_size = len(available_gpus)
    else:
        world_size = torch.cuda.device_count()
        available_gpus = list(range(world_size))
    
    if world_size == 0:
        print("❌ 没有可用的GPU!")
        return
    
    print(f"🖥️ 使用 {world_size} 个GPU: {available_gpus}")
    print(f"📁 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🤖 模型目录: {args.model_dir}")
    
    start_time = time.time()
    
    # 多GPU并行提取embeddings
    print("🚀 开始多GPU并行提取embeddings...")
    
    # 使用multiprocessing.spawn启动多个GPU进程
    mp.spawn(
        extract_embeddings_on_gpu,
        args=(world_size, args, audio_files),
        nprocs=world_size,
        join=True
    )
    
    # 合并统计信息
    merge_gpu_stats(args.output_dir, world_size)
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 多GPU embedding提取完成！")
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"📊 处理效率: {len(audio_files)/total_time:.1f} 个文件/秒")
    print(f"🖥️ GPU加速比: 预计比单GPU快 {world_size:.1f}x")
    print(f"💾 结果保存到: {args.output_dir}")
    
    # 显示目录结构示例
    print(f"\n📂 输出目录结构示例:")
    count = 0
    for root, dirs, files in os.walk(args.output_dir):
        if count >= 10:  # 只显示前10个示例
            print("   ...")
            break
        for file in files[:3]:  # 每个目录最多显示3个文件
            if file.endswith('.pkl'):
                rel_path = os.path.relpath(os.path.join(root, file), args.output_dir)
                print(f"   {rel_path}")
                count += 1
                if count >= 10:
                    break

if __name__ == "__main__":
    main()
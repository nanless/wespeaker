#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeSpeaker diarization inference script.
完全仿照3D-Speaker的diarization流程：
- 使用TenVad进行VAD
- VAD后处理（平滑+形态学填充）和边界细化（基于能量）
- 不做滑窗，直接在VAD segments上提取embedding
- 使用AHC聚类

Usage:
    1. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --model_dir [model_dir]
    2. python infer_diarization.py --wav [wav_list] --out_dir [out_dir] --model_dir [model_dir] --nprocs [n]
"""

import os
import sys
import argparse
import warnings
import json
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torchaudio
import numpy as np
from tqdm import tqdm

# Add wespeaker to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wespeaker.cli.speaker import load_model_local
from wespeaker.diar.ahc_clusterer import cluster as ahc_cluster

warnings.filterwarnings("ignore")


def get_voice_activity_detection_model(device=None, threshold=0.5):
    """
    使用TenVad进行VAD。
    返回可调用对象，接受1-D波形tensor/ndarray，输出flags和wav_data。
    """
    try:
        from ten_vad import TenVad
    except ImportError:
        try:
            sys.path.append('/root/code/gitlab_repos/se_train')
            from ten_vad import TenVad  # type: ignore
        except Exception as e:
            raise ImportError('ten_vad is required for VAD. Please install/ensure it is available.') from e

    class TenVadWrapper:
        def __init__(self, sample_rate=16000, frame_ms=16.0, threshold=0.5):
            self.sample_rate = sample_rate
            self.hop_size = int(frame_ms * sample_rate / 1000)
            self.engine = TenVad(self.hop_size, threshold)

        def __call__(self, wav_1d):
            # 转换为numpy float32，范围[-1, 1]
            if hasattr(wav_1d, 'detach'):
                x = wav_1d.detach().cpu().numpy().astype(np.float32)
            else:
                x = np.asarray(wav_1d).astype(np.float32)
            if x.size == 0:
                return [], x
            x = np.clip(x, -1.0, 1.0)
            x_i16 = (x * 32767).astype(np.int16)

            num_frames = len(x_i16) // self.hop_size
            flags = []
            for i in range(num_frames):
                frame = x_i16[i*self.hop_size:(i+1)*self.hop_size]
                if len(frame) == self.hop_size:
                    _, f = self.engine.process(frame)
                    flags.append(int(f))
                else:
                    flags.append(0)
            
            # 返回原始flags用于后处理
            return flags, x

    # 默认使用16ms hop，匹配数据集设置
    return TenVadWrapper(sample_rate=16000, frame_ms=16.0, threshold=threshold)


def load_audio(input, ori_fs=None, obj_fs=None):
    """加载音频文件，支持路径、numpy数组或tensor。"""
    if isinstance(input, str):
        wav, fs = torchaudio.load(input)
        wav = wav.mean(dim=0, keepdim=True)
        if obj_fs is not None and fs != obj_fs:
            wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=obj_fs)
        return wav
    elif isinstance(input, np.ndarray) or isinstance(input, torch.Tensor):
        wav = torch.from_numpy(input) if isinstance(input, np.ndarray) else input
        if wav.dtype in (torch.int16, torch.int32, torch.int64):
            wav = wav.type(torch.float32)
            wav = wav / 32768
        wav = wav.type(torch.float32)
        assert wav.ndim <= 2
        if wav.ndim == 2:
            if wav.shape[0] > wav.shape[1]:
                wav = torch.transpose(wav, 0, 1)
            wav = wav.mean(dim=0, keepdim=True)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if ori_fs is not None and obj_fs is not None and ori_fs != obj_fs:
            wav = torchaudio.functional.resample(wav, orig_freq=ori_fs, new_freq=obj_fs)
        return wav
    else:
        return input


def circle_pad(x, target_len, dim=0):
    """循环填充tensor到目标长度。"""
    xlen = x.shape[dim]
    if xlen >= target_len:
        return x
    n = int(np.ceil(target_len / xlen))
    xcat = torch.cat([x for _ in range(n)], dim=dim)
    return torch.narrow(xcat, dim, 0, target_len)


def compressed_seg(seg_list):
    """压缩连续的相同speaker segments。"""
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1] + seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list


def _save_vad_waveform_png(wav_path, fs, vad_time_raw, vad_time_processed, vad_time_refined, out_png):
    """
    保存PNG图片，显示波形和raw、processed、refined VAD活动区间。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        wav = load_audio(wav_path, obj_fs=fs)
        if hasattr(wav, 'detach'):
            y = wav.detach().cpu().numpy()
        else:
            y = np.asarray(wav)
        y = y[0] if y.ndim > 1 else y
        if y.size == 0:
            return
        t = np.arange(y.shape[0], dtype=np.float32) / float(fs)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # 顶部: RAW VAD
        ax = axes[0]
        ax.plot(t, y, color='#1f77b4', linewidth=0.5)
        for st, ed in (vad_time_raw or []):
            try:
                ax.axvspan(float(st), float(ed), color='crimson', alpha=0.25, label='Raw VAD')
            except Exception:
                continue
        ax.set_xlim(0, t[-1])
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform + Raw VAD')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 中间: PROCESSED VAD (after _post_process_speech_flags)
        ax2 = axes[1]
        ax2.plot(t, y, color='#1f77b4', linewidth=0.5)
        for st, ed in (vad_time_processed or []):
            try:
                ax2.axvspan(float(st), float(ed), color='orange', alpha=0.3, label='Processed VAD')
            except Exception:
                continue
        ax2.set_xlim(0, t[-1])
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Waveform + Processed VAD (after smoothing & morphological fill)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 底部: REFINED VAD (after _refine_vad_boundaries_with_energy)
        ax3 = axes[2]
        ax3.plot(t, y, color='#1f77b4', linewidth=0.5)
        for st, ed in (vad_time_refined or []):
            try:
                ax3.axvspan(float(st), float(ed), color='green', alpha=0.3, label='Refined VAD')
            except Exception:
                continue
        ax3.set_xlim(0, t[-1])
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Waveform + Refined VAD (after energy-based boundary refinement)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        # best-effort; 如果出现问题（如matplotlib缺失）则跳过
        print(f"警告: 保存VAD可视化图片失败: {e}")
        pass


parser = argparse.ArgumentParser(description='WeSpeaker diarization inference (仿照3D-Speaker流程).')
parser.add_argument('--wav', type=str, required=True, help='输入wav文件或wav列表文件')
parser.add_argument('--out_dir', type=str, required=True, help='输出结果目录')
parser.add_argument('--model_dir', type=str, required=True, help='包含wespeaker模型文件的目录')
parser.add_argument('--out_type', choices=['rttm', 'json'], default='json', type=str, help='结果格式，rttm或json')
parser.add_argument('--disable_progress_bar', action='store_true', help='禁用进度条')
parser.add_argument('--nprocs', default=None, type=int, help='进程数（默认：根据GPU数量自动检测）')
parser.add_argument('--speaker_num', default=None, type=int, help='已知的说话人数量（可选）')
parser.add_argument('--vad_threshold', type=float, default=0.5, help='TenVad阈值（默认：0.5）')
parser.add_argument('--vad_min_speech_ms', type=float, default=200.0, help='VAD后处理：最小语音段时长（毫秒，默认：200.0）')
parser.add_argument('--vad_max_silence_ms', type=float, default=300.0, help='VAD后处理：最大静音间隙（毫秒，默认：300.0）')
parser.add_argument('--vad_energy_threshold', type=float, default=0.05, help='VAD能量阈值（默认：0.05）')
parser.add_argument('--vad_boundary_expansion_ms', type=float, default=10.0, help='VAD边界扩展（毫秒，默认：10.0）')
parser.add_argument('--vad_boundary_energy_percentile', type=float, default=10.0, help='VAD边界能量百分位（默认：10.0）')
parser.add_argument('--cluster_fix_cos_thr', type=float, default=0.3, help='AHC聚类固定余弦阈值（默认：0.3）')
parser.add_argument('--cluster_mer_cos', type=float, default=0.3, help='AHC聚类合并余弦阈值（默认：0.3）')
parser.add_argument('--cluster_min_cluster_size', type=int, default=0, help='AHC聚类最小簇大小（默认：0）')
parser.add_argument('--batch_size', type=int, default=64, help='embedding提取的batch size（默认：64）')


class DiarizationWeSpeaker:
    """
    WeSpeaker diarization pipeline，完全仿照3D-Speaker流程。
    """
    def __init__(self, model_dir, device=None, rank=0,
                 vad_threshold=0.5,
                 vad_min_speech_ms=200.0,
                 vad_max_silence_ms=300.0,
                 vad_energy_threshold=0.05,
                 vad_boundary_expansion_ms=10.0,
                 vad_boundary_energy_percentile=10.0,
                 cluster_fix_cos_thr=0.3,
                 cluster_mer_cos=0.3,
                 cluster_min_cluster_size=0,
                 batch_size=64,
                 speaker_num=None):
        
        self.device = self.normalize_device(device)
        self.speaker_num = speaker_num
        self.batch_size = batch_size
        self.fs = 16000  # 采样率
        
        # VAD后处理参数
        self.vad_frame_size_ms = 16.0
        self.vad_min_speech_ms = float(vad_min_speech_ms)
        self.vad_max_silence_ms = float(vad_max_silence_ms)
        self.vad_energy_threshold = float(vad_energy_threshold)
        self.vad_boundary_expansion_ms = float(vad_boundary_expansion_ms)
        self.vad_boundary_energy_percentile = float(vad_boundary_energy_percentile)
        
        # 聚类参数
        self.cluster_fix_cos_thr = cluster_fix_cos_thr
        self.cluster_mer_cos = cluster_mer_cos
        self.cluster_min_cluster_size = cluster_min_cluster_size
        
        # VAD信息保存（用于后续保存）
        self.last_vad_time_raw = None
        self.last_vad_time_processed = None
        self.last_vad_time = None  # refined VAD
        self.last_vad_processed_mask = None
        self.last_vad_refined_mask = None
        self.last_vad_masked_audio = None
        self.last_wav_data = None
        self.output_field_labels = None
        
        # 加载模型
        if rank == 0:
            print(f"正在从 {model_dir} 加载模型...")
        self.model = load_model_local(model_dir)
        self.model.set_device(str(self.device))
        # Speaker对象在初始化时已经设置了model.eval()，无需再次调用
        
        # 加载VAD模型
        self.vad_model = get_voice_activity_detection_model(self.device, threshold=vad_threshold)
        
        if rank == 0:
            print(f"模型已加载到设备: {self.device}")
    
    def normalize_device(self, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device
    
    def do_vad(self, wav):
        """执行VAD。wav: [1, T]"""
        speech_flags, wav_data = self.vad_model(wav[0])
        return speech_flags, wav_data
    
    def postprocess_vad(self, speech_flags, wav_data):
        """
        应用VAD后处理：_post_process_speech_flags 和 _refine_vad_boundaries_with_energy
        返回 processed_mask, refined_mask, 和 vad_time intervals
        """
        # 转换flags为处理后的flags
        processed_flags = self._post_process_speech_flags(speech_flags)
        
        # 转换处理后的flags为mask (processed_mask)
        hop_size = int(self.vad_frame_size_ms * self.fs / 1000)
        processed_mask = np.zeros(len(wav_data), dtype=np.float32)
        for i, flag in enumerate(processed_flags):
            s = i * hop_size
            e = min((i + 1) * hop_size, len(wav_data))
            processed_mask[s:e] = flag
        
        # 使用能量细化边界 (refined_mask)
        refined_mask = self._refine_vad_boundaries_with_energy(wav_data, processed_mask)
        
        # 转换mask为时间间隔
        vad_time = self._mask_to_intervals(refined_mask)
        return processed_mask, refined_mask, vad_time
    
    def _post_process_speech_flags(self, flags):
        """平滑 + 形态学填充（简单实现）"""
        flags = np.array(flags, dtype=np.float32)
        
        # 简单移动平均平滑
        win = 3
        pad = np.pad(flags, (win // 2, win // 2), mode='edge')
        smooth = np.convolve(pad, np.ones(win) / win, mode='valid')
        smooth = (smooth > 0.5).astype(np.float32)

        # 最小语音段 / 最大静音段约束（基于帧）
        min_speech_frames = max(1, int(self.vad_min_speech_ms / self.vad_frame_size_ms))
        max_silence_frames = max(1, int(self.vad_max_silence_ms / self.vad_frame_size_ms))

        res = smooth.copy()
        # 填充短静音间隙
        count0 = 0
        for i in range(len(res)):
            if res[i] == 0:
                count0 += 1
            else:
                if 0 < count0 <= max_silence_frames:
                    res[i - count0 : i] = 1
                count0 = 0
        # 移除过短的语音段
        count1 = 0
        for i in range(len(res)):
            if res[i] == 1:
                count1 += 1
            else:
                if 0 < count1 < min_speech_frames:
                    res[i - count1 : i] = 0
                count1 = 0
        return res.astype(np.float32)
    
    def _refine_vad_boundaries_with_energy(self, audio_data, vad_mask):
        """使用基于能量的方法细化VAD边界"""
        refined_mask = vad_mask.copy()
        window_size = int(0.02 * self.fs)  # 20ms
        hop_length = int(0.01 * self.fs)   # 10ms
        n_frames = (len(audio_data) - window_size) // hop_length + 1
        if n_frames <= 0:
            return refined_mask

        frame_energy = np.zeros(len(audio_data), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_length
            e = min(s + window_size, len(audio_data))
            en = float(np.mean(audio_data[s:e] ** 2))
            frame_energy[s:e] = max(frame_energy[s:e].max(), en)

        vad_diff = np.diff(np.concatenate(([0], vad_mask, [0])))
        speech_starts = np.where(vad_diff > 0)[0]
        speech_ends = np.where(vad_diff < 0)[0]
        if len(speech_starts) == 0 or len(speech_ends) == 0:
            return refined_mask

        lookahead_frames = 10
        lookahead_samples = lookahead_frames * hop_length
        energy_floor = float(self.vad_energy_threshold)
        energy_percentile = float(self.vad_boundary_energy_percentile)
        boundary_expand_ms = float(self.vad_boundary_expansion_ms)
        boundary_expand_samples = int(boundary_expand_ms * self.fs / 1000.0)

        for start, end in zip(speech_starts, speech_ends):
            seg_energy = frame_energy[start:end]
            if len(seg_energy) == 0:
                continue
            dynamic_th = max(np.percentile(seg_energy, energy_percentile), energy_floor)
            
            # Step 1: 前向收缩 - 移除低能量后找到新的起始点
            new_start = start
            for i in range(start, min(end, start + lookahead_samples)):
                if frame_energy[i] < dynamic_th:
                    refined_mask[start:i] = 0
                    new_start = i
                    break
            
            # Step 2: 后向收缩 - 移除低能量后找到新的结束点
            new_end = end
            for i in range(end - 1, max(new_start, end - lookahead_samples), -1):
                if frame_energy[i] < dynamic_th:
                    refined_mask[i:end] = 0
                    new_end = i + 1
                    break
            
            # Step 3: 从收缩位置扩展边界
            if boundary_expand_samples > 0:
                expand_start_begin = max(start, new_start - boundary_expand_samples)
                expand_start_end = new_start
                refined_mask[expand_start_begin:expand_start_end] = 1
                
                expand_end_begin = new_end
                expand_end_end = end
                refined_mask[expand_end_begin:expand_end_end] = 1
        return refined_mask.astype(np.float32)
    
    def _mask_to_intervals(self, mask):
        """将VAD mask转换为时间间隔（秒）"""
        if len(mask) == 0:
            return []
        
        # 找到转换点
        diff = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(diff > 0)[0]
        ends = np.where(diff < 0)[0]
        
        if len(starts) == 0:
            return []
        
        intervals = []
        for s, e in zip(starts, ends):
            start_sec = float(s) / self.fs
            end_sec = float(e) / self.fs
            if end_sec > start_sec:
                intervals.append([start_sec, end_sec])
        
        return intervals
    
    def _flags_to_intervals(self, flags, wav_data):
        """将原始speech flags转换为时间间隔（秒）"""
        if len(flags) == 0:
            return []
        
        hop_size = int(self.vad_frame_size_ms * self.fs / 1000)
        intervals = []
        i = 0
        N = len(flags)
        while i < N:
            if flags[i]:
                j = i + 1
                while j < N and flags[j]:
                    j += 1
                start_sec = float(i * hop_size) / self.fs
                end_sec = float(min(j * hop_size, len(wav_data))) / self.fs
                if end_sec > start_sec:
                    intervals.append([start_sec, end_sec])
                i = j
            else:
                i += 1
        
        return intervals
    
    def _apply_vad_mask_from_mask(self, wav_data, vad_mask):
        """从mask数组应用VAD mask到音频：将非VAD区域置零"""
        # 转换为numpy如果需要
        if hasattr(wav_data, 'detach'):
            wav_np = wav_data.detach().cpu().numpy()
        else:
            wav_np = np.asarray(wav_data)
        
        # 处理不同的输入形状
        if wav_np.ndim == 2:
            audio = wav_np[0].copy()
        else:
            audio = wav_np.copy()
        
        # 确保mask长度匹配音频长度
        mask_len = min(len(vad_mask), len(audio))
        mask = vad_mask[:mask_len].copy()
        
        # 如果mask长度小于音频长度，进行padding
        if len(mask) < len(audio):
            mask = np.pad(mask, (0, len(audio) - len(mask)), mode='constant', constant_values=0.0)
        
        # 应用mask
        masked_audio = audio * mask
        
        # 返回与输入相同的形状
        if wav_np.ndim == 2:
            return masked_audio.reshape(1, -1)
        else:
            return masked_audio
    
    def do_emb_extraction(self, chunks, wav):
        """
        在chunks上提取embeddings。
        chunks: [[st1, ed1]...] (秒)
        wav: [1, T]
        """
        # 提取每个chunk的音频并提取embedding
        embeddings = []
        valid_chunk_indices = []  # 记录成功提取embedding的chunk索引
        
        for i, (st, ed) in enumerate(chunks):
            start_idx = int(st * self.fs)
            end_idx = int(ed * self.fs)
            if end_idx <= start_idx:
                continue
            
            # 提取chunk音频 [T]
            chunk_wav = wav[0, start_idx:end_idx]
            
            # extract_embedding_from_pcm期望输入为[1, T]格式
            chunk_wav = chunk_wav.unsqueeze(0).to(self.device)  # [1, T]
            
            try:
                emb = self.model.extract_embedding_from_pcm(chunk_wav, self.fs)
                if emb is not None:
                    # emb是tensor，需要转换为numpy
                    if isinstance(emb, torch.Tensor):
                        emb_np = emb.detach().cpu().numpy().flatten()
                    else:
                        emb_np = np.array(emb).flatten()
                    embeddings.append(emb_np)
                    valid_chunk_indices.append(i)
            except Exception as e:
                # 如果提取失败，跳过这个chunk
                if len(embeddings) == 0:  # 只在第一次失败时打印警告
                    print(f"警告: 提取embedding失败 (chunk {i}): {e}")
                continue
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            # 返回有效的chunks（只包含成功提取embedding的chunks）
            valid_chunks = [chunks[i] for i in valid_chunk_indices]
        else:
            embeddings = np.array([])
            valid_chunks = []
        return embeddings, valid_chunks
    
    def do_clustering(self, chunks, embeddings, speaker_num=None):
        """执行AHC聚类"""
        if len(embeddings) == 0:
            return 0, []
        
        cluster_labels = ahc_cluster(
            embeddings,
            fix_cos_thr=self.cluster_fix_cos_thr,
            min_cluster_size=self.cluster_min_cluster_size,
            mer_cos=self.cluster_mer_cos,
            speaker_num=speaker_num if speaker_num is not None else self.speaker_num
        )
        
        speaker_num = int(cluster_labels.max() + 1)
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels
    
    def __call__(self, wav_path, utt_id=None):
        """
        对单个音频文件执行diarization。
        返回: list of [start_sec, end_sec, speaker_id]
        """
        if utt_id is None:
            utt_id = Path(wav_path).stem
        
        try:
            # 加载音频
            wav_data = load_audio(wav_path, obj_fs=self.fs)
            
            # 保存原始音频数据
            self.last_wav_data = wav_data
            
            # stage 1-1: 执行VAD (原始)
            speech_flags, wav_data_for_vad = self.do_vad(wav_data)
            
            # 保存原始VAD时间（用于可视化）
            self.last_vad_time_raw = self._flags_to_intervals(speech_flags, wav_data_for_vad)
            
            # stage 1-1.5: VAD后处理（返回processed_mask, refined_mask和vad_time）
            vad_processed_mask, vad_refined_mask, vad_time = self.postprocess_vad(speech_flags, wav_data_for_vad)
            
            # 保存VAD信息
            self.last_vad_processed_mask = vad_processed_mask
            self.last_vad_refined_mask = vad_refined_mask
            self.last_vad_time_processed = self._mask_to_intervals(vad_processed_mask)
            self.last_vad_time = vad_time  # refined VAD
            
            # 生成VAD masked audio
            self.last_vad_masked_audio = self._apply_vad_mask_from_mask(wav_data, vad_refined_mask)
            
            # stage 2: 准备segments用于embedding提取
            # 不做滑窗，直接在VAD segments上提取embedding
            chunks = [[st, ed] for (st, ed) in vad_time]
            
            # 如果没有有效的chunks，返回空结果
            if len(chunks) == 0:
                self.output_field_labels = []
                return []
            
            # stage 3: 提取embeddings
            embeddings, valid_chunks = self.do_emb_extraction(chunks, wav_data)
            
            if len(embeddings) == 0 or len(valid_chunks) == 0:
                self.output_field_labels = []
                return []
            
            # stage 4: 聚类（使用有效的chunks）
            speaker_num, output_field_labels = self.do_clustering(valid_chunks, embeddings, self.speaker_num)
            
            # 保存输出结果
            self.output_field_labels = output_field_labels
            
            # 添加utt_id到输出
            result = [[utt_id, seg[0], seg[1], seg[2]] for seg in output_field_labels]
            
            return result
        except Exception as e:
            print(f"处理 {wav_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_diar_output(self, out_file, wav_id, output_field_labels):
        """保存diarization结果到文件。"""
        if out_file.endswith('rttm'):
            line_str = "SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
            with open(out_file, 'w') as f:
                for seg in output_field_labels:
                    if len(seg) >= 4:
                        utt, seg_st, seg_ed, cluster_id = seg[0], seg[1], seg[2], seg[3]
                        f.write(line_str.format(utt, seg_st, seg_ed - seg_st, cluster_id))
        elif out_file.endswith('json'):
            out_json = {}
            for seg in output_field_labels:
                if len(seg) >= 4:
                    utt, seg_st, seg_ed, cluster_id = seg[0], seg[1], seg[2], seg[3]
                    item = {
                        'start': float(seg_st),
                        'stop': float(seg_ed),
                        'speaker': int(cluster_id),
                    }
                    segid = f"{utt}_{seg_st:.3f}_{seg_ed:.3f}"
                    out_json[segid] = item
            with open(out_file, 'w') as f:
                json.dump(out_json, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError('目前支持的文件格式仅限于RTTM和JSON。')


def main_process(rank, nprocs, args, wav_list):
    """每个进程的主处理函数。"""
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        ngpus = torch.cuda.device_count()
        device = torch.device(f'cuda:{rank % ngpus}')
    
    # 初始化diarization pipeline
    diarization = DiarizationWeSpeaker(
        model_dir=args.model_dir,
        device=device,
        rank=rank,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_max_silence_ms=args.vad_max_silence_ms,
        vad_energy_threshold=args.vad_energy_threshold,
        vad_boundary_expansion_ms=args.vad_boundary_expansion_ms,
        vad_boundary_energy_percentile=args.vad_boundary_energy_percentile,
        cluster_fix_cos_thr=args.cluster_fix_cos_thr,
        cluster_mer_cos=args.cluster_mer_cos,
        cluster_min_cluster_size=args.cluster_min_cluster_size,
        batch_size=args.batch_size,
        speaker_num=args.speaker_num
    )
    
    # 在进程间分配工作
    wav_list = wav_list[rank::nprocs]
    if rank == 0 and (not args.disable_progress_bar):
        wav_list = tqdm(wav_list, desc=f"Rank 0 处理中")
    
    for wav_path in wav_list:
        t0 = time.time()
        
        # 获取utterance ID
        wav_id = Path(wav_path).stem
        
        # 执行diarization
        output = diarization(wav_path, utt_id=wav_id)
        
        elapsed = time.time() - t0
        
        # 保存结果
        if args.out_dir is not None:
            out_file = os.path.join(args.out_dir, f"{wav_id}.{args.out_type}")
        else:
            out_file = f"{wav_path.rsplit('.', 1)[0]}.{args.out_type}"
        
        if output:
            diarization.save_diar_output(out_file, wav_id, output)
        
        # 保存VAD可视化图片
        try:
            png_path = os.path.join(os.path.dirname(out_file), f"{wav_id}.vad.png")
            _save_vad_waveform_png(
                wav_path,
                diarization.fs,
                diarization.last_vad_time_raw or [],
                diarization.last_vad_time_processed or [],
                diarization.last_vad_time or [],
                png_path
            )
        except Exception as e:
            if rank == 0:
                print(f"警告: 保存VAD可视化图片失败: {e}")
        
        # 保存VAD masked audio
        try:
            if diarization.last_vad_masked_audio is not None:
                masked_audio_path = os.path.join(os.path.dirname(out_file), f"{wav_id}.vad_masked.wav")
                import soundfile as sf
                masked_audio = diarization.last_vad_masked_audio
                # 处理不同的形状
                if hasattr(masked_audio, 'detach'):
                    masked_audio = masked_audio.detach().cpu().numpy()
                if masked_audio.ndim == 2:
                    masked_audio = masked_audio[0]
                sf.write(masked_audio_path, masked_audio, diarization.fs)
        except Exception as e:
            if rank == 0:
                print(f"警告: 保存VAD masked audio失败: {e}")
        
        # 保存VAD info (raw, processed, refined)
        try:
            vad_info = {
                'wav_path': wav_path,
                'sample_rate': diarization.fs,
                'vad_raw': {
                    'intervals': [[float(st), float(ed)] for st, ed in (diarization.last_vad_time_raw or [])],
                    'num_segments': len(diarization.last_vad_time_raw or []),
                    'total_duration': sum([float(ed) - float(st) for st, ed in (diarization.last_vad_time_raw or [])])
                },
                'vad_processed': {
                    'intervals': [[float(st), float(ed)] for st, ed in (diarization.last_vad_time_processed or [])],
                    'num_segments': len(diarization.last_vad_time_processed or []),
                    'total_duration': sum([float(ed) - float(st) for st, ed in (diarization.last_vad_time_processed or [])])
                },
                'vad_refined': {
                    'intervals': [[float(st), float(ed)] for st, ed in (diarization.last_vad_time or [])],
                    'num_segments': len(diarization.last_vad_time or []),
                    'total_duration': sum([float(ed) - float(st) for st, ed in (diarization.last_vad_time or [])])
                }
            }
            vad_info_file = os.path.join(os.path.dirname(out_file), f"{wav_id}.vad_info.json")
            with open(vad_info_file, 'w') as vf:
                json.dump(vad_info, vf, indent=2, ensure_ascii=False)
        except Exception as e:
            if rank == 0:
                print(f"警告: 保存VAD info失败: {e}")
        
        # 获取音频时长
        def _get_duration_seconds(path):
            try:
                import soundfile as sf
                info = sf.info(path)
                if info.samplerate > 0:
                    return float(info.frames) / float(info.samplerate)
            except Exception:
                pass
            try:
                import wave
                with wave.open(path, 'rb') as w:
                    frames = w.getnframes()
                    rate = w.getframerate() or 0
                    return (float(frames) / float(rate)) if rate > 0 else None
            except Exception:
                pass
            return None
        
        duration_sec = _get_duration_seconds(wav_path)
        
        # 计算成对的余弦相似度
        pairwise_list = []
        pair_min = None
        pair_mean = None
        try:
            # output_field_labels格式: [[start, end, speaker_id], ...]
            segs = diarization.output_field_labels or []
            if len(segs) >= 2:
                # 提取每个segment的embedding
                seg_times = [[float(s[0]), float(s[1])] for s in segs]
                wav_full = load_audio(wav_path, obj_fs=diarization.fs)
                embs, _ = diarization.do_emb_extraction(seg_times, wav_full)
                if len(embs) >= 2 and embs.shape[0] >= 2:
                    # 归一化embeddings
                    Z = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
                    # 计算相似度矩阵
                    S = Z @ Z.T
                    # 获取上三角索引
                    triu_idx = np.triu_indices(S.shape[0], k=1)
                    vals = S[triu_idx]
                    if vals.size > 0:
                        pair_min = float(vals.min())
                        pair_mean = float(vals.mean())
                    # 构建成对相似度列表
                    for i in range(S.shape[0]):
                        for j in range(i+1, S.shape[0]):
                            pairwise_list.append({
                                'i': int(i),
                                'j': int(j),
                                'seg_i': {'start': float(segs[i][0]), 'stop': float(segs[i][1]), 'speaker': int(segs[i][2])},
                                'seg_j': {'start': float(segs[j][0]), 'stop': float(segs[j][1]), 'speaker': int(segs[j][2])},
                                'cosine': float(S[i, j]),
                            })
        except Exception as e:
            if rank == 0:
                print(f"警告: 计算成对相似度失败: {e}")
        
        # 保存元数据
        meta = {
            'wav_path': wav_path,
            'duration_sec': duration_sec,
            'processing_time_sec': elapsed,
            'rtf': (elapsed / duration_sec) if (duration_sec and duration_sec > 0) else None,
            'num_segments': len(output) if output else 0,
            'pairwise_min_cosine': pair_min,
            'pairwise_mean_cosine': pair_mean,
        }
        
        meta_file = os.path.join(os.path.dirname(out_file), f"{wav_id}.meta.json")
        try:
            with open(meta_file, 'w') as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
        except Exception:
            pass
        
        # 保存成对相似度文件
        try:
            pairs_file = os.path.join(os.path.dirname(out_file), f"{wav_id}.pairs.json")
            with open(pairs_file, 'w') as pf:
                json.dump({'pairs': pairwise_list}, pf, indent=2, ensure_ascii=False)
        except Exception:
            pass


def main():
    args = parser.parse_args()
    
    # 检查模型目录
    if not os.path.exists(args.model_dir):
        parser.error(f"模型目录不存在: {args.model_dir}")
    
    if not os.path.exists(os.path.join(args.model_dir, 'avg_model.pt')):
        parser.error(f"在 {args.model_dir}/avg_model.pt 找不到模型文件")
    
    # 解析输入wav列表
    if args.wav.endswith('.wav') or args.wav.endswith('.flac') or args.wav.endswith('.mp3'):
        # 输入是单个音频文件
        wav_list = [args.wav]
    else:
        # 输入应该是wav列表文件
        try:
            with open(args.wav, 'r') as f:
                wav_list = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            raise Exception(f'[错误]: 读取wav列表文件 {args.wav} 失败: {e}')
    
    if len(wav_list) == 0:
        raise Exception('[错误]: 输入中未找到音频文件。')
    
    # 确定进程数
    if args.nprocs is None:
        ngpus = torch.cuda.device_count()
        if ngpus > 0:
            print(f'[信息]: 检测到 {ngpus} 个GPU。')
            args.nprocs = ngpus
        else:
            print('[信息]: 未检测到GPU，使用CPU。')
            args.nprocs = 1
    
    args.nprocs = min(len(wav_list), args.nprocs)
    print(f'[信息]: 使用 {args.nprocs} 个进程进行diarization。')
    
    # 创建输出目录
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
    
    # 运行多进程diarization
    if args.nprocs > 1:
        mp.spawn(main_process, nprocs=args.nprocs, args=(args.nprocs, args, wav_list))
    else:
        main_process(0, 1, args, wav_list)
    
    print(f'[信息]: Diarization完成！结果已保存到: {args.out_dir}')


if __name__ == '__main__':
    main()

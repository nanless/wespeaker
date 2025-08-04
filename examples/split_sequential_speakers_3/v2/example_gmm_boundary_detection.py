#!/usr/bin/env python3
"""
GMM边界检测演示脚本
展示混合高斯模型在说话人边界检测中的应用
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import argparse
import os

def generate_synthetic_embeddings(n_speakers=3, embeddings_per_speaker=1000, dim=512):
    """
    生成合成的说话人embedding数据
    
    Args:
        n_speakers: 说话人数量
        embeddings_per_speaker: 每个说话人的embedding数量
        dim: embedding维度
    
    Returns:
        (embeddings, labels): embedding数组和对应的说话人标签
    """
    print(f"🔧 生成合成数据: {n_speakers}个说话人, 每人{embeddings_per_speaker}个embedding, 维度{dim}")
    
    embeddings = []
    labels = []
    
    for speaker_id in range(n_speakers):
        # 为每个说话人生成两个高斯中心（模拟不同说话状态）
        center1 = np.random.randn(dim) * 0.5 + speaker_id * 2
        center2 = np.random.randn(dim) * 0.5 + speaker_id * 2 + 0.8
        
        speaker_embeddings = []
        
        # 生成embedding：一半来自center1，一半来自center2
        for i in range(embeddings_per_speaker):
            if i < embeddings_per_speaker // 2:
                # 第一种状态（如正常语音）
                embedding = center1 + np.random.randn(dim) * 0.3
            else:
                # 第二种状态（如激动语音）
                embedding = center2 + np.random.randn(dim) * 0.3
            
            speaker_embeddings.append(embedding)
            labels.append(speaker_id)
        
        embeddings.extend(speaker_embeddings)
        center1_str = ", ".join([f"{x:.2f}" for x in center1[:3]])
        center2_str = ", ".join([f"{x:.2f}" for x in center2[:3]])
        print(f"  说话人 {speaker_id+1}: 中心1=[{center1_str}]..., 中心2=[{center2_str}]...")
    
    return np.array(embeddings), np.array(labels)

def train_speaker_gmm_demo(embeddings, speaker_id, n_components=2):
    """演示说话人GMM训练"""
    print(f"\n🧠 为说话人 {speaker_id+1} 训练GMM模型...")
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=42,
        max_iter=100,
        n_init=3
    )
    
    gmm.fit(embeddings)
    
    # 显示训练结果
    print(f"  ✅ 训练完成: {n_components}个组件")
    print(f"  📊 BIC: {gmm.bic(embeddings):.2f}")
    print(f"  📊 AIC: {gmm.aic(embeddings):.2f}")
    print(f"  🎯 组件权重: {gmm.weights_}")
    
    # 计算每个样本属于各组件的概率
    probs = gmm.predict_proba(embeddings)
    print(f"  📈 平均组件1概率: {np.mean(probs[:, 0]):.3f}")
    print(f"  📈 平均组件2概率: {np.mean(probs[:, 1]):.3f}")
    
    return gmm

def demo_boundary_detection(embeddings, labels, segment_size=1000):
    """演示边界检测过程"""
    print(f"\n🎯 演示边界检测过程...")
    
    n_speakers = len(np.unique(labels))
    theoretical_boundaries = [i * segment_size for i in range(1, n_speakers)]
    
    print(f"  📏 理论边界位置: {theoretical_boundaries}")
    
    # 为每个说话人段训练GMM
    gmm_models = []
    for i in range(n_speakers):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(embeddings))
        speaker_embeddings = embeddings[start_idx:end_idx]
        
        gmm = train_speaker_gmm_demo(speaker_embeddings, i)
        gmm_models.append(gmm)
    
    # 在边界附近评估GMM概率
    print(f"\n🔍 评估边界区域的GMM概率...")
    
    for i, boundary in enumerate(theoretical_boundaries):
        print(f"\n  边界 {i+1} (位置 {boundary}):")
        
        # 获取边界附近的embedding
        window_size = 10
        start_idx = max(0, boundary - window_size)
        end_idx = min(len(embeddings), boundary + window_size)
        
        left_gmm = gmm_models[i]
        right_gmm = gmm_models[i + 1]
        
        for pos in range(start_idx, end_idx):
            embedding = embeddings[pos:pos+1]
            
            left_prob = left_gmm.score_samples(embedding)[0]
            right_prob = right_gmm.score_samples(embedding)[0]
            
            separation = left_prob - right_prob if pos < boundary else right_prob - left_prob
            
            print(f"    位置 {pos}: 左GMM概率={left_prob:.2f}, 右GMM概率={right_prob:.2f}, 分离度={separation:.2f}")

def create_visualization(embeddings, labels, output_dir):
    """创建可视化图表"""
    print(f"\n📊 创建可视化图表...")
    
    # 使用PCA降维到2D进行可视化
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 创建散点图
    plt.figure(figsize=(12, 8))
    
    n_speakers = len(np.unique(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, n_speakers))
    
    for speaker_id in range(n_speakers):
        mask = labels == speaker_id
        speaker_embeddings = embeddings_2d[mask]
        
        plt.scatter(speaker_embeddings[:, 0], speaker_embeddings[:, 1], 
                   c=[colors[speaker_id]], label=f'说话人 {speaker_id+1}', 
                   alpha=0.6, s=20)
        
        # 为每个说话人训练GMM并显示椭圆
        if np.sum(mask) > 10:  # 确保有足够的样本
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(speaker_embeddings)
            
            # 绘制GMM组件的椭圆
            for j in range(gmm.n_components):
                mean = gmm.means_[j]
                cov = gmm.covariances_[j]
                
                # 计算椭圆参数
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals)
                
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(mean, width, height, angle=angle, 
                                 facecolor='none', edgecolor=colors[speaker_id], 
                                 linewidth=2, linestyle='--', alpha=0.8)
                plt.gca().add_patch(ellipse)
    
    plt.xlabel('PCA 第一主成分')
    plt.ylabel('PCA 第二主成分')
    plt.title('说话人Embedding分布与GMM建模\n(虚线椭圆表示GMM组件)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    visualization_path = os.path.join(output_dir, 'gmm_embedding_visualization.png')
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 可视化图表已保存: {visualization_path}")

def main():
    parser = argparse.ArgumentParser(description="GMM边界检测演示")
    parser.add_argument('--output_dir', type=str, default='gmm_demo_output',
                       help='输出目录路径')
    parser.add_argument('--n_speakers', type=int, default=3,
                       help='合成说话人数量')
    parser.add_argument('--embeddings_per_speaker', type=int, default=1000,
                       help='每个说话人的embedding数量')
    parser.add_argument('--dim', type=int, default=512,
                       help='embedding维度')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎭 GMM说话人边界检测演示")
    print("=" * 50)
    
    # 生成合成数据
    embeddings, labels = generate_synthetic_embeddings(
        args.n_speakers, args.embeddings_per_speaker, args.dim
    )
    
    # 演示边界检测
    demo_boundary_detection(embeddings, labels, args.embeddings_per_speaker)
    
    # 创建可视化
    create_visualization(embeddings, labels, args.output_dir)
    
    print(f"\n🎉 演示完成！结果保存在: {args.output_dir}")
    print(f"\n📋 关键概念总结:")
    print(f"  🧠 GMM建模: 每个说话人用{2}个高斯组件建模，捕捉多样性")
    print(f"  📊 概率评估: 使用对数概率评估边界音频与相邻说话人的契合度")
    print(f"  🎯 边界检测: 选择使分离度最大的位置作为精确边界")
    print(f"  🔄 自适应: 样本不足时自动减少组件数或回退到余弦相似度")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
说话人相似度计算演示脚本
展示相似度计算的详细过程和原理
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def manual_cosine_similarity(vec1, vec2):
    """手动计算余弦相似度，展示计算过程"""
    print("🔍 手动计算余弦相似度过程:")
    print(f"向量A: {vec1}")
    print(f"向量B: {vec2}")
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    print(f"点积: {dot_product:.4f}")
    
    # 计算L2范数
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    print(f"||A||: {norm_a:.4f}")
    print(f"||B||: {norm_b:.4f}")
    
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_a * norm_b)
    print(f"余弦相似度: {cosine_sim:.4f}")
    
    return cosine_sim

def demonstrate_similarity_calculation():
    """演示相似度计算"""
    print("=" * 60)
    print("说话人相似度计算演示")
    print("=" * 60)
    
    # 模拟两个说话人的embedding
    print("\n📊 示例1: 高相似度说话人")
    speaker_A = np.array([0.5, 0.8, 0.1, 0.6])
    speaker_B = np.array([0.4, 0.9, 0.2, 0.5])
    
    manual_sim = manual_cosine_similarity(speaker_A, speaker_B)
    sklearn_sim = cosine_similarity([speaker_A], [speaker_B])[0][0]
    print(f"sklearn结果: {sklearn_sim:.4f}")
    print(f"差异: {abs(manual_sim - sklearn_sim):.6f}")
    
    print("\n📊 示例2: 低相似度说话人")
    speaker_C = np.array([0.1, 0.2, 0.9, 0.1])
    speaker_D = np.array([0.9, 0.1, 0.1, 0.8])
    
    manual_sim2 = manual_cosine_similarity(speaker_C, speaker_D)
    sklearn_sim2 = cosine_similarity([speaker_C], [speaker_D])[0][0]
    print(f"sklearn结果: {sklearn_sim2:.4f}")
    
    print("\n📊 示例3: 完全相同的说话人")
    manual_sim3 = manual_cosine_similarity(speaker_A, speaker_A)
    print(f"自相似度: {manual_sim3:.4f}")

def simulate_speaker_embeddings(num_speakers=5, embedding_dim=4):
    """模拟生成说话人embeddings"""
    print(f"\n🎭 模拟 {num_speakers} 个说话人的 {embedding_dim} 维embedding")
    
    # 生成随机embeddings
    np.random.seed(42)  # 确保可重复
    embeddings = {}
    
    for i in range(num_speakers):
        # 模拟真实embedding的分布特性
        embedding = np.random.normal(0.5, 0.2, embedding_dim)
        embedding = np.clip(embedding, 0, 1)  # 限制在[0,1]范围
        embeddings[f'speaker_{i+1}'] = embedding
        print(f"说话人{i+1}: {embedding}")
    
    return embeddings

def compute_similarity_matrix(embeddings):
    """计算完整的相似度矩阵"""
    print("\n📈 计算相似度矩阵...")
    
    speaker_names = list(embeddings.keys())
    embedding_matrix = np.array(list(embeddings.values()))
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    print("相似度矩阵:")
    print("=" * 50)
    
    # 打印表头
    print(f"{'':12}", end='')
    for name in speaker_names:
        print(f"{name:12}", end='')
    print()
    
    # 打印矩阵内容
    for i, name in enumerate(speaker_names):
        print(f"{name:12}", end='')
        for j in range(len(speaker_names)):
            print(f"{similarity_matrix[i][j]:12.4f}", end='')
        print()
    
    return similarity_matrix, speaker_names

def analyze_similarities(similarity_matrix, speaker_names):
    """分析相似度结果"""
    print("\n🔍 相似度分析:")
    print("=" * 40)
    
    # 找到最相似的pair (排除自己和自己)
    max_sim = 0
    max_pair = None
    min_sim = 1
    min_pair = None
    
    n = len(speaker_names)
    for i in range(n):
        for j in range(i+1, n):  # 只看上三角矩阵
            sim = similarity_matrix[i][j]
            if sim > max_sim:
                max_sim = sim
                max_pair = (speaker_names[i], speaker_names[j])
            if sim < min_sim:
                min_sim = sim
                min_pair = (speaker_names[i], speaker_names[j])
    
    print(f"最相似的说话人对: {max_pair[0]} vs {max_pair[1]}")
    print(f"相似度: {max_sim:.4f}")
    print(f"最不相似的说话人对: {min_pair[0]} vs {min_pair[1]}")
    print(f"相似度: {min_sim:.4f}")
    
    # 相似度分布统计
    upper_triangle = similarity_matrix[np.triu_indices(n, k=1)]
    print(f"\n相似度统计:")
    print(f"平均相似度: {np.mean(upper_triangle):.4f}")
    print(f"标准差: {np.std(upper_triangle):.4f}")
    print(f"最大相似度: {np.max(upper_triangle):.4f}")
    print(f"最小相似度: {np.min(upper_triangle):.4f}")

def demonstrate_utterance_aggregation():
    """演示说话人utterance聚合过程"""
    print("\n🎤 演示说话人utterance聚合")
    print("=" * 40)
    
    # 模拟一个说话人的多个utterance embeddings
    utterance_embeddings = [
        np.array([0.5, 0.8, 0.1, 0.6]),
        np.array([0.4, 0.9, 0.2, 0.5]),
        np.array([0.6, 0.7, 0.1, 0.7]),
        np.array([0.3, 0.8, 0.3, 0.4])
    ]
    
    print("原始utterance embeddings:")
    for i, emb in enumerate(utterance_embeddings):
        print(f"  Utterance {i+1}: {emb}")
    
    # 计算平均embedding
    avg_embedding = np.mean(utterance_embeddings, axis=0)
    print(f"\n平均embedding: {avg_embedding}")
    
    # 分析聚合效果
    print(f"\n聚合分析:")
    original_matrix = np.array(utterance_embeddings)
    print(f"原始embedding标准差: {np.std(original_matrix, axis=0)}")
    print(f"平均值: {np.mean(original_matrix, axis=0)}")
    
    # 计算utterance之间的相似度
    print(f"\nUtterance间相似度:")
    for i in range(len(utterance_embeddings)):
        for j in range(i+1, len(utterance_embeddings)):
            sim = cosine_similarity([utterance_embeddings[i]], [utterance_embeddings[j]])[0][0]
            print(f"  Utterance {i+1} vs {j+1}: {sim:.4f}")

def demonstrate_boundary_detection_similarity():
    """演示边界检测中的相似度应用"""
    print("\n🎯 演示边界检测相似度应用")
    print("=" * 40)
    
    # 模拟音频sequence的embeddings
    # 前5个属于说话人A，后5个属于说话人B
    speaker_A_embeddings = [
        np.array([0.8, 0.2, 0.1, 0.9]) + np.random.normal(0, 0.05, 4)
        for _ in range(5)
    ]
    speaker_B_embeddings = [
        np.array([0.2, 0.8, 0.9, 0.1]) + np.random.normal(0, 0.05, 4)
        for _ in range(5)
    ]
    
    all_embeddings = np.array(speaker_A_embeddings + speaker_B_embeddings)
    
    # 计算段中心
    left_center = np.mean(speaker_A_embeddings, axis=0)
    right_center = np.mean(speaker_B_embeddings, axis=0)
    
    print(f"左段中心 (说话人A): {left_center}")
    print(f"右段中心 (说话人B): {right_center}")
    print(f"段间相似度: {cosine_similarity([left_center], [right_center])[0][0]:.4f}")
    
    # 测试不同边界位置的质量
    print(f"\n边界位置评估:")
    for boundary in range(3, 8):  # 测试边界位置3-7
        # 计算左侧与左中心的相似度
        left_sims = [cosine_similarity([emb], [left_center])[0][0] 
                     for emb in all_embeddings[:boundary]]
        left_avg = np.mean(left_sims)
        
        # 计算右侧与右中心的相似度
        right_sims = [cosine_similarity([emb], [right_center])[0][0] 
                      for emb in all_embeddings[boundary:]]
        right_avg = np.mean(right_sims)
        
        # 边界质量得分
        boundary_score = left_avg + right_avg
        
        print(f"  边界位置 {boundary}: 左侧相似度 {left_avg:.4f}, 右侧相似度 {right_avg:.4f}, 总分 {boundary_score:.4f}")

def create_similarity_visualization(similarity_matrix, speaker_names):
    """创建相似度可视化图"""
    try:
        plt.figure(figsize=(10, 8))
        
        # 创建热力图
        sns.heatmap(similarity_matrix, 
                   xticklabels=speaker_names, 
                   yticklabels=speaker_names,
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   square=True)
        
        plt.title('说话人相似度矩阵')
        plt.xlabel('说话人')
        plt.ylabel('说话人')
        plt.tight_layout()
        
        # 保存图片
        output_file = 'examples/extract_and_conclude_similarities/v2/similarity_matrix_demo.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 相似度矩阵可视化已保存: {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠️ 可视化创建失败: {e}")

def main():
    """主函数"""
    print("🎙️ 说话人相似度计算演示程序")
    print("详细展示相似度计算的数学原理和实际应用\n")
    
    # 1. 基础相似度计算演示
    demonstrate_similarity_calculation()
    
    # 2. 模拟说话人embeddings
    embeddings = simulate_speaker_embeddings(num_speakers=6, embedding_dim=8)
    
    # 3. 计算相似度矩阵
    similarity_matrix, speaker_names = compute_similarity_matrix(embeddings)
    
    # 4. 分析相似度结果
    analyze_similarities(similarity_matrix, speaker_names)
    
    # 5. 演示utterance聚合
    demonstrate_utterance_aggregation()
    
    # 6. 演示边界检测应用
    demonstrate_boundary_detection_similarity()
    
    # 7. 创建可视化
    create_similarity_visualization(similarity_matrix, speaker_names)
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("💡 关键要点:")
    print("  1. 相似度使用余弦相似度计算")
    print("  2. 取值范围 [0, 1]，越接近1越相似")
    print("  3. 说话人embedding通过utterance平均获得")
    print("  4. 可用于说话人识别、聚类、边界检测等任务")
    print("=" * 60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
从真实数据计算Feature-PID相关性
Author: AI Assistant
Date: 2025-01-30
"""

import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置期刊风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

colors = {
    'primary': '#0173B2',
    'secondary': '#DE8F05',
    'success': '#029E73',
    'danger': '#D55E00',
    'purple': '#CC78BC',
    'neutral': '#949494',
}

def load_data(filepath='augmented_pid_data_filtered.json'):
    """加载过滤后的数据"""
    print(f"📂 加载数据: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✅ 成功加载 {len(data)} 个样本")
    return data

def extract_features_and_pid(data):
    """提取特征和PID参数"""
    features_list = []
    pid_list = []
    
    for sample in data:
        features = sample['features']
        pid = sample['optimal_pid']
        
        # 提取特征
        feature_vec = [
            features['dof'],
            features['total_mass'],
            features['avg_link_mass'],
            features['total_inertia'],
            features['max_reach'],
            features.get('payload_mass', 0.0),
        ]
        
        # 提取PID
        pid_vec = [
            pid['kp'],
            pid['kd'],
            pid.get('ki', 0.0),  # 有些可能没有ki
        ]
        
        features_list.append(feature_vec)
        pid_list.append(pid_vec)
    
    # 转换为numpy数组
    features_array = np.array(features_list)
    pid_array = np.array(pid_list)
    
    return features_array, pid_array

def calculate_correlation_with_pvalue(features_array, pid_array):
    """
    计算相关性矩阵和p值
    
    返回:
        correlation_matrix: 相关系数矩阵
        pvalue_matrix: p值矩阵
    """
    n_features = features_array.shape[1]
    n_pids = pid_array.shape[1]
    
    correlation_matrix = np.zeros((n_features, n_pids))
    pvalue_matrix = np.zeros((n_features, n_pids))
    
    for i in range(n_features):
        for j in range(n_pids):
            # 计算Pearson相关系数和p值
            corr, pval = pearsonr(features_array[:, i], pid_array[:, j])
            correlation_matrix[i, j] = corr
            pvalue_matrix[i, j] = pval
    
    return correlation_matrix, pvalue_matrix

def plot_correlation_heatmap(correlation_matrix, pvalue_matrix, save_path='feature_correlation_real.png'):
    """绘制相关性热力图"""
    
    features = ['DOF', 'Total Mass', 'Avg Link Mass', 'Total Inertia', 'Max Reach', 'Payload Mass']
    pid_params = ['Kp', 'Kd', 'Ki']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 绘制热力图
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(pid_params)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(pid_params, fontsize=11, fontweight='bold')
    ax.set_yticklabels(features, fontsize=10)
    
    # 在每个格子中显示数值和显著性
    for i in range(len(features)):
        for j in range(len(pid_params)):
            corr = correlation_matrix[i, j]
            pval = pvalue_matrix[i, j]
            
            # 根据p值添加显著性标记
            if pval < 0.001:
                significance = '***'
            elif pval < 0.01:
                significance = '**'
            elif pval < 0.05:
                significance = '*'
            else:
                significance = ''
            
            text_color = 'white' if abs(corr) > 0.5 else 'black'
            ax.text(j, i, f'{corr:.2f}\n{significance}',
                   ha="center", va="center", color=text_color,
                   fontsize=9, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, pad=0.03, shrink=0.95)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20, fontsize=10)
    
    # 设置标题
    n_samples = correlation_matrix.shape[0] * correlation_matrix.shape[1]
    ax.set_title('Correlation Between Robot Features and Optimal PID Parameters\n' + 
                '(232 Filtered Samples, *** p<0.001, ** p<0.01, * p<0.05)',
                fontsize=11, fontweight='bold', pad=15)
    
    ax.set_xlabel('PID Parameters', fontsize=11, fontweight='bold')
    ax.set_ylabel('Robot Features', fontsize=11, fontweight='bold')
    
    # 添加网格
    ax.set_xticks(np.arange(len(pid_params)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(features)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✅ 已保存: {save_path}")
    plt.close()

def print_correlation_analysis(correlation_matrix, pvalue_matrix):
    """打印详细的相关性分析"""
    features = ['DOF', 'Total Mass', 'Avg Link Mass', 'Total Inertia', 'Max Reach', 'Payload Mass']
    pid_params = ['Kp', 'Kd', 'Ki']
    
    print("\n" + "="*80)
    print("📊 Feature-PID 相关性分析（基于真实数据）")
    print("="*80)
    
    for j, pid in enumerate(pid_params):
        print(f"\n🎯 {pid} 的相关性排名:")
        print("-" * 60)
        
        # 创建相关性列表
        corr_list = [(features[i], correlation_matrix[i, j], pvalue_matrix[i, j]) 
                     for i in range(len(features))]
        
        # 按相关性绝对值排序
        corr_list.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feat, corr, pval in corr_list:
            # 显著性标记
            if pval < 0.001:
                sig = '***'
            elif pval < 0.01:
                sig = '**'
            elif pval < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            # 相关性强度描述
            if abs(corr) > 0.7:
                strength = "极强"
            elif abs(corr) > 0.5:
                strength = "强"
            elif abs(corr) > 0.3:
                strength = "中等"
            elif abs(corr) > 0.1:
                strength = "弱"
            else:
                strength = "极弱"
            
            print(f"  {feat:20s}: {corr:+6.3f} {sig:3s}  (p={pval:.4f}) - {strength}相关")
    
    print("\n" + "="*80)
    
    # 统计显著相关的数量
    n_total = correlation_matrix.size
    n_sig_001 = np.sum(pvalue_matrix < 0.001)
    n_sig_01 = np.sum((pvalue_matrix >= 0.001) & (pvalue_matrix < 0.01))
    n_sig_05 = np.sum((pvalue_matrix >= 0.01) & (pvalue_matrix < 0.05))
    n_nonsig = np.sum(pvalue_matrix >= 0.05)
    
    print(f"📈 显著性统计:")
    print(f"  p < 0.001 (***): {n_sig_001}/{n_total} ({n_sig_001/n_total*100:.1f}%)")
    print(f"  p < 0.01  (** ): {n_sig_01}/{n_total} ({n_sig_01/n_total*100:.1f}%)")
    print(f"  p < 0.05  (*  ): {n_sig_05}/{n_total} ({n_sig_05/n_total*100:.1f}%)")
    print(f"  p ≥ 0.05  (ns ): {n_nonsig}/{n_total} ({n_nonsig/n_total*100:.1f}%)")
    print("="*80 + "\n")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔬 从真实数据计算Feature-PID相关性")
    print("="*80 + "\n")
    
    # 1. 加载数据
    data = load_data('augmented_pid_data_filtered.json')
    
    # 2. 提取特征和PID
    features_array, pid_array = extract_features_and_pid(data)
    print(f"\n📊 数据维度:")
    print(f"  特征矩阵: {features_array.shape} (样本数 × 特征数)")
    print(f"  PID矩阵: {pid_array.shape} (样本数 × PID参数数)")
    
    # 3. 计算相关性
    print(f"\n🔍 计算Pearson相关系数和p值...")
    correlation_matrix, pvalue_matrix = calculate_correlation_with_pvalue(features_array, pid_array)
    
    # 4. 打印分析
    print_correlation_analysis(correlation_matrix, pvalue_matrix)
    
    # 5. 绘制热力图
    print("🎨 生成相关性热力图...")
    plot_correlation_heatmap(correlation_matrix, pvalue_matrix)
    
    # 6. 保存数据为CSV供查看
    features = ['DOF', 'Total Mass', 'Avg Link Mass', 'Total Inertia', 'Max Reach', 'Payload Mass']
    pid_params = ['Kp', 'Kd', 'Ki']
    
    df_corr = pd.DataFrame(correlation_matrix, index=features, columns=pid_params)
    df_pval = pd.DataFrame(pvalue_matrix, index=features, columns=pid_params)
    
    df_corr.to_csv('correlation_coefficients.csv')
    df_pval.to_csv('correlation_pvalues.csv')
    
    print(f"\n💾 数据已保存:")
    print(f"  - correlation_coefficients.csv")
    print(f"  - correlation_pvalues.csv")
    
    print("\n✅ 分析完成！")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()


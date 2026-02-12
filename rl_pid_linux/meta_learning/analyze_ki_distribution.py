#!/usr/bin/env python3
"""分析Ki分布"""
import json
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
with open('augmented_pid_data_filtered.json', 'r') as f:
    data = json.load(f)

# 提取Ki
ki_values = [s['optimal_pid'].get('ki', 0.0) for s in data]
ki_array = np.array(ki_values)

print(f"Ki统计:")
print(f"  平均值: {np.mean(ki_array):.4f}")
print(f"  中位数: {np.median(ki_array):.4f}")
print(f"  标准差: {np.std(ki_array):.4f}")
print(f"  最小值: {np.min(ki_array):.4f}")
print(f"  最大值: {np.max(ki_array):.4f}")
print(f"  零值比例: {np.sum(ki_array == 0) / len(ki_array) * 100:.1f}%")

# 绘制分布
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(ki_array, bins=30, edgecolor='black')
plt.xlabel('Ki Value')
plt.ylabel('Frequency')
plt.title('Ki Distribution')

plt.subplot(1, 2, 2)
plt.boxplot(ki_array)
plt.ylabel('Ki Value')
plt.title('Ki Boxplot')
plt.tight_layout()
plt.savefig('ki_distribution.png', dpi=150)
print("\n✅ 已保存: ki_distribution.png")


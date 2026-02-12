#!/usr/bin/env python3
"""
过滤掉Laikago虚拟样本，只保留高质量样本
"""

import json
from pathlib import Path

# 加载优化后的数据
data_path = Path(__file__).parent / 'augmented_pid_data_optimized.json'
with open(data_path, 'r') as f:
    data = json.load(f)

print("=" * 80)
print("过滤Laikago虚拟样本")
print("=" * 80)

print(f"\n原始数据: {len(data)}个样本")

# 统计
types_count = {}
for d in data:
    name = d['name']
    if d['type'] == 'real':
        key = f"真实-{name}"
    elif 'laikago' in name:
        key = "虚拟-Laikago"
    elif 'panda' in name:
        key = "虚拟-Panda"
    else:
        key = "虚拟-KUKA"
    types_count[key] = types_count.get(key, 0) + 1

print("\n样本分布:")
for key, count in sorted(types_count.items()):
    print(f"   {key}: {count}")

# 过滤：保留所有真实样本 + Panda虚拟样本 + KUKA虚拟样本
filtered_data = [
    d for d in data 
    if d['type'] == 'real' or 'laikago' not in d['name'].lower()
]

print(f"\n过滤后: {len(filtered_data)}个样本")
print(f"   排除: {len(data) - len(filtered_data)}个Laikago虚拟样本")

# 统计过滤后的优化误差
errors = [d.get('optimization_error_deg', 0) for d in filtered_data]
import numpy as np

print(f"\n过滤后优化误差统计:")
print(f"   平均: {np.mean(errors):.2f}°")
print(f"   中位: {np.median(errors):.2f}°")
print(f"   最小: {np.min(errors):.2f}°")
print(f"   最大: {np.max(errors):.2f}°")
print(f"   <10°: {sum(1 for e in errors if e < 10)} 样本")
print(f"   10-30°: {sum(1 for e in errors if 10 <= e < 30)} 样本")
print(f"   ≥30°: {sum(1 for e in errors if e >= 30)} 样本")

# 保存过滤后的数据
output_path = Path(__file__).parent / 'augmented_pid_data_filtered.json'
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=2)

print(f"\n💾 过滤后数据已保存: {output_path}")

print("\n" + "=" * 80)
print("✅ 过滤完成！")
print("=" * 80)
print(f"\n🎯 下一步:")
print(f"   python train_with_filtered_data.py")


#!/usr/bin/env python3
"""
验证MAE计算方法的差异
检查Table 1的28.67°和Table 3 per-joint平均7.51°之间的关系
"""

import numpy as np

# Table 3的per-joint MAE数据（Pure Meta-PID）
per_joint_mae_pure = np.array([2.57, 12.36, 4.10, 6.78, 5.41, 4.31, 11.45, 10.23, 10.36])
per_joint_mae_rl = np.array([2.26, 2.42, 3.87, 6.49, 5.32, 4.19, 11.26, 10.19, 10.33])

# Table 1的overall MAE数据
table1_mae_pure = 28.67
table1_mae_rl = 24.88

print("="*80)
print("MAE计算方法验证")
print("="*80)

print("\n【Table 3的per-joint数据】")
print(f"Pure Meta-PID per-joint: {per_joint_mae_pure}")
print(f"Meta-PID+RL per-joint:   {per_joint_mae_rl}")

print("\n【方法1：简单算术平均】")
avg_pure_simple = np.mean(per_joint_mae_pure)
avg_rl_simple = np.mean(per_joint_mae_rl)
print(f"Pure Meta-PID平均: {avg_pure_simple:.2f}°")
print(f"Meta-PID+RL平均:   {avg_rl_simple:.2f}°")
print(f"改进率: {(avg_pure_simple - avg_rl_simple) / avg_pure_simple * 100:.1f}%")

print("\n【Table 1的数据】")
print(f"Pure Meta-PID MAE: {table1_mae_pure:.2f}°")
print(f"Meta-PID+RL MAE:   {table1_mae_rl:.2f}°")
print(f"改进率: {(table1_mae_pure - table1_mae_rl) / table1_mae_pure * 100:.1f}%")

print("\n【差异分析】")
ratio_pure = table1_mae_pure / avg_pure_simple
ratio_rl = table1_mae_rl / avg_rl_simple
print(f"Table 1 / Table 3 比例 (Pure): {ratio_pure:.2f}x")
print(f"Table 1 / Table 3 比例 (RL):   {ratio_rl:.2f}x")

print("\n【可能的解释】")
print(f"1. 如果是RMS而非平均: √(Σ e²/n)")
rms_pure = np.sqrt(np.mean(per_joint_mae_pure**2))
rms_rl = np.sqrt(np.mean(per_joint_mae_rl**2))
print(f"   Pure: {rms_pure:.2f}°, RL: {rms_rl:.2f}° (还是不匹配)")

print(f"\n2. 如果是加权平均（按关节负载权重）:")
print(f"   可能J2、J7-J9等关节有更高权重")

print(f"\n3. 如果是不同的测试数据:")
print(f"   Table 1可能使用了不同的测试episode或更长的时间序列")

print(f"\n4. 如果Table 1的MAE是joint space norm的平均:")
print(f"   MAE_t = mean_over_time(||e_t||_1 / n)")
print(f"   但这仍应该等于simple average")

print("\n" + "="*80)
print("⚠️  警告：Table 1和Table 3的数据不一致！")
print("建议检查：")
print("1. 这两个表的数据是否来自同一次测试？")
print("2. MAE的定义是否在论文中有明确说明？")
print("3. 是否需要重新运行测试以确保数据一致性？")
print("="*80)

